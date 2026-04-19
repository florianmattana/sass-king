# Kernel 09 warp shuffle and vote (warp-level communication primitives)

First chapter covering warp-synchronous communication. Introduces the SHFL family for value exchange between threads, the VOTE and MATCH families for warp-wide consensus, and the synchronization primitives `__syncwarp` and `__activemask`. Twelve variants tested to map every warp-level communication primitive available on SM120.

Hardware warp reduction (REDUX, added in Ampere) is covered separately in chapter 10 because it belongs to a different class of operation: arithmetic reduction rather than communication.

## Source (baseline 09a)

```cuda
__global__ void warp_reduce(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? a[i] : 0.0f;

    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);

    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = val;
    }
}
```

Classic butterfly reduction: 5 SHFL.BFLY stages, one lane per warp stores the partial sum.

## SASS main body structure (09a, 30 instructions)

```
[Prologue: 7 instructions]
  Standard 8-instruction skeleton (see FINDINGS.md "Canonical prologue").
  Note: BSSY.RECONVERGENT inserted after prologue BEFORE the ID computation
  because ptxas foresees a divergent bounds check.

[BSSY opens reconvergence scope: 1 instruction]
  BSSY.RECONVERGENT B0, 0xf0    ; reconvergence target at 0xf0

[Val initialization: 1 instruction]
  HFMA2 R2, -RZ, RZ, 0, 0        ; val = 0.0f (fallback for out-of-bounds threads)

[Bounds check: 2 instructions]
  IMAD R5, R6, UR6, R9
  ISETP.GE.AND P0, PT, R5, UR4, PT
  @P0 BRA 0xe0                   ; out-of-bounds threads skip LDG

[Conditional global load: 3 instructions]
  LDC.64 R2, c[0x0][0x380]
  IMAD.WIDE R2, R5, 0x4, R2
  LDG.E R2, desc[UR4][R2.64]

[BSYNC closes reconvergence: 1 instruction]
  BSYNC.RECONVERGENT B0          ; all lanes of warp are now active

[5 SHFL.BFLY + 5 FADD: 10 instructions]
  SHFL.BFLY PT, R3, R2, 0x10, 0x1f ; FADD R3, R3, R2
  SHFL.BFLY PT, R0, R3, 0x8,  0x1f ; FADD R0, R3, R0
  SHFL.BFLY PT, R5, R0, 0x4,  0x1f ; FADD R5, R0, R5
  SHFL.BFLY PT, R4, R5, 0x2,  0x1f ; FADD R4, R5, R4
  SHFL.BFLY PT, R7, R4, 0x1,  0x1f

[Lane-zero test: 1 instruction]
  LOP3.LUT P0, RZ, R9, 0x1f, RZ, 0xc0, !PT   ; P0 = (tid & 31) != 0

[Predicated exit: 1 instruction]
  @P0 EXIT                       ; non-lane-0 threads exit

[Warp-id derivation and store address: 4 instructions]
  SHF.R.U32.HI R0, RZ, 0x5, R6   ; blockDim.x / 32
  SHF.R.U32.HI R5, RZ, 0x5, R9   ; tid / 32
  IMAD R5, R0, UR6, R5           ; warp_id_global
  IMAD.WIDE.U32 R2, R5, 0x4, R2  ; &c[warp_id_global]

[Store and exit: 2 instructions]
  STG.E desc[UR4][R2.64], R7
  EXIT
```

## Solid observations (verified across variants 09a to 09m)

### 1. Four distinct SHFL opcodes on SM120

| Intrinsic | Opcode | Format | Notes |
|---|---|---|---|
| `__shfl_sync` | SHFL.IDX | `Pdst, Rdst, Rsrc, lane_source, 0x1f` | Broadcast from a specific lane |
| `__shfl_xor_sync` | SHFL.BFLY | `Pdst, Rdst, Rsrc, xor_mask, 0x1f` | Butterfly pattern |
| `__shfl_up_sync` | SHFL.UP | `Pdst, Rdst, Rsrc, delta, RZ` | Lane N reads lane N-delta |
| `__shfl_down_sync` | SHFL.DOWN | `Pdst, Rdst, Rsrc, delta, 0x1f` | Lane N reads lane N+delta |

**Asymmetric encoding in SHFL.UP.** The 5th operand is `RZ` for UP but `0x1f` for IDX, BFLY, DOWN. The other three shuffle variants encode the segment mask (width of the shuffle group). UP does not, likely because of its distinct out-of-range semantics: when lane N does not have a valid lane N-delta (small N), the lane keeps its original value. A separate mode bit is needed.

**Variants are physically distinct opcodes**, not a single SHFL with mode bits. The opcode encoding visible in the instruction bytes differs:
* SHFL.BFLY: `0x...f7f89` pattern
* SHFL.IDX: `0x...f7589` pattern
* SHFL.UP: `0x...7989` pattern
* SHFL.DOWN: `0x...f7f89` pattern (same family as BFLY)

**Practical use cases.** Each SHFL variant has a canonical use:
* BFLY: warp reductions in log2(N) stages
* IDX: broadcast a value (often from lane 0 of the warp)
* DOWN: shift values toward lower lane ids (for warp-wide scan)
* UP: shift values toward higher lane ids (also for warp-wide scan)

### 2. BSSY.RECONVERGENT and BSYNC.RECONVERGENT are conditional

Kernel 09a had a bounds check before the SHFL sequence, producing:

```
BSSY.RECONVERGENT B0, 0xf0
...
@P0 BRA 0xe0
...
BSYNC.RECONVERGENT B0
SHFL.BFLY ...
```

Kernel 09b removed the bounds check. The BSSY/BSYNC pair disappeared entirely. SHFL instructions executed directly after LDG without reconvergence wrappers.

**Resolved.** `BSSY.RECONVERGENT` and `BSYNC.RECONVERGENT` are emitted only when SHFL (or any warp-synchronous instruction) follows code that may have partial thread participation. They ensure the full warp is reconverged before the next warp-level operation.

The immediate operand of BSSY (0xf0 in 09a) is the address of the reconvergence point, which is where the matching BSYNC lives. This pair is the SASS-level implementation of the Independent Thread Scheduling reconvergence mechanism.

### 3. Initialization of `val = 0.0f` via HFMA2

Kernel 09a uses the trick observed earlier in kernels 04 and 05:

```
HFMA2 R2, -RZ, RZ, 0, 0
```

Loads `0.0f` into R2 via the FP16 packed FMA pipeline. Used before the divergent LDG so that out-of-bounds threads have a defined value for the subsequent SHFL.

**Why this matters for SHFL.** SHFL requires all lanes active, including those that would have exited via `@P0 EXIT` in a simpler kernel. Here ptxas cannot let them exit (the SHFL would stall waiting for them), so they must stay and participate with a defined (zero) value. Without the zero init, their register would hold garbage and corrupt the reduction.

This is the pattern to recognize for any warp-synchronous kernel with input bounds: HFMA2 zero-init BEFORE the bounds check, followed by BSSY/BSYNC wrapping the divergent section.

### 4. VOTE has two forms with different destination types

| Form | SASS | Destination |
|---|---|---|
| Predicate form | `VOTE.{ANY,ALL} Pdst, Psrc` | A predicate (true/false) |
| Register form | `VOTE.ANY Rdst, Pmask, Psrc` | A register (mask of threads with true predicate) |

**The register form takes two predicates, not one.** The first predicate `Pmask` is the set of lanes to poll (usually `PT` = all active). The second predicate `Psrc` is the per-lane condition being voted on.

Intrinsic-to-form mapping:
* `__any_sync(mask, pred)` → predicate form `VOTE.ANY P, P`
* `__all_sync(mask, pred)` → predicate form `VOTE.ALL P, P`
* `__ballot_sync(mask, pred)` → register form `VOTE.ANY R, PT, P`
* `__activemask()` → register form `VOTE.ANY R, PT, PT`

**Resolved.** VOTE.ANY and VOTE.ALL are distinct SASS opcodes, not the same opcode with a mode bit. Confirmed by side-by-side emission in kernel 09h. VOTE.ALL has no register form in our observations (would require a different intrinsic to test).

**No dedicated BALLOT or ACTIVEMASK opcode.** Both are syntactic sugar for VOTE.ANY register form.

### 5. MATCH opcodes produce different result shapes

| Intrinsic | SASS | Output |
|---|---|---|
| `__match_any_sync` | `MATCH.ANY Rdst, Rsrc` | Register: mask of lanes with matching value |
| `__match_all_sync` | `MATCH.ALL Pdst, Rdst, Rsrc` | Predicate (all match) + register (mask) |

**MATCH.ALL is the first dual-output instruction observed in this project.** A single SASS instruction writes both a predicate (indicating whether all lanes have the same value) and a register (holding the mask). The CUDA intrinsic signature `__match_all_sync(mask, value, int* pred)` reflects this dual output.

### 6. `__syncwarp` has no dedicated SASS opcode on SM120

Kernel 09e inserted `__syncwarp()` between each SHFL. Result:
* First `__syncwarp()` (before the first SHFL): replaced by **6 NOPs** at addresses 0x00a0 to 0x00f0.
* All subsequent `__syncwarp()` calls: eliminated entirely, producing no SASS output.

Kernel 09k used `__syncwarp(0x0000ffff)` with a partial mask. ptxas emitted:
* `MOV R11, 0xffff` loading the mask into R11, but R11 is never consumed afterwards.
* A single NOP before the SHFL.
* The subsequent SHFL.BFLY still used segment mask `0x1f` (full warp), ignoring the partial mask.

**Conclusion.** On SM120, `__syncwarp` is a scheduling hint, not a real instruction. ptxas converts it to NOP padding when necessary (to stall until a prior memory operation completes) or eliminates it entirely. The mask argument is ignored at the SASS level for full-warp synchronization.

**Why 6 NOPs specifically?** The LDG that feeds the first SHFL has a variable latency. Without `__syncwarp`, ptxas interleaves useful work (like LOP3 or IMAD.WIDE) between LDG and SHFL to hide the latency. With `__syncwarp`, ptxas is forced to place a barrier and fills the space with NOPs. The count of 6 matches approximately the LDG-to-use latency that ptxas would otherwise have hidden.

**Practical consequence.** Avoid inserting `__syncwarp()` redundantly. It can only slow kernels down and cannot accelerate them. The sync semantics are implicit in SHFL, VOTE, and MATCH on SM120.

### 7. 64-bit SHFL is two 32-bit SHFL instructions

Kernel 09m used `__shfl_sync` on a `double` (64-bit). ptxas emitted:

```
SHFL.IDX PT, R5, R3, RZ, 0x1f    // upper half (R3 from R2:R3)
SHFL.IDX PT, R4, R2, RZ, 0x1f    // lower half
DADD R4, R2, R4
STG.E.64 desc[UR4][R6.64], R4
```

**Observations on order.** The high half is shuffled first, then the low half. This is the reverse of what a simple read might suggest. Hypothesis: ptxas emits the high half first because R3 is the source producing R5, which is used in the DADD only after R4 is ready. Scheduling the independent high-half shuffle earlier hides its latency behind the low-half shuffle and the subsequent DADD.

**No SHFL.IDX.64 or wider SHFL variant.** 64-bit values are always split. A similar rule likely applies to BFLY, UP, DOWN (not tested directly with 64-bit operands).

### 8. REDUX writes to the uniform register file

From kernel 09i (briefly shown here, full treatment in chapter 10):

```
REDUX.SUM.S32 UR7, R2      // per-thread input, uniform output
MOV R5, UR7                 // copy back to per-thread register
```

**First observation of a cross-register-file write.** REDUX takes a per-thread register input (each lane's value) and writes a uniform register output (the reduced scalar, identical across the warp). This is architecturally elegant because the result of a reduction is by definition uniform.

Downstream per-thread consumers need a MOV from UR back to R. If the consumer is another uniform operation, no MOV is needed.

### 9. Scheduling patterns around SHFL

ptxas consistently interleaves useful work between LDG and SHFL. In kernel 09b (no bounds check), the sequence is:

```
LDG.E R2, desc[UR4][R2.64]
LOP3.LUT P0, RZ, R11, 0x1f, RZ, 0xc0, !PT   ; lane-zero test, hidden behind LDG
SHFL.BFLY PT, R5, R2, 0x10, 0x1f
```

The LOP3 between LDG and SHFL serves two purposes: it does useful work (computes the lane-zero predicate for the later EXIT), and it hides LDG latency before the SHFL consumes R2.

In kernel 09d (VOTE with ballot), a similar pattern:

```
LDG.E R2, desc[UR4][R2.64]
LOP3.LUT P1, ...              ; independent work
FSETP.GT.AND P0, PT, R2, RZ, PT
VOTE.ANY R5, PT, P0
```

**VOTE is hoisted when it has no data dependency.** In kernel 09l (`__activemask`), VOTE.ANY appears at address 0x0020, before the pointer loads at 0x0050:

```
/*0010*/  S2R R7, SR_TID.X
/*0020*/  VOTE.ANY R5, PT, PT      ; __activemask, no dependency, placed early
/*0030*/  LOP3.LUT P0, ...
...
/*0050*/  LDC R0, c[0x0][0x360]    ; pointer loads later
```

This is unusual because LDC normally happens in the first few instructions. Here ptxas prioritizes emitting VOTE early to start its latency clock as soon as possible.

### 10. Secondary observations

**`IMAD.WIDE.U32` variant.** New SASS modifier observed in kernels with warp-id arithmetic. Used specifically where ptxas can prove the index is non-negative (warp_id derived from tid/32). Appears in 09a, 09b, 09d, 09h, 09i, 09j, 09l. Not seen in earlier kernels because they did not have warp-id arithmetic in the output path.

**`LOP3.LUT P, ..., 0xc0, !PT` as lane-zero idiom.** This exact encoding appears in almost every kernel of chapter 09. The LUT value `0xc0` combined with `!PT` in the predicate input position encodes `(AND, then test zero)`. Single instruction fuses AND with compare-to-zero.

**Division by 32 never uses modulo.** Throughout chapter 09, `threadIdx.x / 32` and `blockDim.x / 32` compile to `SHF.R.U32.HI R, RZ, 0x5, R`. Consistent with the power-of-2 division rule from kernel 06. Reminds us that ptxas is good at this optimization; warp-id arithmetic is always cheap.

**Dead MOV in kernel 09k.** `MOV R11, 0xffff` loads the `__syncwarp` partial mask into R11, but R11 is never consumed. Possibly dead code, possibly a scheduler hint. Not fully explained.

## Variants summary with instruction count and compute ratio

| Variant | Primitive | Total instr | Compute (useful) | Ratio | Notes |
|---|---|---|---|---|---|
| 09a | SHFL.BFLY + bounds | 30 | 5 FADD | 17% | BSSY/BSYNC present |
| 09b | SHFL.BFLY no bounds | 21 | 5 FADD | 24% | BSSY/BSYNC absent, cleaner |
| 09c | SHFL.IDX broadcast | 14 | 1 FADD | 7% | Single shuffle + add |
| 09d | VOTE ballot | 18 | 0 | 0% | Vote output, no compute |
| 09e | __syncwarp full | 27 | 5 FADD | 19% | 6 NOPs added before first SHFL |
| 09f | SHFL.UP | 14 | 1 FADD | 7% | 5th operand is RZ |
| 09g | SHFL.DOWN | 14 | 1 FADD | 7% | 5th operand is 0x1f like BFLY |
| 09h | VOTE.ALL + VOTE.ANY | 23 | 0 | 0% | Two vote opcodes confirmed |
| 09j | MATCH.ANY + MATCH.ALL | 20 | 0 | 0% | MATCH.ALL dual output |
| 09k | __syncwarp(0xffff) | 16 | 0 | 0% | Dead MOV of mask |
| 09l | __activemask alone | 14 | 0 | 0% | VOTE hoisted to start |
| 09m | SHFL.IDX on double | 16 | 1 DADD | 6% | 2 SHFL for 64-bit |

Compute ratio is (FADD or DADD count) / (total useful instructions). For pure communication kernels (vote, match) the compute ratio is 0 by design, which is why they are typically combined with useful downstream work.

## What this chapter adds to SASS reading skills

**Identify warp-level primitives at a glance.** Opcodes starting with SHFL, VOTE, MATCH, or preceded by BSSY/BSYNC.RECONVERGENT signal warp-synchronous code. The specific variant tells the communication pattern.

**Recognize the `VOTE.ANY R, PT, PT` idiom for `__activemask`.** This exact encoding (both predicates always-true) is syntactic sugar, not a real query.

**Spot `__syncwarp` as NOP padding.** Isolated NOPs clustered before a SHFL or VOTE often indicate a forced `__syncwarp`. If the NOPs are not necessary (data dependency already resolved), they are pure overhead.

**Detect 64-bit warp ops.** Two consecutive SHFL instructions with adjacent source registers signal a 64-bit shuffle. The 32-bit halves are transferred separately, often with the high half emitted first.

**Distinguish predicate-form from register-form VOTE.** Predicate output (`VOTE.ANY P, P`) for boolean aggregations, register output (`VOTE.ANY R, PT, P`) for ballots. The latter takes two predicate operands.

**Understand the "HFMA2 zero init + BSSY wrap" pattern for warp-sync after bounds.** Any kernel that combines a bounds check with SHFL/VOTE/MATCH must keep all 32 lanes alive for the warp-sync instruction. HFMA2 initializes the value to zero as a safe fallback, and BSSY/BSYNC enforce reconvergence.

## Practical consequences for kernel writers

* **Warp reduction of int32?** Use `__reduce_add_sync` (REDUX, 1 instruction). Described in chapter 10.
* **Warp reduction of float?** Use the 5-stage butterfly with `__shfl_xor_sync`. No hardware shortcut exists for FP.
* **Broadcasting a value from lane 0 to the warp?** Use `__shfl_sync(mask, val, 0)`, compiles to one SHFL.IDX.
* **Warp-wide scan?** Use `__shfl_up_sync` (exclusive scan pattern) or `__shfl_down_sync` (reverse scan), in log2(N) stages.
* **Divergence detection?** Use `__activemask()`, compiles to one VOTE.ANY with trivial inputs.
* **Avoid `__syncwarp()` between warp-synchronous operations.** It produces NOP padding that prevents ptxas from interleaving useful work. Rely on the implicit synchronization of SHFL/VOTE/MATCH.
* **Divergence before a warp-sync op?** Initialize the value to a safe fallback (zero or identity for your reduction) using HFMA2 or IMAD before the divergent code. ptxas will insert BSSY/BSYNC automatically.
* **64-bit shuffles cost 2 SHFL.** If the 64-bit value can be split and worked on as two 32-bit halves in parallel, the shuffle cost is the same.

## Hypotheses opened or deferred

1. **Asymmetric encoding of SHFL.UP.** The 5th operand is `RZ` instead of `0x1f`. Is this purely encoding convention, or does it produce a different memory order / clamp behavior? Would require comparing cycle-accurate behavior of UP vs DOWN for the same shuffle distance.
2. **Dead MOV in `__syncwarp(partial_mask)`.** Kernel 09k emitted `MOV R11, 0xffff` that is never consumed. Is it truly dead code (optimization residue), or does the hardware decoder read the immediate as a scheduler hint even without a subsequent register consumer? Not verified.
3. **Pipeline assignment for SHFL.** Documentation says LSU, consistent with the scoreboard usage visible in the control codes. Not confirmed by gpuasm annotation in this session.
4. **Does `WARPSYNC` opcode exist on SM120?** The gist herrmann SASS evolution shows WARPSYNC in the Volta-Turing addition. We never observed it on SM120, even with partial masks. Either ptxas does not emit it, or it was removed. To verify by constructing a deliberately divergent kernel where a partial-mask sync is unavoidable.
5. **Behavior of SHFL under genuine divergence.** All our tests had either full-warp participation or a simple bounds check. Complex divergence patterns (half the warp taking one path, half another, each needing its own SHFL) not tested.
6. **Order of 64-bit SHFL halves.** Kernel 09m emitted high half first. Is this deterministic, or does ptxas make this choice based on register availability?

## New instructions and idioms observed in this chapter

| Opcode / Idiom | Usage |
|---|---|
| SHFL.BFLY | Butterfly shuffle (XOR pattern) |
| SHFL.IDX | Broadcast from specified lane |
| SHFL.UP | Lane N reads lane N-delta (asymmetric 5th operand) |
| SHFL.DOWN | Lane N reads lane N+delta |
| VOTE.ANY (predicate form) | Predicate OR across warp |
| VOTE.ALL (predicate form) | Predicate AND across warp |
| VOTE.ANY (register form) | Ballot with two predicate inputs (mask, condition) |
| MATCH.ANY | Register output: mask of matching lanes |
| MATCH.ALL | Dual output: predicate (all match) + register (mask) |
| BSSY.RECONVERGENT | Declare reconvergence scope with target address |
| BSYNC.RECONVERGENT | Close reconvergence scope, reactivate all lanes |
| IMAD.WIDE.U32 | Unsigned-explicit 64-bit multiply-add |
| `HFMA2 R, -RZ, RZ, 0, 0` | Load 0.0f into FP32 register (safety init pattern) |
| `VOTE.ANY R, PT, PT` | Idiom for `__activemask()` |
| `LOP3.LUT P, RZ, R, 0x1f, RZ, 0xc0, !PT` | Idiom for `(x & 0x1f) == 0` test |