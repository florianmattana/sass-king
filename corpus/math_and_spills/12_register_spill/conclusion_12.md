# Kernel 12: Register spill (STL, LDL, stack frame, local memory)

## Scope

Eleven variants exploring when and how ptxas emits register spill on SM120 with CUDA 13.2. The chapter establishes the boundary between "ptxas finds a way without spilling" and "ptxas spills to local memory", documents the SASS shapes that appear in each case, and catalogs the spill-related opcodes.

## Variants

| Variant | Source | Flag | Spill observed |
|---|---|---|---|
| 12a | 10 FFMA accumulators, scalar | none | No |
| 12b | same | `-maxrregcount=32` | No (SASS identical to 12a) |
| 12c | same | `-maxrregcount=24` | No (SASS identical to 12a) |
| 12d | same | `-maxrregcount=16` | No — ptxas warns and raises floor to 24 |
| 12e | 16 FFMA accumulators in loop | none | No (uses up to R34) |
| 12f | same | `-maxrregcount=24` | No — ptxas restructures the loop instead |
| 12g | 12 array pointers | none | No (uses up to R30) |
| 12h | `__noinline__` function, 3 args | `-maxrregcount=24` | No (function call without register preservation) |
| 12i | 32 FFMA accumulators in loop | `-maxrregcount=24` | **Yes — massive spill** |
| 12j | 16 float arguments to CALL | none | No (16 arguments passed in registers) |
| 12k | `int arr[64]` local array | none | **Yes — STL.128 vectorized init + LDL reads** |

## Major structural findings

### Finding 1: SM120 has a floor of 24 registers per thread

When compiling with `-maxrregcount=16`, ptxas emits the warning:
```
ptxas warning : For profile sm_120 adjusting per thread register count of 16 to lower bound of 24
```
and silently produces a 24-register binary. The resulting SASS is byte-identical to the same kernel compiled at `-maxrregcount=24` (or `=32`, or no flag at all, for the simpler 12a-d kernels).

This establishes an architectural invariant: **24 registers per thread is the minimum ptxas will allocate on SM120 profile**. Requests below 24 are silently adjusted.

### Finding 2: Ptxas restructures before spilling

Comparing 12e (no flag, 16-accumulator loop) to 12f (same source, `-maxrregcount=24`):
* 12e uses up to R34 naturally and does not spill.
* 12f uses up to R23 exactly and does not spill — instead ptxas restructures the loop body to reduce live range overlap, using a different scheduling that keeps peak register usage within the budget.

This means `-maxrregcount` acts as a hint that triggers restructuring first, spill only as last resort. Ptxas prefers to change the algorithm's shape than to inject local memory traffic.

### Finding 3: Argument passing through CALL does not require spill

Kernel 12j passes 16 float arguments to a `__noinline__` function. Expected: spill to local memory for argument passing. Observed: the 16 arguments are passed entirely in registers R0-R23 (plus the return address in R12). No STL, no LDL. The callee reads arguments from the exact register positions the caller prepared.

**This is not a stack-based ABI**. Ptxas coordinates the register layout globally between caller and callee within the same compilation unit. On GPU with 255 registers per thread, there is usually enough register budget for arbitrary argument counts.

### Finding 4: Static local arrays are always spilled

Kernel 12k's `int arr[64]` (256 bytes) cannot fit in registers and is allocated in local memory. Ptxas decrements the stack pointer (R1) by exactly 256 bytes at kernel entry and accesses the array via LDL / STL / STL.128. This is deterministic — any array larger than the register budget spills to local memory.

### Finding 5: Spill preserves the unroll

In 12i (32 accumulators, 24-register limit), ptxas keeps the 4× loop unroll AND spills. The result is a mixture of inline FFMA chains and STL / LDL / LDL.LU operations rolling accumulators through a small register window. The unroll multiplies the kernel size by 10× (from ~50 to ~550 instructions) but preserves the ILP structure.

## Spill mechanics: dissecting kernel 12i

### Stack frame allocation

```
/*0040*/ IADD R1, R1, -0x128 ;           // allocate 0x128 = 296 bytes on stack
```

The stack pointer R1 (initialized from `c[0x0][0x37c]` in the prologue) is decremented to carve out a local-memory frame. Frame size is determined by ptxas based on the number of values that need to be spilled plus alignment padding.

Observed frame sizes in the project so far:
* 12i: 0x128 (296 bytes) for 32 FFMA accumulators
* 12k: 0x100 (256 bytes) for `int arr[64]`

[HYP] Frame alignment is on 8-byte boundaries at minimum. Both observed sizes are multiples of 8 (0x128 = 296 = 37×8, 0x100 = 256 = 32×8). Not yet confirmed on smaller sizes.

### STL and LDL opcodes

Three spill-related opcodes observed:

* **`STL [R1+offset], R`** — store one 32-bit register to local memory.
* **`LDL R, [R1+offset]`** — load one 32-bit register from local memory.
* **`LDL.LU R, [R1+offset]`** — load with "last use" hint. [HYP] `.LU` signals to the local memory cache that this line will not be reused, allowing eviction after the read. Observed systematically on loads whose destination is consumed once by the next FFMA and then overwritten.
* **`STL.128 [R1+offset], R`** — vectorized store of 4 consecutive 32-bit registers. Observed in 12k for initializing `int arr[64]` four elements at a time.

### Rolling window pattern

The canonical spill pattern observed in 12i is a "rolling window": at each FFMA, one accumulator is spilled (STL) and another is reloaded (LDL), while the FFMA computes. Example from 12i around 0x04a0-0x04e0:

```
STL  [R1+0x38], R7       ; spill R7 (result of previous FFMA)
FFMA R6, R21, R5, R6     ; compute with already-live R6, R21, R5
LDL.LU R7, [R1+0x11c]    ; reload next accumulator into R7
STL  [R1+0x34], R6       ; spill R6 (result just computed)
FFMA R5, R21, R4, R5     ; compute with already-live R5, R21, R4
LDL.LU R6, [R1+0x118]    ; reload next accumulator into R6
STL  [R1+0x30], R5       ; spill R5
FFMA R4, R21, R3, R4
LDL.LU R5, [R1+0x114]
...
```

This is effectively **pipelined spill**: at steady state, there is exactly one STL and one LDL per FFMA, keeping the 24-register window full at all times. Ptxas does not gather the spills into batches — it interleaves them with compute to hide the local memory latency.

### Frame region layout

In 12i, the 0x128-byte frame is divided into regions with distinct purposes:

* **`[R1+0x00]` to `[R1+0x3c]`** — rotating accumulator storage (16 slots × 4 bytes). The primary spill-and-reload region for the FFMA chain.
* **`[R1+0x40]` to `[R1+0x68]`** — secondary accumulator region for the second half of the unrolled body (the algorithm unrolls by 4 so two "banks" of accumulators are needed).
* **`[R1+0x6c]`** — persistent save of R21 (= `a[i] + (float)k`, the loop variable). Not recomputed on every iteration.
* **`[R1+0x70]` to `[R1+0xd4]`** — scratch staging for register swaps during the unroll rollover. Ptxas stores the "next iteration" accumulators here before they are rolled into the live set.
* **`[R1+0xd8]` to `[R1+0x124]`** — more scratch for the rollover, upper half.

The regions are semantically distinct even though they share the same stack frame — ptxas assigns disjoint offsets to values that have different lifetimes.

### Register assignment under pressure

In 12i, certain registers are **never spilled**:
* R1: stack pointer (never touched after the initial IADD)
* R0: `blockDim.x * blockIdx.x + tid` (computed once, used at the end for STG)
* R2: pointer to `a[i]` (stays live for the full kernel)
* R17: `val` (result of LDG.E, read early and consumed only at STG)
* R21: `x = v + (float)k` (loop-variant value computed fresh each iteration)

Ptxas prioritizes keeping pointers and small-number-of-uses values in registers, spilling the accumulators which have many future uses but always in predictable positions (enabling clean spill slots).

### Kernel size explosion

Comparison:
* 12e (16 accumulators, no spill): ~300 instructions
* 12i (32 accumulators with spill): ~550 instructions
* 12a (10 FFMA chain, no loop, no spill): ~50 instructions

The ratio is roughly:
* **Spill overhead per accumulator per iteration**: 2 instructions (1 STL + 1 LDL)
* **Kernel size multiplier due to spill**: 1.5× to 2× for a moderate-pressure loop kernel

## Local array spill: dissecting kernel 12k

### R2UR conversion opcode

```
/*0040*/ IADD R1, R1, -0x100 ;
/*0050*/ R2UR UR7, R1 ;
```

**`R2UR UR7, R1`** — new opcode. Copies per-thread register R1 into uniform register UR7. Observed specifically for the stack pointer.

[HYP] Why copy R1 (per-thread) into a uniform register? Because the LDL / STL addressing mode supports `[R_index + UR_base]` but perhaps not `[R_index + R_base]` for register combinations. Or because ptxas knows that in practice, each thread's stack pointer (R1) will be at the same offset from the lane-0 stack pointer and can be treated as "effectively uniform" for addressing purposes. Needs more data points to confirm.

### Vectorized spill with STL.128

```
/*0180*/ STL.128 [R1], R4 ;          // stores R4, R5, R6, R7
/*0200*/ STL.128 [R1+0x10], R20 ;    // stores R20, R21, R22, R23
/*0260*/ STL.128 [R1+0x20], R24 ;    // stores R24, R25, R26, R27
...
```

**`STL.128 [addr], R`** — stores 4 consecutive registers as a 128-bit transaction. Used for initializing `int arr[64]` by computing 4 values at a time in R4..R7 or R20..R23 then spilling them together.

This parallels LDG.E.128 / STG.E.128 for global memory. The 128-bit width reduces the number of LSU transactions by 4× compared to 4 individual STL operations.

### Indexed load pattern

```
LDL R24, [R37+UR7] ;
LDL R25, [R35+UR7] ;
LDL R7,  [R33+UR7] ;
...
```

Runtime-indexed access to `arr[(idx + k) & 0x3f]` becomes `LDL R, [R_byteoffset + UR_base]` where:
* `UR_base = UR7` = the uniform copy of the stack pointer
* `R_byteoffset` = per-thread offset computed as `(index & 0x3f) * 4`

The mask-and-shift step is visible in the SASS as:
```
LOP3.LUT R, R_idx, 0xfc, RZ, 0xc0, !PT  ; R = R_idx AND 0xfc (aligns to 4 bytes, masks mod 64)
```

`0xfc = 11111100b` simultaneously (a) aligns to 4-byte boundary and (b) keeps only the low 6 bits masked with 0xfc = limits to 64 × 4 byte range. Elegant compilation of `(idx & 0x3f) * 4`.

## Argument passing through CALL: dissecting kernel 12j

The `__noinline__` function receiving 16 floats does not trigger spill. Observations:

```
/*00c0*/ FADD R7, R0, 1 ;           // arg0 (R0 holds v)
/*00d0*/ FADD R9, R0, 2 ;           // arg1
/*00e0*/ FADD R4, R0, 3 ;
/*00f0*/ FADD R11, R0, 4 ;
/*0100*/ FADD R13, R0, 5 ;
/*0110*/ FADD R6, R0, 6 ;
/*0120*/ FADD R15, R0, 7 ;
/*0130*/ FADD R17, R0, 8 ;
/*0140*/ FADD R8, R0, 9 ;
/*0150*/ FADD R19, R0, 10 ;
/*0160*/ FADD R3, R0, 11 ;
/*0170*/ FADD R2, R0, 12 ;
/*0180*/ FADD R21, R0, 13 ;
/*0190*/ FADD R23, R0, 14 ;
/*01a0*/ FADD R10, R0, 15 ;
/*01b0*/ MOV R12, 0x1d0 ;            // return address in R12
/*01c0*/ CALL.REL.NOINC 0x220 ;
```

Sixteen arguments end up in R0, R2, R3, R4, R6, R7, R8, R9, R10, R11, R13, R15, R17, R19, R21, R23 (non-consecutive register numbers, interleaved around R12 which holds the return address). The callee body:

```
/*0220*/ FFMA R4, R4, R11, R13 ;     // reads R4, R11, R13
/*0230*/ FFMA R7, R0, R7, R9 ;       // reads R0, R7, R9
/*0240*/ FFMA R6, R6, R15, R17 ;
...
```

The callee reads directly from these registers. There is no prolog or epilog in the callee related to argument unpacking or register save/restore — the "ABI" is simply "caller places arguments where callee reads them".

**Implication.** The term "caller-saved" and "callee-saved" from classical ABIs does not apply. Ptxas performs a single-pass global register allocation that considers all functions in the compilation unit and places values where they are needed. There is no convention; there is only optimal assignment.

[HYP] This might break if the callee is in a separate compilation unit (separate `.cu` file with `__device__` function, compiled with `-rdc=true`). In that case, ptxas can't see both sides, and a classical ABI must apply. Not tested.

## Anti-spill strategies ptxas uses

Before spilling, ptxas applies these techniques in order of preference:

1. **Recomputation.** A value that is cheap to recompute (e.g., `(float)k` in the loop) is not spilled; it is recomputed each time it is needed. Observed in 12e where the counter is re-converted with UI2FP.F32.U32 inside the loop rather than cached in a register.

2. **Restructuring.** Change the algorithm's scheduling to reduce live range overlap. Observed in 12f: the "round-robin" FFMA chain is rewritten so that each iteration writes to only one accumulator at a time, reducing peak register usage. The result is semantically equivalent but the live set shrinks.

3. **Register file reuse.** Dead registers are recycled across semantic boundaries (see chapter 02 liveness observations). Ptxas maximizes this before spilling.

4. **Global allocation across CALL.** Caller and callee share the register namespace (see 12j). No artificial ABI boundary forces a spill.

5. **Unroll reduction.** As a last resort before spilling, ptxas could reduce the unroll factor, but this was not observed in 12i — ptxas chose to spill rather than reduce the 4× unroll.

6. **Spill.** Only when none of the above is sufficient. Uses STL / LDL with rolling window pattern.

## Opcode inventory (new in this chapter)

| Opcode | Usage | Semantics |
|---|---|---|
| STL | 12i, 12k | Store to local memory (32-bit) |
| STL.128 | 12k | Store 128-bit (4 registers) to local memory |
| LDL | 12i, 12k | Load from local memory (32-bit) |
| LDL.LU | 12i | Load from local memory with last-use hint |
| R2UR | 12k | Copy per-thread register to uniform register |
| IADD3 dst, PT, PT, a, b, c | 12k | Three-input add with explicit carry predicates (first in integer context) |

## Addressing modes observed

* **`[R1]`** — direct access at stack pointer (offset 0).
* **`[R1+offset]`** — small immediate offset from stack pointer (observed up to 0x124 = 292).
* **`[R_idx + UR_base]`** — per-thread index + uniform base, used when R2UR copied R1 to UR7.

[GAP] Addressing mode `[R_idx + R_base]` not observed. Possibly not supported by LDL/STL, which would explain why R2UR is needed to copy R1 to UR.

## Canonical spill patterns worth flagging

### Pattern A: Stack frame at prologue

```
LDC R1, c[0x0][0x37c]        ; standard prologue
... (usual prologue)
IADD R1, R1, -<frame_size>   ; allocate local memory frame (only if spilling)
```

The IADD on R1 appears immediately after the standard prologue if and only if local memory is needed. Presence of this instruction is a reliable signal of spill or local array.

### Pattern B: Rolling window in compute loop

```
STL [R1+<old_slot>], R_spilled
FFMA R_dst, ...
LDL.LU R_loaded, [R1+<new_slot>]
```

Repeating triplet inside a register-pressured compute loop. Each FFMA is surrounded by a spill of the previous result and a reload of the next input.

### Pattern C: Vectorized init + scalar reads

```
<compute 4 values in R4-R7>
STL.128 [R1+offset], R4
<compute next 4 values>
STL.128 [R1+offset+0x10], R8
...
LDL R, [R_idx + UR_base]     ; scalar reads at random indices
LDL R, [R_idx + UR_base]
...
```

Used when the program initializes a static local array sequentially then reads from it at computed indices.

### Pattern D: R2UR for stack addressing

```
IADD R1, R1, -<frame_size>
R2UR UR_base, R1
...
LDL R, [R_offset + UR_base]
```

Observed in 12k. [HYP] The R2UR is required because LDL does not accept `[R + R]` addressing modes; the "base" must be a uniform register.

## Cost analysis

Comparing 12i (spilled) to 12e (not spilled) with the same type of workload:

| Metric | 12e (16 acc, no spill) | 12i (32 acc, spilled) |
|---|---|---|
| Kernel size | ~300 instructions | ~550 instructions |
| Spill instructions | 0 | ~170 STL + LDL pairs |
| FFMA instructions | ~110 | ~130 |
| Useful compute ratio | ~37% | ~24% |
| Per-iteration cost | ~4 cycles/FFMA | ~6 cycles/FFMA (estimated with spill overhead) |

The spill reduces the useful compute ratio by ~35% and adds approximately 1.5 instructions of memory traffic per FFMA. On a memory-bound workload this would be catastrophic; on a compute-bound workload it is tolerable if the alternative (more blocks × fewer threads) reduces occupancy further.

[HYP] The exact cycle cost of STL / LDL on SM120 is not measured. Based on kernel 06's MUFU.RCP latency hypothesis, LDL from L1-resident local memory is probably 20-30 cycles. LDL.LU may be faster due to eviction policy. Not microbenchmarked.

## Observations worth flagging

* [OBS] **Stack pointer R1 is the conventional spill base.** Every STL/LDL uses R1 (or UR7 = R1 copy) as base. No evidence of ptxas ever choosing a different register for this purpose.
* [OBS] **Frame size is exactly sized to need + alignment.** 0x100 for 64 ints (exact), 0x128 for 32 FFMA accumulators (exact + padding).
* [OBS] **`.LU` suffix on LDL is hint-only.** The load still functions without it. Semantics: tell the cache that this line will not be reused, allowing eviction. Useful for scratch reloads.
* [OBS] **No STL.128 observed in 12i.** Ptxas uses only scalar STL for the rolling accumulator window, not vectorized. Probably because the accumulators are spilled at different times (not in a burst).
* [OBS] **STL.128 used in 12k.** Four consecutive 32-bit stores fuse when they follow a predictable initialization pattern.
* [OBS] **LDL does not vectorize in our observations.** Only STL has the `.128` variant in these dumps. [GAP] Does LDL.128 exist? Not observed. Possibly exists but ptxas did not trigger it here.
* [OBS] **Fixed-latency instructions (FFMA, IADD) still have stall counts.** Spill adds variable-latency LDL to the critical path but the fixed-latency interior of the unroll is unchanged.
* [OBS] **Scoreboard usage for STL/LDL.** The control codes `0x001fe80000100800` on STL and `0x001ea80000300800` on LDL suggest specific scoreboard patterns. [GAP] Exact decoding of the `0x100` and `0x300` fields not completed.
* [OBS] **Descriptor-based addressing NOT used for local memory.** Unlike LDG/STG which use `desc[UR][R.64]`, LDL/STL use direct `[R+offset]` or `[R+UR]`. Local memory does not go through the global memory descriptor system.

## Gaps explicitly acknowledged

1. The exact cycle latency of STL vs LDL vs LDL.LU not measured.
2. The bit-level encoding of the `.LU` suffix not extracted.
3. Why ptxas needs R2UR to copy R1 before using it as LDL base — hypothesis only, not confirmed by testing addressing mode `[R+R]`.
4. Whether LDL.128 exists as an opcode; not observed in these dumps.
5. Control code scoreboard fields for LSU-local pipeline (STL/LDL) not fully decoded.
6. Stack frame alignment rule: 8 bytes observed, but smaller frames not tested.
7. Interaction between `-rdc=true` (separate compilation) and the register-sharing CALL pattern of 12j: probably breaks, not tested.
8. Whether ptxas ever chooses a register other than R1 as spill base: no evidence against, but not explicitly tested.

## Open questions for future work

* Does ptxas ever spill pointer values (the 2-register pairs for `desc[R.64]`)? In 12g with 12 pointers it did not, but extreme cases with 20+ pointers could.
* What is the exact threshold at which ptxas switches from restructure to spill? The 12i kernel is above it; 12e/12f are below. The boundary is somewhere between 16 and 32 FFMA accumulators under `-maxrregcount=24`.
* Does tensor core MMA (future chapter) force spill patterns we have not seen yet? Tensor core operations have mandatory register layout constraints that may interact with spill rules.
* How does `__launch_bounds__` interact with `-maxrregcount`? Both affect the register budget; their combination's behavior is not tested.

## Placement in the project

Chapter 12 closes the infrastructure foundation of the SM120 SASS vocabulary:
* Prologue, bounds check, addressing (chapters 01-03)
* Compile-time vs runtime control flow (chapters 04-05)
* Shared memory (chapters 06-07)
* Vectorized global memory (chapter 08)
* Warp primitives (chapters 09-10)
* Math and arithmetic slowpath (chapter 11)
* Register spill and local memory (chapter 12)

What remains on the Phase 1 roadmap: tensor core instructions (HMMA, QMMA, OMMA) for chapter 13+. After that, Phase 2 audits of production kernels becomes feasible with the vocabulary built here.