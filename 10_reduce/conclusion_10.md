# Kernel 10 REDUX (hardware warp reduction)

Chapter dedicated to the REDUX family of instructions: a single SASS opcode that replaces the multi-stage butterfly pattern (5 SHFL.BFLY + 5 arithmetic) with one variable-latency instruction writing to the uniform register file. Added in Ampere (SM80), present on SM120.

Eight variants tested to map the full operation space: SUM, MIN signed, MAX signed, MIN unsigned, MAX unsigned, AND, OR, XOR.

## Source (baseline 10a)

```cuda
__global__ void reduce_sum(const int* a, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (i < n) ? a[i] : 0;
    int res = __reduce_add_sync(0xffffffff, val);
    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = res;
    }
}
```

Canonical warp reduction with bounds check, identity fallback for out-of-bounds lanes, per-warp output written by lane 0.

## SASS main body structure (10a, 20 instructions)

```
[Prologue: 5 instructions]
  LDC R1, c[0x0][0x37c]                  ; stack pointer init
  S2R R7, SR_TID.X                        ; threadIdx.x
  S2UR UR6, SR_CTAID.X                    ; blockIdx.x (uniform)
  LDCU UR4, c[0x0][0x390]                 ; n (scalar kernel arg)
  BSSY.RECONVERGENT B0, 0xf0              ; open reconvergence scope for REDUX

[Identity init: 1 instruction]
  HFMA2 R2, -RZ, RZ, 0, 0                 ; val = 0 (identity for SUM)

[Bounds check: 2 instructions]
  LDC R0, c[0x0][0x360]                   ; blockDim.x
  IMAD R5, R0, UR6, R7                    ; i = blockDim.x * blockIdx.x + tid
  ISETP.GE.AND P0, PT, R5, UR4, PT        ; P0 = (i >= n)

[Conditional global load: 4 instructions]
  LDCU.64 UR4, c[0x0][0x358]              ; descriptor for global addressing
  @P0 BRA 0xe0                            ; skip LDG if out of bounds
  LDC.64 R2, c[0x0][0x380]                ; load a pointer
  IMAD.WIDE R2, R5, 0x4, R2               ; &a[i]
  LDG.E R2, desc[UR4][R2.64]              ; val = a[i]

[Reconvergence and reduction: 2 instructions]
  BSYNC.RECONVERGENT B0                   ; all 32 lanes now active
  REDUX.SUM.S32 UR7, R2                   ; UR7 = sum of val across warp

[Lane-zero test and predicated exit: 3 instructions]
  LOP3.LUT P0, RZ, R7, 0x1f, RZ, 0xc0, !PT ; P0 = (tid & 31) != 0
  MOV R9, UR7                              ; cross-file: R9 <- UR7 (waits for REDUX scoreboard)
  @P0 EXIT                                 ; non-lane-0 threads exit

[Warp-id derivation and store: 4 instructions]
  LDC.64 R2, c[0x0][0x388]                 ; c pointer
  SHF.R.U32.HI R0, RZ, 0x5, R0             ; blockDim.x / 32
  SHF.R.U32.HI R5, RZ, 0x5, R7             ; tid / 32
  IMAD R5, R0, UR6, R5                     ; warp_id_global
  IMAD.WIDE.U32 R2, R5, 0x4, R2            ; &c[warp_id_global]

[Store and exit: 2 instructions]
  STG.E desc[UR4][R2.64], R9
  EXIT
```

Instructions 0x01a0 and following are safety trap (BRA to self) and NOP padding.

## Solid observations

### 1. REDUX operation is encoded in the control code, not the opcode

All 8 variants share the **exact same opcode bytes**: `0x00000000020773c4`. Only the control code differs.

Control codes across variants (middle bytes, left and right stripped):

```
Variant           ctrl (relevant bits)
SUM.S32          ...0c02...
MIN.S32          ...1002...
MAX.S32          ...1402...
MIN (unsigned)   ...1000...
MAX (unsigned)   ...1400...
AND              ...0000...
OR               ...0400...
XOR              ...0800...
```

This is a significant architectural observation: REDUX is a single opcode whose semantic operation is selected by bits in the control code. Unlike SHFL (where SHFL.BFLY, SHFL.IDX, SHFL.UP, SHFL.DOWN are four distinct opcodes), REDUX is one opcode with eight behavioral variants.

### 2. Decoded bit semantics of REDUX control code

From pattern matching across the 8 variants, the following bits encode operation and signedness:

**Operation family bits (bits 14-16 of the control code):**

| Bit 16 | Bit 15 | Bit 14 | Operation |
|---|---|---|---|
| 0 | 0 | 0 | AND (default) |
| 0 | 0 | 1 | OR |
| 0 | 1 | 0 | XOR |
| 0 | 1 | 1 | SUM |
| 1 | 0 | 0 | MIN |
| 1 | 0 | 1 | MAX |
| 1 | 1 | x | Not observed (reserved?) |

[HYP] The 3-bit layout suggests two operation classes: bitwise/arithmetic (bit 16 = 0) and comparison (bit 16 = 1). Within bitwise class, bits 14-15 index into {AND, OR, XOR, SUM}. Within comparison class, bit 14 flips MIN to MAX. SUM sharing bits with OR and XOR is consistent with viewing SUM as the "full carry-propagating" variant of XOR.

**Signedness bit (bit 9, value 0x200):**

| Bit 9 | Interpretation |
|---|---|
| 0 | Unsigned (or bitwise, where signedness is irrelevant) |
| 1 | Signed (S32) |

[RES] Only observed as 1 for SUM.S32, MIN.S32, MAX.S32. Always 0 for MIN unsigned, MAX unsigned, AND, OR, XOR. `__reduce_add_sync` on int produces S32; `__reduce_min/max_sync` produce S32 or U32 depending on argument type.

[HYP] `__reduce_add_sync` on unsigned int may produce the same SASS (addition mod 2^32 is identical for signed and unsigned), meaning REDUX.SUM without S32 suffix may never appear in practice. Not directly tested; our 10a only tested signed. A future variant 10i = reduce_sum_unsigned would confirm.

### 3. cuobjdump naming conventions are asymmetric

The disassembly shows inconsistent suffix display:

| SASS text | Operation | Signedness |
|---|---|---|
| `REDUX.SUM.S32` | SUM | signed (explicit) |
| `REDUX.MIN.S32` | MIN | signed (explicit) |
| `REDUX.MAX.S32` | MAX | signed (explicit) |
| `REDUX.MIN` | MIN | unsigned (implicit, no suffix) |
| `REDUX.MAX` | MAX | unsigned (implicit, no suffix) |
| `REDUX.AND` | AND | not applicable... wait, `REDUX` alone |
| `REDUX` (no suffix) | AND | default |
| `REDUX.OR` | OR | not applicable |
| `REDUX.XOR` | XOR | not applicable |

[RES] Two asymmetries:
* `.U32` is never emitted. Unsigned MIN/MAX display as `REDUX.MIN` and `REDUX.MAX`, with the unsigned interpretation implicit.
* `.AND` is never emitted. AND (all operation bits = 0) displays as plain `REDUX` with no suffix at all. AND is the "default" operation in the encoding.

Practical consequence for reading: `REDUX` alone = AND unsigned, `REDUX.MIN` = MIN unsigned, `REDUX.MIN.S32` = MIN signed. Reader must know the naming convention.

### 4. Three identity load patterns depending on the target constant

REDUX on a warp with divergent lanes requires all 32 lanes active. Out-of-bounds lanes must contribute the operation's identity element. Ptxas selects the load strategy based on the identity value:

| Operation | Identity | Load instruction |
|---|---|---|
| SUM | `0x00000000` (0) | `HFMA2 R, -RZ, RZ, 0, 0` |
| MAX unsigned | `0x00000000` (0) | `HFMA2 R, -RZ, RZ, 0, 0` |
| OR | `0x00000000` (0) | `HFMA2 R, -RZ, RZ, 0, 0` |
| XOR | `0x00000000` (0) | `HFMA2 R, -RZ, RZ, 0, 0` |
| MAX signed | `0x80000000` (INT_MIN) | `HFMA2 R, -RZ, RZ, -0.0, 0` |
| MIN signed | `0x7FFFFFFF` (INT_MAX) | `MOV R, 0x7fffffff` |
| MIN unsigned | `0xFFFFFFFF` (UINT_MAX) | `MOV R, 0xffffffff` |
| AND | `0xFFFFFFFF` (all ones) | `MOV R, 0xffffffff` |

**Rule.** HFMA2 is used when the 32-bit target value can be expressed as the concatenation of two half-float constants. Otherwise, a 32-bit immediate MOV is used.

**Why HFMA2 works for integer constants.** HFMA2 computes `-RZ * RZ + imm`, which simplifies to `0 + imm`. The immediate is encoded as two adjacent half-float values in the high and low halves of the target register.
* `0x00000000 = 0.0_h || 0.0_h` via `HFMA2 ..., 0, 0`
* `0x80000000 = -0.0_h || 0.0_h` via `HFMA2 ..., -0.0, 0` (half-float sign bit sets bit 31 of the concatenated result)

[RES] The same trick used for FP32 constants in kernels 04 and 05 is re-purposed here to load specific integer bit patterns, exploiting the fact that the FP bit representation and the integer representation are both 32-bit values.

**Why MOV is used for `0x7FFFFFFF` and `0xFFFFFFFF`.** These values cannot be expressed as a half-float concatenation that HFMA2 can encode. `0xFFFFFFFF = NaN_h || NaN_h` in FP would require encoding NaN in the HFMA2 immediate slot, which ptxas does not use. `0x7FFFFFFF = +NaN_h || NaN_h` similarly problematic.

### 5. Twenty-five of twenty-seven instructions are byte-identical across variants

Instruction-by-instruction comparison of the 8 dumps shows that the kernel skeleton is strictly invariant under operation change. Only two instructions differ:
* Address 0x0050: identity load (HFMA2 or MOV, depending on identity value)
* Address 0x00f0: the REDUX variant

All other 25 instructions (prologue, bounds check, BSSY/BSYNC, LDG, LOP3 lane-zero test, cross-file MOV, epilogue store, safety trap, NOP padding) have byte-identical encoding across all 8 variants.

[RES] Confirms that the compiler's scheduling decisions for this kernel class are driven purely by the identity value and operation. The rest of the kernel is mechanical.

### 6. REDUX writes to uniform register file, requires cross-file MOV

Every variant produces `REDUX.{op} UR7, R2`. The input is a per-thread register (each lane contributes its value); the output is a uniform register (the reduction is identical across all lanes).

Consequence: downstream consumers in per-thread code need a MOV `R, UR` to bring the value back to the per-thread register file. Observed pattern:

```
REDUX.SUM.S32 UR7, R2
LOP3.LUT P0, ...         ; independent work, interleaved
MOV R9, UR7              ; cross-file transfer, waits on REDUX scoreboard
@P0 EXIT
...
STG.E desc[UR4][R2.64], R9   ; uses R9 (per-thread) for the store
```

The MOV is placed after an independent LOP3 computation, hiding some of REDUX's variable latency. The MOV's control code (`0x001fd80008000f00`) suggests a wait on the scoreboard set by REDUX, but full bit-level validation is pending.

[RES] REDUX is the first SASS instruction observed in this project that writes across the per-thread/uniform register file boundary. Every other instruction in our 9 previous chapters kept the input and output in the same file (R→R or UR→UR).

### 7. Instruction count comparison with butterfly reduction

| Kernel | Reduction primitive | Total body | Reduction cost |
|---|---|---|---|
| 09a | SHFL.BFLY FP32 (with bounds) | 30 instructions | 5 SHFL + 5 FADD = 10 instructions, 5 sequential stages |
| 09b | SHFL.BFLY FP32 (no bounds) | 21 instructions | same 10 instructions, 5 stages |
| 10a | REDUX.SUM.S32 (with bounds) | 20 instructions | 1 instruction, 1 stage |

REDUX saves 9 instructions in the reduction path and collapses 5 sequential dependency stages into 1. The critical path latency improvement is proportionally larger than the instruction count reduction.

[GAP] Actual cycle-accurate latency of REDUX not measured. Variable-latency like SHFL, so a precise comparison requires microbenchmarking, deferred per D009.

### 8. Scheduling pattern: LOP3 lane-zero test placed between REDUX and MOV

Every variant schedules the lane-zero test (`LOP3.LUT P0, ...`) between the REDUX and the MOV R←UR that consumes REDUX's result. The LOP3 has no data dependency on REDUX and provides useful work during REDUX's latency window.

This is the same pattern observed around SHFL in chapter 09: place independent work between a variable-latency producer and its consumer. In chapter 10, the producer is REDUX; the technique is preserved.

## Variants summary

| Variant | SASS | Identity | Identity load | Signed |
|---|---|---|---|---|
| 10a | REDUX.SUM.S32 | 0 | HFMA2 | yes |
| 10b | REDUX.MIN.S32 | INT_MAX | MOV | yes |
| 10c | REDUX.MAX.S32 | INT_MIN | HFMA2 (-0.0 trick) | yes |
| 10d | REDUX.MIN | UINT_MAX | MOV | no |
| 10e | REDUX.MAX | 0 | HFMA2 | no |
| 10f | REDUX | UINT_MAX | MOV | no |
| 10g | REDUX.OR | 0 | HFMA2 | no |
| 10h | REDUX.XOR | 0 | HFMA2 | no |

## What this chapter adds to SASS reading skills

**Recognize REDUX as warp reduction at a glance.** A single instruction writing from R to UR in the middle of a kernel, possibly preceded by BSSY/BSYNC for reconvergence, is the REDUX signature.

**Decode the operation from the SASS text or the control code.** Plain `REDUX` = AND. `REDUX.MIN`/`.MAX` without further suffix = unsigned. Explicit `.S32` suffix = signed. `.SUM.S32`, `.OR`, `.XOR` are straightforward.

**Detect the identity load pattern.** An HFMA2 or MOV immediate right before a bounds check, loading a specific integer constant (0, INT_MAX, INT_MIN, 0xFFFFFFFF), is a strong signal that the code uses a warp-synchronous operation (SHFL, REDUX, VOTE, MATCH) where all lanes must remain active.

**Follow the cross-file MOV R←UR.** After REDUX produces a uniform register result, a `MOV R, UR` moves it back to the per-thread register file before any per-thread consumer (store, arithmetic with per-thread data). The MOV is the consumer that waits on REDUX's scoreboard.

**Count stages, not just instructions.** REDUX saves both instructions (1 vs 10) and pipeline stages (1 vs 5). For critical-path-limited reductions, this matters more than the raw instruction count.

## Practical consequences for kernel writers

* **Int32 warp reduction?** Use `__reduce_{add,min,max,and,or,xor}_sync`. Single instruction vs 10 for butterfly.
* **Int32 unsigned reduction?** Same intrinsic, but with unsigned argument. Produces REDUX without `.S32` suffix. Same cost as signed except for MIN/MAX where the unsigned comparison path is used.
* **Float reduction?** No REDUX variant for float. Must use butterfly SHFL + FADD (see chapter 09). The gap is significant on SM120 and earlier.
* **Mixed-type reduction (e.g., float min)?** Requires butterfly.
* **Identity choice for out-of-bounds lanes?** Match the operation's algebraic identity (0 for SUM/OR/XOR/MAX_unsigned, INT_MAX for MIN_signed, INT_MIN for MAX_signed, all-ones for AND/MIN_unsigned). Ptxas will pick the correct load instruction automatically.

## Hypotheses opened or deferred

1. **[HYP]** `__reduce_add_sync` on unsigned int produces the same SASS as on signed int (addition is bitwise identical). A variant 10i = reduce_sum_unsigned would confirm or refute.
2. **[HYP]** The 3-bit operation encoding (bits 14-16) may have reserved combinations (bits 15+16 simultaneously). Could correspond to future Blackwell-specific reductions or not exist at all.
3. **[HYP]** Bit 9 as signedness flag: confirmed for MIN/MAX. For SUM, signedness is probably cosmetic (same hardware path). For AND/OR/XOR, confirmed irrelevant (bit always 0).
4. **[GAP]** Scheduling bits of the REDUX control code (stall, yield, scoreboard) not decoded in detail. The MOV R9, UR7 that follows clearly waits on a scoreboard, but the specific SBS number and wait mask not validated against a parser.
5. **[GAP]** REDUX latency (cycles) not measured. Microbenchmark pending per D009.
6. **[HYP]** `CREDUX` (cluster-scope REDUX, mentioned in herrmann's SM100a delta) may exist on SM120. Not tested since our kernels do not use clusters. Relevant for chapter 13+ if tensor core or async memory uses cluster primitives.
7. **[HYP]** The HFMA2-with-`-0.0` trick to load INT_MIN may generalize. Any constant of the form `0xABCD0000` or `0x0000ABCD` or `0xABCDEFGH` where the halves correspond to encodable half-float values is HFMA2-loadable. Full decomposition rules not explicitly formalized.

## New instructions and idioms observed in this chapter

| Opcode / Idiom | Usage |
|---|---|
| `REDUX UR, R` | Warp AND reduction (unsigned), default operation |
| `REDUX.OR UR, R` | Warp bitwise OR |
| `REDUX.XOR UR, R` | Warp bitwise XOR |
| `REDUX.SUM.S32 UR, R` | Warp integer sum, signed 32-bit |
| `REDUX.MIN UR, R` | Warp min, unsigned 32-bit |
| `REDUX.MIN.S32 UR, R` | Warp min, signed 32-bit |
| `REDUX.MAX UR, R` | Warp max, unsigned 32-bit |
| `REDUX.MAX.S32 UR, R` | Warp max, signed 32-bit |
| `HFMA2 R, -RZ, RZ, -0.0, 0` | Load `0x80000000` (INT_MIN) into R via half-float concat trick |
| `MOV R, 0x7fffffff` | Load INT_MAX when HFMA2 cannot encode the pattern |
| `MOV R, 0xffffffff` | Load UINT_MAX / all-ones when HFMA2 cannot encode |
| `MOV R, UR` cross-file | Bring REDUX result back to per-thread register file |

## Placement in the project

Chapter 10 closes the warp-level primitive thread opened by chapter 09. Chapters 09 and 10 together cover:
* Communication primitives (SHFL variants, VOTE variants, MATCH variants, `__syncwarp`, `__activemask`)
* Arithmetic reduction primitives (REDUX variants)

For a production kernel audit (Phase 3), recognizing any instruction in these two families should be immediate. The next infrastructure chapter (11) will cover the division slowpath, closing another thread left open in the README roadmap.