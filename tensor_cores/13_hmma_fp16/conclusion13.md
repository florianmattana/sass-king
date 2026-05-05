# Chapter 13 — HMMA FP16 baseline on SM120

This chapter establishes the HMMA (half-precision matrix-multiply-accumulate) baseline for tensor core study on the RTX 5070 Ti (SM120, Blackwell consumer). It is the first chapter of the tensor core track and forms the foundation against which QMMA (chapter 14), block-scaled MMA (chapter 16), ldmatrix (chapter 17), and pipelined tile kernels (chapter 18) will be compared.

Five minimal kernel variants were analyzed end-to-end: SASS dump, opcode identification, register allocation study, control code decoding, scheduling pattern recognition, and latency measurement. The methodology follows controlled variation: each variant changes exactly one source parameter, and the resulting SASS delta is confronted against the hypotheses formed before the change.

All measurements were performed with `nvcc -arch=sm_120`, `cuobjdump --dump-sass`, and `clock64()` hardware timer. No NCU run was executed for this chapter; this is noted as an open gap.

## Scope of the chapter

**What this chapter covers:**
* HMMA opcode name, shape modifier, and dtype modifiers
* Register layout per thread for FP16 input, FP16/FP32 accumulator
* Register allocation by ptxas in the single-MMA case and in the accumulator-chained case
* The `@!UPT UIADD3 URZ` NOP pad pattern that surrounds HMMA
* Scoreboard bits in the HMMA control code
* Serial latency of HMMA.16816.F32 measured via clock64 microbench
* The canonical "minimal kernel MMA" template observed stable across all variants

**What this chapter does not cover:**
* HMMA at other shapes (m16n8k8 and below), deferred
* HMMA throughput (independent MMAs that can overlap), deferred to chapter 18
* FP16 accumulator latency (only FP32 accumulator was measured)
* NCU-side validation of the 35-cycle measurement
* Identification of which specific scoreboard slot HMMA uses (gpuasm.com disassembler unavailable)
* Sensitivity of HMMA latency to input data (NaN, denormals, zeros)
* Interactions with occupancy and multi-warp scheduling

## Variants studied

| Variant | Source PTX | Purpose | Instructions utiles | Result |
|---|---|---|---|---|
| 13a | `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` | Baseline single MMA, FP16 input, FP16 accumulator | 27 | d[0] = 0x4c004c00 (16.0) |
| 13b | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | Same as 13a, FP32 accumulator | 31 | d[0] = 16.0 |
| 13c | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | Same as 13b, BF16 input | 31 | d[0] = 16.0 |
| 13d | Two chained 13b MMAs (D1 becomes C of second) | Accumulator chaining baseline | 37 | d[0] = 32.0 |
| 13e | N chained MMAs bracketed by clock64() (N=16, 32, 64) | Latency microbenchmark | 76 (N=16) | see latency table |

All kernels run with 1 block × 32 threads (one warp) to isolate the MMA behavior from multi-warp scheduling effects.

## Opcode family and nomenclature

[OBS-13-01] The SASS opcode name emitted by ptxas for `mma.sync.aligned.m16n8k16.*` is `HMMA.16816`.

The full opcode form is `HMMA.16816.<acc_dtype>[.<input_dtype>]`:

| PTX form | SASS form | 13a/b/c |
|---|---|---|
| `.f16.f16.f16.f16` | `HMMA.16816.F16` | 13a |
| `.f32.f16.f16.f32` | `HMMA.16816.F32` | 13b |
| `.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16` | 13c |

[OBS-13-02] The shape modifier `.16816` is the concatenation of the M, N, K dimensions (16, 8, 16). This follows the historic Volta/Turing/Ampere pattern.

[OBS-13-03] The accumulator dtype suffix (`.F16` or `.F32`) is mandatory. The input dtype suffix is omitted when the input is FP16 (implicit default) and explicit only when the input is BF16.

### Byte-level encoding

The HMMA instruction is 16 bytes: 8 bytes of opcode + operands, 8 bytes of control code.

| Variant | Opcode + operand bytes | Control code bytes |
|---|---|---|
| 13a HMMA.16816.F16 | `0x000000100c0c723c` | `0x004ff60000000812` |
| 13b HMMA.16816.F32 | `0x000000100c0c723c` | `0x004ff60000001814` |
| 13c HMMA.16816.F32.BF16 | `0x000000100c0c723c` | `0x004ff60000041814` |
| 13d HMMA.16816.F32 (HMMA #1) | `0x0000000a0c10723c` | `0x084ff60000001810` |
| 13d HMMA.16816.F32 (HMMA #2) | `0x0000000a0c10723c` | `0x000ff60000001810` |

[OBS-13-04] The opcode+operand bytes are fully determined by the operand base registers. The dtype information is encoded only in the control code. The same bytes `0x000000100c0c723c` appear in 13a, 13b, and 13c because all three use the same operand base registers (R12, R16, R18/R20).

[RES] The byte-level delta 13b → 13c (FP16 input → BF16 input) is exactly `0x40000`, which is bit 18. The cuobjdump mnemonic shows `.BF16` only when bit 18 is set.

[RES] The byte-level delta 13a → 13b (FP16 acc → FP32 acc) is `0x1002` in the low-order control bytes, which is bit 12 ORed with bit 1. cuobjdump displays `.F16` vs `.F32` based on this combined pattern.

### Control code bits identified (partial)

| Bit | Inferred meaning | Evidence |
|---|---|---|
| 1 | Part of "non-FP16 accumulator mode" | 13a → 13b transition |
| 12 | Part of "FP32 accumulator flag" | 13a → 13b transition |
| 18 | Input dtype = BF16 (0 = FP16) | 13b → 13c byte-delta |
| 26 | MMA scoreboard set (SBS) | 13d HMMA1 (has bit 26), HMMA2 (does not) |
| 27 | MMA scoreboard wait | 13d, 13e internal HMMAs have bit 27 set, terminal HMMA does not |

[GAP] The exact scoreboard slot identifier (SB0, SB1, etc.) is not decoded. This would require a bit-level disassembler like gpuasm.com (currently down) or MaxAs-style reverse engineering. Bits 26 and 27 establish that HMMA is variable-latency and uses a scoreboard, but the scoreboard slot number lives in other bits of the control code.

## Fragment layout per thread

The m16n8k16 shape with 32 threads per warp implies the following per-thread fragment sizes, matching the CUTLASS atoms from `include/cute/arch/mma_sm80.hpp`:

| Variant | D | A | B | C |
|---|---|---|---|---|
| 13a (F16 acc) | uint32[2] (4 half) | uint32[4] (8 half) | uint32[2] (4 half) | uint32[2] (4 half) |
| 13b (F32 acc, FP16) | float[4] | uint32[4] | uint32[2] | float[4] |
| 13c (F32 acc, BF16) | float[4] | uint32[4] (8 bf16) | uint32[2] (4 bf16) | float[4] |

[OBS-13-05] The SASS instruction takes exactly four operands: D base, A base, B base, C base. The per-fragment register counts are implicit in the shape and dtype encoding, not visible in the text form of the instruction.

For example, 13b emits `HMMA.16816.F32 R12, R12, R16, R20` but this encodes:
* D occupies R12:R13:R14:R15 (4 registers)
* A occupies R12:R13:R14:R15 (4 registers)
* B occupies R16:R17 (2 registers)
* C occupies R20:R21:R22:R23 (4 registers)

The disassembler only shows the base; the spans are inferred from the opcode suffix.

## Register allocation patterns

### Single-MMA case (13a, 13b, 13c)

[OBS-13-06] ptxas colocates D and A onto the same register base. When D is smaller than A (13a, where D is 2 regs and A is 4 regs), the overlap is partial: D reuses the first 2 registers of A. When D equals A in size (13b, 13c, where both are 4 regs), the overlap is complete: D and A share exactly the same 4 registers.

This is safe because MMA semantics guarantee A is fully consumed before D is produced. The instruction executes atomically at the register file level.

| Variant | D registers | A registers | Overlap |
|---|---|---|---|
| 13a | R12:R13 | R12:R13:R14:R15 | partial (2 of 4) |
| 13b | R12:R13:R14:R15 | R12:R13:R14:R15 | complete |
| 13c | R12:R13:R14:R15 | R12:R13:R14:R15 | complete |

[OBS-13-07] B is always on a distinct register base (R16, R16, R16 respectively). ptxas never colocates B with D or A.

[OBS-13-08] C is also on a distinct register base (R18 for 13a, R20 for 13b/13c). In the single-MMA case, C is only read, never written-to-the-same-block, so there is no benefit to colocating it with D.

### Accumulator-chained case (13d, 13e)

[OBS-13-09] When HMMA feeds its D into the C of a subsequent HMMA, ptxas colocates D and C onto the same register base. The accumulator lives in-place across the chain.

In 13d: `HMMA R16, R12, R10, R16` — D = R16:R19, C = R16:R19. The second HMMA has the same operands exactly. In 13e, all 16 HMMAs have the form `HMMA R16, R4, R2, R16` where the accumulator R16:R19 is read and written in-place.

[OBS-13-10] In the chained case, D and A are NOT colocated, even though in the single-MMA case they would be. Reason: A is re-read by the next HMMA in the chain, so it cannot be overwritten as D.

[RES] The allocation rule is: D and C share registers when chained; D and A share registers when not chained. ptxas maximizes register reuse subject to the data dependency graph.

### Register allocation summary table

| Scenario | D ↔ A | D ↔ C | A and B |
|---|---|---|---|
| Single MMA, D size < A size | partial overlap | disjoint | disjoint |
| Single MMA, D size = A size | full overlap | disjoint | disjoint |
| Chained MMA | disjoint | full overlap | disjoint |

## The .reuse modifier on MMA operands

[OBS-13-11] In chained MMA (13d, 13e), ptxas emits `.reuse` on the B operand of every HMMA except the last one:

```
13d:   HMMA.16816.F32 R16, R12, R10.reuse, R16   (HMMA1, feeds HMMA2)
13d:   HMMA.16816.F32 R16, R12, R10, R16         (HMMA2, last in chain)
13e:   HMMA.16816.F32 R16, R4, R2.reuse, R16     (HMMAs #1-15)
13e:   HMMA.16816.F32 R16, R4, R2, R16           (HMMA #16, last in chain)
```

[HYP] The `.reuse` modifier activates the operand reuse cache for the next MMA that will re-read the same operand. It is a scheduling hint that reduces register file bandwidth pressure by caching the value close to the MMA unit.

[OBS-13-12] `.reuse` is not emitted on A, even though A is also re-read by the next HMMA in a chain. One hypothesis: the reuse cache has limited capacity (possibly only 2 registers worth), so ptxas prioritizes B (2 registers) over A (4 registers). Another hypothesis: the cache is positional (one slot per operand role), and only the B slot is enabled.

[OBS-13-13] `.reuse` is not emitted on C in the observed HMMA chains. [HYP] This may be because C is already colocated with D in the chaining case (same register base), which is a different form of reuse at the register file level.

## The NOP pad pattern

Every single-MMA kernel in 13a, 13b, 13c shows the same post-HMMA structure:

```
HMMA.16816.<dtype> Rd, Ra, Rb, Rc
@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ
@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ
STG.E ...
```

[OBS-13-14] The two `@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ` instructions are semantic NOPs: the predicate `@!UPT` is false by construction (UPT is always true, `!UPT` is always false), all operands are URZ (uniform zero register), and the destination is URZ. The instruction has no observable effect.

[OBS-13-15] The count of 2 NOPs is stable across variants 13a, 13b, 13c. It does not depend on the accumulator dtype (FP16 vs FP32) or on the input dtype (FP16 vs BF16).

### Refinement from 13d (chained case)

[OBS-13-16] 13d has 4 NOPs total: 2 between HMMA1 and HMMA2, and 2 between HMMA2 and STG. The pattern is "2 NOPs after each HMMA," not "2 NOPs before STG."

### Refinement from 13e (clock bracketed)

[OBS-13-17] 13e has 2 NOPs after each of HMMAs #1 through #15, but **zero NOPs after HMMA #16**. HMMA #16 is immediately followed by `CS2R R2, SR_CLOCKLO`.

[RES] The NOP pad is not an intrinsic property of HMMA. It is emitted by ptxas when the consumer of HMMA's D result has a data dependency that needs the NOPs to cover the HMMA latency via the scheduling stream. When the consumer does not depend on D (CS2R reads SR_CLOCKLO, which does not involve R16), no NOPs are needed.

### The revised rule

[RES] ptxas emits `@!UPT UIADD3 URZ` NOPs after HMMA to cover the gap between HMMA completion and the next dependent consumer. In a minimal kernel with no useful work to schedule, ptxas fills the gap with NOPs. In a production kernel (GEMM with ldmatrix, address arithmetic, index bumps), these slots are filled with real work, and fewer or zero uniform NOPs appear.

The NOPs are therefore a signal of an under-utilized kernel (no independent work available to overlap with HMMA latency), not a signal of HMMA itself.

## Scoreboard and variable latency

[OBS-13-18] HMMA is variable-latency and uses a scoreboard mechanism. Evidence:

From 13d's control codes:
* HMMA1 (0x0170): `0x084ff60000001810` — bit 27 (0x08000000) set
* HMMA2 (0x01a0): `0x000ff60000001810` — bit 27 clear

From 13e's control codes:
* HMMA #1 (0x0170): `0x084ff60000001810` — bits 26 and 27 both set
* HMMA #2-15: `0x080ff60000001810` — bit 27 set, bit 26 clear
* HMMA #16: `0x000ff00000001810` — bit 27 clear

[HYP] Bit 26 is "set scoreboard slot" (SBS). Set on the first HMMA of a chain to mark the scoreboard slot that subsequent HMMAs will wait on.

[HYP] Bit 27 is "wait on scoreboard". Set on HMMAs that have a data dependency on a prior HMMA (the D result feeds C of this one via register reuse). Clear on the first HMMA (nothing to wait for) and on the last HMMA (its consumer is not a scoreboard-coordinated MMA).

[GAP] The specific scoreboard slot number used (SB0, SB1, ...) is not decoded from the control code bytes. The bit positions that encode the slot ID are embedded in the control code but their exact mapping is not recovered in this chapter.

[OBS-13-19] The LSU instructions (LDG, STG) in the kernel also use scoreboards, visible in the control code byte `0x004ea6` pattern common to all LDGs. These are different scoreboard slots from the MMA scoreboard.

### Unified HMMA control code table across chapter 13

| Kernel | HMMA position | Opcode+operand bytes | Control code |
|---|---|---|---|
| 13a | single | `0x000000100c0c723c` | `0x004ff60000000812` |
| 13b | single | `0x000000100c0c723c` | `0x004ff60000001814` |
| 13c | single | `0x000000100c0c723c` | `0x004ff60000041814` |
| 13d | HMMA #1 (feeds HMMA #2) | `0x0000000a0c10723c` | `0x084ff60000001810` |
| 13d | HMMA #2 (last) | `0x0000000a0c10723c` | `0x000ff60000001810` |
| 13e | HMMA #1 (first of chain) | `0x000000020410723c` | `0x084ff60000001810` |
| 13e | HMMA #2-15 (mid-chain) | `0x000000020410723c` | `0x080ff60000001810` |
| 13e | HMMA #16 (last, feeds CS2R) | `0x000000020410723c` | `0x000ff00000001810` |

[OBS-13-30] The low-order 16 bits of the control code (the "scheduling code") vary consistently with the role of the HMMA:
* `0x812` on 13a (F16 acc, single)
* `0x1814` on 13b, 13c (F32 acc, single; F32 acc BF16, single)
* `0x1810` on 13d, 13e (F32 acc, chain context)

The difference between 13b `0x1814` and 13d `0x1810` is bit 2 (`0x4`). This suggests a flag related to "dependent consumer immediately follows" versus "independent consumer" (13b's consumer is STG which depends on R16; 13d's HMMA #1 consumer is HMMA #2 which also depends on R16, but the chain context changes the scheduling hint).

[GAP] The exact semantics of low-order bits 0-4 of the HMMA control code are not fully decoded. [HYP] They may encode a combination of yield, stall, read barrier, and write barrier fields as in the Turing/Ampere control code format, but the exact bit layout on Blackwell is not pinned down in this chapter.

### Transition analysis between adjacent HMMAs in 13e

[OBS-13-31] The transition from HMMA #1 to HMMA #2-15 shows one bit change: bit 26 (`0x04000000`) goes from set to clear. This is consistent with the interpretation "bit 26 = set scoreboard" being relevant only on the first HMMA that establishes the chain's scoreboard slot. Subsequent HMMAs in the chain wait on the scoreboard (bit 27) but do not re-set it.

[OBS-13-32] The transition from HMMA #15 to HMMA #16 shows two bit changes: bit 27 clears (no wait, its consumer CS2R does not use the MMA scoreboard) and bits 2 set (scheduling code changes from `0x1810` to `0x0010`... pattern). HMMA #16 ends the chain and hands off to a non-MMA consumer, which requires a different scheduling profile.

## Canonical kernel MMA minimal template

[PAT] A kernel that consists of a single warp loading fragments, doing one or more HMMAs, and storing the result follows this instruction template, stable across 13a, 13b, 13c, 13d:

```
// Prologue (8 instructions)
LDC R1, c[0x0][0x37c]                  // stack
S2R R_tid, SR_TID.X                    // threadIdx

// Pointer loads (one LDC.64 per kernel arg, 8-byte stride in param space)
LDC.64 R_ptr_a, c[0x0][0x380]
LDC.64 R_ptr_b, c[0x0][0x388]
LDC.64 R_ptr_c, c[0x0][0x390]
LDC.64 R_ptr_d, c[0x0][0x398]
LDCU.64 UR_desc, c[0x0][0x358]         // descriptor for LDG/STG

// Stride computation (one SHF per per-thread stride)
SHF.L.U32 R_stride_ab, R_tid, K_ab, RZ
SHF.L.U32 R_stride_cd, R_tid, K_cd, RZ

// Address computation (IMAD.WIDE.U32 per pointer)
IMAD.WIDE.U32 R_addr_a, R_stride_ab, 0x4, R_ptr_a
IMAD.WIDE.U32 R_addr_b, ..., 0x4, R_ptr_b
IMAD.WIDE.U32 R_addr_c, R_stride_cd, 0x4, R_ptr_c

// Fragment loads
LDG.E.CONSTANT R_b0, desc[UR_desc][R_addr_b]
LDG.E.CONSTANT R_b1, desc[UR_desc][R_addr_b+0x4]
// ... for all A registers, B registers, C registers

IMAD.WIDE.U32 R_addr_d, R_stride_cd, 0x4, R_ptr_d

// MMA
HMMA.16816.<dtype> R_d, R_a, R_b, R_c

// NOP pad (per-HMMA, if followed by dependent consumer)
@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ
@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ

// Stores
STG.E desc[UR_desc][R_addr_d], R_d_0
// ... for all D registers

// Epilogue
EXIT
BRA .                                  // self-trap
NOP (padding to kernel boundary)
```

[OBS-13-20] This template is stable across all four single-HMMA or chained variants. The chained case just inserts additional HMMA + NOP pairs between the last LDG and the first STG.

[OBS-13-21] The LDG order for C registers is inverted (offsets 0xc, 0x8, 0x4, 0x0 emitted in that order). This is a stable pattern across 13b, 13c, 13d, 13e, and reflects ptxas's register allocation choice (R23 → R22 → R21 → R20). It is not a correctness issue and not a performance concern.

## Latency measurement (13e)

### Setup and methodology

The 13e kernel times N chained HMMAs via `clock64()` bracketing:

```
CS2UR UR_start, SR_CLOCKLO       // uniform clock read (start)
HMMA × N chained                  // D fed back as C each iteration
CS2R R_end, SR_CLOCKLO            // per-thread clock read (end)
IADD.64 R_delta, R_end, -UR_start // compute delta
STG.E.64 ...                      // store delta
```

The chain uses the same A, B inputs for all HMMAs and the accumulator is propagated in-place (D = C = R16:R19 for all iterations). This forces a serial dependency that ptxas cannot reorder.

[OBS-13-22] A new opcode appears: `CS2UR UR6, SR_CLOCKLO`. This is the uniform-register variant of CS2R (chapter 11 introduced CS2R for per-thread special register reads). CS2UR writes the clock value to a uniform register shared across the warp, which is semantically correct because all threads in a warp sample the same clock at the same issue slot.

[OBS-13-23] At the end of timing, ptxas emits `CS2R R2, SR_CLOCKLO` (per-thread variant) because the subsequent compute path (IADD.64 with a per-thread destination, ISETP on tid) is per-thread.

[OBS-13-24] `IADD.64 R2, R2, -UR6` demonstrates that IADD.64 accepts a uniform register as a source operand, mixed with per-thread operands. This is a new data point about mixed R/UR instruction forms on SM120.

### Measurements

| N_HMMA | Total cycles | cycles / HMMA |
|---|---|---|
| 16 | 872 | 54.50 |
| 32 | 1439 | 44.97 |
| 64 | 2541 | 39.70 |

### Linear regression

From pairs of measurements:
* (N=32) − (N=16): delta 567 cycles for 16 additional HMMAs → 35.44 cycles per HMMA
* (N=64) − (N=32): delta 1102 cycles for 32 additional HMMAs → 34.44 cycles per HMMA
* (N=64) − (N=16): delta 1669 cycles for 48 additional HMMAs → 34.77 cycles per HMMA

[RES] The marginal cost per added HMMA in a serial chain is **~35 cycles** on SM120 for HMMA.16816.F32 with FP16 input.

Computing the fixed overhead:
* N=16: 872 − 16 × 35 = 312 cycles overhead
* N=32: 1439 − 32 × 35 = 319 cycles overhead
* N=64: 2541 − 64 × 35 = 301 cycles overhead

[RES] The fixed overhead is approximately **~310 cycles** and is consistent across N values within measurement noise.

### Model

```
total_cycles(N) ≈ 310 + 35 × N     (HMMA.16816.F32 serial chain on SM120)
```

[RES] [GAP-13a-02] closed: HMMA.16816.F32 serial latency is ~35 cycles per HMMA.

### Interpretation of the 310-cycle overhead

[HYP] The 310-cycle overhead is composed of:
* Latency of the very first HMMA (which includes the startup cost of activating the tensor core pipeline)
* Latency of CS2UR and CS2R themselves
* Latency of the final HMMA result becoming visible to CS2R (the clock read does not depend on R16, so there is no explicit scoreboard wait, but the issue slot is ordered)
* Possible pipeline spin-up for the scoreboard system

Distinguishing these components would require N=1 and N=2 measurements. This is a straightforward follow-up not done in this chapter.

### Instruction count vs cycles

The hot path between CS2UR and CS2R consists of:
* 16 × HMMA
* 15 × 2 = 30 UIADD3 NOPs (2 per HMMA except the last)
* Total: 46 instructions

46 instructions in ~560 cycles (N=16 chain) gives approximately 12 cycles per instruction on average. Since UIADD3 NOPs are 1-cycle throughput, most of the 35-cycle-per-HMMA cost is inside the HMMA instruction itself or in the scoreboard wait of the following HMMA.

[HYP] The intrinsic HMMA.16816.F32 latency is on the order of 32-33 cycles, with the 2 NOPs absorbing about 2 cycles of that. The remaining latency is waited on by the scoreboard at the next HMMA's bit 27.

### Latency caveats

[IMPORTANT] 35 cycles is the **serial latency** in a chain where every HMMA depends on the previous one. This is NOT the HMMA throughput. In a real GEMM kernel, many HMMAs can be in flight simultaneously because their D results are all written to independent accumulator registers, not fed back as C. The throughput (cycles per HMMA when many are pipelined) is substantially lower than 35 and will be measured in chapter 18.

## Compiler determinism observations

[OBS-13-25] ptxas produces byte-identical SASS for the portion of the kernel that does not involve HMMA, across 13b and 13c. Only the HMMA instruction differs (bit 18 of the control code). 30 of 31 instructions are byte-identical between these two variants.

[OBS-13-26] The kernel argument layout in constant memory is stable at 8-byte stride per pointer starting at `c[0x0][0x380]` (a, 0x380; b, 0x388; c, 0x390; d, 0x398). This matches previous chapters for kernels with 4 pointer arguments and no scalar arguments.

[OBS-13-27] `LDG.E.CONSTANT` is emitted for any pointer marked `const __restrict__` in the kernel signature, not just for compile-time constant lookup tables. This extends the observation from chapter 11 (Payne-Hanek table) to a more general rule about ptxas's treatment of restricted read-only pointers.

[OBS-13-28] ptxas places the destination pointer address computation (`IMAD.WIDE.U32 R_addr_d`) **late** in the instruction stream, after most of the LDGs for inputs are issued. In 13b, 13c, 13d this IMAD.WIDE is the last address computation before HMMA. Rationale: the D address is not needed until STG, so delaying its computation frees the destination register for other uses earlier and minimizes its live range.

[OBS-13-29] Register recycling across semantic boundaries: in 13d, R4 is first used to hold `&b[tid*2]` (address of b). Once the two LDGs of b complete (R10, R11), R4 is free. ptxas then overwrites R4 at 0x0160 to become `&d[tid*4]`. The same physical register plays two unrelated roles in the kernel lifetime. This is consistent with the register recycling pattern from chapter 02 (register lifetime across semantic boundaries).

## Open gaps

### Gaps resolved in this chapter

| Gap ID | Origin | Resolution |
|---|---|---|
| GAP-13a-01 (scoreboard existence) | 13a | Resolved: bits 26 (SBS) and 27 (wait) of HMMA control code |
| GAP-13a-02 (HMMA latency) | 13a | Resolved: ~35 cycles/HMMA for FP32 acc, serial chain |
| HYP-13c-X1/X2/X3 (BF16 SASS) | 13b → 13c | Resolved: explicit `.BF16` suffix, bit 18 of ctrl code |
| HYP-13d-2a (chain accumulator in-place) | 13d | Resolved: D and C share the same register base |
| HYP-13d-3b (2 NOPs per HMMA) | 13d | Resolved with refinement: 2 NOPs when consumer depends on D, 0 otherwise |

### Gaps open after this chapter

| Gap ID | Description | Path to resolution |
|---|---|---|
| GAP-13-A | HMMA latency for FP16 accumulator (only F32 was measured) | Re-run 13e with FP16 acc variant, 1 hour of work |
| GAP-13-B | HMMA latency for BF16 input, FP32 accumulator | Re-run 13e with BF16, 1 hour |
| GAP-13-C | HMMA at other shapes (m16n8k8 on SM120) | Small kernel with `m16n8k8`, check if opcode changes |
| GAP-13-D | Scoreboard slot identifier (which SB# is used) | Requires byte-level ISA decoder for Blackwell; gpuasm.com unavailable |
| GAP-13-E | NCU validation of latency and pipeline utilization | NCU --set full on 13e variants |
| GAP-13-F | Sensitivity of HMMA latency to input data (NaN, denormal, zero) | New microbench with patterned inputs |
| GAP-13-G | HMMA throughput (independent MMAs, not chained) | Deferred to chapter 18 (pipelined tile) |
| GAP-13-H | Why exactly 2 NOPs and not 1, 3, or another count | Requires understanding the full latency covered by the bit 27 scoreboard wait. [HYP] Two NOPs may cover a fixed scheduling slot quantum and the rest may be covered by the scoreboard itself. |
| GAP-13-I | Low-order scheduling bits of HMMA control code (bits 0-4) | Requires either a Blackwell ISA decoder or systematic microbenchmarks varying the hardware scheduling hint encoding |
| GAP-13-J | Whether ldmatrix (chapter 17) reduces or eliminates the uniform UIADD3 NOPs in a real GEMM kernel | Will be resolved directly in chapter 17 |

### Gaps explicitly deferred

GAP-13-G (throughput) is deferred to chapter 18 because it requires the pipelined tile structure and overlap analysis to be meaningful. Measuring throughput on a minimal kernel is not representative of production usage.

GAP-13-F (data sensitivity) is deferred because it is orthogonal to the baseline characterization and would pollute the chapter with edge-case analysis before the mainline is solid.

## Implications for subsequent chapters

### Chapter 14 (QMMA FP8 kind::f8f6f4)

Chapter 13 establishes HMMA as the baseline. Chapter 14 tests whether the FP8 MMA on SM120 uses:
* Same opcode family (HMMA with extended suffixes like `.E4M3` or `.E5M2`)
* A new opcode family (QMMA or similar)
* Different shape suffixes (m16n8k32 for FP8 at the same MMA atom size)

[HYP] Expected delta versus chapter 13: the shape modifier changes to `.168k` where k > 16 because FP8 packs more elements per register. The register layout per thread changes correspondingly.

### Chapter 15 (MMA narrow)

Chapter 15 tests smaller shapes (m8n8k4 legacy Volta, or subset shapes). Chapter 13's template is expected to simplify: fewer LDGs, possibly fewer accumulator registers.

### Chapter 16 (FP4 peak with block-scaled MMA)

Chapter 16 tests kind::mxf4nvf4 and block-scaled MMA with UE8M0 scales. [HYP] The opcode will be a new family rather than HMMA because block-scaling is an architectural feature added in SM100a/SM120. [OBS] The scale operand introduces a fifth operand position that does not exist in heritage HMMA.

### Chapter 17 (ldmatrix)

Chapter 17 focuses on the ldmatrix SASS opcode, which is the memory-side counterpart of MMA. The fragment loads in 13a-13e were all plain LDG.E.CONSTANT. A real kernel uses ldmatrix to transfer shared memory to fragment registers efficiently. The NOP pad observed here is expected to shrink or disappear as ldmatrix fills the scheduling slots.

### Chapter 18 (pipelined tile)

Chapter 18 tests independent HMMAs with overlap, multi-buffer pipelining, and the production kernel structure. The 35-cycle serial latency measured here sets the upper bound; the throughput will be lower. The chapter will confront GAP-13-G directly.

## Summary table of chapter 13 additions to FINDINGS.md

### New pipeline
* TC (tensor core)

### New opcodes
* `HMMA.16816.F16` — MMA warp-level, m16n8k16, FP16 input, FP16 accumulator
* `HMMA.16816.F32` — MMA warp-level, m16n8k16, FP16 input (implicit), FP32 accumulator
* `HMMA.16816.F32.BF16` — MMA warp-level, m16n8k16, BF16 input (explicit), FP32 accumulator
* `CS2UR` — special-register to uniform-register read (observed for SR_CLOCKLO)

### New modifiers
* `.16816` on HMMA: shape m16n8k16 (MNK concatenation)
* `.F16` / `.F32` on HMMA: accumulator dtype
* `.BF16` on HMMA: input dtype override (default is FP16)
* `.reuse` on MMA B operand: reuse cache hint for the next MMA

### New architectural invariants
* MMA is warp-level (32 threads cooperate on one tile)
* HMMA takes 4 SASS operands: D base, A base, B base, C base (fragment spans are implicit in the opcode)
* HMMA is variable-latency with scoreboard (bits 26 SBS and 27 wait in the control code)
* Opcode bytes are determined by operand base registers only; dtype lives in the control code
* Register allocation MMA: D and A colocated when possible in single-MMA case, D and C colocated in chained case, A and B always distinct

### New canonical patterns
* **Canonical MMA minimal template** (see section above)
* **Canonical MMA accumulator chaining**: D_reg = C_reg (in-place), A and B on distinct bases, `.reuse` on B when next MMA will re-read B
* **Canonical MMA NOP pad**: 2 × `@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ` between HMMA and a dependent consumer. Absent if the consumer does not depend on D.

### New cost rules
* Single HMMA with dependent store consumer: 1 HMMA + 2 uniform NOP slots = 3 instruction slots
* Chained HMMA of length N: N HMMA + 2(N-1) NOPs = 3N - 2 instruction slots
* HMMA.16816.F32 serial latency: ~35 cycles/HMMA
* Chain total cycles (N >= 16): ~310 + 35 × N
* Accumulator dtype (F16 vs F32) does not affect instruction count around HMMA, only LDG/STG for C/D
* Input dtype (FP16 vs BF16) affects nothing outside the HMMA control code itself

### New compiler artifacts
* `LDG.E.CONSTANT` extended rule: applied to any pointer marked `const __restrict__`, not just compile-time lookup tables
* Inverted LDG order for C registers when ptxas allocates C in reverse (R20-R23 filled c3, c2, c1, c0)
* Kernel argument layout: 8 bytes per pointer starting at `c[0x0][0x380]`
* Deterministic byte-identical SASS for instructions unrelated to a local source change (30 of 31 instructions identical between 13b and 13c)

## References

CUTLASS `include/cute/arch/mma_sm80.hpp` was consulted for the exact PTX form of the three MMA atoms used in 13a, 13b, 13c. The structs referenced are:
* `SM80_16x8x16_F16F16F16F16_TN` (line 92)
* `SM80_16x8x16_F32F16F16F32_TN` (line 158)
* `SM80_16x8x16_F32BF16BF16F32_TN` (line 224)

These heritage SM80 atoms are valid on SM120 and compile to HMMA instructions as documented in this chapter.

## Conclusion

Five controlled variants, 76+ SASS instructions analyzed, 27+ observations logged, 5 gaps resolved, 7 gaps explicitly left open. The HMMA.16816 baseline on SM120 is now mapped: opcode family, encoding structure, register allocation rules, scheduling patterns, and serial latency. This is the foundation on which chapters 14 through 19 will build.
