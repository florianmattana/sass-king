# Chapter 14 — QMMA FP8/FP6/FP4 baseline (kind::f8f6f4, m16n8k32)

## Goal

This chapter establishes the QMMA opcode family on SM120 — the SASS-level realization of `mma.sync.aligned.kind::f8f6f4.*` from PTX. Ten kernel variants (14a-14j) cover every dtype combination available in the family, the chaining pattern for accumulator-in-place, and a serial latency microbenchmark.

The chapter answers four primary questions:

1. **What SASS opcode emerges from `kind::f8f6f4`?** A new family `QMMA`, distinct from HMMA established in chapter 13.
2. **How are input dtypes encoded?** A 3-bit-per-operand field in the control code, fully decoded across the 5 dtypes (E4M3, E5M2, E3M2, E2M3, E2M1) and validated by a Popper-style prediction (variant 14i).
3. **How does QMMA relate to HMMA?** Patterns for register allocation, chaining, scoreboard, NOPs, and accumulator dtype encoding are MMA-family wide invariants — they apply identically to HMMA and QMMA.
4. **What is the cost?** Serial latency model: `total_cycles ≈ 510 + 35 × N`. Marginal cost identical to HMMA. Fixed overhead +200 cycles.

## Toolchain note

The PTX feature `.kind::f8f6f4` requires `-arch=compute_120a -code=sm_120a`. The shortcut `-arch=sm_120a` is not accepted by nvcc 13.2 even though `sm_120a` is supported via the explicit compute/code form. Plain `-arch=sm_120` rejects with `Feature '.kind::f8f6f4' not supported on .target 'sm_120'`. Output binaries carry the `EF_CUDA_ACCELERATORS` flag in their header — a marker for any kernel using sm_120a-specific PTX features.

## Variants

| Variant | Inputs | Acc | Purpose |
|---|---|---|---|
| 14a | E4M3 × E4M3 | F32 | Baseline single QMMA |
| 14b | E4M3 × E5M2 | F32 | First mixed input dtype, identifies B dtype encoding |
| 14c | E5M2 × E5M2 | F32 | Symmetric A dtype, identifies A dtype encoding |
| 14d | E2M1 × E2M1 | F32 | First FP4 test, validates QMMA covers full kind::f8f6f4 family |
| 14e | E4M3 × E4M3 | F32 | Two chained QMMAs, accumulator in-place |
| 14f | E4M3 × E4M3 | F32 | N-chain serial latency (N=16/32/64) |
| 14g | E3M2 × E3M2 | F32 | First FP6 test (e3m2) |
| 14h | E2M3 × E2M3 | F32 | Second FP6 test (e2m3), invalidates initial encoding model |
| 14i | E4M3 × E2M1 | F32 | Mixed inter-family, Popper-style prediction validation |
| 14j | E4M3 × E4M3 | F16 | FP16 accumulator, identifies acc dtype encoding bits |

## Key SASS observations

### The QMMA opcode

```
QMMA.16832.<acc>.<inputA>.<inputB> R_d, R_a, R_b, R_c
```

* `.16832` is the shape m16n8k**32** (vs HMMA's `.16816` for m16n8k**16**)
* `.F32` or `.F16` is the accumulator dtype
* `.E4M3`, `.E5M2`, `.E3M2`, `.E2M3`, `.E2M1` are input dtypes (both A and B explicit, no implicit default)
* 4 SASS operands as in HMMA: D base, A base, B base, C base. Fragment spans implicit in opcode

### Opcode bytes invariance

The opcode bytes `0x000000100c0c727a` are **invariant across 6 variants of input dtype** (E4M3.E4M3, E4M3.E5M2, E5M2.E5M2, E3M2.E3M2, E2M3.E2M3, E2M1.E2M1, E4M3.E2M1) and across both accumulator dtypes (F32 in 14a, F16 in 14j) when the operand bases are the same. Dtype encoding lives entirely in the control code.

### Control code topology — fully decoded

By comparing 10 variants byte-by-byte, 13 bits of the QMMA control code have been decoded:

| Bit | Meaning | Evidence |
|---|---|---|
| 1 | Acc dtype = F16 | 14j vs 14a |
| 2 | Acc dtype = F32 | 14j vs 14a |
| 4 | Always set on QMMA | invariant |
| 10, 11 | Always set on QMMA (MMA family marker) | invariant |
| 13 | F32 acc auxiliary bit | 14j vs 14a |
| 14 | A input mantissa ≠ 3 | 14b/c/d/g/h/i |
| 15 | B input mantissa ≠ 3 | 14b/c/d/g/h/i |
| 18 | A input exp = 3 (E3M2) | 14g |
| 19 | A input exp = 2 (E2M3, E2M1) | 14d/h |
| 20 | B input exp = 3 (E3M2) | 14g |
| 21 | B input exp = 2 (E2M3, E2M1) | 14d/h |
| 26 | MMA scoreboard set (SBS) | 14e/f |
| 27 | MMA scoreboard wait | 14e/f |

### Input dtype encoding model

Each operand is encoded by **3 independent bits** in the control code:

```
For operand A: bits 14, 18, 19
For operand B: bits 15, 20, 21

bit "mant≠3" = 1 if mantissa is 1 or 2 (E5M2, E3M2, E2M1)
bit "exp=3"  = 1 if exponent bits = 3 (E3M2 only)
bit "exp=2"  = 1 if exponent bits = 2 (E2M3, E2M1)
```

| dtype | exp | mant | bits | hex |
|---|---|---|---|---|
| E4M3 | 4 | 3 | 0 0 0 | default, no override |
| E5M2 | 5 | 2 | 1 0 0 | mant≠3 only |
| E3M2 | 3 | 2 | 1 1 0 | mant≠3 + exp=3 |
| E2M3 | 2 | 3 | 0 0 1 | exp=2 only |
| E2M1 | 2 | 1 | 1 0 1 | mant≠3 + exp=2 |

The encoding is **fully orthogonal between A and B** (no inter-operand interaction). The hardware uses 5 of 8 possible 3-bit codepoints, leaving room for future format extensions.

### Validation by prediction

After deriving the model from the 5 symmetric variants (14a, 14c, 14d, 14g, 14h), the asymmetric mixed case 14i (A = E4M3, B = E2M1) was used as a Popper-style test. The model predicted `ctrl = 0x004ff6000020ac14`. The observed value matched **exactly**. This validates the encoding as scientifically robust.

### Architectural interpretation

The encoding suggests the SM120 tensor core has MAC units organized by exponent class (4, 3, 2) with a sub-mode for mantissa-3 vs alternative mantissa widths. This is consistent with a unified FP8/FP6/FP4 hardware that selects internal datapath via these 3 bits per operand.

### Accumulator dtype encoding (MMA-family wide)

Comparing 14j (F16 acc) to 14a (F32 acc), the encoding is:

* **Bit 1** = 1 if accumulator is F16
* **Bit 2** = 1 if accumulator is F32
* Bit 13 (QMMA) or bit 12 (HMMA) is an auxiliary bit set in F32 mode

This encoding is **identical between HMMA and QMMA**. It is the 9th MMA-family wide invariant established in this chapter.

### Register allocation (MMA-family wide)

QMMA follows the same allocation rules as HMMA from chapter 13:

* **Single MMA**: D and A colocated. When D is smaller than A (F16 acc, 2 vs 4 registers), D occupies the first 2 of A's 4 registers. When D and A have the same size (F32 acc, 4 vs 4), they share the same 4 registers entirely.
* **Chained MMA** (14e): D and C colocated (accumulator in-place). A and B keep their own distinct register blocks because they are re-read by each QMMA.

Fragment layouts per thread:

| Acc dtype | D | A | B | C |
|---|---|---|---|---|
| F32 (14a, 14e, 14f, 14i) | float[4] | uint32[4] | uint32[2] | float[4] |
| F16 (14j) | uint32[2] | uint32[4] | uint32[2] | uint32[2] |

A and B register counts are independent of the input dtype: 4 uint32 for A, 2 uint32 for B. The element packing within each uint32 changes (4 e4m3 or 8 e2m1 per uint32), but ptxas always uses the same number of registers.

### Chaining (14e)

Two chained QMMAs with accumulator in-place produce SASS analogous to 13d for HMMA:

```
QMMA #1: R16, R12, R10.reuse, R16    ctrl 0x084ff60000002c10
2 × @!UPT UIADD3 NOPs
QMMA #2: R16, R12, R10, R16          ctrl 0x000ff60000002c10
2 × @!UPT UIADD3 NOPs
```

Same patterns observed in HMMA chapter 13 hold:

* `.reuse` on B operand of every QMMA except the last
* 2 UIADD3 NOPs after each QMMA when consumer depends on D
* D and C colocated (accumulator in-place)
* A on a distinct register block, re-read by each QMMA

### Scoreboard scheme (14e, 14f)

The high byte zone of the control code is byte-identical between HMMA chains (chapter 13e) and QMMA chains:

```
First MMA (chain start):  0x084ff6...  → bits 26 (SBS) + 27 (wait) set
Mid-chain MMAs:           0x080ff6...  → bit 27 only (wait, no SBS)
Last MMA (no MMA after):  0x000ff0...  → no wait, no SBS
```

The MMA scoreboard scheme is fully MMA-family wide.

### Latency (14f)

Three measurements at N = 16, 32, 64 chained QMMAs:

| N | total_cycles | cycles per QMMA |
|---|---|---|
| 16 | 1070 | 66.88 |
| 32 | 1637 | 51.16 |
| 64 | 2742 | 42.84 |

Linear regression on incremental costs:
* ΔN(16→32): 567 cycles for 16 added QMMAs → 35.4 cycles/QMMA
* ΔN(32→64): 1105 cycles for 32 added QMMAs → 34.5 cycles/QMMA

**Model**: `total_cycles ≈ 510 + 35 × N`.

Marginal cost per QMMA (~35 cycles) is **identical to HMMA.16816.F32** measured in chapter 13e. This is striking because QMMA m16n8k32 performs **2× more FMAs internally** than HMMA m16n8k16 (k=32 vs k=16), yet completes in the same serial latency. Effective FMA throughput per cycle is 2× higher for FP8 QMMA than for FP16 HMMA, consistent with Blackwell consumer FP8 throughput specs.

Fixed chain overhead is ~510 for QMMA vs ~310 for HMMA. [HYP] The +200 cycles may reflect a deeper pipeline startup for the FP8 MAC units. [INF] This overhead is amortized in production GEMM when N is much larger than the latency-probe chain sizes.

## MMA-family wide invariants

After chapter 14, the following patterns are confirmed common to HMMA and QMMA:

1. **Single-MMA register allocation**: D and A colocated (with partial overlap when D < A in size).
2. **Chained-MMA register allocation**: D and C colocated. A and B on distinct bases.
3. **`.reuse` on B operand** of every MMA in a chain except the last.
4. **2 UIADD3 NOPs** after each MMA when the consumer depends on D.
5. **MMA scoreboard scheme**: bit 26 (SBS) on first MMA of chain, bit 27 (wait) on dependent MMAs.
6. **Inverted LDG order** for one operand (B or C depending on kernel).
7. **Late destination address**: IMAD.WIDE for store address placed just before the MMA.
8. **Template kernel structure**: prologue + LDG fragments + MMA + NOPs + STG, byte-identical between HMMA and QMMA when operand sizes match.
9. **Accumulator dtype encoding**: bit 1 = F16 acc, bit 2 = F32 acc.

These 9 invariants justify treating MMA patterns as **MMA-family wide** in the cross-chapter summary, rather than opcode-specific.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| QMMA is a new opcode, not an HMMA extension | Confirmed |
| QMMA opcode bytes invariant across input dtypes | Confirmed (6 variants) |
| QMMA opcode bytes invariant across acc dtypes | Confirmed (14j, when operand bases match) |
| QMMA covers FP8, FP6, and FP4 in kind::f8f6f4 family | Confirmed |
| Input dtype encoding is per-operand symmetric | Confirmed by 14c |
| Input dtype encoding is fully orthogonal A/B | Confirmed by 14i Popper test |
| QMMA serial latency ~35 cycles, identical to HMMA | Confirmed |
| Acc dtype encoding identical HMMA/QMMA | Confirmed (14j vs 13a/13b) |
| Chaining patterns identical HMMA/QMMA | Confirmed (14e vs 13d) |
| Scoreboard scheme identical HMMA/QMMA | Confirmed (14f vs 13e) |

## Open gaps

| Gap | Notes |
|---|---|
| Layout fragment FP4 | Variant 14d returned d[0]=2.0 instead of expected 32.0 with naive `0x22222222` packing. The SASS observation is unaffected (opcode/control code do not depend on input values), but the productive use of FP4 inputs requires the actual SM120 FP4 fragment layout, which is not trivial. To investigate in a dedicated chapter for production FP4 attention. |
| Single-MMA vs chain-last control code delta | Some bits beyond 26/27 differ between a single QMMA (14a) and a chain-last QMMA (14e #2). Possibly scoreboard slot ID encoding. Not decoded. |
| QMMA at other shapes | m16n8k32 only tested. Other shapes from CUTLASS (if any in kind::f8f6f4 family) not tested. |
| Block-scaled MMA (kind::mxf8f6f4) | Deferred to chapter 16 (block-scaled FP4 peak). |
| Shape encoding bits | The bits that encode m16n8k32 (vs hypothetical other shapes) are inside bits 10-13 zone. Cannot be isolated until another shape is tested. |

## How to read a QMMA SASS dump

When you see a QMMA in a production kernel:

1. **Identify the dtype** from the mnemonic: `QMMA.16832.<acc>.<inputA>.<inputB>`. Both inputs are explicit.
2. **Verify the opcode bytes** start with `0x000000??0c0c727a` (or similar pattern with operand-base substitutions). This confirms QMMA opcode family.
3. **Decode the input dtype bits** in the control code (bits 14-15 for "alt mantissa", 18-21 for exp class).
4. **Check the chaining context** via control code high bytes:
   * `0x084ff6...` = first of chain (sets scoreboard)
   * `0x080ff6...` = mid-chain (waits on scoreboard)
   * `0x000ff0...` = last (no wait, consumer is non-MMA)
5. **Look at the surrounding NOPs**: 2 UIADD3 NOPs after each QMMA when its D feeds another instruction.
6. **Check for `.reuse` on B**: signals chain context.

## Summary

* 1 new SASS opcode: `QMMA.16832.<acc>.<inputA>.<inputB>`
* 1 new SASS opcode: `CS2UR` (already from chapter 13e, also used in 14f)
* 13 control code bits decoded
* 5 input dtypes encoded with 3 bits each (orthogonal A/B)
* 2 accumulator dtypes encoded
* 9 MMA-family wide invariants confirmed
* Latency model: `total_cycles ≈ 510 + 35 × N`
* Toolchain rule: `kind::f8f6f4` requires `-arch=compute_120a -code=sm_120a`
