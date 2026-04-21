# Chapter 16 — FP4 peak block-scaled MMA on SM120

## Goal

This chapter decodes the block-scaled MMA family on SM120: the `kind::mxf8f6f4` and `kind::mxf4nvf4` variants of `mma.sync.aligned` with scale factors. These are the instructions behind the announced FP4 peak throughput (900+ TFLOPS) on Blackwell consumer.

Four kernel variants (16a-16d) cover:

* The `mxf8f6f4` baseline with scale_vec::1X and ue8m0 scales (16a)
* The `mxf4nvf4` standard path with scale_vec::2X and ue8m0 scales (16b)
* The `mxf4nvf4` peak path with scale_vec::4X and ue4m3 scales (16c)
* Latency microbenchmark on the peak path (16d)

The chapter answers four primary questions:

1. **What SASS opcode does block-scaled MMA emit?** `QMMA.SF` for mxf8f6f4, `OMMA.SF` (new opcode family) for mxf4nvf4.
2. **How are scale_vec and scale dtype encoded at SASS level?** Via mnemonic suffixes and control code bits, not in opcode bytes.
3. **Is OMMA slower than HMMA/QMMA due to extra scaling logic?** No, OMMA is faster per cycle (~29 vs ~35 cycles).
4. **What is the FP4 peak path exactly?** `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` with 4 ue4m3 scales per fragment.

## Variants

| Variant | PTX kind | Shape | Scale vec | Scale dtype | Purpose |
|---|---|---|---|---|---|
| 16a | mxf8f6f4 | m16n8k32 | 1X | ue8m0 | Baseline block-scaled, E4M3 inputs |
| 16b | mxf4nvf4 | m16n8k64 | 2X | ue8m0 | New shape k=64, FP4 standard |
| 16c | mxf4nvf4 | m16n8k64 | 4X | ue4m3 | FP4 peak path (900+ TFLOPS target) |
| 16d | mxf4nvf4 | m16n8k64 | 4X | ue4m3 | Latency microbench (N = 16, 32, 64) |

## Key SASS observations

### Three MMA opcode families on SM120

The block-scaled work revealed a new opcode family, completing the SM120 tensor core landscape:

| Family | Low byte | Shape | PTX kind | Scaled? |
|---|---|---|---|---|
| HMMA | 0x3c | m16n8k16 | mma.sync standard | No |
| QMMA | 0x7a | m16n8k32 | kind::f8f6f4, kind::mxf8f6f4 | Optional (.SF modifier) |
| **OMMA** | **0x7f** | m16n8k64 | kind::mxf4nvf4 | Always (implicit .SF) |

The low byte of the opcode is a reliable identifier of family, consistent across all variants observed.

### QMMA.SF for mxf8f6f4 (16a)

```
QMMA.SF.16832.F32.E4M3.E4M3.E8 R12, R12, R10, R16, R0, R0, URZ
opcode: 0x7000000a0c0c747a   ctrl: 0x004ff60000003e10
```

Mnemonic: existing QMMA opcode (low byte 0x7a) with new `.SF` modifier indicating "scale factor enabled". The scale dtype is appended as `.E8` suffix (ue8m0 = 8-bit exp, 0 mantissa).

Operand layout: 7 operands instead of 4 for plain QMMA:
* D, A, B, C (same as QMMA)
* SFA = scale factor for A (single uint32)
* SFB = scale factor for B (single uint32)
* URZ = uniform register zero for the bid/tid parameters (constants in CUTLASS atoms)

Register allocation observed: when SFA and SFB have equal values, ptxas colocates them in the same register (R0 in 16a).

### OMMA.SF for mxf4nvf4 (16b)

```
OMMA.SF.16864.F32.E2M1.E2M1.E8 R12, R12, R10, R16, R0, R0, URZ
opcode: 0x7000000a0c0c747f   ctrl: 0x004ff60000083e10
```

The mnemonic is structurally identical to QMMA.SF, with two changes:
* Shape is `.16864` (k=64) instead of `.16832`
* Low byte of opcode is `0x7f` instead of `0x7a` — this is a distinct opcode family, not a modifier of QMMA

`OMMA` is a new SASS opcode family introduced on Blackwell (presumably "O" for "Octal" since FP4 packs 8 values per uint32, or for the doubled k dimension). Distinct from QMMA at the opcode bytes level.

Operand layout identical to QMMA.SF (same 7 operands). Scale handling is the same.

### OMMA.SF with fine-grained scaling (16c, the 900 TFLOPS path)

```
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X R12, R12, R10, R16, R0, R0, URZ
opcode: 0x7000000a0c0c747f   ctrl: 0x004ff60000043e10
```

Two new modifiers appear:
* `.UE4M3` = scale dtype is ue4m3 (unsigned E4M3: 4 exp bits, 3 mantissa bits), replacing the implicit `.E8` (ue8m0)
* `.4X` = scale_vec::4X (4 scale factors per fragment, finer granularity than the default 2X for OMMA)

**Opcode bytes identical to 16b.** The entire difference between 16b and 16c lives in the control code.

### Scale vec and scale dtype encoded in control code

Comparing byte 2 of the control code across variants:

| Variant | Config | Byte 2 |
|---|---|---|
| 16a | QMMA.SF 1X ue8m0 | 0x00 |
| 16b | OMMA.SF 2X ue8m0 | 0x08 (bit 19) |
| 16c | OMMA.SF 4X ue4m3 | 0x04 (bit 18) |

Two candidate encodings:

**Interpretation 1 (direct mapping)**: each scale_vec + scale_dtype combination has a unique value in byte 2. The 3 combinations exercised correspond to the 3 production paths supported by CUTLASS on SM120.

**Interpretation 2 (orthogonal bits)**:
* Bit 19 = scale_vec::2X indicator (set only in 16b)
* Bit 18 = ue4m3 + 4X mode (set only in 16c)

Without a fourth variant (e.g. mxf4nvf4 4X with ue8m0), we cannot disambiguate. All CUTLASS atoms observed match one of the three variants decoded, so this ambiguity has no practical impact for audit work.

### Mnemonic convention: default is silent

A useful pattern observed:
* For QMMA (mxf8f6f4), `scale_vec::1X` is default and **not tagged** in the mnemonic.
* For OMMA (mxf4nvf4), `scale_vec::2X` is default and **not tagged** either.
* `scale_vec::4X` is not the default for either family and **requires explicit `.4X` tag**.
* Similarly, `ue8m0` scale dtype is abbreviated `.E8` (the common case), while `ue4m3` is written out fully `.UE4M3`.

This matches broader SASS convention: the common case is silent, deviations are tagged.

### OMMA latency (16d): surprisingly fast

Serial chain latency measurements on OMMA.SF 4X (N = 16, 32, 64):

| N_MMA | Cycles total | Cycles/MMA |
|---|---|---|
| 16 | 789 | 49.3 |
| 32 | 1250 | 39.1 |
| 64 | 2180 | 34.1 |

Linear fit: **total_cycles(N) ≈ 330 + 29 × N**

Asymptotic cycles/MMA: **~29** (vs ~35 for HMMA and QMMA on SM120).

Comparison of chain-latency throughput across MMA families:

| Opcode | Shape | FLOPs/MMA | Cycles/MMA | FLOPs/cycle |
|---|---|---|---|---|
| HMMA | m16n8k16 | 4096 | 35 | 117 |
| QMMA | m16n8k32 | 8192 | 35 | 234 |
| **OMMA 4X** | **m16n8k64** | **16384** | **29** | **565** |

OMMA delivers **4.8× more FLOPs per cycle than HMMA** at the latency floor. The peak 900+ TFLOPS announced by NVIDIA is not achievable by a latency-bound chain (we measured ~254 TFLOPS single-warp from this microbench), but the tensor core pipeline can issue multiple MMAs in flight simultaneously, pushing throughput above the chain-bound figure.

### Register reuse in OMMA chain (16d)

Chained OMMAs show ptxas applies `.reuse` flags on A (R2) and SFB (R8) operands:

```
OMMA R12, R4, R2.reuse, RZ,  R8, R8.reuse, URZ   # first: C=RZ
OMMA R12, R4, R2.reuse, R12, R8, R8.reuse, URZ   # chain: C=R12 (previous D)
```

`.reuse` keeps these operands in the register cache between consecutive OMMAs, avoiding re-fetch from register file. Same pattern as observed in QMMA chain (chap 14e).

### Control code byte 0 change in chain

First OMMA in the chain (no prior dependency) has control code byte 0 = 0xff (wait mask empty). Second and subsequent OMMAs have byte 0 = 0x0c, indicating explicit wait on the scoreboard slot where the previous D was written.

Consistent with the scoreboard-wait mechanism decoded in earlier chapters.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| Block-scaled MMA produces a new SASS opcode | Confirmed for OMMA (low byte 0x7f). QMMA.SF is a modifier variant, not new opcode. |
| Scale factors emit 2 new operands (SFA, SFB) | Confirmed |
| Shape m16n8k64 is specific to mxf4nvf4 | Confirmed |
| `.UE4M3.4X` tags mark the fine-grained mode | Confirmed |
| scale_vec and scale dtype live in control code byte 2 | Confirmed |
| OMMA latency similar to QMMA (~35 cyc) | Partial: OMMA is **faster** at ~29 cyc |
| Peak 900 TFLOPS accessible via chain latency | Rejected: chain gives ~254 TFLOPS; peak requires pipelined throughput |

## Open gaps

| Gap | Notes |
|---|---|
| Orthogonal vs direct-map encoding of byte 2 | CUTLASS exposes 3 variants only; mxf4nvf4 4X with ue8m0 would disambiguate but is not a CUTLASS atom |
| OMMA throughput microbench | Chain latency measures ~29 cyc/MMA; pipelined throughput across independent MMAs could reveal the real 900 TFLOPS peak |
| Scale factor values other than 1.0 | All variants tested with scale=1.0. Real block-scaled kernels use diverse scale values per block |
| Direct comparison with production FP4 GEMM SASS | CUTLASS and Marlin FP4 kernels would validate decoded patterns in production context |
| FP4 register layout (E2M1) | Still unresolved from chap 14d [GAP-14d-1]. Independent of opcode decoding but blocks full audit of FP4 kernels |

## Patterns cristallized

* [OBS] **OMMA is a distinct SASS opcode family** with low byte 0x7f, introduced on Blackwell for the `kind::mxf4nvf4` path. Shape always m16n8k64, always block-scaled.
* [OBS] **Mnemonic convention for block-scaled MMA**: default scale_vec (1X for QMMA, 2X for OMMA) is silent; deviations are tagged explicitly (`.4X`). Scale dtype ue8m0 abbreviates to `.E8`; ue4m3 written fully.
* [OBS] **Byte 2 of control code encodes the scaling mode** (scale_vec × scale_dtype combination) for block-scaled MMA. Opcode bytes do not change between scale_vec::2X and scale_vec::4X with different dtypes.
* [OBS] **OMMA chain latency ~29 cycles/MMA**, lower than HMMA and QMMA (~35). Blackwell tensor cores are optimized for FP4: larger shape + lower latency = 4.8× more FLOPs per cycle than HMMA.
* [OBS] **The 900 TFLOPS peak requires pipelined throughput**, not achievable in a latency-bound chain. Our microbench gives ~254 TFLOPS single-warp, leaving ~3.5× headroom for pipelined execution.

## Summary

* 3 new SASS opcodes/variants: QMMA.SF (block-scaled mxf8f6f4), OMMA (new family for mxf4nvf4), OMMA.SF (block-scaled, always implicit)
* Low byte 0x7f identifies OMMA, distinct from QMMA (0x7a) and HMMA (0x3c)
* Scale vec and scale dtype encoded in control code byte 2 (not in opcode bytes)
* `.UE4M3.4X` tags the fine-grained FP4 peak path
* OMMA latency measured at ~29 cycles/MMA, faster than HMMA/QMMA
* OMMA throughput: 565 FLOPs/cycle single-warp chain (254 TFLOPS), 4.8× HMMA
* Full landscape of SM120 tensor core opcodes now documented: HMMA (FP16/BF16), QMMA (FP8/FP6/FP4), OMMA (FP4 peak)
