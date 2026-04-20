# 15 MMA narrow precision (FP6, FP4 at k=32)

FP6 and FP4 variants at shape m16n8k32 with `kind::f8f6f4`. FP4 at k=32 does not reach peak performance (peak FP4 requires k=64, chapter 16). This chapter documents the sub-peak form for reference.

## Variants planned

| # | A type | B type | C/D type | K | Note |
|---|---|---|---|---|---|
| 15a | e3m2 | e3m2 | f32 | 32 | First FP6 |
| 15b | e2m3 | e2m3 | f32 | 32 | Alternate FP6 format |
| 15c | e3m2 | e2m3 | f32 | 32 | Mixed FP6 |
| 15d | e2m1 | e2m1 | f32 | 32 | FP4 at k=32 (sub-peak reference) |
| 15e | e4m3 | e2m1 | f32 | 32 | Mixed FP8 + FP4 (cross-precision) |

## Key questions

1. Does FP6 have a distinct SASS opcode from FP8? Or same opcode with dtype modifier?
2. How are 6-bit elements packed into 32-bit registers? Likely 5 per register with 2 bits padding.
3. How are 4-bit elements packed? Likely 8 per register.
4. Why does FP4 at k=32 reach only 238 TFLOPS (1/4 of peak)? Hardware underutilization or wiring preference for larger K?
5. Does the SASS instruction count differ between k=32 FP4 (this chapter) and k=64 FP4 (chapter 16)?

## Context from FINDINGS.md

**From chapter 14**: `kind::f8f6f4` SASS mapping established for FP8. Chapter 15 extends the observation to FP6 and FP4 inputs under the same kind.

**Peak performance data from Lei Mao benchmark**: `SM120_16x8x32_TN<e2m1, e2m1, f32>` = 238 TFLOPS. `SM120_16x8x64_TN_VS<e2m1, e2m1, f32, ue8m0, 32>` = 933 TFLOPS. The ~4× gap must have a SASS-level explanation.

## Status

* [ ] 15a e3m2 e3m2 f32
* [ ] 15b e2m3 e2m3 f32
* [ ] 15c e3m2 e2m3 f32 mixed
* [ ] 15d e2m1 e2m1 f32 at k=32 (sub-peak)
* [ ] 15e e4m3 e2m1 f32 mixed
* [ ] conclusion15.md

## Dependencies

Chapter 14 establishes the `kind::f8f6f4` baseline for FP8. This chapter extends to narrower dtypes at the same shape.