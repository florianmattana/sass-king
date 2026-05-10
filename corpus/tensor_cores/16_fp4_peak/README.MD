# 16 FP4 peak (k=64, block scaling, 900 TFLOPS)

FP4 MMA at shape m16n8k64. Non-scaled (`kind::f8f6f4`) and block-scaled (`kind::mxf8f6f4` with `scale_vec::2X`). This is where SM120 reaches its advertised peak of 900.4 TFLOPS for FP4.

Also covers `kind::mxf4` and `kind::mxf4nvf4` (CUDA 13.1+).

## Variants planned

| # | PTX atom | SF type | CUTLASS name |
|---|---|---|---|
| 16a | `mma.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e2m1.e2m1.f32` | none | `SM120_16x8x64_TN<e2m1,e2m1,f32>` |
| 16b | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0` | ue8m0 | `SM120_16x8x64_TN_VS<e2m1,e2m1,f32,ue8m0,32>` (**peak**) |
| 16c | `...scale_vec::2X.m16n8k64...ue4m3` | ue4m3 | `SM120_16x8x64_TN_VS<e2m1,e2m1,f32,ue4m3,32>` (peak) |
| 16d | `...kind::mxf4.block_scale.scale_vec::4X.m16n8k64...` (CUDA 13.1+) | ue8m0 | block16 smaller blocks |
| 16e | `...kind::mxf4nvf4...` | ue4m3 | NVIDIA FP4 variant |
| 16f | same as 16b, compiled `sm_120` vs `sm_120a` | — | Arch conditionality test |
| 16g | Lei Mao peak benchmark template (tight MMA loop, no memory) | — | Reproduce 900 TFLOPS |

## Key questions

1. Does block scaling introduce a new SASS opcode, or is it a modifier on the base MMA opcode?
2. How are the scale factors (SF_A, SF_B) passed: separate register operands, packed into A/B, or in a dedicated slot?
3. Why is the 900 TFLOPS peak reached only at k=64 and not at k=32 (same hardware)? SASS-level answer expected.
4. What does `scale_vec::2X` mean concretely? [HYP] 2 scales per 64-element K block (one per 32 elements). To verify.
5. Does `kind::mxf4nvf4` differ from `kind::mxf4` at SASS, or only at front-end?
6. Is there a SASS-level performance difference between `ue8m0` and `ue4m3` scale factor types?
7. Does `-arch=sm_120a` produce different SASS than `-arch=sm_120` for the same block-scaled atom?

## Context from FINDINGS.md

**From chapter 15**: sub-peak FP4 at k=32 reaches 238 TFLOPS. The 4× gap to 933 TFLOPS in this chapter must have structural causes.

**Architectural hypothesis (to verify)**: block-scaled atoms require `-arch=sm_120a`. Plain `sm_120` rejects or silently fails to generate.

**Cost rule extension**: this chapter adds cost rules for SF register load (pre-MMA) and for block-scaled MMA itself.

## Status

* [ ] 16a non-scaled FP4 k=64
* [ ] 16b block-scaled FP4 peak (ue8m0)
* [ ] 16c block-scaled FP4 peak (ue4m3)
* [ ] 16d kind::mxf4 block16
* [ ] 16e kind::mxf4nvf4
* [ ] 16f sm_120 vs sm_120a delta
* [ ] 16g peak reproduction benchmark
* [ ] conclusion16.md

## Dependencies

Chapter 14 (FP8 non-scaled) provides the `kind::f8f6f4` baseline. Chapter 15 (FP4 k=32) provides the sub-peak FP4 reference. This chapter is the main event of tensor core on SM120.