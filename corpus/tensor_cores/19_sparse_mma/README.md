# 19 Sparse MMA (2:4 structured sparsity)

## Scope

* [OBS] This chapter studies warp-level `mma.sp::ordered_metadata` on SM120 using inline PTX compiled with `nvcc -arch=compute_120a -code=sm_120a`.
* [OBS] The accepted probes cover non-scaled `kind::f8f6f4`, F16 accumulator, metadata value changes, block-scaled `kind::mxf8f6f4`, block-scaled `kind::mxf4nvf4`, and a sparse QMMA dependency chain.
* [GAP] Runtime numeric validation and latency timing are blocked by the unavailable NVIDIA driver in the local environment.

## Sources

| File | Purpose |
|---|---|
| `19_sparse_qmma_probe.cu` | [OBS] Parameterized non-scaled sparse `kind::f8f6f4` probe. |
| `19_sparse_block_scale_probe.cu` | [OBS] Parameterized sparse block-scaled probe for `kind::mxf8f6f4` and `kind::mxf4nvf4`. |
| `19_sparse_qmma_chain.cu` | [OBS] Parameterized sparse `kind::f8f6f4` dependency-chain probe. |

## Variants studied

| Variant | PTX form | SASS result |
|---|---|---|
| 19a | `kind::f8f6f4.f32.e4m3.e4m3.f32`, metadata `0xaaaaaaaa`, selector `0` | [OBS] `QMMA.SP.16864.F32.E4M3.E4M3` |
| 19b | `kind::f8f6f4.f32.e4m3.e5m2.f32`, metadata `0xaaaaaaaa`, selector `0` | [OBS] `QMMA.SP.16864.F32.E4M3.E5M2` |
| 19c | `kind::f8f6f4.f32.e3m2.e2m3.f32`, metadata `0xaaaaaaaa`, selector `0` | [OBS] `QMMA.SP.16864.F32.E3M2.E2M3` |
| 19d | `kind::f8f6f4.f32.e2m1.e2m1.f32`, metadata `0xaaaaaaaa`, selector `0` | [OBS] `QMMA.SP.16864.F32.E2M1.E2M1` |
| 19e | Same as 19a, metadata `0x55555555` | [OBS] Same `QMMA.SP` encoding as 19a, metadata producer changes to `MOV R0, 0x55555555` |
| 19f | Same as 19a, metadata `0xffffffff` | [OBS] Same `QMMA.SP` encoding as 19a, metadata producer changes to `MOV R0, 0xffffffff` |
| 19g | Same as 19a, selector `1` | [OBS] Rejected by ptxas: expected selector `0` |
| 19h | `kind::f8f6f4.f16.e3m2.e2m1.f16`, metadata `0xaaaaaaaa`, selector `0` | [OBS] `QMMA.SP.16864.F16.E3M2.E2M1` |
| 19i | `kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0` | [OBS] `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8` |
| 19j | `kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0` | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8` |
| 19k | `kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` |
| 19l | `kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0` | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8.4X` |
| 19m | 16 x sparse `kind::f8f6f4.f32.e4m3.e4m3.f32` chain | [OBS] 16 x `QMMA.SP.16864.F32.E4M3.E4M3` |

## Key answers

* [RES] Sparse non-scaled `kind::f8f6f4` is not a new SASS opcode family in the tested SM120 form. It emits `QMMA.SP` with low byte `0x7a`.
* [RES] Sparse block-scaled `kind::mxf8f6f4` emits `QMMA.SF.SP` with low byte `0x7a`.
* [RES] Sparse block-scaled `kind::mxf4nvf4` emits `OMMA.SF.SP` with low byte `0x7f`.
* [RES] Sparse metadata is an explicit SASS register operand.
* [OBS] The selector appears as the final immediate operand in SASS.
* [OBS] Selector value `1` is rejected by ptxas for the tested `kind::f8f6f4` sparse form.
* [INF] Sparse forms double the displayed SASS K field relative to the dense family baseline: dense `QMMA.16832` becomes sparse `QMMA.SP.16864`, and dense `OMMA.SF.16864` becomes sparse `OMMA.SF.SP.168128`.

## Commands

```bash
nvcc -arch=compute_120a -code=sm_120a -DA_DTYPE=e4m3 -DB_DTYPE=e4m3 -DACC_F32=1 -DMETA_VALUE=0xaaaaaaaau -DSELECTOR_VALUE=0 -o build/19a_sparse_e4m3_e4m3 19_sparse_qmma_probe.cu
cuobjdump --dump-sass build/19a_sparse_e4m3_e4m3 > 19a_sparse_e4m3_e4m3.sass
```

## Open gaps

* [GAP] Runtime validity of individual metadata bit patterns is not known from SASS alone.
* [GAP] Sparse chain latency and sparse throughput are not measured in this environment.
* [GAP] Exact bit placement of the `.SP` mode in SASS encoding is only partially bounded by opcode/control-code comparisons.
