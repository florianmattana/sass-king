# 24 Production mini-GEMM audit

## Scope

* [OBS] This chapter studies production-like mini-GEMM SASS structure through 30 controlled probes, 24a through 24ad.
* [OBS] The probes compile with `nvcc -arch=compute_120a -code=sm_120a` and are dumped with `cuobjdump --dump-sass`.
* [OBS] The chapter covers HMMA, QMMA, OMMA, sparse QMMA/OMMA, cp.async, LDSM, STSM, global-store epilogues, predicated bounds, loop structure, register pressure, descriptor arithmetic, scale loads, metadata loads, split-K reduction, and cold error paths.
* [OBS] Runtime smoke execution succeeds on the host RTX 5070 Ti for all 30 variants.
* [GAP] These are structural audit probes, not a full numeric GEMM correctness suite.

## Sources

| File | Purpose |
|---|---|
| `24_production_mini_gemm.cu` | [OBS] Parameterized source for variants 24a through 24ad. |
| `compile.sh` | [OBS] Rebuilds all binaries and SASS dumps. |

## Variants studied

| Variant | Probe | Main SASS result |
|---|---|---|
| 24a | Minimal HMMA tile | [OBS] `HMMA.16816.F32`, `STG.E.128`. |
| 24b | Minimal QMMA E4M3 tile | [OBS] `QMMA.16832.F32.E4M3.E4M3`, `STG.E.128`. |
| 24c | Minimal QMMA E2M1 tile | [OBS] `QMMA.16832.F32.E2M1.E2M1`, `STG.E.128`. |
| 24d | Minimal scaled OMMA FP4 tile | [OBS] `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, `STG.E.128`. |
| 24e | cp.async single stage | [OBS] `LDGSTS`, `LDGDEPBAR`, `DEPBAR.LE`, `BAR`, `LDSM`, `HMMA`, `STG`. |
| 24f | cp.async double buffer | [OBS] Two `LDGSTS` groups before `DEPBAR.LE SB0, 0x0`, then `LDSM`, `HMMA`, `STG`. |
| 24g | LDSM order A/B | [OBS] `LDSM.16.M88.2` and `LDSM.16.M88.4` feed `HMMA`. |
| 24h | Accumulator chain depth | [OBS] Four chained `QMMA.16832.F32.E4M3.E4M3` instructions. |
| 24i | Epilogue STG scalar baseline | [OBS] Store-only baseline emits `STG.E.128` with no MMA. |
| 24j | Epilogue STSM shared | [OBS] `QMMA`, `STSM.16.M88.4`, `BAR`, `STG`. |
| 24k | Epilogue shared to global | [OBS] `QMMA`, `STSM.16.M88.4`, `BAR`, shared reload, `STG`. |
| 24l | Predicated bounds epilogue | [OBS] `HMMA` followed by bounded `STG.E.128`. |
| 24m | Preserved tile loop | [OBS] Dynamic loop retains repeated `HMMA` structure. |
| 24n | Unrolled tile loop | [OBS] Fixed loop emits five visible `HMMA` instructions. |
| 24o | Guarded divergent tile | [OBS] Guarded path emits predicated `HMMA` and `WARPSYNC.ALL`. |
| 24p | Register pressure variant | [OBS] 112 useful instructions, `HMMA`, arithmetic pressure, `STG`. |
| 24q | Shared alignment variant | [OBS] Offset `LDSM` addresses feed `HMMA`. |
| 24r | Sparse QMMA tile | [OBS] `QMMA.SP.16864.F32.E4M3.E4M3`, `STG`. |
| 24s | Scaled sparse OMMA tile | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X`, `STG`. |
| 24t | Full audit checklist dump | [OBS] Combines double-buffer `LDGSTS`, `DEPBAR`, `BAR`, `LDSM`, `HMMA`, `STG`. |
| 24u | Warpgroup absence check | [OBS] Emits warp-level `HMMA`; no `WGMMA` or `TCGEN` mnemonics appear. |
| 24v | Uniform register path | [OBS] `S2UR`, parameter `LDG`, `HMMA`, `STG`. |
| 24w | Descriptor/address arithmetic | [OBS] Parameter load and address arithmetic feed `HMMA`, then `STG`. |
| 24x | Barrier/wait variants | [OBS] Three `LDGSTS` groups and waits `SB0,2`, `SB0,1`, `SB0,0`. |
| 24y | Vectorized global store epilogue | [OBS] `HMMA` followed by `STG.E.128`. |
| 24z | Split-K / multi-CTA reduction stub | [OBS] `HMMA`, `REDG.E.ADD.F32...`, and `STG.E.128`. |
| 24aa | Scale load path | [OBS] Parameter `LDG.E.CONSTANT` loads feed `OMMA.SF...UE4M3.4X`. |
| 24ab | Metadata load path | [OBS] Parameter `LDG.E.CONSTANT` feeds `QMMA.SP` metadata operand. |
| 24ac | Nontrivial layout strides | [OBS] Stride parameters and address arithmetic feed `HMMA`, then `STG`. |
| 24ad | Cold error/assert path | [OBS] `BSSY`, `HMMA`, `BSYNC`, `BPT.TRAP`, `STG`. |

## Commands

```bash
cd tensor_cores/24_production_mini_gemm
bash compile.sh
for exe in build/24*; do "$exe"; done
```

## Key answers

* [RES] A production-like SM120 mini-GEMM audit can be decomposed into visible SASS regions: global-to-shared copy, async dependency wait, shared matrix load, tensor-core compute, optional shared epilogue, global store, and optional reduction.
* [RES] The tested warp-level path stays in `HMMA`, `QMMA`, `OMMA`, `QMMA.SP`, and `OMMA.SF.SP`; the warpgroup-only `WGMMA` and datacenter Blackwell `TCGEN` families do not appear in these SM120 probes.
* [RES] Scale values and sparse metadata remain separate audit channels: scale loads feed OMMA scale operands, and metadata loads feed sparse MMA metadata operands.
* [RES] Split-K style accumulation is visible as a reduction instruction (`REDG.E.ADD.F32...`) separate from the normal vectorized `STG.E.128` epilogue.
* [GAP] The probes validate structural SASS signatures and runtime launchability; full numerical validation of GEMM outputs remains outside this chapter.
