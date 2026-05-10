# QMMA

## Scope

QMMA is the Blackwell SASS MMA family observed for dense, scaled, and sparse FP8/FP6/FP4 `mma.sync` / `mma.sp` forms on SM120a.

## Evidence

| Source | Evidence |
|---|---|
| `corpus/tensor_cores/14_qmma_fp8/` | [OBS] Dense `kind::f8f6f4` dtype variants and latency chains. |
| `corpus/tensor_cores/16_fp4_peak/` | [OBS] `QMMA.SF` block-scaled forms. |
| `corpus/tensor_cores/19_sparse_mma/` | [OBS] `QMMA.SP` and `QMMA.SF.SP` sparse forms. |
| `corpus/tensor_cores/23_fragment_layout/` | [OBS] FP4/FP6 layout and LDSM-fed QMMA probes. |
| `corpus/tensor_cores/24_production_mini_gemm/` | [OBS] Production-like QMMA, sparse metadata, and epilogue context. |
| `corpus/tensor_cores/25_stsm_epilogue/` | [OBS] QMMA-to-STSM epilogue path. |

## Canonical Forms

| Key | Distilled form | Meaning | Why it matters | Evidence | Notes |
|---|---|---|---|---|---|
| `QMMA_D_A_B_C` | `QMMA.16832.F32.<A>.<B> D, A, B, C` | Dense low-precision MMA accumulating into F32. | Baseline QMMA matcher for FP8/FP6/FP4 compute paths. | `corpus/tensor_cores/14_qmma_fp8/14a_qmma_e4m3_e4m3_f32.sass` | [OBS] Dense FP8/FP6/FP4 accumulator form. |
| `QMMA_F16_D_A_B_C` | `QMMA.16832.F16.<A>.<B> D, A, B, C` | Dense low-precision MMA accumulating into F16. | Separates accumulator dtype effects from input dtype effects. | `corpus/tensor_cores/14_qmma_fp8/14j_qmma_e4m3_e4m3_f16.sass` | [OBS] F16 accumulator variant. |
| `QMMA_SF_D_A_B_C_SCALES` | `QMMA.SF.16832.F32.<A>.<B>.<scale> D, A, B, C, ...` | Dense block-scaled QMMA with extra scale operands. | Requires scale-factor operand tracking in addition to A/B/C/D flow. | `corpus/tensor_cores/23_fragment_layout/23i_scale_factor_interaction.sass` | [OBS] Dense block-scaled QMMA form. |
| `QMMA_SP_D_A_B_C_META` | `QMMA.SP.16864.F32.<A>.<B> D, A, B, C, meta, selector` | Sparse QMMA with metadata and selector operands. | Marks structured sparsity and changes the operand audit model. | `corpus/tensor_cores/24_production_mini_gemm/24r_sparse_qmma_tile.sass` | [OBS] Sparse non-scaled form. |
| `QMMA_SF_SP_D_A_B_C_META_SCALES` | `QMMA.SF.SP.16864.F32.<A>.<B>.<scale> D, A, B, C, meta, ...` | Sparse block-scaled QMMA. | Combines metadata, selector, and scale-factor channels in one compute instruction. | `corpus/tensor_cores/19_sparse_mma/19i_sparse_mxf8f6f4_e3m2_e2m1.sass` | [OBS] Sparse block-scaled form. |

## Operand Model

| Operand | Role | Status |
|---|---|---|
| `D` | Destination accumulator register base | [OBS] First SASS operand. |
| `A` | A fragment register base | [OBS] Second SASS operand. |
| `B` | B fragment register base | [OBS] Third SASS operand. |
| `C` | Input accumulator register base | [OBS] Fourth SASS operand; chained MMAs colocate D and C after the first instruction. |
| `meta` | Sparse metadata register | [OBS] Present on `.SP` forms. |
| Scale operands | Scale-factor registers / uniform placeholders | [OBS] Present on `.SF` forms. |

## Modifier / Field Model

| Field | Values observed | Evidence | Status |
|---|---|---|---|
| Shape | `16832`, `16864` | Chapters 14, 19, 23, 24 | [OBS] Dense non-scaled QMMA uses `16832`; sparse observed forms use `16864`. |
| Accumulator dtype | `F32`, `F16` | Chapter 14 | [OBS] Accumulator dtype appears after shape. |
| Input dtype | `E4M3`, `E5M2`, `E3M2`, `E2M3`, `E2M1` | Chapters 14, 15, 23 | [OBS] A/B dtypes are explicit in mnemonic. |
| Scale marker | `.SF` | Chapters 16, 19, 23 | [OBS] Marks block-scaled form. |
| Sparse marker | `.SP` | Chapters 19, 24, 25 | [OBS] Marks sparse metadata form. |
| Scale dtype | `.E8`, `.UE4M3` in related scaled families | Chapters 16, 19, 23 | [OBS] Scale dtype is mnemonic-visible where emitted. |

## Matching Notes

* [RES] `QMMA` low opcode byte observed as `0x7a` across tested dense and sparse forms.
* [RES] Dtype should be read from the mnemonic, not guessed from opcode low byte alone.
* [RES] Sparse metadata and scale operands must be tracked as separate audit channels.
* [OBS] LDSM-fed and direct-register QMMA paths are distinct in surrounding instructions, not in the QMMA mnemonic alone.

## Open Gaps

* [GAP] Full FP4/FP6 lane-to-value decode remains open.
* [GAP] Exact `.SP` bit placement in opcode/control fields remains deferred.
* [INF] QMMA dtype and MMA-chain control bits are decoded in `CONTROL_CODE.md`; full stall/yield and universal scoreboard mapping remain open.
