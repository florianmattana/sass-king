# LDSM

## Scope

LDSM is the SASS matrix-load family observed for `ldmatrix.sync.aligned` on SM120 / SM120a.

## Evidence

| Source | Evidence |
|---|---|
| `corpus/tensor_cores/17_ldmatrix/` | [OBS] Baseline x1/x2/x4 and transpose probes. |
| `corpus/tensor_cores/18_pipelined_tile/` | [OBS] LDSM in pipelined tile context. |
| `corpus/tensor_cores/23_fragment_layout/` | [OBS] LDSM-fed FP4/FP6 QMMA path. |
| `corpus/tensor_cores/24_production_mini_gemm/` | [OBS] Production-like LDSM order and alignment probes. |

## Canonical Forms

| Key | Distilled form | Meaning | Why it matters | Evidence | Notes |
|---|---|---|---|---|---|
| `LDSM_R_R` | `LDSM.16.M88 R, [R]` | Loads one b16 matrix fragment group from shared memory. | Baseline matrix-load signature before MMA operand use. | `corpus/tensor_cores/17_ldmatrix/17a_ldmatrix_x1.sass` | [OBS] x1 non-transposed b16 load. |
| `LDSM_R_R_X2` | `LDSM.16.M88.2 R, [R]` | Loads two b16 matrix fragment groups. | Width suffix changes destination register footprint. | `corpus/tensor_cores/17_ldmatrix/17b_ldmatrix_x2.sass` | [OBS] x2 non-transposed b16 load. |
| `LDSM_R_R_X4` | `LDSM.16.M88.4 R, [R]` | Loads four b16 matrix fragment groups. | Common high-width fragment supply form for tensor-core tiles. | `corpus/tensor_cores/17_ldmatrix/17c_ldmatrix_x4.sass` | [OBS] x4 non-transposed b16 load. |
| `LDSM_T_R_R_X4` | `LDSM.16.MT88.4 R, [R]` | Loads four transposed b16 matrix fragment groups. | Distinguishes transpose-driven layout handling from ordinary fragment loads. | `corpus/tensor_cores/17_ldmatrix/17d_ldmatrix_x4_trans.sass` | [OBS] x4 transposed b16 load. |
| `LDSM_OFFSET_R_R_XN` | `LDSM.16.M88[.2|.4] R, [R+imm]` | Loads matrix fragments from an offset shared-memory address. | Captures alignment and layout variants without changing the instruction family. | `corpus/tensor_cores/24_production_mini_gemm/24q_shared_memory_alignment_variant.sass` | [OBS] Offset shared-memory load addresses. |

## Operand Model

| Operand | Role | Status |
|---|---|---|
| `R` destination | Fragment register base | [OBS] Destination register group depends on width suffix. |
| `[R]` / `[R+imm]` | Shared-memory source address | [OBS] Address can include immediate offset after ptxas address arithmetic. |

## Modifier / Field Model

| Field | Values observed | Evidence | Status |
|---|---|---|---|
| Element size | `.16` | Chapter 17 | [OBS] Tested LDSM forms are b16. |
| Shape | `M88`, `MT88` | Chapter 17 | [OBS] `T` inside shape marks transposed form in observed mnemonics. |
| Width | none, `.2`, `.4` | Chapter 17 | [OBS] Width suffix maps x1/x2/x4. |

## Matching Notes

* [OBS] Production-like HMMA/QMMA tiles commonly show `LDSM.16.M88.2` and `LDSM.16.M88.4` before MMA.
* [INF] The specific A/B fragment role is contextual; do not assign A or B purely from the mnemonic without checking operand flow.
* [OBS] Offset shared-memory forms preserve the LDSM family and expose alignment/layout effects.

## Open Gaps

* [GAP] Complete lane-to-fragment mapping is not fully decoded for all width/transpose combinations.
* [INF] denvdis exposes LDSM `src_rel_sb`, `dst_wr_sb`, `req_bit_set`, and `usched_info` fields in `CONTROL_CODE.md`; complete universal bit placement remains open.
