# STSM

## Scope

STSM is the SASS matrix-store family observed for `stmatrix.sync.aligned` on SM120 / SM120a.

## Evidence

| Source | Evidence |
|---|---|
| `corpus/tensor_cores/22_stmatrix/` | [OBS] First-pass m8n8 b16 STSM forms and STS fallback. |
| `corpus/tensor_cores/25_stsm_epilogue/` | [OBS] Epilogue/storeback probes, b8 target compatibility, conversions, barrier/readback patterns. |

## Canonical Forms

| Key | Distilled form | Meaning | Why it matters | Evidence | Notes |
|---|---|---|---|---|---|
| `STSM_R_R` | `STSM.16.M88 [R], R` | Stores one b16 matrix fragment group to shared memory. | Baseline non-transposed STSM signature for epilogue matching. | `corpus/tensor_cores/25_stsm_epilogue/25a_stsm_x1_runtime_layout.sass` | [OBS] x1 non-transposed b16 store. |
| `STSM_R_R_X2` | `STSM.16.M88.2 [R], R` | Stores two b16 matrix fragment groups. | Width suffix changes register consumption and shared-memory footprint. | `corpus/tensor_cores/25_stsm_epilogue/25b_stsm_x2_runtime_layout.sass` | [OBS] x2 non-transposed b16 store. |
| `STSM_R_R_X4` | `STSM.16.M88.4 [R], R` | Stores four b16 matrix fragment groups. | Wider store form is a stronger signal for tiled epilogue storeback. | `corpus/tensor_cores/25_stsm_epilogue/25c_stsm_x4_runtime_layout.sass` | [OBS] x4 non-transposed b16 store. |
| `STSM_T_R_R` | `STSM.16.MT88 [R], R` | Stores one transposed b16 matrix fragment group. | Distinguishes layout transformation from plain shared-memory storeback. | `corpus/tensor_cores/25_stsm_epilogue/25d_stsm_x1_trans_runtime_layout.sass` | [OBS] x1 transposed b16 store. |
| `STSM_T_R_R_X2` | `STSM.16.MT88.2 [R], R` | Stores two transposed b16 matrix fragment groups. | Combines transpose and width suffix in the matcher key. | `corpus/tensor_cores/25_stsm_epilogue/25e_stsm_x2_trans_runtime_layout.sass` | [OBS] x2 transposed b16 store. |
| `STSM_T_R_R_X4` | `STSM.16.MT88.4 [R], R` | Stores four transposed b16 matrix fragment groups. | Common high-width transposed form for matrix storeback classification. | `corpus/tensor_cores/25_stsm_epilogue/25f_stsm_x4_trans_runtime_layout.sass` | [OBS] x4 transposed b16 store. |
| `STSM_B8_T_R_R_XN` | `STSM.8.MT168[.2|.4] [R], R` | Stores transposed 8-bit matrix fragments. | Captures the `sm_120a`-qualified b8 matrix-store path. | `corpus/tensor_cores/25_stsm_epilogue/25q_sm120a_b8_stsm.sass` | [OBS] sm_120a b8 forms; plain sm_120 rejects the PTX. |

## Operand Model

| Operand | Role | Status |
|---|---|---|
| `[R]` | Shared-memory destination address | [OBS] Direct shared address register. |
| `R` | Source register base | [OBS] Register group consumed by STSM. |

## Modifier / Field Model

| Field | Values observed | Evidence | Status |
|---|---|---|---|
| Element size | `.16`, `.8` | Chapters 22 and 25 | [OBS] `.16` for m8n8 b16; `.8` for sm_120a m16n8 b8. |
| Shape | `M88`, `MT88`, `MT168` | Chapters 22 and 25 | [OBS] `T` inside shape marks transposed form in observed mnemonics. |
| Width | none, `.2`, `.4` | Chapters 22 and 25 | [OBS] Width suffix maps x1/x2/x4. |
| Target | `sm_120`, `sm_120a` | `25q_negative_invalid_b8_stsm.log`, `25q_sm120a_b8_stsm.sass` | [RES] b8 is target-qualified. |

## Matching Notes

* [RES] First-pass epilogue matcher should recognize `STSM -> BAR -> LDS -> STG` as the cross-thread storeback pattern.
* [RES] `STS.128` is a fallback/shared-store family, not STSM.
* [OBS] No-barrier same-thread contrast exists as adjacent `STSM` and `LDS` in 25n, but it does not prove cross-thread visibility without a barrier.

## Open Gaps

* [GAP] Full lane-to-value semantic layout remains undecoded from runtime words.
* [GAP] STSM latency remains undecoded.
* [INF] denvdis exposes STSM `src_rel_sb`, `req_bit_set`, and `usched_info` fields in `CONTROL_CODE.md`; complete universal bit placement remains open.
