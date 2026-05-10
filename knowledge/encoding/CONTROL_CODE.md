# CONTROL_CODE

## Scope

This page records the reusable SM120 / SM120a control-code facts that are backed by local SASS evidence and denvdis cross-checks.

It is not a complete hardware bit map. SASS King treats local dump bytes as primary evidence and denvdis output as field-name cross-validation.

## Evidence

| Source | Evidence |
|---|---|
| `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [OBS] QMMA dtype, accumulator, and MMA chain scoreboard bits decoded by controlled local comparisons. |
| `corpus/tensor_cores/18_pipelined_tile/conclusion18.md` | [OBS] `DEPBAR.LE SB0, N` wait-group argument decoded in local control-code bits 38-39. |
| `knowledge/DENVDIS_INTEGRATION.md` | [INF] `nvd -O -S -p` recognizes representative SM120 / SM120a families with `not_found 0` and exposes control/dependency fields. |
| denvdis validation run 2026-05-10 | [INF] `nvd` output exposes `cword`, `usched_info`, `req_bit_set`, `src_rel_sb`, `dst_wr_sb`, reuse labels, and `S> tab` scheduling rows. |

## denvdis Field Model

denvdis renders each instruction with a compact control suffix and then expands selected fields in the operand dump.

Example shape:

```text
> QMMA.16832.F32.E4M3.E4M3 R16,R12,R10.reuse,R16 &0x4 ?trans11 ?NOP ?PMN
 E usched_info: USCHED_INFO 1B trans11
 E reuse_src_b: REUSE 1 reuse
 V req_bit_set: 0x4 type 0
```

| Field | denvdis surface | Meaning for SASS King | Status |
|---|---|---|---|
| Raw compact word | `cword <hex>` | denvdis compact control/scheduling word for the instruction. Do not equate it directly with the full local SASS control-code hex until mapped. | [INF] Exposed by denvdis. |
| Required/wait mask | first `&...`; `V req_bit_set` | Dependency bit mask used by denvdis for required scoreboard waits or request bits. | [INF] Cross-check field. Exact bit placement in local control word remains per-family. |
| Source release scoreboard | second `&...`; `V src_rel_sb` | Source dependency/release scoreboard selector where the instruction form has one. | [INF] Cross-check field. |
| Destination write scoreboard | third `&...`; `V dst_wr_sb` | Destination write scoreboard selector for producer instructions such as loads. | [INF] Cross-check field. |
| Scheduling transform | `?transN`, `?WAITN_END_GROUP`, `?OFF_DECK_DRAIN`; `E usched_info` | Encoded scheduling class / wait-end-group state. | [INF] Cross-check field. |
| Operand reuse | `.reuse`; `E reuse_src_*: REUSE 1 reuse` | Operand reuse flag for a named source operand. | [RES] Local `.reuse` suffix and denvdis operand field agree in tested scalar and MMA fixtures. |
| Scheduling table | `S> tab ...` | denvdis dependency/scheduling-table match for producer/consumer classes and operands. | [INF] Useful for audit tooling, not local primary evidence. |

## Resolved Local Bit Fields

These fields are backed by local controlled comparisons. denvdis is useful as a naming cross-check, but the observation comes from local SASS/control-code deltas.

| Field | Local bit position | Values observed | Evidence | Status |
|---|---:|---|---|---|
| QMMA accumulator dtype | 1, 2, 13 | F16 sets bit 1; F32 sets bit 2 and auxiliary bit 13. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] Dense QMMA accumulator dtype bits decoded. |
| QMMA family marker | 4, 10, 11 | Always set on tested QMMA forms. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] Marker bits for tested QMMA forms. |
| QMMA A dtype mantissa class | 14 | Set when A mantissa is not 3. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] Orthogonal A dtype bit. |
| QMMA B dtype mantissa class | 15 | Set when B mantissa is not 3. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] Orthogonal B dtype bit. |
| QMMA A exponent class | 18, 19 | Bit 18 for E3M2; bit 19 for E2M3/E2M1. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] Orthogonal A dtype bits. |
| QMMA B exponent class | 20, 21 | Bit 20 for E3M2; bit 21 for E2M3/E2M1. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] Orthogonal B dtype bits. |
| MMA chain scoreboard set | 26 | Set on first MMA in tested HMMA/QMMA chains. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] MMA-family chain bit. |
| MMA chain scoreboard wait | 27 | Set on dependent MMA chain instructions, cleared on final chain instruction. | `corpus/tensor_cores/14_qmma_fp8/conclusion14.md` | [RES] MMA-family chain bit. |
| `DEPBAR.LE SB0, N` wait-group argument | 38-39 | `00` = 0, `01` = 1, `10` = 2; `11` predicted for 3. | `corpus/tensor_cores/18_pipelined_tile/conclusion18.md` | [RES] Wait-group field for tested N values. |

## Cross-Checked Family Coverage

| Family | denvdis field evidence | SASS King use |
|---|---|---|
| `HMMA` | `HMMA.16816.F16`, `cword`, `req_bit_set`, `usched_info`, `S> tab` | Cross-check MMA dependency and scheduling claims. |
| `QMMA` | `QMMA.16832...`, `.reuse`, `req_bit_set`, `usched_info`, `S> tab` | Cross-check dtype/modifier parsing and chain scheduling. |
| `QMMA.SF` | `QMMA.SF.16832...`, scale dtype suffix, control fields | Cross-check block-scale operand model. |
| `QMMA.SP` | `QMMA.SP.16864...`, metadata operand, control fields | Cross-check sparse non-scaled operand model. |
| `QMMA.SF.SP` | `QMMA.SF.SP.16864...`, metadata operand, scale operands, `1X`, control fields | Cross-check sparse block-scaled QMMA operand model. |
| `OMMA.SF` | `OMMA.SF.16864...`, scale dtype suffix, control fields | Cross-check FP4 peak operand model. |
| `OMMA.SF.SP` | `OMMA.SF.SP.168128...`, metadata operand, scale operands, `4X`, control fields | Cross-check sparse block-scaled OMMA operand model. |
| `LDSM` | `LDSM.16.M88.[2|4]`, `src_rel_sb`, `dst_wr_sb`, `usched_info` | Cross-check matrix-load dependency fields. |
| `STSM.16` | `STSM.16.M88.4`, `src_rel_sb`, `req_bit_set`, `usched_info` | Cross-check matrix-store dependency fields. |
| `STSM.8` | `STSM.8.MT168[.2|.4]` on tested `sm_120a`, control fields | Cross-check target-qualified b8 store support. |
| `LDGSTS` | `LDGSTS.E.LTC128B.128`, `src_rel_sb`, `dst_wr_sb`, `req_bit_set` | Cross-check async-copy scoreboard fields. |
| `LDGDEPBAR` | `LDGDEPBAR`, control fields | Cross-check commit-group marker. |
| `DEPBAR` | `DEPBAR.LE SB0, N`, `WAIT*_END_GROUP`, `req_bit_set` | Cross-check wait-group semantics. |
| `BSSY` / `BSYNC` | `BSSY.RECONVERGENT`, `BSYNC.RECONVERGENT`, predicates, `usched_info` | Cross-check divergence/reconvergence control markers. |
| `WARPSYNC` | `@P0 WARPSYNC.ALL`, predicate, `usched_info`, `req_bit_set` | Cross-check guarded warp-level synchronization markers. |
| `REDUX` | `REDUX.SUM.S32 UR7,R2`, `req_bit_set`, uniform destination, `S> tab` | Cross-check warp-reduction control and cross-file consumer flow. |

## Matching Notes

* [RES] For production audit matching, read low-precision MMA dtypes from the mnemonic and the normalized encoding pages, not from opcode low byte alone.
* [RES] For QMMA, local comparisons establish the dtype and chain bits listed above. denvdis confirms the rendered mnemonic and dependency fields for representative cubins.
* [RES] For `cp.async`, local comparisons establish that `DEPBAR.LE SB0, N` encodes N in bits 38-39 and denvdis confirms `SB0` and the wait-group rendering.
* [INF] `req_bit_set`, `src_rel_sb`, and `dst_wr_sb` are suitable field names for SASS King audit tooling when backed by denvdis output.
* [INF] `usched_info` and `S> tab` are suitable for scheduling-class cross-checks, but scheduling latency claims still need local measurements or local controlled comparisons.

## Open Gaps

* [GAP] Full stall/yield bit placement is still not decoded. denvdis exposes scheduling classes such as `transN`, `WAITN_END_GROUP`, and `OFF_DECK_DRAIN`, but this pass did not map those names to every local bit in the full control word.
* [GAP] The exact relationship between denvdis `cword` and the full local SASS control-code hex is not fully documented.
* [GAP] Non-MMA scoreboard fields should be resolved per family before claiming a universal scoreboard bit layout.
* [RES] `WARPSYNC`, `REDUX`, `QMMA.SF.SP`, and `OMMA.SF.SP` have denvdis cubin cross-checks with `not_found 0` in the 2026-05-10 targeted gap pass.
