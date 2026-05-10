# denvdis Integration Plan

This page tracks the validation pass for using `redplait/denvdis` as the bit-level SM120 / SM120a cross-check backend for SASS King.

## Positioning

[OBS] SASS King already has local SM120 / SM120a SASS dumps for tensor-core, matrix-memory, async-copy, control-flow, and warp-collective studies through Kernel 25.

[INF] redplait/denvdis lists `sm120` in `sm_version.txt` and includes SM120 descriptor files under `data12/sm120_1.txt`, `data12/sm120_2.txt`, and `data12/sm120_3.txt`, based on external repository inspection.

[INF] denvdis exposes `nvd` options for encoding fields (`-O`), scheduling tables (`-S`), predicates (`-p`), and register tracking (`-T`) according to its README.

[INF] denvdis is the correct low-level cross-check candidate for bit placement, scheduling-resource, predicate, and register-flow claims. It does not replace local SASS dumps as SASS King primary evidence.

## Tagging Rule

| Source | SASS King status |
|---|---|
| Bit value visible in local dump | `[OBS]` |
| Semantic interpretation from local dump plus denvdis output | `[INF]` |
| Family listed by denvdis but absent from local dumps | `[HYP]` or watchlist |
| Gap resolved by local dump plus denvdis cross-check | `[RES]` |
| denvdis and local evidence disagree | `[GAP]` until explained |

## Validation Commands

Build and run denvdis in a separate tooling checkout. Do not vendor denvdis into this repository unless explicitly approved.

[INF] Validation run 2026-05-10 used `redplait/denvdis` commit `317fa63` with `nvd` and `sm120.so` built in an external denvdis build directory. The build used external ELFIO and FP16 header checkouts:

```bash
make nvd sm120.so ELFIO=-I<ELFIO-checkout> FP16=-I<FP16-include-dir> CLANG=clang++ PERL_OPTS=
```

[INF] The local probe binaries are host ELF executables with embedded CUDA cubins. Running `nvd` directly on those executables reports `not CUBIN`; the validation extracts device ELFs first:

```bash
cuobjdump -xelf all path/to/kernel
SM_DIR=<denvdis-build-dir> <denvdis-build-dir>/nvd -O -S -p path/to/*.cubin
```

Representative command shape after extraction:

```bash
nvd -O -S -p path/to/kernel.cubin
nvd -O -S -p -T path/to/kernel.cubin
```

`SM_DIR=<denvdis-build-dir>` is required when running outside the denvdis build directory so that `nvd` can load `sm120.so`.

## Representative Coverage Set

Use local cubins or dumps that already exist in the project. Prefer cubins when `nvd` needs binary input; keep `cuobjdump` output as the comparison baseline.

| Area | Candidate evidence | Families to validate | Required checks | Status |
|---|---|---|---|---|
| HMMA baseline | `corpus/tensor_cores/13_hmma_fp16/13a_hmma_f16_f16` | `HMMA` | family, dtype, operands, scheduling fields | [RES] `HMMA.16816.F16`, `not_found 0`. |
| QMMA dense | `corpus/tensor_cores/14_qmma_fp8/14a_qmma_e4m3_e4m3_f32` and `14e_qmma_chain` | `QMMA` | dtype modifiers, operands, control fields | [RES] `QMMA.16832.F32.E4M3.E4M3`, including `.reuse` on chain fixture. |
| QMMA block-scaled | `corpus/tensor_cores/16_fp4_peak/16a_mxf8f6f4_e4m3_baseline` | `QMMA.SF` | scale operands, dtype modifiers, control fields | [RES] `QMMA.SF.16832.F32.E4M3.E4M3.E8.1X`, `not_found 0`. |
| OMMA block-scaled | `corpus/tensor_cores/16_fp4_peak/16b_mxf4nvf4_2x` | `OMMA.SF` | family, scale operands, scheduling fields | [RES] `OMMA.SF.16864.F32.E2M1.E2M1.E8`, `not_found 0`. |
| Sparse MMA | `corpus/tensor_cores/25_stsm_epilogue/25v_sparse_qmma_to_stsm_epilogue` | `QMMA.SP` | sparse modifiers, metadata operands, `.SP` field evidence | [RES] `QMMA.SP.16864.F32.E4M3.E4M3`, `not_found 0`. |
| Sparse block-scaled MMA | Scratch builds of `corpus/tensor_cores/19_sparse_mma/19_sparse_block_scale_probe.cu` variants 19i and 19k | `QMMA.SF.SP`, `OMMA.SF.SP` | sparse modifiers, metadata operands, `.SP` field evidence | [RES] `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8.1X` and `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X`, `not_found 0`. |
| Matrix load | `corpus/tensor_cores/17_ldmatrix/17c_ldmatrix_x4` | `LDSM` | `.M88`, `.MT88`, `.2`, `.4`, control fields | [RES] `LDSM.16.M88.4`, `not_found 0`. |
| Async copy | `corpus/tensor_cores/18_pipelined_tile/18a_pipelined_tile` | `LDGSTS`, `LDGDEPBAR`, `DEPBAR` | SB0, wait group, scheduling fields | [RES] `LDGSTS.E.LTC128B.128`, `LDGDEPBAR`, `DEPBAR.LE SB0`, `not_found 0`. |
| Matrix store b16 | Scratch compile of `corpus/tensor_cores/22_stmatrix/22c_stmatrix_x4` | `STSM.16` | `.M88`, `.MT88`, `.2`, `.4`, operands | [RES] `STSM.16.M88.4`, `not_found 0`. |
| Matrix store b8 | Scratch compile of `corpus/tensor_cores/25_stsm_epilogue/25q_sm120a_b8_stsm.o` | `STSM.8` | target gate, `.MT168`, `.2`, `.4`, operands | [RES] `STSM.8.MT168`, `.2`, and `.4` recognized for `sm_120a`, `not_found 0`. |
| Divergence | Scratch compile of `corpus/tensor_cores/21_divergence_reconvergence/21c_lane_divergent_if` and `21n_divergent_mma_guard` | `BSSY`, `BSYNC`, `BRA`, `EXIT`, `WARPSYNC` | predicates, branch targets, scheduling fields | [RES] `BSSY.RECONVERGENT`, `BSYNC.RECONVERGENT`, `BRA`, `EXIT`, and `@P0 WARPSYNC.ALL` recognized. |

## Results Table

Fill this table after running denvdis. Do not mark a gap resolved until the row cites both local evidence and the denvdis command result.

| Family | Local evidence | denvdis command | Recognized | Modifiers OK | Control fields exposed | Scheduling exposed | SASS King action |
|---|---|---|---|---|---|---|---|
| `HMMA` | `corpus/tensor_cores/13_hmma_fp16/13a_hmma_f16_f16` | `nvd -O -S -p 13a_hmma_f16_f16.*.cubin` | Yes: `HMMA.16816.F16` | Yes: size and F16 dtype preserved | Partial: `cword`, `req_bit_set`, reuse labels, SB fields where present | Yes: `usched_info` and `S> tab` | Use denvdis as bit-level cross-check for HMMA pages. |
| `QMMA` | `corpus/tensor_cores/14_qmma_fp8/14a_qmma_e4m3_e4m3_f32`, `14e_qmma_chain` | `nvd -O -S -p 14*.cubin` | Yes: `QMMA.16832.F32.E4M3.E4M3` | Yes: dtype modifiers and `.reuse` preserved | Partial: `cword`, `req_bit_set`, reuse labels, SB fields where present | Yes: `usched_info` and `S> tab` | Use denvdis for QMMA dtype/modifier cross-checks. |
| `QMMA.SF` | `corpus/tensor_cores/16_fp4_peak/16a_mxf8f6f4_e4m3_baseline` | `nvd -O -S -p 16a*.cubin` | Yes: `QMMA.SF.16832.F32.E4M3.E4M3.E8.1X` | Yes: SF and scale dtype rendered | Partial: `cword`, `req_bit_set`, SB fields where present | Yes: `usched_info` and `S> tab` | Replace bit-placement speculation with denvdis cross-reference plus local bytes. |
| `QMMA.SP` | `corpus/tensor_cores/25_stsm_epilogue/25v_sparse_qmma_to_stsm_epilogue` compiled from a scratch build | `nvd -O -S -p 25v*.cubin` | Yes: `QMMA.SP.16864.F32.E4M3.E4M3` | Yes: `.SP` and dtype modifiers rendered | Partial: `cword`, `req_bit_set`, SB fields where present | Yes: `usched_info` and `S> tab` | Sparse dense-QMMA path can move from parser risk to denvdis-backed cross-check. |
| `QMMA.SF.SP` | Scratch build of `corpus/tensor_cores/19_sparse_mma/19_sparse_block_scale_probe.cu` for 19i | `nvd -O -S -p 19i*.cubin` | Yes: `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8.1X` | Yes: `.SF.SP`, sparse metadata, scale dtype, and `1X` rendered | Partial: `cword`, `req_bit_set`, SB fields where present | Yes: `usched_info` and `S> tab` | Use denvdis for sparse block-scaled QMMA field naming. |
| `OMMA.SF` | `corpus/tensor_cores/16_fp4_peak/16b_mxf4nvf4_2x` | `nvd -O -S -p 16b*.cubin` | Yes: `OMMA.SF.16864.F32.E2M1.E2M1.E8` | Yes: OMMA, SF, and dtype modifiers rendered | Partial: `cword`, `req_bit_set`, SB fields where present | Yes: `usched_info` and `S> tab` | Use denvdis for OMMA.SF field naming, keep algorithmic semantics in SASS King. |
| `OMMA.SF.SP` | Scratch build of `corpus/tensor_cores/19_sparse_mma/19_sparse_block_scale_probe.cu` for 19k | `nvd -O -S -p 19k*.cubin` | Yes: `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` | Yes: `.SF.SP`, sparse metadata, scale dtype, and `4X` rendered | Partial: `cword`, `req_bit_set`, SB fields where present | Yes: `usched_info` and `S> tab` | Use denvdis for sparse block-scaled OMMA field naming. |
| `LDSM` | `corpus/tensor_cores/17_ldmatrix/17c_ldmatrix_x4` | `nvd -O -S -p 17c*.cubin` | Yes: `LDSM.16.M88.4` | Yes: width and layout rendered | Partial: `cword`, `src_rel_sb`, `dst_wr_sb`, `req_bit_set` | Yes: `usched_info` and `S> tab` | Use denvdis to cross-check LDSM operand fields. |
| `STSM.16` | Scratch build of `corpus/tensor_cores/22_stmatrix/22c_stmatrix_x4` | `nvd -O -S -p 22c*.cubin` | Yes: `STSM.16.M88.4` | Yes: width and layout rendered | Partial: `cword`, `src_rel_sb`, `req_bit_set` | Yes: `usched_info` and `S> tab` | Use denvdis to cross-check STSM.16 pages. |
| `STSM.8` | Scratch build of `corpus/tensor_cores/25_stsm_epilogue/25q_sm120a_b8_stsm.o` | `nvd -O -S -p 25q*.cubin` | Yes: `STSM.8.MT168`, `.2`, `.4` | Yes: b8, MT168, and width modifiers rendered | Partial: `cword`, `src_rel_sb`, `req_bit_set` | Yes: `usched_info` and `S> tab` | Keep the target-qualified claim: supported in tested `sm_120a`, rejected in tested plain `sm_120`. |
| `LDGSTS` | `corpus/tensor_cores/18_pipelined_tile/18a_pipelined_tile` | `nvd -O -S -p 18a*.cubin` | Yes: `LDGSTS.E.LTC128B.128` | Yes: `.E`, `.LTC128B`, and size rendered | Partial: `cword`, `src_rel_sb`, `dst_wr_sb`, `req_bit_set` | Yes: `usched_info` and `S> tab` | Use denvdis for async-copy scoreboard cross-checks. |
| `LDGDEPBAR` | `corpus/tensor_cores/18_pipelined_tile/18a_pipelined_tile` | `nvd -O -S -p 18a*.cubin` | Yes: `LDGDEPBAR` | Yes | Partial: `cword`, `src_rel_sb`, `req_bit_set` | Yes: `usched_info` and `S> tab` | Cross-link from async-copy documentation. |
| `DEPBAR` | `corpus/tensor_cores/18_pipelined_tile/18a_pipelined_tile` | `nvd -O -S -p 18a*.cubin` | Yes: `DEPBAR.LE SB0` | Yes: condition and SB operand rendered | Partial: `cword`, `WAIT*_END_GROUP`, `req_bit_set` | Yes: `usched_info` and `S> tab` | Use for wait-group interpretation, but do not claim full bit map yet. |
| `BSSY` / `BSYNC` | Scratch build of `corpus/tensor_cores/21_divergence_reconvergence/21c_lane_divergent_if` | `nvd -O -S -p 21c*.cubin` | Yes: `BSSY.RECONVERGENT`, `BSYNC.RECONVERGENT` | Yes: reconvergence modifier and barrier operand rendered | Partial: `cword`, predicates, `req_bit_set` | Yes: `usched_info` and `S> tab` | Use for divergence page cross-check. |
| `WARPSYNC` | Scratch build of `corpus/tensor_cores/21_divergence_reconvergence/21n_divergent_mma_guard` | `nvd -O -S -p 21n*.cubin` | Yes: `@P0 WARPSYNC.ALL` | Yes: predicate and `.ALL` modifier rendered | Partial: `cword`, predicate, `req_bit_set` | Yes: `usched_info` | Use for guarded warp-level synchronization cross-check. |
| `REDUX` | `corpus/warp_collectives/10_reduce/sm_120/10a_reduce_sum` | `nvd -O -S -p 10a*.cubin` | Yes: `REDUX.SUM.S32` | Yes: operation and signedness rendered | Partial: `cword`, `req_bit_set`, uniform destination | Yes: `usched_info` and `S> tab` | Use for warp-reduction field naming and MOV consumer checks. |

## Control-Code Target

[INF] The validation confirms that `nvd -O -S -p` exposes raw control words and decoded dependency/scheduling fields across the tested families: `cword`, `usched_info`, `req_bit_set`, `src_rel_sb`, `dst_wr_sb`, reuse labels, and `S> tab` scheduling rows.

[GAP] GAP-control-code-1 remains open for a dedicated bit map. This pass proves denvdis is useful for cross-validation, but it does not yet provide a SASS King page that maps every local byte bit to wait, stall, yield, scoreboard, and reuse semantics.

Minimum validation set:

| Field | Existing SASS King claim | denvdis check | Resolution rule |
|---|---|---|---|
| Wait mask | Byte 0 contains SB wait bits. | `nvd -O` on `LDGSTS`, `DEPBAR`, `LDSM`, MMA chains. | `[RES]` only if local bytes and denvdis field labels agree. |
| Stall | Bits 8-15 approximately encode stall count. | `nvd -O` on varied arithmetic, memory, and branch instructions. | Correct or replace existing `[INF]` with denvdis-backed interpretation. |
| Yield | Bit 13 approximately encodes yield. | `nvd -O` on instructions with differing yield display. | Keep `[GAP]` if denvdis does not expose the field cleanly. |
| Write scoreboard | Bits 17-19 approximately encode write scoreboard assignment. | `nvd -O` on MMA/LDSM/LDGSTS chains. | Resolve only per family if needed. |
| Reuse | Bits 58-61 approximately encode operand reuse. | `nvd -O` on MMA chains and scalar FFMA examples. | Resolve if denvdis field labels and local `.reuse` output match. |

If enough fields validate, create `knowledge/encoding/CONTROL_CODE.md` and update `knowledge/FINDINGS.md`.

## Watchlist Audit

[GAP] Families listed in denvdis but not observed locally remain watchlist items. The denvdis pass may classify each watchlist family as `present_in_denvdis`, `absent_in_denvdis`, or `unknown`, but that classification is not a local observation.

Initial watchlist groups:

- Blackwell uniform families: `CREDUX`, `LDT`, `LDTM`, `STT`, `STTM`, `UF*`, `UI*`, `UV*`.
- Tensor and memory management families: `UTC*`, `UTMA*`, `UBLK*`, `UMEMSETS`.
- Cluster or datacenter-adjacent families that may be out of SM120 consumer scope.

## Exit Criteria

Phase 2.5 is complete when:

- denvdis builds in the local tooling environment or a reproducible build blocker is documented.
- At least eight representative local kernels are checked with `nvd -O -S -p`.
- The results table above is filled.
- `knowledge/SASS_INSTRUCTIONS_SM120.md` is updated for any confirmed cross-check policy changes.
- `knowledge/ISA_COVERAGE.md` records denvdis watchlist status separately from local observation status.
- `knowledge/encoding/CONTROL_CODE.md` is created if the control-code fields are sufficiently validated. [RES] Created for the partial field model; full stall/yield bit placement remains open.

After this pass, Phase 3 pattern work should start with a high-evidence pattern such as warp reduction, then proceed to tensor-core chain patterns.
