# SASS Instructions on SM120 / SM120a

This inventory tracks instruction families observed in this repository. It is not a complete ISA reference. Full coverage status lives in `ISA_COVERAGE.md`.

SASS King uses local SM120 / SM120a dumps as primary evidence. External disassembly tooling, especially `redplait/denvdis`, is used as a bit-level cross-check for encoding fields, scheduling tables, predicates, and register tracking. A family listed by denvdis but not observed in local SASS remains unobserved for SASS King until a local dump contains it. A field interpretation confirmed by denvdis should be tagged `[INF]` unless the bit value itself is directly visible in a local dump.

Claims use the project tags:

* [OBS] Directly observed in local SASS dumps or ptxas logs.
* [INF] Inferred from controlled comparisons.
* [GAP] Not yet validated or only partially decoded.

## Tensor-core and matrix instructions

| Family | Mnemonics observed | Meaning | Why it matters | Target | First evidence | Source trigger | Status |
|---|---|---|---|---|---|---|---|
| HMMA | `HMMA.16816.F32`, `HMMA.16816.F16` | Warp-level FP16/BF16 matrix multiply-accumulate. | Baseline tensor-core family used to anchor fragment, accumulator, and latency reasoning before Blackwell low-precision forms. | `sm_120`, `sm_120a` | `corpus/tensor_cores/13_hmma_fp16/` | `mma.sync.aligned.m16n8k16` | [OBS] Warp-level FP16/BF16 MMA family. |
| QMMA | `QMMA.16832.*` | Dense Blackwell FP8/FP6/FP4 matrix multiply-accumulate. | Core SM120a low-precision tensor-core compute family for production GEMM and attention audits. | `sm_120a` | `corpus/tensor_cores/14_qmma_fp8/` | `mma.sync.aligned.kind::f8f6f4.m16n8k32` | [OBS] Dense FP8/FP6/FP4 MMA family. |
| QMMA.SF | `QMMA.SF.16832.*` | Dense QMMA with explicit scale-factor operands. | Separates block-scaled numeric behavior from ordinary dense MMA operand flow. | `sm_120a` | `corpus/tensor_cores/16_fp4_peak/` | `kind::mxf8f6f4.block_scale` | [OBS] Block-scaled QMMA form. |
| QMMA.SP | `QMMA.SP.16864.*` | Sparse QMMA with metadata operands and no scale-factor marker. | Exposes the SASS path for structured sparsity and its extra audit channel. | `sm_120a` | `corpus/tensor_cores/19_sparse_mma/` | `mma.sp::ordered_metadata...kind::f8f6f4` | [OBS] Sparse non-scaled QMMA form. |
| QMMA.SF.SP | `QMMA.SF.SP.16864.*` | Sparse block-scaled QMMA. | Combines sparse metadata tracking with scale-factor tracking, so audits must follow both operand classes. | `sm_120a` | `corpus/tensor_cores/19_sparse_mma/` | `mma.sp::ordered_metadata...kind::mxf8f6f4.block_scale` | [OBS] Sparse block-scaled QMMA form. |
| OMMA.SF | `OMMA.SF.16864.*` | Block-scaled FP4 peak MMA family. | Marks the highest-compression FP4 compute path observed in the project. | `sm_120a` | `corpus/tensor_cores/16_fp4_peak/` | `kind::mxf4nvf4.block_scale` | [OBS] Block-scaled FP4 peak MMA family. |
| OMMA.SF.SP | `OMMA.SF.SP.168128.*` | Sparse block-scaled FP4 peak MMA family. | Shows the sparse FP4 peak path and increases the number of metadata/scale operands to audit. | `sm_120a` | `corpus/tensor_cores/19_sparse_mma/` | `mma.sp::ordered_metadata...kind::mxf4nvf4.block_scale` | [OBS] Sparse block-scaled FP4 MMA form. |
| LDSM | `LDSM.16.M88`, `LDSM.16.M88.2`, `LDSM.16.M88.4`, `LDSM.16.MT88.4` | Loads matrix fragments from shared memory into registers. | It is the bridge between shared-memory tiling and MMA fragment operands. | `sm_120`, `sm_120a` | `corpus/tensor_cores/17_ldmatrix/` | `ldmatrix.sync.aligned` | [OBS] Shared-memory matrix-fragment load family. |
| STSM | `STSM.16.M88`, `STSM.16.M88.2`, `STSM.16.M88.4`, `STSM.16.MT88`, `STSM.16.MT88.2`, `STSM.16.MT88.4` | Stores matrix fragments from registers into shared memory. | It explains epilogue layout, shared-memory storeback, and cross-thread readback patterns. | `sm_120`, `sm_120a` | `corpus/tensor_cores/22_stmatrix/` | `stmatrix.sync.aligned.m8n8.shared.b16` | [OBS] Shared-memory matrix-fragment store family. |
| STSM b8 | `STSM.8.MT168`, `STSM.8.MT168.2`, `STSM.8.MT168.4` | Stores 8-bit matrix fragments through the STSM family. | It is target-qualified evidence that some matrix-store forms require `sm_120a`, not plain `sm_120`. | `sm_120a` | `corpus/tensor_cores/25_stsm_epilogue/25q_sm120a_b8_stsm.sass` | `stmatrix.sync.aligned.m16n8.*.shared.b8` | [OBS] Plain `sm_120` rejects this form; `sm_120a` accepts it. |

## Memory and synchronization instructions

| Family | Mnemonics observed | Meaning | Why it matters | Target | First evidence | Source trigger | Status |
|---|---|---|---|---|---|---|---|
| LDGSTS | `LDGSTS.E.LTC128B.128` | Asynchronous global-to-shared copy. | It is the SASS realization of pipelined tile staging before LDSM and MMA. | `sm_120`, `sm_120a` | `corpus/tensor_cores/18_pipelined_tile/` | `cp.async.ca.shared.global.L2::128B` | [OBS] Async global-to-shared copy realization. |
| LDGDEPBAR | `LDGDEPBAR` | Dependency helper for async-copy groups. | It marks compiler-managed bookkeeping around staged global-to-shared transfers. | `sm_120`, `sm_120a` | `corpus/tensor_cores/18_pipelined_tile/` | `cp.async.commit_group` / wait path | [OBS] Async-copy dependency barrier helper. |
| DEPBAR | `DEPBAR.LE SB0, N` | Waits on scoreboard-backed async-copy progress. | It defines when a staged tile is safe to consume from shared memory. | `sm_120`, `sm_120a` | `corpus/tensor_cores/18_pipelined_tile/` | `cp.async.wait_group` | [OBS] Waits on async-copy scoreboard. |
| BAR | `BAR.SYNC.DEFER_BLOCKING` | CTA-wide synchronization barrier. | It separates producer and consumer phases for shared-memory data visible across threads. | `sm_120`, `sm_120a` | `corpus/basics/06_shared_memory_scalar/` | `__syncthreads()` | [OBS] Block barrier form used before shared readback. |
| LDS | `LDS`, `LDS.128` | Scalar or vector shared-memory load. | It is the ordinary shared-memory read path, distinct from matrix-fragment LDSM. | `sm_120`, `sm_120a` | `corpus/basics/06_shared_memory_scalar/` | Shared-memory loads | [OBS] Shared-memory load family. |
| STS | `STS`, `STS.128` | Scalar or vector shared-memory store. | It is the ordinary shared-memory write path and the fallback contrast for STSM. | `sm_120`, `sm_120a` | `corpus/basics/06_shared_memory_scalar/` | Shared-memory stores | [OBS] Scalar/vector shared-store family, distinct from STSM. |
| LDG | `LDG`, `LDG.E`, `LDG.E.64`, `LDG.E.128`, `LDG.E.CONSTANT`, `LDG.E.ENL2.256` | Global-memory load. | It identifies global input traffic, cache qualifiers, and vectorized access width in audits. | `sm_120`, `sm_120a` | `corpus/basics/08_vectorized_load/` | Global loads | [OBS] Global-load family. |
| STG | `STG`, `STG.E`, `STG.E.64`, `STG.E.128`, `STG.E.ENL2.256` | Global-memory store. | It identifies final output traffic and epilogue writeback width. | `sm_120`, `sm_120a` | `corpus/basics/01_vector_add/` | Global stores | [OBS] Global-store family. |
| REDG | `REDG.E.ADD.F32...` | Global-memory reduction or atomic accumulation form. | It marks split-K or multi-CTA accumulation paths that are not plain stores. | `sm_120a` | `corpus/tensor_cores/24_production_mini_gemm/24z_split_k_or_multi_cta_reduction_stub.sass` | `atomicAdd` reduction stub | [OBS] Global reduction form used for split-K style accumulation. |

## Control-flow and warp instructions

| Family | Mnemonics observed | Meaning | Why it matters | Target | First evidence | Source trigger | Status |
|---|---|---|---|---|---|---|---|
| BRA | `BRA`, `BRA.U` | Branch instruction. | It is the primary marker for explicit loops, back edges, and non-predicated control flow. | `sm_120` | `corpus/basics/04_simple_loop/`, `corpus/tensor_cores/20_control_flow/` | Branches and loops | [OBS] Branch family. |
| EXIT | `EXIT`, predicated `EXIT` | Kernel exit instruction. | It marks return paths, bounds-check exits, and final control-flow convergence. | `sm_120` | `corpus/basics/01_vector_add/` | Bounds checks and returns | [OBS] Kernel exit form. |
| BSSY / BSYNC | `BSSY`, `BSYNC`, `BSSY.RECONVERGENT`, `BSYNC.RECONVERGENT` | Divergence and reconvergence markers. | They expose compiler-managed reconvergence regions around divergent branches and cold paths. | `sm_120`, `sm_120a` | `corpus/tensor_cores/21_divergence_reconvergence/` | Divergence/reconvergence and cold paths | [OBS] Reconvergence markers. |
| WARPSYNC | `WARPSYNC.ALL` | Warp-level synchronization. | It distinguishes warp-scoped ordering from CTA-scoped barriers. | `sm_120`, `sm_120a` | `corpus/tensor_cores/21_divergence_reconvergence/` | Guarded warp-level synchronization | [OBS] Warp synchronization form. |
| SHFL | `SHFL.BFLY`, `SHFL.IDX`, `SHFL.UP`, `SHFL.DOWN` | Cross-lane register exchange inside a warp. | It identifies warp-level data movement without shared memory. | `sm_120` | `corpus/warp_collectives/09_warp_shuffling/` | Warp shuffles | [OBS] Warp shuffle family. |
| VOTE | `VOTE.ANY`, `VOTE.ALL` | Warp-level predicate vote. | It shows collective boolean decisions across lanes. | `sm_120` | `corpus/warp_collectives/09_warp_shuffling/` | Warp votes | [OBS] Warp vote family. |
| REDUX | `REDUX`, `REDUX.SUM`, `REDUX.MIN`, `REDUX.MAX`, `REDUX.OR`, `REDUX.XOR` | Warp-level reduction. | It is the compact SASS signature for warp reductions that do not expand into shuffle trees. | `sm_120` | `corpus/warp_collectives/10_reduce/` | Warp reductions | [OBS] Warp reduction family. |

## Uniform-register instructions

| Family | Mnemonics observed | Meaning | Why it matters | Target | First evidence | Source trigger | Status |
|---|---|---|---|---|---|---|---|
| S2UR | `S2UR` | Moves special-register values into uniform registers. | It shows when warp-uniform state is represented in the uniform register file. | `sm_120`, `sm_120a` | `corpus/basics/07_umov/` | Uniform special-register movement | [OBS] Moves special-register values into uniform registers. |
| R2UR | `R2UR` | Moves a general register value into a uniform register. | It marks compiler-selected promotion from per-thread register flow into uniform flow. | `sm_120`, `sm_120a` | `corpus/math_and_spills/12_register_spill/`, `corpus/tensor_cores/21_divergence_reconvergence/` | Per-thread to uniform path selected by ptxas | [OBS] Register-to-uniform-register move. |
| UMOV | `UMOV`, `UMOV.64` | Uniform-register move or immediate materialization. | It is the basic data movement instruction inside uniform register flow. | `sm_120`, `sm_120a` | `corpus/basics/07_umov/` | Uniform immediate moves | [OBS] Uniform MOV family. |
| ULEA | `ULEA` | Uniform address arithmetic. | It identifies address computation done once per warp-uniform path rather than per lane. | `sm_120`, `sm_120a` | `corpus/basics/06_shared_memory_scalar/` | Uniform address arithmetic | [OBS] Uniform LEA family. |
| LDCU | `LDCU`, `LDCU.64` | Uniform constant-memory load. | It marks kernel parameter and constant loads that feed uniform computation. | `sm_120`, `sm_120a` | `corpus/basics/01_vector_add/` | Uniform constant loads | [OBS] Uniform constant-load family. |

## Not observed / out of SM120 scope

| Family | Meaning | Why it matters | Status |
|---|---|---|---|
| WGMMA / warpgroup MMA | Warpgroup-level matrix multiply family known from other NVIDIA architecture contexts. | Its absence prevents importing SM90-style warpgroup assumptions into SM120 findings. | [OBS] Not observed in the SM120/SM120a dumps through Kernel 25; warpgroup MMA is out of scope for consumer SM120. |
| TCGEN / TMEM | Tensor memory and tcgen-style datacenter Blackwell path. | Its absence keeps SM120 consumer analysis separate from SM100a datacenter mechanisms. | [OBS] Not observed in the SM120/SM120a dumps through Kernel 25; Blackwell datacenter SM100a feature, not consumer SM120 scope. |
| Control-code field model | Partial decoder for QMMA dtype/chain bits, `DEPBAR.LE SB0, N`, reuse, denvdis scoreboard fields, and scheduling classes. | It gives audit tooling stable names for the fields already backed by local evidence plus denvdis cross-checks. | [INF] See `knowledge/encoding/CONTROL_CODE.md`; full stall/yield bit placement remains [GAP]. |

## Glossary

### Tensor-core and matrix path

`HMMA` is the established FP16/BF16 warp-level MMA family used as the baseline for tensor-core analysis. It matters because its fragment and accumulator structure gives a reference point for newer low-precision families. [OBS]

`QMMA` is the Blackwell low-precision dense MMA family observed for FP8/FP6/FP4 forms. It matters because most SM120a tensor-core audits depend on recognizing its dtype, accumulator, sparse, and scale-factor variants. [OBS]

`OMMA` is the observed FP4 peak MMA family in block-scaled paths. It matters because it marks a different low-precision compute family from dense QMMA and carries separate operand-shape expectations. [OBS]

`LDSM` loads matrix fragments from shared memory into registers. It matters because MMA operands are often prepared by LDSM, so its ordering and width suffixes explain the fragment supply path. [OBS]

`STSM` stores matrix fragments from registers into shared memory. It matters because it defines storeback and epilogue patterns that cannot be explained by scalar `STS` alone. [OBS]

### Memory and synchronization path

`LDGSTS`, `LDGDEPBAR`, and `DEPBAR` form the observed async-copy staging path. They matter because production tensor-core kernels overlap global-memory movement with shared-memory tile consumption. [OBS]

`LDS` and `STS` are ordinary shared-memory load/store families. They matter because they are the contrast set for matrix-specific `LDSM` and `STSM` instructions. [OBS]

`LDG`, `STG`, and `REDG` identify global-memory input, output, and reduction traffic. They matter because production audits need to distinguish plain epilogue stores from reduction accumulation. [OBS]

### Control and uniform path

`BRA`, `EXIT`, `BSSY`, and `BSYNC` define explicit branch, exit, and reconvergence structure. They matter because control-flow shape is part of the SASS signature, not only a byproduct of source-level branching. [OBS]

`SHFL`, `VOTE`, and `REDUX` are warp-level collective families. They matter because they expose cross-lane computation without shared-memory traffic. [OBS]

`S2UR`, `R2UR`, `UMOV`, `ULEA`, and `LDCU` define uniform-register flow. They matter because uniform operands and addresses must be tracked separately from per-lane register flow. [OBS]
