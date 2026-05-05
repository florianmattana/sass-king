# Tensor cores (SM120)

Chapter 13 onwards. Systematic SASS-level documentation of NVIDIA tensor core instructions on SM120 (Blackwell consumer, RTX 50 series).

## Why this section exists

Jia et al. (Citadel, 2018-2019) dissected tensor cores on Volta and Turing. Nothing systematic has been published since for Ampere, Hopper, or Blackwell. For SM120 specifically: zero public documentation of the SASS form emitted by `mma.sync`. This section fills that gap.

NVIDIA publishes PTX documentation (`mma.sync`, `ldmatrix`, block scaling qualifiers). What each PTX atom becomes at SASS level is undocumented for Blackwell consumer.

## Scope

Warp-level tensor core on SM120:

* `mma.sync.aligned` with `kind::f8f6f4`, `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4`
* `ldmatrix` and `stmatrix` families
* `cp.async` for pipelined shared memory loads
* Sparse MMA via `mma.sp::ordered_metadata`
* Shapes m16n8k16, m16n8k32, m16n8k64, m16n8k128 (sparse)

Not in scope on SM120 (architecturally unavailable):

* `wgmma.mma_async` (Hopper SM90+)
* `tcgen05.mma` and TMEM (Blackwell datacenter SM100a+)
* Multicast clusters

## Chapter plan

| Chapter | Topic | Key finding |
|---|---|---|
| [13_hmma_fp16/](./13_hmma_fp16/) | HMMA baseline: f16, bf16 at m16n8k16 | New SASS opcode family `HMMA.16816.<acc>`, serial latency ~35 cycles/HMMA |
| [14_qmma_fp8/](./14_qmma_fp8/) | FP8/FP6/FP4 non-scaled, kind::f8f6f4, m16n8k32 | New SASS opcode `QMMA.16832.<acc>.<A>.<B>`, dtype encoding decoded by Popper-style prediction |
| [15_mma_narrow/](./15_mma_narrow/) | FP6 and FP4 mixed-precision variants | Complete for SASS encoding: mixed FP6, reversed mixed FP6, FP16 accumulator, and latency/probe sources added |
| [16_fp4_peak/](./16_fp4_peak/) | FP4 peak block-scaled (kind::mxf8f6f4 and kind::mxf4nvf4, 900+ TFLOPS path) | New SASS opcode family `OMMA` (low byte 0x7f). `.SF` modifier for block scaling. `.UE4M3.4X` suffix identifies the peak path. OMMA cycles/MMA ~29 (vs ~35 HMMA/QMMA) |
| [17_ldmatrix/](./17_ldmatrix/) | `ldmatrix` (6 variants: x1/x2/x4 × trans/no-trans) | New opcode `LDSM.16.M[T]88[.N]`, production pattern captured |
| [18_pipelined_tile/](./18_pipelined_tile/) | Full pipelined GEMM tile with cp.async, LDSM, HMMA | Three new opcodes: LDGSTS, LDGDEPBAR, DEPBAR.LE. Full production pipeline decoded |
| [19_sparse_mma/](./19_sparse_mma/) | 2:4 structured sparsity MMA | Sparse forms are `QMMA.SP`, `QMMA.SF.SP`, and `OMMA.SF.SP`; metadata is an explicit register operand |
| [20_control_flow/](./20_control_flow/) | Control flow | Constant loops fully unroll by default; preserved loops expose back-edge BRA; 8 x 2 HMMA probe emits 16 HMMAs |
| [21_divergence_reconvergence/](./21_divergence_reconvergence/) | Divergence and reconvergence | Lane divergence alone does not always force BSSY/BSYNC; ptxas also uses predication, FSEL/UFSEL, predicated EXIT, VOTE, WARPSYNC.ALL, and local CALL patterns |
| [22_stmatrix/](./22_stmatrix/) | stmatrix / matrix store | STSM exists for m8n8 b16: `STSM.16.M[T]88[.2|.4]`; tested m16n8 b8 forms are target-qualified |
| [23_fragment_layout/](./23_fragment_layout/) | FP4 / FP6 fragment layout | 22 probes: dense QMMA, LDSM-fed QMMA, scale-vector OMMA, sparse metadata; runtime smoke done, full decode open |
| [24_production_mini_gemm/](./24_production_mini_gemm/) | Production mini-GEMM audit | 30 production-like probes: LDGSTS, LDSM, HMMA/QMMA/OMMA, STSM, STG, REDG, cold paths |
| [25_stsm_epilogue/](./25_stsm_epilogue/) | STSM epilogue layout and storeback semantics | 26 probes: STSM layout, MMA epilogues, STS fallback, conversions, storeback, b8 compatibility |

## Phase 3 gate

Phase 3 does not start until the SM120 audit prerequisites below are complete.

Required before Phase 3:

- [x] Chapter 20 - Control flow
- [x] Chapter 21 - Divergence and reconvergence
- [x] Chapter 22 - stmatrix / matrix store

Strongly recommended before Phase 3:

- [x] Chapter 23 - FP4 / FP6 fragment layout
- [x] Chapter 24 - Production mini-GEMM audit
- [x] Chapter 25 - STSM epilogue layout and storeback semantics
- [ ] Audit confidence framework - methodology checklist for qualifying production-kernel conclusions

Deferred and non-blocking for Phase 3:

- [ ] Sparse QMMA / OMMA latency once hardware runtime is available
- [ ] Exact `.SP` bit placement in opcode/control fields
- [ ] Full control-code bit decoder

## Methodology

Same controlled variation methodology as chapters 01-12. One minimal change per variant, recompile, dump SASS via `cuobjdump --dump-sass`, diff against baseline, document. Each kernel is a standalone `.cu` with full `main()`.

Kernels use inline PTX `asm volatile` rather than CUTLASS wrappers to minimize compiler noise. CUTLASS wrappers may be used for comparison in specific variants.

## Relationship with knowledge/

Tensor core findings live in `../knowledge/FINDINGS.md`, under dedicated `## Kernel <N>` sections. Reusable instruction-family facts live in `../knowledge/SASS_INSTRUCTIONS_SM120.md` and `../knowledge/encoding/`.

The per-chapter `conclusion{N}.md` files here document the narrative of the chapter (what we tried, in what order, what the deltas were). They complement but do not duplicate the global knowledge files.

## Status

| Chapter | Status | Variants | Key opcode(s) |
|---|---|---|---|
| 13 | **Done** | 13a-13e (5) | HMMA.16816.F32, HMMA.16816.F16 |
| 14 | **Done** | 14a-14j (10) | QMMA.16832.\<acc>.\<A>.\<B> |
| 15 | **Done** | 15c, 15f-15k + chapter 14 cross-references | QMMA.16832.F32.E3M2.E2M3, QMMA.16832.F16.E3M2.E3M2 |
| 16 | **Done** | 16a-16d (4) | QMMA.SF, OMMA.SF.16864, OMMA.SF.16864.UE4M3.4X |
| 17 | **Done** | 17a-17f (6) | LDSM.16.M\[T\]88\[.N\] |
| 18 | **Done** | 18a-18c (3) | LDGSTS.E.LTC128B.128, LDGDEPBAR, DEPBAR.LE SB0, N |
| 19 | **Done** | 19a-19m (13) | QMMA.SP.16864, QMMA.SF.SP.16864, OMMA.SF.SP.168128 |
| 20 | **Done** | 20a-20v (22) | BRA, BRA.U, predication, BSSY/BSYNC for break |
| 21 | **Done** | 21a-21t (20) | BSSY/BSYNC, VOTE.ANY, SHFL.IDX, WARPSYNC.ALL, predicated EXIT |
| 22 | **Done** | 22a-22l (12) | STSM.16.M88, STSM.16.MT88, STS.128 fallback |
| 23 | **SASS + runtime smoke done** | 23a-23v (22) | QMMA.16832 FP4/FP6, QMMA.SF, OMMA.SF.4X, QMMA.SF.SP, LDSM-fed QMMA |
| 24 | **SASS + runtime smoke done** | 24a-24ad (30) | LDGSTS, LDSM, HMMA/QMMA/OMMA, STSM, STG, REDG |
| 25 | **SASS + runtime smoke done** | 25a-25z (26) | STSM.16.M88/MT88, STS.128, F2F.F16/BF16, MMA-to-STSM, STG |

Chapters 13, 14, 16, 17, 18, and 19 decode the core tensor-core instruction families. Chapter 20 closes the first control-flow gate for loop lowering and back-edge detection. Chapter 21 closes the first divergence/reconvergence gate for the tested lane-divergent patterns. Chapter 22 closes the first matrix-store gate for m8n8 b16 STSM. Chapter 23 closes first-pass FP4/FP6 SASS coverage and runtime smoke execution for fragment-layout probes, with full lane-to-value decode still open. Chapter 24 closes first-pass production-like mini-GEMM SASS coverage and runtime smoke execution, but remains structural rather than a full numeric GEMM correctness suite. Chapter 25 closes first-pass STSM epilogue/storeback SASS coverage and runtime smoke execution. The audit confidence framework remains open so production-kernel conclusions can be qualified explicitly.

## SM120 MMA opcode landscape (after chapter 19)

| Family | Low byte | Shape | PTX kind | Scaled? |
|---|---|---|---|---|
| HMMA | 0x3c | m16n8k16 | mma.sync standard | No |
| QMMA | 0x7a | m16n8k32 | kind::f8f6f4, kind::mxf8f6f4 | Optional (.SF modifier) |
| OMMA | 0x7f | m16n8k64 | kind::mxf4nvf4 | Always (implicit .SF) |
| QMMA.SP | 0x7a | m16n8k64 | mma.sp kind::f8f6f4 | Sparse |
| QMMA.SF.SP | 0x7a | m16n8k64 | mma.sp kind::mxf8f6f4 | Sparse and scaled |
| OMMA.SF.SP | 0x7f | m16n8k128 | mma.sp kind::mxf4nvf4 | Sparse and scaled |

## Hardware and toolchain

* GPU: NVIDIA RTX 5070 Ti (SM120, compute capability 12.0, 46 SMs, 12 GB GDDR7)
* Compile: `nvcc -arch=sm_120` for base MMA, `nvcc -arch=compute_120a -code=sm_120a` for arch-conditional features (kind::f8f6f4, kind::mxf8f6f4, kind::mxf4nvf4, block scaling)
* Dump: `cuobjdump --dump-sass <binary>`
* Profile: Nsight Compute (NCU) with `--set full`
* Annotation: gpuasm.com for scoreboards, stalls, pipeline visualization
