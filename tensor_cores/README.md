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
| [15_mma_narrow/](./15_mma_narrow/) | FP6 and FP4 mixed-precision variants | Planned (partially covered by chapter 14) |
| [16_fp4_peak/](./16_fp4_peak/) | FP4 at k=64 with block scaling (900 TFLOPS peak) | Planned |
| [17_ldmatrix/](./17_ldmatrix/) | `ldmatrix` (6 variants: x1/x2/x4 × trans/no-trans) | New opcode `LDSM.16.M[T]88[.N]`, production pattern captured |
| [18_pipelined_tile/](./18_pipelined_tile/) | Full pipelined GEMM tile with cp.async, LDSM, HMMA | Three new opcodes: LDGSTS, LDGDEPBAR, DEPBAR.LE. Full production pipeline decoded |
| [19_sparse_mma/](./19_sparse_mma/) | 2:4 structured sparsity MMA (1801 TOPS peak) | Planned |

## Methodology

Same controlled variation methodology as chapters 01-12. One minimal change per variant, recompile, dump SASS via `cuobjdump --dump-sass`, diff against baseline, document. Each kernel is a standalone `.cu` with full `main()`.

Kernels use inline PTX `asm volatile` rather than CUTLASS wrappers to minimize compiler noise. CUTLASS wrappers may be used for comparison in specific variants.

## Relationship with FINDINGS.md

Tensor core findings live in the main `FINDINGS.md` at the root of the repo, under dedicated `## Kernel <N>` sections. This is consistent with the project's principle of avoiding document proliferation: new opcodes extend the Pipelines table, new canonical patterns go in the Canonical patterns section, new architectural invariants go in Architectural invariants.

The per-chapter `conclusion{N}.md` files here document the narrative of the chapter (what we tried, in what order, what the deltas were). They complement but do not duplicate FINDINGS.md.

## Status

| Chapter | Status | Variants | Key opcode(s) |
|---|---|---|---|
| 13 | **Done** | 13a-13e (5) | HMMA.16816.F32, HMMA.16816.F16 |
| 14 | **Done** | 14a-14j (10) | QMMA.16832.\<acc>.\<A>.\<B> |
| 15 | Planned | - | - |
| 16 | Planned | - | - |
| 17 | **Done** | 17a-17f (6) | LDSM.16.M\[T\]88\[.N\] |
| 18 | **Done** | 18a-18c (3) | LDGSTS.E.LTC128B.128, LDGDEPBAR, DEPBAR.LE SB0, N |
| 19 | Planned | - | - |

Chapters 13, 14, 17, 18 together form a complete toolkit for auditing production GEMM and attention kernels on SM120.

## Hardware and toolchain

* GPU: NVIDIA RTX 5070 Ti (SM120, compute capability 12.0, 46 SMs, 12 GB GDDR7)
* Compile: `nvcc -arch=sm_120` for base MMA, `nvcc -arch=compute_120a -code=sm_120a` for arch-conditional features (kind::f8f6f4, block scaling)
* Dump: `cuobjdump --dump-sass <binary>`
* Profile: Nsight Compute (NCU) with `--set full`
* Annotation: gpuasm.com for scoreboards, stalls, pipeline visualization
