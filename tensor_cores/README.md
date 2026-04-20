# Tensor cores (SM120)

Chapter 13 onwards. Systematic SASS-level documentation of NVIDIA tensor core instructions on SM120 (Blackwell consumer, RTX 50 series).

## Why this section exists

Jia et al. (Citadel, 2018-2019) dissected tensor cores on Volta and Turing. Nothing systematic has been published since for Ampere, Hopper, or Blackwell. For SM120 specifically: zero public documentation of the SASS form emitted by `mma.sync`. This section fills that gap.

NVIDIA publishes PTX documentation (`mma.sync`, `ldmatrix`, block scaling qualifiers). What each PTX atom becomes at SASS level is undocumented for Blackwell consumer.

## Scope

Warp-level tensor core on SM120:
* `mma.sync.aligned` with `kind::f8f6f4`, `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4`
* `ldmatrix` and `stmatrix` families
* Sparse MMA via `mma.sp::ordered_metadata`
* Shapes m16n8k16, m16n8k32, m16n8k64, m16n8k128 (sparse)

Not in scope on SM120 (architecturally unavailable):
* `wgmma.mma_async` (Hopper SM90+)
* `tcgen05.mma` and TMEM (Blackwell datacenter SM100a+)
* Multicast clusters

## Chapter plan

| Chapter | Topic | Key question |
|---|---|---|
| [13_hmma_fp16/](./13_hmma_fp16/) | HMMA baseline: f16, bf16 at m16n8k16 | What is the SASS opcode for heritage MMA? |
| [14_qmma_fp8/](./14_qmma_fp8/) | FP8 non-scaled, kind::f8f6f4, m16n8k32 | How does `kind::` prefix appear at SASS? |
| [15_mma_narrow/](./15_mma_narrow/) | FP6 and FP4 at k=32 | How are narrow dtypes packed in registers? |
| [16_fp4_peak/](./16_fp4_peak/) | FP4 at k=64 with block scaling (900 TFLOPS peak) | Why does k=64 reach peak and k=32 does not? |
| [17_ldmatrix/](./17_ldmatrix/) | `ldmatrix` and `stmatrix` variants | What is the SASS opcode and addressing mode? |
| [18_pipelined_tile/](./18_pipelined_tile/) | Full MMA tile with accumulator reuse | How does a production GEMM microkernel look in SASS? |
| [19_sparse_mma/](./19_sparse_mma/) | 2:4 structured sparsity MMA (1801 TOPS peak) | How is sparsity metadata encoded? |

## Methodology

Same controlled variation methodology as chapters 01-12. One minimal change per variant, recompile, dump SASS via `cuobjdump --dump-sass`, diff against baseline, document. Each kernel is a standalone `.cu` with full `main()`.

Kernels use inline PTX `asm volatile` rather than CUTLASS wrappers to minimize compiler noise. CUTLASS wrappers may be used for comparison in specific variants.

## Relationship with FINDINGS.md

Tensor core findings live in the main `FINDINGS.md` at the root of the repo, under a dedicated "Tensor core" section. This is consistent with the project's principle of avoiding document proliferation: new opcodes extend the Pipelines table, new canonical patterns go in the Canonical patterns section, new architectural invariants go in Architectural invariants.

The per-chapter `conclusion{N}.md` files here document the narrative of the chapter (what we tried, in what order, what the deltas were). They complement but do not duplicate FINDINGS.md.

## Status

| Chapter | Status |
|---|---|
| 13 | Planned |
| 14 | Planned |
| 15 | Planned |
| 16 | Planned |
| 17 | Planned |
| 18 | Planned |
| 19 | Planned |

## Hardware and toolchain

* GPU: NVIDIA RTX 5070 Ti (SM120, compute capability 12.0, 46 SMs, 12 GB GDDR7)
* Compile: `nvcc -arch=sm_120` for base MMA, `nvcc -arch=sm_120a` for arch-conditional features (block scaling)
* Dump: `cuobjdump --dump-sass <binary>`
* Profile: Nsight Compute (NCU) with `--set full`
* Annotation: gpuasm.com for scoreboards, stalls, pipeline visualization