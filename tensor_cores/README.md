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
| 15 | **Done** | 15c, 15f-15k + chapter 14 cross-references | QMMA.16832.F32.E3M2.E2M3, QMMA.16832.F16.E3M2.E3M2 |
| 16 | **Done** | 16a-16d (4) | QMMA.SF, OMMA.SF.16864, OMMA.SF.16864.UE4M3.4X |
| 17 | **Done** | 17a-17f (6) | LDSM.16.M\[T\]88\[.N\] |
| 18 | **Done** | 18a-18c (3) | LDGSTS.E.LTC128B.128, LDGDEPBAR, DEPBAR.LE SB0, N |
| 19 | Planned | - | - |

Chapters 13, 14, 16, 17, 18 together form a complete toolkit for auditing production GEMM, attention, and block-scaled FP4 kernels on SM120.

## SM120 MMA opcode landscape (after chapter 16)

| Family | Low byte | Shape | PTX kind | Scaled? |
|---|---|---|---|---|
| HMMA | 0x3c | m16n8k16 | mma.sync standard | No |
| QMMA | 0x7a | m16n8k32 | kind::f8f6f4, kind::mxf8f6f4 | Optional (.SF modifier) |
| OMMA | 0x7f | m16n8k64 | kind::mxf4nvf4 | Always (implicit .SF) |

## Hardware and toolchain

* GPU: NVIDIA RTX 5070 Ti (SM120, compute capability 12.0, 46 SMs, 12 GB GDDR7)
* Compile: `nvcc -arch=sm_120` for base MMA, `nvcc -arch=compute_120a -code=sm_120a` for arch-conditional features (kind::f8f6f4, kind::mxf8f6f4, kind::mxf4nvf4, block scaling)
* Dump: `cuobjdump --dump-sass <binary>`
* Profile: Nsight Compute (NCU) with `--set full`
* Annotation: gpuasm.com for scoreboards, stalls, pipeline visualization
