# SASS King

Systematic reverse engineering of NVIDIA SASS, with the goal of enabling kernel-level audits on compiled CUDA binaries. Starting with SM120 (consumer Blackwell, RTX 50 series), expanding to other architectures over time.

The last systematic public work on SASS was Jia et al. (Citadel) on Volta and Turing in 2018. Nothing equivalent exists for Ampere, Hopper, or Blackwell. For SM120 specifically: zero. SASS King fills that gap.

Each study pairs SASS reading (cuobjdump + gpuasm.com) with NCU profiling to correlate instructions with measured performance.

Full context: Part 1 — Reading NVIDIA SASS from First Principles

## Goal

Build the tools, documentation, and pattern library that let a kernel engineer open a SASS dump and understand what the compiler did, why, and what can be improved at the source level.

Concretely, the project works toward an audit tool that takes a cubin and produces an optimization report. The public side is the knowledge: patterns, observations, annotated kernels. That's what SASS King is.

## Methodology

**Controlled variation.** Two kernels differ by exactly one variable (dtype, operand order, unroll factor, compilation flag). The SASS diff isolates the compiler decision. Microbenchmarks via `%clock` for latency measurement. Results are reproducible from the kernel sources in the repo.

**Strict claim tagging.** Every observation is tagged: [OBS] for measured facts, [INF] for logical deductions, [HYP] for speculation, [RES] when a hypothesis becomes confirmed or refuted, [GAP] for open questions documented honestly rather than papered over.

**Top-down and bottom-up combined.** Top-down: pattern recognition on real kernels (which structures recur, which are anomalies). Bottom-up: microbenchmarks isolating single instructions or sequences. Each approach validates the other.

## Roadmap

### Phase 1 — Teaching kernels on SM120 (controlled variation)

- [x] Part 1 — Kernels 01 to 04: baseline, FMA fusion, scoreboards, unroll cascade. [Read](https://florianmattana.com/p/reading-nvidia-sass-from-first-principles)
- [x] Kernel 05 — Loop with small fixed trip count
- [x] Kernel 06 — Shared memory scalar (LDS, STS, BAR.SYNC), runtime modulo CALL
- [x] Kernel 07 — Shared memory patterns (bank conflicts, padding, multi-buffer)
- [x] Kernel 08 — Vectorized global memory (LDG.E.128, LDG.E.ENL2.256, FP64)
- [x] Kernel 09 — Warp primitives (SHFL.BFLY, VOTE, MATCH)
- [x] Kernel 10 — Warp reduction patterns (REDUX, butterfly, lane-zero)
- [x] Kernel 11 — Slowpath arithmetic (MUFU.RCP/LG2/EX2/RSQ, inline division, log2f, expf, sinf, sqrtf, Payne-Hanek)
- [x] Kernel 12 — Register spill and local memory (STL, LDL, LDL.LU, STL.128, stack frame, R2UR)

### Phase 2 - Teaching and tensor core kernels on SM120 (controlled variation)

- [x] Kernel 13 — HMMA baseline (FP16, BF16, m16n8k16): opcode family, register allocation, accumulator chaining, serial latency model
- [x] Kernel 14 — QMMA baseline FP8/FP6/FP4 (kind::f8f6f4, m16n8k32): new opcode family, dtype encoding decoded across 5 input dtypes, MMA-family wide invariants, serial latency model
- [x] Kernel 15 - MMA narrow (FP6, FP4 standalone variants, mixed-precision combinations)
- [x] Kernel 16 — FP4 peak (kind::mxf8f6f4 block-scaled, m16n8k64, scale factor encoding)
- [x] Kernel 17 — ldmatrix and stmatrix (LDSM, .trans modifier, latency)
- [x] Kernel 18 — Pipelined MMA tile (ldmatrix + MMA scoreboard interleaving, accumulator chains)
- [x] Kernel 19 - Sparse MMA (sparsity metadata encoding)
- [ ] Kernel 20 - Control flow (back-edge BRA detection, loop detection, predication vs branching)
- [ ] Kernel 21 - Divergence and reconvergence (BSSY/BSYNC, warp-divergent branches, predicated arithmetic)
- [ ] Kernel 22 - stmatrix / matrix store (STSM if present, fallback STS sequence if not present)
- [ ] Kernel 23 - FP4 / FP6 fragment layout (E2M1, E3M2, E2M3 packing and runtime validation)
- [ ] Kernel 24 - Production mini-GEMM audit (LDGSTS + LDSM + QMMA/OMMA + STG end-to-end)

### Phase 3 gate

Phase 3 does not start until the remaining SM120 coverage needed for production audits is complete.

Required before Phase 3:

- [ ] Kernel 20 - Control flow
- [ ] Kernel 21 - Divergence and reconvergence
- [ ] Kernel 22 - stmatrix / matrix store

Strongly recommended before Phase 3:

- [ ] Kernel 23 - FP4 / FP6 fragment layout
- [ ] Kernel 24 - Production mini-GEMM audit

Deferred and non-blocking for Phase 3:

- [ ] Sparse QMMA / OMMA latency once hardware runtime is available
- [ ] Exact `.SP` bit placement in opcode/control fields
- [ ] Full control-code bit decoder

### Phase 3 — Pattern library

Formalize patterns extracted from chapter studies and production kernels. Each pattern gets a machine-readable signature, a reference kernel, and annotated SASS.

Running target: 20-30 patterns in the first pass. Examples in scope:

- Warp reduction (sum, max, min, and, or, xor)
- Chain HMMA (FP16, BF16) / QMMA (FP8 variants) / OMMA (FP4 peak)
- cp.async pipeline (2-stage, 3-stage)
- LDSM fragment loading (x1/x2/x4, with and without transpose)
- Online softmax chunk (FlashAttention-style)
- Block scaling computation (UE8M0, UE4M3)
- FP4 encoding cascade
- Register spill signature and recovery patterns

### Phase 4 — Library audits

Real production kernels, annotated end to end using the pattern library. Targets below; kernel counts reflect what is available on gpuasm.com.

| Library | Kernels | Status |
|---|---|---|
| flash_attn2 | 138 | Planned |
| flash_attn4 | 49 | Planned |
| cutlass (SM120a) | 113 | Planned |
| cute-tutorial | 13 | Planned |
| xformers | 36 | Planned |
| transformer_engine | 109 | Planned |
| flashinfer | 36 | Planned |
| flashmla | 9 | Planned |
| deepep | 2 | Planned |
| llamacpp / ggml | 218 | Planned |
| sglang | 14 | Planned |
| llmc | 8 | Planned |
| tinygrad | 12 | Planned |
| nunchaku | 37 | Planned |
| fouroversix | 57 | Planned |
| bitsandbytes | 2 | Planned |
| arcquant | 24 | Planned |
| qerl | 6 | Planned |
| sgemm | 60 | Planned |
| quack | 83 | Planned |

Prioritization will favor coverage diversity (different algorithmic patterns) over raw count. Not all 1000+ kernels will be audited individually; representative kernels per library are the goal.

### Phase 5 — Audit tool

A pipeline that takes a cubin, detects patterns via the public pattern library, and produces an optimization report. Scope and distribution to be finalized as the library matures.

### Phase 6 — Cross-architecture

Same studies replayed on:

- [ ] SM80 (A100)
- [ ] SM86 (RTX 3090) — gpuasm example corpus
- [ ] SM89 (RTX 4090)
- [ ] SM90a (H100)
- [ ] SM100a (B200)

Via direct hardware access, contributors, or public dumps. The methodology and pattern format transfer; architecture-specific data requires its own measurements.

## Target architectures

| Arch | GPU | Why |
|---|---|---|
| SM80 | A100 | Datacenter Ampere baseline |
| SM86 | RTX 3090 | Consumer Ampere, gpuasm example corpus |
| SM89 | RTX 4090 | Most common consumer inference card |
| SM90a | H100 | TMA, WGMMA, warp specialization, mbarrier, clusters |
| SM100a | B200 | tcgen05.mma, TMEM |
| SM120 | RTX 5070 Ti/5090 | Hybrid SM90/SM100 ISA, mma.sync with mxf8f6f4 |

Work starts on SM120 (direct hardware access). Other architectures via public dumps and contributors.

## Repository layout

- `01_vector_add/` to `12_register_spill/` — Phase 1 teaching kernels (basic SASS patterns, controlled variation)
- `tensor_cores/` — Phase 2 tensor core kernels organized by chapter
  - `13_hmma_fp16/` — HMMA opcode family baseline
  - `14_qmma_fp8/` — QMMA opcode family baseline
  - `15_mma_narrow/` - FP6, FP4, and mixed narrow QMMA variants
  - `16_fp4_peak/` — OMMA, block-scaled FP4
  - `17_ldmatrix/` — LDSM variants
  - `18_pipelined_tile/` — cp.async pipeline
  - `19_sparse_mma/` - sparse MMA metadata and sparse QMMA/OMMA forms
  - `20_control_flow/`, `21_divergence_reconvergence/`, `22_stmatrix/`, `23_fragment_layout/`, `24_production_mini_gemm/` - planned before Phase 3
- `patterns/` (coming) — formalized pattern library
- `production/` (coming) — production kernel audits
- `FINDINGS.md` — running log of observations, hypotheses, and resolutions, organized by chapter, with cross-chapter summary of pipelines, invariants, canonical patterns, and open gaps

Each chapter folder contains the kernel sources (.cu), compiled binaries, and a `conclusion<N>.md` writeup. SASS dumps (.sass) are reproducible from the binaries via `cuobjdump --dump-sass` and are not committed.

## Tools

- `cuobjdump --dump-sass` for raw disassembly
- `gpuasm.com` for scoreboards, stalls, pressure, and dependency arrows
- Nsight Compute (NCU) for per-kernel profiling, SASS-to-source mapping, and stall attribution
- `nvcc -Xptxas -v` for register usage and spill warnings

## Related work

SASS King stands on the shoulders of several reverse engineering efforts and remains complementary to them.

- [redplait/denvdis](https://github.com/redplait/denvdis) extracts opcode tables and latency data directly from ptxas and provides `dg.pl` for scheduling analysis and `ced` for cubin patching. Essential low-level tooling.
- [kuterdinel.com/nv_isa](https://kuterdinel.com/nv_isa/) provides a fuzzed ISA specification for SM90a.
- Jia et al. 2018, "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." Foundational methodology.
- NVIDIA `cuda-binary-utilities` documentation, the official opcode list for Blackwell.

At this stage SASS King operates at a different layer: algorithmic pattern recognition on production kernels, mapped to source-level optimization opportunities.

## Contributing

Contributions are welcome, especially dumps from hardware we don't have direct access to (A100, H100, RTX 4090, B200). See `CONTRIBUTING.md` for what's useful: raw SASS dumps with metadata, kernel studies following the controlled variation methodology, corrections to observations, or new pattern proposals.

If you work on SASS-level tooling (redplait, kuterdinel, others), collaboration on complementary layers is welcome. Reach out via GitHub issues or at [florianmattana.com](https://florianmattana.com).

## Author

Florian Mattana. [florianmattana.com](https://florianmattana.com).
