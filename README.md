# SASS King

Systematic reverse engineering of NVIDIA SASS across architectures.

Each study pairs **SASS reading** (`cuobjdump` + [gpuasm.com](https://gpuasm.com/)) with **NCU profiling** to correlate instructions with measured performance.

Full context: [Part 1 — Reading NVIDIA SASS from First Principles](https://florianmattana.com/posts/sass_king/)

## Roadmap

### Phase 1 — Teaching kernels on SM120 (controlled variation)

* [x] **Part 1** — Kernels 01 to 04: baseline, FMA fusion, scoreboards, unroll cascade. [Read](https://florianmattana.com/posts/sass_king/)
* [x] Kernel 05 — Loop with small fixed trip count
* [x] Kernel 06 — Shared memory scalar (`LDS`, `STS`, `BAR.SYNC`), runtime modulo `CALL`
* [x] Kernel 07 — Shared memory patterns (bank conflicts, padding, multi-buffer)
* [x] Kernel 08 — Vectorized global memory (`LDG.E.128`, `LDG.E.ENL2.256`, FP64)
* [x] Kernel 09 — Warp primitives (`SHFL.BFLY`, `VOTE`, `MATCH`)
* [x] Kernel 10 — Warp reduction patterns (`REDUX`, butterfly, lane-zero)
* [x] Kernel 11 — Slowpath arithmetic (`MUFU.RCP/LG2/EX2/RSQ`, inline division, `log2f`, `expf`, `sinf`, `sqrtf`, Payne-Hanek)
* [x] Kernel 12 — Register spill and local memory (`STL`, `LDL`, `LDL.LU`, `STL.128`, stack frame, `R2UR`)

### Phase 2 — Tensor core kernels on SM120 (controlled variation)

* [x] Kernel 13 — `HMMA` baseline (FP16, BF16, m16n8k16): opcode family, register allocation, accumulator chaining, serial latency model
* [x] Kernel 14 — `QMMA` baseline FP8/FP6/FP4 (`kind::f8f6f4`, m16n8k32): new opcode family, dtype encoding decoded across 5 input dtypes, MMA-family wide invariants, serial latency model
* [ ] Kernel 15 — MMA narrow (FP6, FP4 standalone variants, mixed-precision combinations)
* [ ] Kernel 16 — FP4 peak (`kind::mxf8f6f4` block-scaled, `m16n8k64`, scale factor encoding)
* [ ] Kernel 17 — `ldmatrix` and `stmatrix` (`LDSM`, `.trans` modifier, latency)
* [ ] Kernel 18 — Pipelined MMA tile (ldmatrix + MMA scoreboard interleaving, accumulator chains)
* [ ] Kernel 19 — Sparse MMA (sparsity metadata encoding)

### Phase 3 — Classical algorithms

* [ ] Vector Add
* [ ] Prefix Sum (scan)
* [ ] SGEMM
* [ ] Reduction (sum, max)
* [ ] Softmax
* [ ] LayerNorm / RMSNorm

### Phase 4 — Library audits

Real production kernels, annotated end to end. Targets below; kernel counts reflect what is available on gpuasm.com.

| Library            | Kernels | Status  |
|--------------------|---------|---------|
| flash_attn2        | 138     | Planned |
| flash_attn4        | 49      | Planned |
| cutlass (SM120a)   | 113     | Planned |
| cute-tutorial      | 13      | Planned |
| xformers           | 36      | Planned |
| transformer_engine | 109     | Planned |
| flashinfer         | 36      | Planned |
| flashmla           | 9       | Planned |
| deepep             | 2       | Planned |
| llamacpp / ggml    | 218     | Planned |
| sglang             | 14      | Planned |
| llmc               | 8       | Planned |
| tinygrad           | 12      | Planned |
| nunchaku           | 37      | Planned |
| fouroversix        | 57      | Planned |
| bitsandbytes       | 2       | Planned |
| arcquant           | 24      | Planned |
| qerl               | 6       | Planned |
| sgemm              | 60      | Planned |
| quack              | —       | Planned |

### Phase 5 — Cross-architecture

Same studies replayed on:

* [ ] SM80 (A100)
* [ ] SM86 (RTX 3090) — gpuasm example corpus
* [ ] SM89 (RTX 4090)
* [ ] SM90a (H100)
* [ ] SM100a (B200)

### Phase 6 — Reference

* [ ] Per-instruction SASS reference, one page per opcode, per architecture, with empirical latency, throughput, pipeline, and dual-issue rules.

## Target architectures

| Arch   | GPU              | Why                                                 |
|--------|------------------|-----------------------------------------------------|
| SM80   | A100             | Datacenter Ampere baseline                          |
| SM86   | RTX 3090         | Consumer Ampere, gpuasm example corpus              |
| SM89   | RTX 4090         | Most common consumer inference card                 |
| SM90a  | H100             | TMA, WGMMA, warp specialization, mbarrier, clusters |
| SM100a | B200             | `tcgen05.mma`, TMEM                                 |
| SM120  | RTX 5070 Ti/5090 | Hybrid SM90/SM100 ISA, `mma.sync` with `mxf8f6f4`   |

Work starts on SM120 (direct hardware access). Other architectures via public dumps and contributors.

## Repository layout

* `01_vector_add/` to `12_register_spill/` — Phase 1 teaching kernels (basic SASS patterns, controlled variation)
* `tensor_cores/` — Phase 2 tensor core kernels organized by chapter
  * `13_hmma_fp16/` — HMMA opcode family baseline
  * `14_qmma_fp8/` — QMMA opcode family baseline
  * `15_mma_narrow/`, `16_fp4_peak/`, `17_ldmatrix/`, `18_pipelined_tile/`, `19_sparse_mma/` — planned
* `FINDINGS.md` — running log of observations, hypotheses, and resolutions, organized by chapter, with cross-chapter summary of pipelines, invariants, and canonical patterns

Each chapter folder contains the kernel sources (`.cu`), compiled binaries, and a `conclusion<N>.md` writeup. SASS dumps (`.sass`) are reproducible from the binaries via `cuobjdump --dump-sass` and are not committed.

## Tools

* `cuobjdump --dump-sass` for raw disassembly
* [gpuasm.com](https://gpuasm.com/) for scoreboards, stalls, pressure, and dependency arrows
* Nsight Compute (NCU) for per-kernel profiling, SASS-to-source mapping, and stall attribution

## Author

Florian Mattana. [florianmattana.com](https://florianmattana.com)