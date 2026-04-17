# SASS King

Systematic reverse engineering of NVIDIA SASS across architectures.

Each study pairs **SASS reading** (`cuobjdump` + [gpuasm.com](https://gpuasm.com/)) with **NCU profiling** to correlate instructions with measured performance.

Full context: [Part 1 — Reading NVIDIA SASS from First Principles](https://florianmattana.com/posts/sass_king/)

## Roadmap

### Phase 1 — Teaching kernels on SM120 (controlled variation)

- [x] **Part 1** — Kernels 01–04: baseline, FMA fusion, scoreboards, unroll cascade. [Read](https://florianmattana.com/posts/sass_king/)
- [ ] Kernel 05 — Loop with small fixed trip count
- [ ] Kernel 06 — Shared memory scalar (`LDS`, `STS`, `BAR.SYNC`)
- [ ] Kernel 07 — Vectorized load (`LDG.E.128`)
- [ ] Kernel 08 — Warp-level reduction (`SHFL.BFLY`)
- [ ] Kernel 09 — Division slowpath (`CALL.REL.NOINC`)
- [ ] Kernel 10 — Forced register spill (`STL`, `LDL`)
- [ ] Kernel 11 — First tensor core (`QMMA`)

### Phase 2 — Classical algorithms

- [ ] Vector Add
- [ ] Prefix Sum (scan)
- [ ] SGEMM
- [ ] Reduction (sum, max)
- [ ] Softmax
- [ ] LayerNorm / RMSNorm

### Phase 3 — Library audits

Real production kernels, annotated end to end. Targets below; kernel counts reflect what is available on gpuasm.com.

| Library              | Kernels | Status |
|----------------------|---------|--------|
| flash_attn2          | 138     | Planned |
| flash_attn4          | 49      | Planned |
| cutlass (SM120a)     | 113     | Planned |
| cute-tutorial        | 13      | Planned |
| xformers             | 36      | Planned |
| transformer_engine   | 109     | Planned |
| flashinfer           | 36      | Planned |
| flashmla             | 9       | Planned |
| deepep               | 2       | Planned |
| llamacpp / ggml      | 218     | Planned |
| sglang               | 14      | Planned |
| llmc                 | 8       | Planned |
| tinygrad             | 12      | Planned |
| nunchaku             | 37      | Planned |
| fouroversix          | 57      | Planned |
| bitsandbytes         | 2       | Planned |
| arcquant             | 24      | Planned |
| qerl                 | 6       | Planned |
| sgemm                | 60      | Planned |
| quack                | —       | Planned |

### Phase 4 — Cross-architecture

Same studies replayed on:

- [ ] SM80 (A100)
- [ ] SM86 (RTX 3090) — gpuasm example corpus
- [ ] SM89 (RTX 4090)
- [ ] SM90a (H100)
- [ ] SM100a (B200)

### Phase 5 — Reference

- [ ] Per-instruction SASS reference, one page per opcode, per architecture, with empirical latency / throughput / pipeline / dual-issue rules

## Target architectures

| Arch    | GPU              | Why                                                    |
|---------|------------------|--------------------------------------------------------|
| SM80    | A100             | Datacenter Ampere baseline                             |
| SM86    | RTX 3090         | Consumer Ampere, gpuasm example corpus                 |
| SM89    | RTX 4090         | Most common consumer inference card                    |
| SM90a   | H100             | TMA, WGMMA, warp specialization, mbarrier, clusters    |
| SM100a  | B200             | `tcgen05.mma`, TMEM                                    |
| SM120   | RTX 5070 Ti/5090 | Hybrid SM90/SM100 ISA, `mma.sync` with `mxf8f6f4`      |

Work starts on SM120 (direct hardware access). Other architectures via public dumps and contributors.

## Tools

- `cuobjdump --dump-sass` — raw disassembly
- [gpuasm.com](https://gpuasm.com/) — scoreboards, stalls, pressure, dependency arrows
- Nsight Compute (NCU) — per-kernel profiling, SASS-to-source mapping, stall attribution

## Author

Florian Mattana. [florianmattana.com](https://florianmattana.com)
