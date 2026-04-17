# SASS King

Systematic reverse engineering of NVIDIA SASS across architectures.

Full context and methodology: [Part 1 — Reading NVIDIA SASS from First Principles](https://florianmattana.com/posts/sass_king/)

## Roadmap

### Phase 1 — Vocabulary and reading method (SM120)

- [x] **Part 1** — Four minimal kernels, the six-section skeleton, scoreboards in practice. [Read](https://florianmattana.com/posts/sass_king/)
- [ ] Kernel 05 — Loop with small fixed trip count (full unroll vs cascade)
- [ ] Kernel 06 — Shared memory scalar (`LDS`, `STS`, `BAR.SYNC`)
- [ ] Kernel 07 — Vectorized load (`LDG.E.128`)
- [ ] Kernel 08 — Warp-level reduction (`SHFL.BFLY`)
- [ ] Kernel 09 — Division slowpath (`CALL.REL.NOINC`)
- [ ] Kernel 10 — Forced register spill (`STL`, `LDL`)
- [ ] Kernel 11 — First tensor core (`QMMA` on SM120)

### Phase 2 — Cross-architecture comparison

Same controlled variations, reproduced on:

- [ ] SM80 (A100)
- [ ] SM89 (RTX 4090)
- [ ] SM90a (H100)
- [ ] SM100a (B200)

### Phase 3 — Reference and real-kernel audits

- [ ] Instruction-level SASS reference, one page per instruction, per architecture
- [ ] Audit: FlashAttention decode
- [ ] Audit: Marlin W4A16
- [ ] Audit: FlashInfer GEMM FP8
- [ ] Audit: CUTLASS GEMM mainloop
- [ ] Audit: FP4 fused attention

## Target architectures

| Arch    | GPU              | Why                                                    |
|---------|------------------|--------------------------------------------------------|
| SM80    | A100             | Datacenter Ampere baseline                             |
| SM89    | RTX 4090         | Most common consumer inference card                    |
| SM90a   | H100             | TMA, WGMMA, warp specialization, mbarrier, clusters    |
| SM100a  | B200             | `tcgen05.mma`, TMEM                                    |
| SM120   | RTX 5070 Ti/5090 | Hybrid SM90/SM100 ISA, `mma.sync` with `mxf8f6f4`      |

Work starts on SM120 (direct hardware access). Other architectures via public dumps and contributors.

## Tools

- `cuobjdump --dump-sass` — raw disassembly
- [gpuasm.com](https://gpuasm.com/) — scoreboards, stalls, pressure, dependency arrows

## Author

Florian Mattana. [florianmattana.com](https://florianmattana.com)
