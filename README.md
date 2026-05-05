<p align="center">
  <img src="assets/sass-king-logo.svg" alt="SASS King logo" width="220">
</p>

<h1 align="center">SASS King</h1>

<p align="center">
  <strong>Reverse engineering NVIDIA SASS from controlled kernels to production audits.</strong>
</p>

<p align="center">
  <a href="knowledge/README.md">Knowledge base</a> ·
  <a href="knowledge/SASS_INSTRUCTIONS_SM120.md">SM120 instruction glossary</a> ·
  <a href="knowledge/encoding/">Encoding notes</a> ·
  <a href="tensor_cores/README.md">Tensor-core chapters</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="Architecture" src="https://img.shields.io/badge/architecture-SM120%20%2F%20SM120a-0b6d55">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20knowledge%20base-f4c95d">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

SASS King is a systematic reverse-engineering project for NVIDIA SASS, the native GPU instruction set emitted inside compiled CUDA binaries. The project starts with SM120 / SM120a consumer Blackwell hardware and expands toward a full cross-architecture ISA and pattern library over time.

The goal is practical: help a kernel engineer open a SASS dump, recognize compiler patterns, identify performance-relevant structures, and connect the binary back to source-level optimization decisions.

## Why It Exists

The last broad public SASS reverse-engineering work comparable in spirit was Jia et al. on Volta and Turing in 2018. Ampere, Hopper, and Blackwell have changed the instruction mix substantially: async copy paths, tensor-core families, matrix load/store instructions, sparse and scaled MMA forms, and new uniform-register flows.

SASS King fills that gap by combining controlled micro-kernels, raw SASS reading, runtime probes, and production-kernel audits.

## Current State

| Area | Status | Where |
|---|---|---|
| SM120 teaching kernels | Complete through kernels 01-12 | `01_vector_add/` to `12_register_spill/` |
| Tensor-core studies | Complete through Kernel 25 | `tensor_cores/` |
| Global findings | Active source of truth | `knowledge/FINDINGS.md` |
| SM120 instruction glossary | Active, evidence-backed | `knowledge/SASS_INSTRUCTIONS_SM120.md` |
| Encoding pilots | Started with `LDSM`, `STSM`, `QMMA` | `knowledge/encoding/` |
| Pattern library | Next phase | `patterns/` |
| Production audits | Planned | `production/` |

## Start Here

- New to the project: read the [knowledge base index](knowledge/README.md).
- Want the current instruction map: read [SASS instructions on SM120 / SM120a](knowledge/SASS_INSTRUCTIONS_SM120.md).
- Want the raw source of truth: read [findings](knowledge/FINDINGS.md).
- Want tensor-core evidence: start with [tensor-core chapters](tensor_cores/README.md).
- Want to contribute dumps or corrections: read [contributing](CONTRIBUTING.md).

Full context for the first public writeup: [Part 1 - Reading NVIDIA SASS from First Principles](https://florianmattana.com/p/reading-nvidia-sass-from-first-principles).

## Methodology

**Controlled variation.** Two kernels differ by exactly one variable: dtype, operand order, unroll factor, memory layout, or compilation target. The SASS diff isolates the compiler decision.

**Strict claim tags.** Every technical claim uses a tag:

| Tag | Meaning |
|---|---|
| `[OBS]` | Directly observed in a dump, log, runtime output, or profile. |
| `[INF]` | Inferred from observed evidence. |
| `[HYP]` | Plausible but not confirmed. |
| `[RES]` | A prior hypothesis resolved by later evidence. |
| `[GAP]` | Open question documented explicitly. |

**Top-down and bottom-up together.** Micro-kernels isolate individual instructions and compiler decisions. Production-like kernels show which patterns matter in real code.

## What Is Covered

The first pass focuses on the SM120 tensor-core and memory pipeline:

- `HMMA`, `QMMA`, `OMMA`
- `LDSM`, `STSM`
- `LDGSTS`, `LDGDEPBAR`, `DEPBAR`
- `LDG`, `STG`, `LDS`, `STS`, `REDG`
- `BRA`, `EXIT`, `BSSY`, `BSYNC`, `WARPSYNC`
- `SHFL`, `VOTE`, `REDUX`
- uniform-register flow: `S2UR`, `R2UR`, `UMOV`, `ULEA`, `LDCU`

The project does not pretend the ISA is complete yet. The public glossary tracks what is observed and explained; deeper pages under `knowledge/encoding/` track families with enough evidence for matcher-style documentation.

## Roadmap

### Phase 1 - Teaching Kernels

Kernels 01-12 establish baseline SASS concepts: FMA fusion, scoreboard behavior, loop lowering, shared memory, global memory, warp primitives, slow-path math, and local-memory spills.

### Phase 2 - Tensor-Core And SM120 Coverage

Kernels 13-25 cover the current SM120 tensor-core path:

| Kernel | Topic |
|---|---|
| 13 | HMMA baseline, register allocation, accumulator chaining |
| 14 | QMMA FP8 / FP6 / FP4 baseline |
| 15 | Narrow MMA variants |
| 16 | FP4 peak and block-scaled OMMA/QMMA |
| 17 | LDSM and matrix-load behavior |
| 18 | Pipelined MMA tile and async copy staging |
| 19 | Sparse MMA metadata |
| 20 | Control flow and back-edge detection |
| 21 | Divergence and reconvergence |
| 22 | STSM matrix-store behavior |
| 23 | FP4 / FP6 fragment layout probes |
| 24 | Production mini-GEMM audit |
| 25 | STSM epilogue layout and storeback semantics |

### Phase 3 - Pattern Library

Formalize recurring structures into reusable signatures:

- `LDGSTS -> DEPBAR -> LDSM -> MMA`
- chained `HMMA` / `QMMA` / `OMMA`
- `STSM -> BAR -> LDS -> STG`
- warp reductions and cross-lane collectives
- register-spill signatures
- scalar and uniform control-flow patterns

### Phase 4 - Production Audits

Apply the pattern library to real kernels from libraries such as FlashAttention, CUTLASS, xFormers, Transformer Engine, FlashInfer, llama.cpp / ggml, tinygrad, and related projects. The goal is representative coverage by algorithmic pattern, not one markdown file per kernel.

### Phase 5 - Audit Tool

Build a pipeline that takes a cubin, detects known patterns, and emits an optimization-oriented report.

### Phase 6 - Cross-Architecture

Replay the methodology on additional targets:

| Arch | Representative GPU | Why |
|---|---|---|
| SM80 | A100 | Datacenter Ampere baseline |
| SM86 | RTX 3090 | Consumer Ampere corpus |
| SM89 | RTX 4090 | Common consumer inference card |
| SM90a | H100 | TMA, WGMMA, warp specialization, clusters |
| SM100a | B200 | tcgen05.mma, TMEM |
| SM120 | RTX 5070 Ti / 5090 | Consumer Blackwell starting point |

## Repository Map

```text
.
├── 01_vector_add/ ... 12_register_spill/   # Phase 1 teaching kernels
├── tensor_cores/                           # Phase 2 tensor-core studies
├── knowledge/                              # Findings, glossary, encoding notes
│   ├── FINDINGS.md
│   ├── SASS_INSTRUCTIONS_SM120.md
│   └── encoding/
├── patterns/                               # Coming: formal pattern library
├── production/                             # Coming: production-kernel audits
└── guide/                                  # SASS reading guide material
```

Each chapter folder contains source kernels, compiled artifacts when relevant, SASS dumps when they are part of the validated evidence set, and a `conclusion<N>.md` writeup.

## Tooling

- `cuobjdump --dump-sass` for raw disassembly.
- `gpuasm.com` for scoreboards, stalls, pressure, and dependency arrows.
- Nsight Compute for profiling and stall attribution.
- `%clock` microbenchmarks for instruction latency probes.
- `nvcc -Xptxas -v` for register and spill metadata.

## Related Work

- [redplait/denvdis](https://github.com/redplait/denvdis) for opcode tables, latency extraction, scheduling analysis, and cubin patching.
- [kuterdinel.com/nv_isa](https://kuterdinel.com/nv_isa/) for a fuzzed ISA specification.
- Jia et al. 2018, "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking."
- NVIDIA `cuda-binary-utilities` documentation.

SASS King operates at the algorithmic pattern layer: recognizing how compiled kernels are structured and connecting those structures to source-level optimization decisions.

## Contributing

Contributions are welcome, especially:

- raw SASS dumps from hardware not directly available here;
- controlled kernel studies that isolate one compiler decision;
- corrections to existing observations;
- new production-kernel pattern proposals;
- cross-architecture comparisons.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the expected metadata and writing standard.

## Author

Florian Mattana. [florianmattana.com](https://florianmattana.com)
