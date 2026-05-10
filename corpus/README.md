# Corpus

This directory contains the reproducible CUDA kernel studies and NVIDIA SASS dumps used by SASS King.

Each chapter is a controlled experiment: compile a small CUDA kernel, dump the generated SASS, compare variants, and record only evidence-backed claims in the chapter conclusion and global knowledge base.

## Sections

| Section | Chapters | Purpose |
|---|---:|---|
| [basics/](basics/) | 01-08 | Baseline CUDA-to-SASS behavior: loads, stores, FMA fusion, loops, shared memory, uniform flow, vectorized memory. |
| [warp_collectives/](warp_collectives/) | 09-10 | Warp-level collectives: shuffles, votes, sync behavior, and hardware reductions. |
| [math_and_spills/](math_and_spills/) | 11-12 | Slow-path math, helper calls, local memory, and register-spill signatures. |
| [tensor_cores/](tensor_cores/) | 13-25 | SM120 / SM120a tensor-core, matrix-memory, async-copy, control-flow, divergence, and production-like mini-GEMM studies. |

## Reading Model

Start from a section README, then open an individual chapter directory. The chapter-local `conclusion*.md` file explains the controlled variants and records the local narrative. Project-wide facts are promoted into [../knowledge/FINDINGS.md](../knowledge/FINDINGS.md), [../knowledge/SASS_INSTRUCTIONS_SM120.md](../knowledge/SASS_INSTRUCTIONS_SM120.md), and [../knowledge/encoding/](../knowledge/encoding/).

## Reproduction Model

The executable part of the project is the controlled CUDA kernels. A typical reproduction flow is:

```bash
cd corpus/basics/01_vector_add
nvcc -arch=sm_120 kernel1.cu -o vector_add
cuobjdump --dump-sass vector_add > sm_120.sass
```

Then compare the dump with the corresponding chapter conclusion and checked-in SASS artifacts.
