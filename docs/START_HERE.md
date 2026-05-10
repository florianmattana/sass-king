# Start Here

This guide is the shortest path through SASS King v0.1.

SASS King is a research corpus and knowledge base for NVIDIA SASS. It is not yet a standalone disassembler, assembler, Ghidra plugin, or audit CLI. The executable part today is the controlled CUDA kernels in `corpus/`.

## Recommended reading path

| Goal | Path |
|---|---|
| Learn the basic reading vocabulary | `corpus/basics/01_vector_add/` -> `corpus/basics/03_vector_fma/` -> `corpus/basics/06_shared_memory_scalar/` -> `corpus/basics/08_vectorized_load/` |
| Understand warp-level primitives | `corpus/warp_collectives/09_warp_shuffling/` -> `corpus/warp_collectives/10_reduce/` |
| Understand tensor-core evidence | `corpus/tensor_cores/13_hmma_fp16/` -> `corpus/tensor_cores/14_qmma_fp8/` -> `corpus/tensor_cores/17_ldmatrix/` -> `corpus/tensor_cores/18_pipelined_tile/` |
| Understand the knowledge base | `knowledge/FINDINGS.md` -> `knowledge/SASS_INSTRUCTIONS_SM120.md` -> `knowledge/encoding/` |

## Read one chapter

Each chapter follows the same model:

1. Open the chapter directory.
2. Read the `conclusion*.md` file first.
3. Inspect the CUDA source that produced the dump.
4. Inspect the checked-in `.sass` dump.
5. Check whether project-wide facts were promoted into `knowledge/FINDINGS.md`.

Example:

```bash
cd corpus/basics/01_vector_add
ls
```

Then read `conclusion01.md` before opening the SASS dump.

## Reproduce one result

The simplest reproducible path is to compile one controlled kernel and dump its NVIDIA SASS:

```bash
cd corpus/basics/01_vector_add
nvcc -arch=sm_120 kernel1.cu -o vector_add
cuobjdump --dump-sass vector_add > sm_120.sass
```

Compare the generated `sm_120.sass` with the checked-in chapter evidence and `conclusion01.md`.

Toolchain and driver versions can change output. If your dump differs, treat the difference as evidence to document, not as an error to hide.

## Understand claim tags

| Tag | Meaning |
|---|---|
| `[OBS]` | Direct observation from a local SASS dump, log, runtime output, or profile. |
| `[INF]` | Inference from observations; the evidence chain should be stated. |
| `[HYP]` | Plausible hypothesis that still needs a controlled test. |
| `[RES]` | Former hypothesis or gap resolved by later evidence. |
| `[GAP]` | Open question not answered by the current evidence. |

Do not upgrade a claim to `[OBS]` because an external source says it. External sources can support an interpretation, but local dumps remain the primary evidence.

## Add evidence from another architecture

For a useful dump contribution:

1. Record GPU model, compute capability, CUDA toolkit version, driver version, OS, and exact compile command.
2. Keep the source kernel minimal and controlled.
3. Include `cuobjdump --dump-sass` output.
4. State what changed relative to the closest existing chapter.
5. Use `[OBS]`, `[INF]`, `[HYP]`, `[RES]`, and `[GAP]` tags in any new claim.

Start from `CONTRIBUTING.md` before opening a correction or dump contribution.
