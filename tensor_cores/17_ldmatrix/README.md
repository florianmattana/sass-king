# 17 ldmatrix and stmatrix

`ldmatrix` and `stmatrix` move data between shared memory and register fragments formatted for MMA consumption. They are the canonical partners of every tensor core instruction. Without efficient ldmatrix, MMA cannot reach peak throughput.

## Variants planned

| # | PTX atom | Notes |
|---|---|---|
| 17a | `ldmatrix.sync.aligned.m8n8.x1.shared.b16` | Simplest: 1 matrix, 16-bit elements |
| 17b | `ldmatrix.sync.aligned.m8n8.x2.shared.b16` | 2 matrices |
| 17c | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 4 matrices (common in tiled MMA) |
| 17d | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | With transpose |
| 17e | `ldmatrix.sync.aligned.m16n16.x1.shared.b8x16` | FP8 element format |
| 17f | `ldmatrix.sync.aligned.m8n8.x4.shared.b8x8` | Scale factor loads |
| 17g | `stmatrix.sync.aligned.m8n8.x4.shared.b16` | Symmetric store |
| 17h | Pipelined `ldmatrix + MMA` | Overlap pattern with scoreboard |

## Key questions

1. What is the SASS opcode for ldmatrix? Candidate: `LDSM` (from Jia et al. Volta/Turing naming).
2. How does ldmatrix compute the per-lane shared memory address?
3. What is the latency of ldmatrix x1, x2, x4?
4. Does ldmatrix use a specific scoreboard slot? Same as LDS or dedicated?
5. What is the exact effect of `.trans`? SASS modifier, or modified address computation?
6. Can ldmatrix.x4 be dual-issued with MMA?
7. Does ldmatrix bank conflict behavior match regular LDS?

## Context from FINDINGS.md

**From chapter 6 (shared memory scalar)**: shared addressing on SM120 does not use descriptors. Direct `[R]` or `[R+UR]`. `__syncthreads()` compiles to `BAR.SYNC.DEFER_BLOCKING 0x0` on ADU pipeline.

**From chapter 8 (vectorized memory)**: LDG width on SM120 caps at 256 bits (`.ENL2.256`). `ldmatrix.x4` at 16-bit = 64 bits per matrix × 4 = 256 bits total, aligns with this cap.

**From chapters 13-16**: the MMA instructions whose operands ldmatrix prepares. Register fragment count per thread (4 for A, 2 for B, 4 for C/D) informs which ldmatrix variant (x1/x2/x4) is used for which matrix.

## Status

* [ ] 17a ldmatrix x1 b16
* [ ] 17b ldmatrix x2 b16
* [ ] 17c ldmatrix x4 b16
* [ ] 17d ldmatrix x4 trans b16
* [ ] 17e ldmatrix b8x16 for FP8
* [ ] 17f ldmatrix b8x8 for SF
* [ ] 17g stmatrix x4 b16
* [ ] 17h ldmatrix + MMA pipelining
* [ ] conclusion17.md

## Dependencies

Chapter 6 provides the LDS baseline. Chapters 13-16 provide the MMA context where ldmatrix is used. Can be studied in parallel with chapter 18 (pipelined tile) once chapter 16 is done.