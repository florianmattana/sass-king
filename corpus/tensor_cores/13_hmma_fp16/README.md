# 13 HMMA baseline (FP16, BF16, m16n8k16)

First tensor core chapter. Establishes the SASS opcode and operand convention for the simplest MMA available on SM120: FP16 and BF16 at shape m16n8k16, inherited from SM80 (Ampere).

## Variants planned

| # | A type | B type | C/D type | PTX atom |
|---|---|---|---|---|
| 13a | f16 | f16 | f16 | `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` |
| 13b | f16 | f16 | f32 | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| 13c | bf16 | bf16 | f32 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` |
| 13d | [defined after 13a-13c observations] | — | — | Accumulator chaining (D fed back as C) |
| 13e | [defined after 13a-13c observations] | — | — | Latency microbenchmark |

## Key questions

1. What SASS opcode does ptxas emit for `mma.sync.aligned.m16n8k16.row.col.*`? Candidate: `HMMA` (Volta/Turing/Ampere heritage).
2. Is the opcode parametrized by modifiers (e.g., `HMMA.16816.F32.F16.F16`), or are the dtypes encoded in operand fields?
3. How are the A/B/C/D register fragments laid out at SASS level?
4. Does ptxas preserve the PTX-documented thread-to-element mapping, or reorder it?
5. What scoreboard does MMA use? Fixed-latency or variable-latency?
6. Is the `.reuse` flag ever set on MMA operands?
7. Does the f16 vs f32 accumulator change the opcode, or just the C/D register count?

## Context from FINDINGS.md

Before starting analysis, the following elements from the project corpus apply:

**Canonical prologue**: every SM120 kernel starts with the 8-instruction skeleton documented in FINDINGS.md. Expected to hold here too.

**Architectural invariants**: 6 scoreboards per warp, per-thread R0-R255 and uniform UR0-UR63, fixed-latency vs variable-latency distinction. TC pipeline likely uses scoreboards (variable-latency hypothesis).

**HFMA2 idiom for FP constants**: if MMA coefficients need materialization, HFMA2 should appear. If instead MOV 32-bit is used, it means FP constants for MMA operands cannot split into two FP16 halves.

**Scoreboard assignment rules**: independent MMAs should get distinct scoreboards; accumulator-chained MMAs should share.

## Status

* [ ] 13a baseline f16 f16 f16 f16
* [ ] 13b f16 f16 f32 f32
* [ ] 13c bf16 bf16 f32 f32
* [ ] 13d accumulator chaining
* [ ] 13e latency microbenchmark
* [ ] conclusion13.md

## Files

Kernels: `13a_*.cu` through `13e_*.cu`. Each is standalone with `main()`.
SASS dumps: `13a_*.sass` next to each kernel.
Conclusion: `conclusion13.md` at chapter end.