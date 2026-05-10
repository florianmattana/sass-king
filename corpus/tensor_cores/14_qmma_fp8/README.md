# 14 QMMA FP8 (kind::f8f6f4, m16n8k32)

First SM120-specific MMA. FP8 inputs at shape m16n8k32 with the `kind::f8f6f4` qualifier. Distinct from SM89's FP8 MMA which does not use the `kind::` prefix.

## Variants planned

| # | A type | B type | C/D type | PTX atom |
|---|---|---|---|---|
| 14a | e4m3 | e4m3 | f32 | `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32` |
| 14b | e5m2 | e5m2 | f32 | `...kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e5m2.f32` |
| 14c | e4m3 | e5m2 | f32 | `...kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e5m2.f32` |
| 14d | e5m2 | e4m3 | f32 | `...kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e4m3.f32` |
| 14e | e4m3 | e4m3 | f16 | `...kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16` |

## Key questions

1. Does the SASS opcode change between `mma.sync` without `kind::` (chapter 13) and `mma.sync.kind::f8f6f4` (this chapter)? Candidate new opcode: `QMMA`.
2. Where does the dtype information end up at SASS level: in the opcode, in modifiers, or in operand encoding?
3. How are FP8 inputs packed into 32-bit registers? 4 e4m3 per register expected.
4. What is the cycle throughput delta between e4m3×e4m3→f32 and e4m3×e4m3→f16?
5. Does mixed FP8 (e4m3 × e5m2) differ from symmetric FP8 at SASS level?

## Context from FINDINGS.md

**From chapter 13**: HMMA SASS form established. Delta with chapter 14 isolates what the `kind::f8f6f4` qualifier changes.

**Architectural invariant** (to verify here): all MMA on SM120 is warp-level. No wgmma, no tcgen05. Chapter 14 confirms this for FP8.

**Pipelines table**: if QMMA is a new opcode, it extends the TC pipeline entry in FINDINGS.md.

## Status

* [ ] 14a e4m3 e4m3 f32 baseline
* [ ] 14b e5m2 e5m2 f32
* [ ] 14c e4m3 e5m2 f32 mixed
* [ ] 14d e5m2 e4m3 f32 mixed reverse
* [ ] 14e e4m3 e4m3 f16 for 2x throughput
* [ ] conclusion14.md

## Dependencies

Chapter 13 conclusion must be complete. The HMMA baseline is the reference against which QMMA delta is measured.