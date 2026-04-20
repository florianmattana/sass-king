# 18 Pipelined MMA tile (full GEMM microkernel)

Synthesis chapter. Brings chapters 13-17 together into a complete MMA-based compute pattern: pipelined tile with ldmatrix + MMA + accumulator reuse + stmatrix. This is the SASS structure that appears in every production tensor core kernel (CUTLASS, FlashAttention, FlashInfer).

## Variants planned

| # | Description |
|---|---|
| 18a | Single MMA in a loop (establishes baseline compute ratio) |
| 18b | 2× MMA with accumulator reuse (same C/D register pair) |
| 18c | 2× MMA with ldmatrix in-flight (classic pipelining) |
| 18d | Full tile: 8× ldmatrix + 8× MMA for 128×128×K inner product |
| 18e | Same tile compiled via CUTLASS wrappers, compared to hand-inline PTX |
| 18f | Multi-warp tile (4 warps cooperating) |

## Key questions

1. Does ptxas automatically pipeline ldmatrix and MMA, or does it require explicit ordering via `asm volatile`?
2. How does accumulator reuse appear in SASS? Same register pair with scoreboard chaining?
3. What is the realistic compute ratio (MMA instructions / total instructions) in a production-quality tile?
4. Does CUTLASS-generated SASS match hand-written inline PTX byte-for-byte?
5. What is the register pressure profile across the tile? Does ptxas spill?
6. How does multi-warp coordination appear in SASS (shared memory barriers, MBAR)?

## Context from FINDINGS.md

**Cumulative from chapters 13-17**: by this point, opcodes, modifiers, fragment layouts, scoreboard rules, and ldmatrix patterns are all established. Chapter 18 is where they compose.

**From chapter 12 (register spill)**: ptxas restructures before spilling. In a dense MMA tile, restructuring may not be an option because fragments are fixed-size. Watch for unexpected spill signals (IADD R1, -<frame>; STL/LDL).

**From chapter 6 (shared memory scalar)** and **chapter 7 (UMOV 0x400)**: shared memory addressing pattern is established. MMA tiles with shared staging will use the same pattern.

**From chapter 10 (REDUX)**: warp-level patterns. Some GEMM epilogues use REDUX for reductions; not relevant here but available.

## Status

* [ ] 18a single MMA in loop
* [ ] 18b 2× MMA with accumulator reuse
* [ ] 18c 2× MMA with pipelined ldmatrix
* [ ] 18d full tile 128×128×K
* [ ] 18e CUTLASS vs hand-inline comparison
* [ ] 18f multi-warp tile
* [ ] conclusion18.md

## Dependencies

Hard prerequisite: chapters 13, 14, 16, 17 complete. Optional: 15, 19 for generality.