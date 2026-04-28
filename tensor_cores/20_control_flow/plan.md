# Chapter 20 - Control flow: loops, unroll, back-edges, BSSY/BSYNC

Status: **executed**. See `README.md`, `conclusion20.md`, and `FINDINGS.md`.

This plan was drafted after an initial attempt to audit a production kernel (fused FP4 attention on SM120) revealed a critical blocker: we could not confidently map C++ loops to their SASS form. The kernel contained constant-trip-count loops (N_TILES=8, K_TILES=2) but the SASS dump showed only 2 QMMAs with no back-edge BRA. This is inconsistent with both "full unroll" (expected 16 QMMAs) and "preserved loop" (expected at least one back-edge BRA).

## Goal

Decode how ptxas transforms C++ loops into SASS on SM120. Answer four concrete questions:

1. Under what conditions does ptxas fully unroll a loop vs preserve it as a SASS loop?
2. What is the SASS signature of a preserved loop on Blackwell: BRA back-edge, BSSY/BSYNC, or something else?
3. How does ptxas handle nested constant-trip-count loops (the common GEMM pattern)?
4. Can a loop exist in SASS without a BRA back-edge, and if so through what mechanism?

## Hypotheses to test

* [RES] HYP-20-1: constant trip count at compile time triggers full unroll for tested scalar and HMMA loops unless an unroll pragma restricts it.
* [RES] HYP-20-2: `#pragma unroll 1` forces a real loop with BRA back-edge for the tested N=16 loop.
* [RES] HYP-20-3: dynamic trip count produces back-edges, but often as a multi-path cascade rather than one compact loop.
* [RES] HYP-20-4: nested constant loops do not partially unroll in tested 4 x 2 and 8 x 2 scalar/HMMA cases. They fully unroll.
* [RES] HYP-20-5: BSSY/BSYNC is not used for ordinary loops in tested variants, but the tested `break` loop emits BSSY/BSYNC.
* [RES] HYP-20-6: no SASS-level loop without a BRA back-edge was observed in tested variants.

## Planned variants

Executed scope was expanded after review. The final studied set is 20a through 20v and is listed in `README.md` and `conclusion20.md`.

| Variant | Kernel | Tests |
|---|---|---|
| 20a | Simple constant loop, N=4, trivial body | Full unroll expected. Baseline. |
| 20b | Same, but N=16 | Find the unroll threshold |
| 20c | Dynamic loop, trip count in kernel arg | Real loop with BRA back-edge expected |
| 20d | Nested constant loops, N=4 outer, K=2 inner, HMMA body | Check unroll behavior. Closest to production GEMM pattern. |
| 20e | `#pragma unroll 1` on constant loop | Verify forces real loop |
| 20f | Reproduce the FP4 attention mystery: N_TILES=8, K_TILES=2, trivial MMA body | Check if 2 QMMAs pattern reproduces |

## Minimal version (4 variants)

If time-constrained, 20a + 20c + 20d + 20f cover the critical ground. 20b and 20e can be added later if 20a/c/d/f raise open questions.

## Expected outputs

* SASS signature of a "real loop" on SM120 (BRA back-edge format, counter register, ISETP predicate, control code pattern)
* Unroll threshold for constant loops (at which N does ptxas stop unrolling)
* Explanation of the FP4 attention case: either reproducing the mystery in a minimal kernel, or discovering it was a stale binary artifact
* Pattern documentation: if N_TILES × K_TILES with MMA body partially unrolls, document the rule
* If the FP4 case is reproducible in a minimal kernel, this reveals a compiler optimization that was not documented in any chapter

## Why this chapter matters

Chapters 13-18 documented kernels with trivial or absent loops (1 thread block, no iteration, direct MMA calls). This was sufficient to decode opcodes and operand encoding, but not sufficient to audit real production kernels. Production kernels always have nested loops (K tile, N tile, warp, etc.). Without this chapter, we cannot reliably audit CUTLASS, Marlin, FlashInfer, or any hand-written kernel with loops. The audit blocker found in the FP4 attention case will reoccur on every production kernel.

## Notes on ptxas behavior to verify

From the aborted FP4 attention audit:

* The binary (Apr 13 timestamp, potentially stale vs current common.h) had 2 QMMAs in a kernel that should have 16 if fully unrolled.
* Zero BRA back-edges in the kernel, which is inconsistent with preserved loops.
* The 71 BSSY/BSYNC pairs observed are presumably for divergence reconvergence, not loops. Chapter 20 confirms ordinary loops do not need BSSY/BSYNC in tested variants; Chapter 21 still needs to test real warp-divergent reconvergence.
* Possible explanations:
  * The binary was stale and contained a specialized compile with N_TILES=1 or similar
  * ptxas applied an optimization that dedupes MMAs across iterations (unlikely if each MMA has distinct operands, but A/B register re-use was observed)
  * A compiler transformation we do not recognize

This chapter should resolve which explanation is correct.

## Dependencies

None. Can be executed independently.

## Estimated effort

4 variants × 15 minutes per variant = ~1 hour of kernel writing and SASS analysis.
