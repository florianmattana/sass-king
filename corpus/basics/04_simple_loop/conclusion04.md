# Kernel 04 — vector_loop (runtime trip count)

Minimal delta designed to force a true backward branch: a runtime-bounded `for` loop with a non-foldable body. The result was richer than expected: ptxas cascaded the loop into four distinct unrolled paths and used several non-obvious optimizations.

## Source

```cuda
__global__ void vector_loop(const float* a, float* c, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < K; k++) {
            x = x * 1.001f + 0.5f;
        }
        c[i] = x;
    }
}
```

Two protections against compiler elimination:
- **K** is a runtime argument: ptxas cannot unroll by a known constant.
- **x = x * 1.001f + 0.5f** does not simplify algebraically: ptxas cannot fold the loop into a closed-form expression.

Both protections were necessary. Either alone would have been defeated by the optimizer.

## Compile

```
nvcc -arch=sm_120 -o 04_vector_loop 04_vector_loop.cu
cuobjdump --dump-sass 04_vector_loop
```

## Surprising result: cascaded unrolling

Instead of emitting a single scalar loop with BRA backward, ptxas generated **four execution paths** based on the runtime value of K:

```
if K < 1      → skip entirely                       (.L_x_560)
if K < 4      → scalar tail loop (1 FFMA per iter)   (.L_x_500)
if 4 ≤ K < 16 → unrolled by 4 + tail                 (.L_x_450)
if K ≥ 16     → unrolled by 16 + tails               (.L_x_230)
```

This is a generalized Duff's device. The compiler generates the large-block path for the common case and smaller-block paths for the remainder, minimizing the amortized cost per iteration.

The actual backward branch is at address 0x350, jumping back to 0x230, forming a loop that executes 16 FFMA per iteration.

## Structural decomposition

The kernel splits into several identifiable regions. The prologue and pointer loads are standard. The new structure is in the loop dispatch and execution.

### Prologue and load of initial x (0x00–0xc0)

Identical in structure to kernels 01–03. Computes i, does the bounds check, loads the pointers, and issues `LDG` to fetch the initial `a[i]` into R5 (which will be reused as `x` throughout the loop).

### Loop dispatch (0xd0–0x200)

Cascade of predicate tests and branches that select which unrolled path to execute:

```
UISETP.GE.AND      UP0, UPT, UR5, 0x1, UPT    // K >= 1?
BRA.U -UP0, .L_x_560                           // skip if no iterations

UISETP.GE.U32.AND  UP0, UPT, UR5, 0x4, UPT    // K >= 4?
ULOP3.LUT          UR4, UR5, 0x3, URZ, 0xc0   // UR4 = K & 3 = K % 4
BRA.U -UP0, .L_x_500                           // scalar tail if K < 4

// K >= 4 path continues
ULOP3.LUT          UR5, UR5, 0x7ffffffc, ...  // UR5 = K & ~3 = floor(K/4)*4
UIADD3             UR5, ..., -UR5, ...         // UR5 = -UR5 (negate for countdown)
UISETP.GE.U32.AND  UP1, UPT, UR5, URZ, UPT    // one more test
BRA.U -UP1, .L_x_1e0
```

Three observations in this dispatch block:

**K % 4 precomputed.** `UR4 = K AND 3` extracts the low 2 bits, which for a positive integer equals K mod 4. This value will later gate the scalar tail loop that cleans up the remainder.

**Signed versus unsigned comparison.** The first test against 1 is signed (no `.U32`) to correctly handle K ≤ 0 (negative integers return false, skipping the loop). The subsequent tests use `.U32` because once K ≥ 1 is established, unsigned comparison is semantically equivalent and potentially faster (hypothesis to validate).

**Loop counter inverted.** ptxas negates UR5 to count *down* from -K toward 0 instead of counting up from 0 to K. This saves a register (no need to keep K) and allows comparison against a small immediate. The termination condition becomes `UR5 >= -12` (continue while at least 16 iterations remain).

### First 4-unrolled block (0x170–0x1d0)

The first partial block of 4 FFMA, preceded by a constant load:

```
HFMA2 R2, -RZ, RZ, 1.875, 0.00931549072265625
UIADD3 UR5, ..., UR5, 0x4, URZ                 // UR5 += 4 (one unroll)
FFMA R5, R5, R2, 0.5                           // 4 FFMA on the chain
FFMA R5, R5, R2, 0.5
FFMA R5, R5, R2, 0.5
FFMA R5, R5, R2, 0.5
```

The `HFMA2` line is a constant load trick. See detailed explanation below.

### Main loop (0x210–0x350) — 16 FFMA per iteration

```
0x210  HFMA2 R2, -RZ, RZ, 1.875, 0.00931549072265625   // preload 1.001f
0x220  UPLOP3.LUT UP0, ...                              // predicate init
0x230  FFMA R5, R5, R2, 0.5                             ← loop target
0x240  UIADD3 UR5, ..., UR5, 0x10, URZ                  // UR5 += 16
0x250  FFMA R5, R5, R2, 0.5
0x260  UISETP.GE.AND UP1, ..., UR5, -0xc, UPT           // test UR5 >= -12
0x270–0x340  FFMA × 14 (the remaining 14 of 16 unrolled FFMA)
0x350  BRA.U -UP1, .L_x_230                             // backward jump
```

This is the real loop body. Loop structure:
- 16 FFMA per iteration, all on the chain `R5 = R5 * R2 + 0.5`.
- Counter update (`UR5 += 16`) inserted between FFMAs to overlap with compute.
- Termination test (`UR5 >= -12`) also inserted in the body, not at the tail. The branch at 0x350 consumes the predicate computed at 0x260.
- Backward branch: `BRA.U -UP1, .L_x_230` jumps back to the first FFMA.

### Cleanup blocks (0x360 onward)

After the 16-unrolled loop exits, cleanup proceeds:
- 8-FFMA block if 8 more iterations remain
- 4-FFMA block if 4 more iterations remain
- Scalar tail loop for the last K % 4 iterations

Each block has its own constant setup and branch guard. The full decision tree is resolved at runtime without any divergence within a warp, because all predicates are uniform (same K for every thread of the warp).

### Store and exit

```
LDC.64     R2, c[0x0][0x388]    // pointer c, loaded late
IMAD.WIDE  R2, R0, 0x4, R2
STG.E      [R2.64], R5
EXIT
```

Note the late loading of the `c` pointer. ptxas did not load it in the prologue alongside `a`. Hypothesis: since R5 would not be ready until after the loop (a long wait), there is no benefit to computing the store address early, so ptxas defers it to minimize register live ranges.

## Deep dive: the HFMA2 constant-loading trick

The line `HFMA2 R2, -RZ, RZ, 1.875, 0.00931549072265625` does not look like a constant load, but it is. It loads the FP32 value `1.001f` into R2 using the half-precision FMA pipeline.

### Why it works

HFMA2 is the half-precision (FP16) packed fused multiply-add. It operates on pairs of FP16 values packed into a 32-bit register.

The bit pattern of `1.001f` as a 32-bit float is `0x3F8020C5`. Broken into two 16-bit halves:
- High 16 bits: `0x3F80` → interpreted as FP16, this is `1.875`
- Low 16 bits: `0x20C5` → interpreted as FP16, this is `0.00931549072265625`

The HFMA2 instruction does `packed(-0 * 0 + 1.875, -0 * 0 + 0.00931549)`. The multiply terms are zero (both operands are `RZ` which is hard-wired to zero). The result is just the packed pair `{1.875, 0.00931549}` written into R2.

The bit pattern of this packed pair equals `0x3F8020C5`, which when interpreted as FP32 is exactly `1.001f`. The downstream FFMA reads R2 as FP32 and gets the correct constant.

### Why ptxas does this

The instruction scheduling is the reason. Loading `1.001f` could also be done via `MOV R2, 0x3F8020C5`. Both produce the same bit pattern in R2. But:
- MOV runs on the ALU/ADU pipeline.
- HFMA2 runs on the FMA pipeline.

In a compute-heavy block with many FFMA (16 in the main loop), the FMA pipeline is the bottleneck. Adding one more MOV on the already-loaded ALU would waste a cycle, while HFMA2 can fill otherwise-idle FMA slot on the warp scheduler, or at least balance the load differently.

Consistent with this theory, the SASS shows that HFMA2 is used only in the 16-unrolled and 4-unrolled blocks, while `MOV R3, 0x3F8020C5` is used in the 8-unrolled block. The compiler is making pipeline balance decisions that depend on the block size.

### Source confirmation

The trick is documented in reverse-engineered SASS references: it is used to encode literal float zero using 8 bits instead of 32 (RZ is 8 bits), and to move the constant through an underutilized pipeline to improve throughput.

Recent community observations (Henry Zhu, December 2025) report that ptxas's decision to use HFMA2 versus IMAD versus MOV for constant loading can be affected by surprising inputs — including the **name** of the CUDA function — with performance impacts of ±20% in real kernels. This is a clear signal that ptxas's scheduling heuristics for constant loading are not fully stable across compiler versions.

## Deep dive: UPLOP3.LUT on predicates

Several instructions like `UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x80, 0x8` appear in the dispatch and cleanup sections. This is LOP3's predicate-logic cousin: a 3-input lookup table applied to uniform predicates.

With all four predicate sources being UPT (always true), the output is determined entirely by the LUT. Hypothesis: these are idiomatic "initialize UP0 to a constant value" instructions, equivalent to `UP0 = true` or `UP0 = false` depending on the LUT. The 0x8 trailing argument may be a mask over the output predicates.

Not fully reverse-engineered. Worth a microbenchmark pass to confirm semantics.

## Patterns identified (new)

1. **Cascaded unrolling for runtime-bounded loops.** ptxas generates multiple unrolled paths (by 16, 8, 4, 1) and selects one at runtime based on the trip count. This is the default for simple dependent-chain loops on scalar arithmetic.

2. **HFMA2 as a constant loader.** Packed FP16 HFMA2 can load an arbitrary FP32 bit pattern by exploiting the concatenation of the two half-precision results. Used when the compiler wants to route constant loading through the FMA pipeline instead of MOV.

3. **Inverted loop counter.** ptxas negates the counter and tests against a small immediate instead of tracking the positive bound. Saves a register and compacts the termination condition.

4. **Signed test for initial bound, unsigned for subsequent bounds.** The first guard (`K >= 1`) is signed to reject negative K; later guards use unsigned comparison once the sign is known.

5. **Precomputed remainder for cleanup tails.** `UR4 = K & 3` (= K % 4) is computed during the dispatch block and later consumed by the scalar tail loop to determine how many iterations remain.

6. **Pipeline balancing via MOV/HFMA2 choice.** The compiler selects between MOV and HFMA2 for constant loading based on the expected pipeline load in the surrounding block. This is a non-obvious heuristic with measurable performance impact.

7. **Counter update interleaved with compute.** In the 16-unrolled body, `UIADD3` (counter increment) and `UISETP` (termination test) are placed between FFMAs, not at the loop tail, so that arithmetic pipelines are saturated throughout the iteration.

8. **Late pointer load for store destination.** When the store value has a long dependency chain, ptxas delays loading the store pointer to minimize the live range of the pointer register.

## Hypotheses to validate by microbenchmark

- Relative throughput of HFMA2 magic versus MOV versus IMAD for constant loading in compute-heavy blocks.
- Impact of block size on the compiler's choice of constant-loading instruction.
- Semantics of UPLOP3.LUT with all-UPT inputs and specific LUT values.
- Stall anomaly: UIADD3 at 0x360 has stall=12, larger than expected for a uniform-datapath arithmetic instruction.

## Instruction count breakdown (approximate)

| Role | Count |
|---|---|
| Prologue + bounds | 8 |
| Pointer loads | 3 |
| Initial load (a[i]) | 2 |
| Loop dispatch (predicate cascade) | ~15 |
| 16-unrolled loop body | ~20 |
| 8-unrolled cleanup | ~12 |
| 4-unrolled cleanup | ~7 |
| Scalar tail loop | ~6 |
| Store + exit | 4 |

Approximately 77 instructions for what the source expresses as a 5-line loop. The expansion ratio here is ~15x source-to-SASS, driven by the unrolling cascade.

## Takeaway

Kernel 04 demonstrates that ptxas does far more than straightforward code generation for even simple source-level patterns. A "smallest possible runtime loop" produced a multi-path code expansion, an undocumented constant-loading trick, and several optimization decisions that depend on kernel-wide heuristics not visible at the source level.

This reinforces the methodology: controlled variation is the only reliable way to map source patterns to SASS patterns, because the compiler's transformations are sufficiently aggressive that any a priori model of "what the SASS should look like" will be wrong in at least one significant way.