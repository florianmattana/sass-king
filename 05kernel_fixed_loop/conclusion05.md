# Kernel 05 — vector_loop_fixed (compile time loop count)

Minimal delta from kernel 04: replace runtime `K` with a compile time constant. The intent is to confirm that the cascade of unrolled paths observed in kernel 04 was driven entirely by the compiler not knowing the trip count. The test also became an opportunity to deeply investigate the HFMA2 constant loading idiom.

## Source

```cuda
__global__ void vector_loop_fixed(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < 8; k++) {       // K = 8, known at compile time
            x = x * 1.001f + 0.5f;
        }
        c[i] = x;
    }
}
```

Two changes from kernel 04: `K` is gone from the signature (only 3 args now), and the loop bound is the literal `8`.

## Compile

```
nvcc -arch=sm_120 -o 05_vector_loop_fixed 05_vector_loop_fixed.cu
cuobjdump --dump-sass 05_vector_loop_fixed
```

## SASS dump

```
0x0000  LDC R1, c[0x0][0x37c]
0x0010  S2R R0, SR_TID.X                          SBS=0
0x0020  S2UR UR4, SR_CTAID.X                      SBS=0
0x0030  LDCU UR5, c[0x0][0x390]                   SBS=1
0x0040  LDC R7, c[0x0][0x360]
0x0050  IMAD R7, R7, UR4, R0           wait={SB0} stall=5 yield
0x0060  ISETP.GE.AND P0, PT, R7, UR5, PT  wait={SB1} stall=13 yield
0x0070  @P0 EXIT
0x0080  LDC.64 R2, c[0x0][0x380]                  SBS=0
0x0090  LDCU.64 UR4, c[0x0][0x358]                SBS=1
0x00a0  HFMA2 R9, -RZ, RZ, 1.875, 0.00931549      stall=12
0x00b0  LDC.64 R4, c[0x0][0x388]                  SBS=2
0x00c0  IMAD.WIDE R2, R7, 0x4, R2      wait={SB0} stall=4
0x00d0  LDG.E R2, desc[UR4][R2.64]                SBS=4
0x00e0  IMAD.WIDE R4, R7, 0x4, R4      stall=4
0x00f0  FFMA R0, R2, R9, 0.5           wait={SB4} stall=8
0x0100  FFMA R0, R0, R9, 0.5
0x0110  FFMA R0, R0, R9, 0.5
0x0120  FFMA R0, R0, R9, 0.5
0x0130  FFMA R0, R0, R9, 0.5
0x0140  FFMA R0, R0, R9, 0.5
0x0150  FFMA R0, R0, R9, 0.5
0x0160  FFMA R3, R0, R9, 0.5
0x0170  STG.E desc[UR4][R4.64], R3
0x0180  EXIT
```

17 useful instructions, no backward branch, no loop control machinery.

## Diff vs kernel 04

| | Kernel 04 (K runtime) | Kernel 05 (K=8 const) |
|---|---|---|
| Total instructions | ~80 | 17 |
| Execution paths | 4 (cascade) | 1 (straight line) |
| Backward branches | 1 | 0 |
| ISETP for loop | yes | none |
| Counter register | yes | none |
| Cleanup blocks | yes (8, 4, scalar) | none |
| FFMA in body | 16 (then more) | 8 (exact) |

Removing `K` from the signature eliminated the entire dispatch cascade. The compiler knew exactly how many iterations to emit and produced exactly that many FFMA in a straight line.

## Confirmed hypothesis

The cascade observed in kernel 04 was caused entirely by `K` being a runtime variable. With the count known at compile time, ptxas does the simplest possible thing: emit N copies of the body. The cost of the cascade (about 60 extra instructions) was the price of handling unknown trip counts. When that uncertainty is removed, the cost vanishes.

This is a useful actionable insight for kernel developers. If you have a loop whose trip count is bounded by a value that could be made compile time (template parameter, `constexpr`, `#define`), refactoring to make it compile time can collapse the SASS size significantly.

## Detailed line analysis

### Prologue (0x00 to 0x70)

Identical structure to all previous kernels. Six section skeleton: stack pointer, thread and block ID, index computation, bounds check.

Note that ptxas chose to compute the index in R7 instead of R0 this time (kernel 04 used R0). This is a register allocation decision that varies based on the global liveness analysis of the kernel. With no loop counter to track, R7 was a convenient choice. The choice has no observable performance impact, it just changes which register holds which value.

The bounds check still uses ISETP with stall=13, consistent with the cross pipeline predicate transfer hypothesis from kernel 03.

### Constant materialization (0x0a0)

```
HFMA2 R9, -RZ, RZ, 1.875, 0.00931549072265625
```

This single line is one of the most surprising patterns in SM120 SASS. It loads the FP32 constant `1.001f` into R9 by exploiting the bit pattern equivalence between two packed FP16 values and one FP32.

The instruction does two FP16 fused multiply adds in parallel:
- High half: `-0 * 0 + 1.875 = 1.875` in FP16 (bit pattern 0x3F80)
- Low half: `-0 * 0 + 0.00931549 = 0.00931549` in FP16 (bit pattern 0x20C5)

Concatenated, the 32 bits of R9 are 0x3F8020C5, which when read as FP32 equals exactly 1.001f.

Why this idiom is used instead of a simpler `MOV R9, 0x3F8020C5` is investigated below.

### Pointer loads and address computation (0x80 to 0xe0)

All pointers are loaded early in the prologue via LDC.64 and LDCU.64. Each pointer gets its own scoreboard so dependent address computations can proceed independently. The IMAD.WIDE for the source array (0x0c0) is dispatched as soon as its pointer is ready, then the LDG fires.

Note that the destination pointer is loaded earlier than in kernel 04. Kernel 04 deferred the destination pointer load because the long FFMA chain would have stalled on the source data anyway. Here the chain is shorter (8 FFMA instead of 16), so the deferral is less beneficial and ptxas loads everything upfront.

### The FFMA chain (0x0f0 to 0x160)

Eight FFMA instructions in straight line, all writing to R0 except the last which writes to R3. Each FFMA does `R0 = previous_R0 * R9 + 0.5`, implementing one iteration of the source loop.

The chain is strictly sequential. Each FFMA depends on the result of the previous one. ptxas could not parallelize this chain because doing so would change the order of floating point rounding operations and produce a different result. IEEE arithmetic is not associative, and ptxas respects the source order strictly.

The last FFMA writes to R3 instead of R0. This choice avoids a potential dependency between the FFMA result and the STG that follows. R3 was free at this point (it was the high half of the source pointer R2:R3, no longer needed after the LDG), so ptxas reused it.

### Store and exit (0x170 to 0x180)

Standard store of R3 to the destination address, followed by EXIT.

## Deep dive: the HFMA2 constant loading idiom

Kernel 05 was used to investigate why ptxas systematically chose HFMA2 over alternative encodings to load FP32 constants. Five variants were tested, all using the same kernel structure but with different constants.

| Variant | Constant | Bit pattern | High FP16 | Low FP16 | Loop iterations | Encoding chosen |
|---|---|---|---|---|---|---|
| 05 | 1.001f | 0x3F8020C5 | 1.875 | 0.0093 | 8 | HFMA2 |
| 05b | 3.14159f | 0x40490FD0 | 2.142 | 0.000476 | 8 | HFMA2 |
| 05c | 2.0f | 0x40000000 | 2.0 | 0.0 | 8 | HFMA2 |
| 05d | 1e20f | 0x60AD78EC | 598.5 | 40320 | 8 | HFMA2 |
| 05e | 1.001f | 0x3F8020C5 | 1.875 | 0.0093 | 1 (no loop) | HFMA2 |
| 05f | 0.5f (mul and add) | 0x3F000000 | 1.75 | 0 | 1 (no loop) | HFMA2 |

The result is unambiguous. ptxas matterializes every FP32 constant used as the multiplier of an FFMA via HFMA2 in a register, regardless of:

- The numerical value of the constant (small, large, irrational, exact power of 2)
- The number of times the constant is used (one or many)
- Whether the kernel has a loop or is straight line code
- Whether the constant could be encoded as a simple immediate

### The actual rule

Inspection of every FFMA in every variant reveals the underlying constraint. FFMA on SM120 has the form:

```
FFMA dst, src1, src2, src3
```

where `dst` is the destination register, `src1` and `src2` are the multiply operands, and `src3` is the addend. Across all observed dumps, only `src3` ever appears as an immediate. `src1` and `src2` are always registers.

The constant `0.5` in every FFMA is the addend, not the multiplier. It is encoded as an immediate in `src3`. The multiplier `1.001f` (or `3.14159f`, etc.) cannot be encoded as an immediate in `src2` because the FFMA opcode reserves only one immediate slot, and that slot is for the addend.

This forces any non trivial FP32 constant used as a multiplier to be loaded into a register first. HFMA2 is the encoding ptxas chose for this load.

### Why HFMA2 specifically

This part is hypothesis. Three plausible reasons:

1. HFMA2 routes the constant load through the FMA pipeline, which is the same pipeline used by the consuming FFMAs. Keeping the constant load on the same pipeline avoids cross pipeline transfer overhead.

2. HFMA2 with `-RZ * RZ` is a compact encoding. The instruction needs to fit in 128 bits including control codes. Using two 16 bit immediates instead of one 32 bit immediate may free up bits in the instruction word for other purposes.

3. Heuristic decision baked into ptxas, possibly suboptimal in some contexts. Henry Zhu (December 2025) has documented cases where this choice has measurable performance impact and where the choice depends on surprising factors like the function name.

Without microbenchmarking the alternatives directly (HFMA2 vs MOV vs IMAD via PTX inline assembly) it is not possible to know which of these is the dominant reason. This is left for a future microbenchmark pass.

### Practical consequence for kernel developers

Every FP32 constant used as a multiplier costs one HFMA2 instruction of materialization plus a register slot for its lifetime. This is not free.

For kernels with many distinct constants (FIR filters, polynomial evaluation, RoPE, normalization layers), the materialization cost can be a non trivial fraction of the kernel. Three optimization strategies are visible from this analysis:

1. **Reuse constants.** Restructure expressions so that the same constant is used multiple times. ptxas materializes each unique constant only once, so reusing means amortizing the cost.

2. **Use constants as addends when possible.** Constants in the addend position (`src3`) are encoded as immediates with no materialization cost. If your formula can be rewritten so that constants are added rather than multiplied, you save instructions.

3. **Pass constants as kernel arguments.** Constants in kernel arguments live in constant memory and are loaded via LDC, which uses a different pipeline. This can be advantageous if the FMA pipeline is the bottleneck and the LDC can be hidden behind other work in the prologue.

## Instruction count breakdown

| Section | Count | % |
|---|---|---|
| Prologue + bounds | 8 | 47% |
| Pointer loads + address arithmetic | 4 | 24% |
| LDG | 1 | 6% |
| Constant materialization (HFMA2) | 1 | 6% |
| Compute (FFMA) | 8 | 47% |
| Store + EXIT | 2 | 12% |

For the first time in this kernel series, the compute section reaches a high fraction of the total. With 8 FFMAs in 17 instructions, useful compute is 47% of the kernel, compared to 5% in kernel 01 and 4% in kernel 03. This is exactly the effect of raising arithmetic intensity that was discussed in the Part 1 guide.

## Patterns identified (new)

1. **Compile time loop bounds eliminate the unrolling cascade.** When ptxas knows the trip count, it emits exactly that many copies of the body in straight line code, with no dispatch and no backward branch. Refactoring runtime trip counts to compile time when possible can drastically reduce SASS size.

2. **FFMA reserves only one immediate slot, in the addend position.** The multiplier sources of FFMA must be registers. This forces all non trivial FP32 constants used as multipliers to be materialized in a register beforehand.

3. **HFMA2 is the systematic materialization idiom.** ptxas uses HFMA2 with `-RZ, RZ, FP16_high, FP16_low` to load any FP32 constant into a register, exploiting the bit pattern equivalence between two packed FP16 and one FP32. This happens regardless of constant value, usage count, or kernel structure.

4. **Register allocation is context dependent.** The same value can land in different registers in different kernels (R0 in kernel 04, R7 in kernel 05) based on global liveness analysis. This has no functional impact but explains why diffing two related kernels may show register renames that are not strictly necessary at the source level.

5. **Last operation in a chain may target a different register to break dependencies.** The last FFMA in the chain writes to R3 instead of continuing the chain in R0, freeing R0 for potential later use and avoiding a tight dependency with the STG.

## Hypotheses to validate by microbenchmark

- HFMA2 vs MOV vs IMAD throughput for constant loading on SM120. Force each via PTX inline and measure.
- Whether FFMA with immediate in `src1` exists at all in any SM120 SASS dump (would invalidate the "addend only" rule).
- Whether the HFMA2 idiom changes on SM80 or SM89 (older architectures may use different defaults).

## Summary

Kernel 05 confirmed that the unroll cascade of kernel 04 was driven by runtime trip count uncertainty, and exposed the systematic use of HFMA2 for FP32 constant materialization on SM120. The investigation produced a clean rule (FFMA reserves only one immediate slot, in the addend position) and three actionable optimization strategies for kernel developers working with floating point constants.

The kernel itself is short, fast, and a reasonable baseline for further work. With 8 FFMAs in straight line, it is also a clean platform for testing other compiler decisions in subsequent variants without the noise of dispatch logic.