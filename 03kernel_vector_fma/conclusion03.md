# Kernel 03 — vector_fma (a*b+c)

Minimal delta that triggers FMA fusion. Replaces the addition in kernel 01 with a multiply-add.

## Source

```cuda
__global__ void vector_fma(const float* a, const float* b, const float* c, float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d[i] = a[i] * b[i] + c[i];
    }
}
```

Two changes from kernel 01: the operation changes from `+` to `*+`, and a fourth operand is added.

## Compile

```
nvcc -arch=sm_120 -o 03_vector_fma 03_vector_fma.cu
cuobjdump --dump-sass 03_vector_fma
```

## SASS dump

```
/*0000*/  LDC       R1, c[0x0][0x37c]
/*0010*/  S2R       R0, SR_TID.X                        SBS=0
/*0020*/  S2UR      UR4, SR_CTAID.X                     SBS=0
/*0030*/  LDCU      UR5, c[0x0][0x3a0]                  SBS=1
/*0040*/  LDC       R11, c[0x0][0x360]                  SBS=0
/*0050*/  IMAD      R11, R11, UR4, R0          stall=5  wait={SB0}  yield
/*0060*/  ISETP.GE.AND P0, PT, R11, UR5, PT   stall=13 wait={SB1} yield
/*0070*/  @P0 EXIT
/*0080*/  LDC.64    R2, c[0x0][0x380]                   SBS=0
/*0090*/  LDCU.64   UR4, c[0x0][0x358]                  SBS=1
/*00a0*/  LDC.64    R4, c[0x0][0x388]                   SBS=2
/*00b0*/  LDC.64    R6, c[0x0][0x390]                   SBS=3
/*00c0*/  IMAD.WIDE R2, R11, 0x4, R2           stall=7  wait={SB0}  yield
/*00d0*/  LDC.64    R8, c[0x0][0x398]                   SBS=0
/*00e0*/  LDG.E     R2, desc[UR4][R2.64]                SBS=4  wait={SB1}
/*00f0*/  IMAD.WIDE R4, R11, 0x4, R4           stall=6  wait={SB2}  yield
/*0100*/  LDG.E     R5, desc[UR4][R4.64]                SBS=4
/*0110*/  IMAD.WIDE R6, R11, 0x4, R6           stall=6  wait={SB3}  yield
/*0120*/  LDG.E     R7, desc[UR4][R6.64]                SBS=4
/*0130*/  IMAD.WIDE R8, R11, 0x4, R8           stall=4  wait={SB0}  yield
/*0140*/  FFMA      R11, R2, R5, R7            stall=5  wait={SB4}  yield
/*0150*/  STG.E     desc[UR4][R8.64], R11
/*0160*/  EXIT
...
```

23 useful instructions. One FFMA replaces what would have been separate FMUL + FADD.

## Structural decomposition

Same six-section skeleton as kernel 01. The sections grow to accommodate the extra pointer, but the organization is identical.

### Compute: the fusion itself

```
FFMA R11, R2, R5, R7    wait={SB4}
```

A single instruction computes `R11 = R2 * R5 + R7`. In terms of the source: `d[i] = a[i] * b[i] + c[i]`.

Ptxas detected the multiply-add pattern in the source expression and emitted a single fused instruction instead of separate FMUL and FADD. This is in contrast to kernel 02, where `a + b + c` produced two FADDs because it did not match the FMA template.

The operand order of FFMA is `dst, srcA, srcB, srcC` where srcC is the addend. So `FFMA R11, R2, R5, R7` computes `R2 * R5 + R7`, matching the source order `a * b + c`. Same ordering as PTX `fma.f32`.

### Three loads on one scoreboard

```
LDG.E R2, ...[R2.64]    SBS=4
LDG.E R5, ...[R4.64]    SBS=4
LDG.E R7, ...[R6.64]    SBS=4
```

All three global loads share SB4. The FFMA does a single `wait={SB4}` to block until all three are complete. This is the scoreboard economy pattern first observed in kernel 01, extended to three producers instead of two.

The SB budget (6 available) is tight when many independent data streams are in flight. Grouping co-consumed producers on one SB frees SBs for other purposes.

### Four LDC with distinct scoreboards, back-to-back issue

The four pointer loads at 0x80 to 0xb0 illustrate the opposite pattern: each pointer load gets its own scoreboard (SB0 through SB3), even though they are all LDCs.

```
LDC.64    R2, c[0x0][0x380]      SBS=0  stall=1
LDCU.64   UR4, c[0x0][0x358]     SBS=1  stall=1
LDC.64    R4, c[0x0][0x388]      SBS=2  stall=1
LDC.64    R6, c[0x0][0x390]      SBS=3  stall=1
```

Why distinct scoreboards rather than grouping? The downstream IMAD.WIDE instructions depend on different pointers and can proceed independently. If all four LDCs shared one SB, the first IMAD would wait for all four to complete, losing the parallelism between the independent address computations.

The stall count of 1 on each LDC reflects the fact that constant cache responds fast (unlike global memory). Ptxas does not need to insert larger cadence gaps between LDCs.

Contrast with global loads (LDG), which have larger effective latencies and where scoreboard grouping is the economizing pattern, not the parallelizing pattern.

### Interleaved address/load scheduling

Ptxas does not emit `LDG LDG LDG` in sequence. Instead it interleaves address computation with load emission:

```
IMAD.WIDE R2, R11, 4, R2     // &a[i]
LDG.E     R2, ...[R2.64]      SBS=4   → fire a[i]
IMAD.WIDE R4, R11, 4, R4     // &b[i]
LDG.E     R5, ...[R4.64]      SBS=4   → fire b[i]  (same SB)
IMAD.WIDE R6, R11, 4, R6     // &c[i]                (no LDG, this is for the store)
```

Four address computations, three loads, and one address for the store, all packed so that the arithmetic is done while the loads are in flight. The FFMA at 0x140 is the first instruction that stalls waiting for data.

## Deep dive: scoreboards and stalls

Kernel 03 is the first kernel where the scoreboard mechanics are clearly visible. Two independent synchronization mechanisms are used in parallel.

### Stall count

Every instruction has a `stall` field (1 to 15 cycles). It tells the warp scheduler how many cycles to wait after emitting this instruction before emitting the next one **from this warp**. Other warps are unaffected and can be dispatched during the stall.

Stall count covers fixed-latency operations. When ptxas knows that FFMA takes exactly 5 cycles on the FMA pipeline, it emits `stall=5` on each FFMA. The consumer downstream can safely read the destination register without additional synchronization.

### Scoreboards

Six scoreboards are available per warp (SB0 through SB5). A scoreboard is a counter, not a flag.

An instruction with `SBS=N` increments SBN when it is issued, and decrements SBN when the data is ready for consumers. An instruction with `wait={N}` blocks until SBN is zero.

Variable-latency operations (LDG, LDS, S2R, MUFU, BAR) must use scoreboards because ptxas cannot know their latency at compile time. The compiler still emits a small stall count to allow the scheduler to issue subsequent instructions (typically stall=1), but the actual data synchronization is carried by the scoreboard.

### The two in combination

A warp is stalled if and only if at least one of these is true:
- Its internal stall counter is nonzero
- The scoreboard wait mask on its next instruction is not satisfied

The two mechanisms cover orthogonal cases. Stall count controls the **cadence** of instruction issue for a single warp. Scoreboards control the **correctness** of data dependencies across variable-latency boundaries.

### Observed anomaly: ISETP stall=13

The instruction `ISETP.GE.AND P0, PT, R11, UR5, PT` has `stall=13`, much larger than the typical 5 for an ALU instruction. Hypothesis: the predicate produced by ISETP must be transferred across pipelines to the CBU (control branch unit) to feed the `@P0 EXIT` that follows. This cross-pipeline transfer adds latency.

This exact stall=13 pattern recurs in every subsequent kernel's bounds check. It is not an isolated anomaly but a universal signature of "predicate producer feeding a predicate consumer on a different pipeline".

To be validated by microbenchmark: measure latency of predicate-producer → predicated-branch-consumer chains versus predicate-producer → arithmetic-consumer chains.

### Yield flag

Every instruction with `wait={...}` on a scoreboard also has the yield flag set. The scheduler is being told: "if this warp must stall here, let another warp run in the meantime."

Instructions without scoreboard waits do not have yield set, because there is no significant stall expected. The pattern is consistent across all kernels observed so far: **yield flag appears on exactly those instructions where the warp might genuinely wait for a slow operation**. This is a universal rule on SM120.

## Uniform registers explained

Kernel 03 made the uniform register distinction visible enough to warrant explanation.

The GPU has two register files, physically separate:
- Per-thread registers R0-R255, with 32 copies per warp (one per thread), housed in the main register file SRAM
- Uniform registers UR0-UR63, with 1 copy per warp, housed in a smaller dedicated SRAM with its own datapath

Uniform values (same across all 32 threads) go into UR. Ptxas detects uniformity automatically. Typical uniform values:
- blockIdx.x, blockDim.x, gridDim.x
- Kernel arguments (pointers and scalars)
- Constants computed from the above

Typical per-thread values:
- threadIdx.x
- Computed indices
- Loaded data

A single instruction can mix sources from both files. `IMAD R11, R11, UR4, R0` multiplies a per-thread R11 by a uniform UR4 and adds a per-thread R0. The hardware reads UR4 once via the uniform datapath and broadcasts to all 32 threads simultaneously.

Benefits of the uniform path:
- No duplication of the value in the main register file
- Separate pipeline, no contention with per-thread arithmetic
- Lower energy per operation (one broadcast instead of 32 reads)

Cost: the uniform file is smaller (64 registers on SM80-SM89, expanded to 256 on SM100+) and the uniform pipeline has a narrower set of supported operations.

## Instruction count breakdown

| Role | Count | % |
|---|---|---|
| Setup | 5 | 22% |
| Bounds check | 2 | 9% |
| Pointer loads | 5 | 22% |
| Address arithmetic | 4 | 17% |
| Memory ops (LDG + STG) | 4 | 17% |
| Compute (FFMA) | **1** | **4%** |
| Exit | 1 | 4% |
| **Total useful** | **23** | |

### Adding a pointer costs three plumbing instructions

Kernel 01 had three pointers (a, b, c): 20 useful instructions total. Kernel 03 has four pointers (a, b, c, d): 23 useful instructions. Delta = +3 for one additional pointer. The three extra instructions are:
- One LDC.64 to load the pointer from constant memory.
- One IMAD.WIDE to compute the per-thread address.
- Zero in the compute section because FFMA fused the multiply-add into a single instruction.

This is the "pointer tax" rule: each additional array argument costs approximately three SASS instructions of plumbing, independent of how the array is used in compute. Kernels that touch many arrays pay a proportional overhead even if each array is accessed only once.

## Patterns identified (new)

1. **FMA fusion is active.** The `a*b+c` expression compiles to a single FFMA. Fusion is syntactic: the multiply and the add must be direct operands of each other. FFMA operand order is `dst, a, b, c` with c as the addend.

2. **Scoreboards and stall count are independent mechanisms.** Stall count is for instruction cadence; scoreboards are for data-ready signaling. Variable-latency ops use both; fixed-latency ops use only stall count.

3. **Scoreboard grouping for co-consumed producers.** Multiple LDGs that all feed the same downstream FFMA go onto the same SB so the consumer waits once.

4. **Distinct scoreboards for parallelizable consumers.** Multiple LDCs each feeding their own independent IMAD.WIDE get distinct SBs so the IMADs can proceed in parallel. The grouping vs distinct choice depends on whether the consumers share or not.

5. **Yield on scoreboard waits.** A universal rule on SM120: any instruction that waits on a scoreboard has the yield flag set to allow the scheduler to switch warps.

6. **ISETP → @P EXIT signature stall=13.** Recurs in every kernel's bounds check. Cross-pipeline transfer latency. Recognized pattern, not an anomaly.

7. **Pointer tax of three instructions per array argument.** One LDC + one IMAD.WIDE + potentially one more memory op (LDG or STG).

8. **Register file separation is physical.** Uniform registers live in a distinct SRAM with a separate pipeline, not a mode of the main register file.

## Hypotheses to validate by microbenchmark

- Latency of predicate → branch (expected: ~13 cycles on SM120).
- Latency of predicate → arithmetic consumer (expected: lower, to quantify).
- Throughput of LDG with SB grouping versus separate SBs.
- Whether the "6 scoreboards per warp" budget is actually 6, or effectively smaller due to hardware constraints.
- Whether LDCs with distinct SBs really execute in parallel or whether the constant cache serializes them internally.