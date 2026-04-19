# Kernel 02 — vector_add + 1.0f

Smallest possible delta from kernel 01: a single extra addition in the expression.

## Source

```cuda
__global__ void vector_add_plus1(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i] + 1.0f;
    }
}
```

One character difference in the source: `+ 1.0f` appended to the expression.

## Compile

```
nvcc -arch=sm_120 -o 02_vector_add_plus1 02_vector_add_plus1.cu
cuobjdump --dump-sass 02_vector_add_plus1
```

## SASS diff from kernel 01

Instructions 0x00 through 0x100 are **byte-for-byte identical** to kernel 01. The entire prologue, bounds check, pointer loads, address arithmetic, and memory loads do not change.

The only difference is in the compute section:

| Address | Kernel 01 | Kernel 02 |
|---------|-----------|-----------|
| 0x110 | `FADD R9, R2, R5` | `FADD R0, R2, R5` |
| 0x120 | `STG.E ..., R9` | `FADD R9, R0, 1` |
| 0x130 | `EXIT` | `STG.E ..., R9` |
| 0x140 | (padding) | `EXIT` |

One extra instruction. One register renamed. Everything else identical.

## Analysis

### No fusion

Two FADDs stay two FADDs. ptxas does not combine them into an FFMA because FFMA computes `a * b + c`, not `a + b + c`. The two operations do not fit the FMA algebraic template.

This establishes the rule: **FMA fusion is syntactically strict**. The source expression must literally contain a multiply followed by an add to trigger fusion. Adding three operands does not.

### Register recycling

In kernel 01, the single FADD wrote to R9 and STG consumed R9. Done.

In kernel 02, there are two FADDs. The first one produces an intermediate value that must live somewhere while the second FADD reads it. ptxas chose to write the first FADD into R0.

Why R0? Because R0 held threadIdx.x during the prologue (via `S2R R0, SR_TID.X`), and after the IMAD at 0x50 computed `i` from it, R0 is never read again. The value is dead. ptxas's liveness analysis detects this and recycles R0 as scratch space for the intermediate `a[i] + b[i]` value.

Net effect: one additional arithmetic operation, zero additional register pressure.

This is a general pattern: ptxas tracks register liveness precisely and reuses dead registers whenever possible, regardless of their original semantic purpose.

### Liveness analysis is global, not local

The mechanism underlying the register recycling deserves explicit statement. Ptxas does not apply register renaming one instruction at a time. It performs a global liveness pass over the entire kernel: for each register, it computes the range of instructions where the register's value is needed, and allocates physical registers to minimize overlap.

At the moment of the first FADD in kernel 02, the liveness analysis has already determined that:
- R0's semantic role (threadIdx.x) ended at 0x50 when IMAD consumed it.
- R0 is free from 0x50 onwards until something else reuses it.
- No subsequent instruction reads R0 as threadIdx.x again.

Therefore R0 is a valid target for the first FADD. This is a **global** decision, not a local peephole transformation. The consequence is that when diffing two related kernels, register renames may appear in sections of the code where nothing else changed, as a ripple effect of a source change elsewhere.

### Delta of one instruction is exact

Kernel 02 has 21 useful instructions, kernel 01 has 20. The delta is exactly +1 for exactly one extra source-level operation. No hidden overhead, no optimization surprises, no compiler-inserted helpers.

This is a rare case where the source-to-SASS ratio is exactly 1:1 for the change. Most kernels have a larger expansion factor because of the infrastructure amortization (prologue, pointer loads, address arithmetic). Here the change is purely in the compute section, which has no amortized overhead.

### Prologue byte-identity confirms ptxas determinism

The first 17 instructions of kernel 02 have byte-identical opcodes and control codes to kernel 01. Same bit pattern, same stall counts, same scoreboards, same register choices.

This confirms that ptxas is deterministic in the absence of source change: identical source produces identical SASS. The corollary is actionable for diffing: if a prologue byte changes between two kernels whose sources are supposed to be identical, the source has actually changed somewhere. Conversely, prologue byte-identity is a strong confirmation that nothing before the changed section drifted.

### The "stable infrastructure" observation

Every non-compute instruction is identical between the two kernels. The prologue, bounds check, and memory operations form a stable infrastructure that does not change when the algorithm changes. Only the compute section grows.

This confirms the method: **controlled variation isolates exactly one pattern at a time**. A one-character source change produces a one-instruction SASS change, and the register recycling logic is the only surprise. If we had changed more at once, we would not know which part of the source caused which part of the SASS change.

## Patterns identified (new)

Four observations on top of kernel 01:

1. **FMA fusion is strict.** Only `a * b + c` patterns fuse. `a + b + c` produces two separate FADDs.

2. **Dead register recycling.** Ptxas reuses registers whose values are no longer live, even across semantic boundaries (threadIdx's R0 becomes an FP scratch register). No pressure increase for additional computation as long as there is dead room in the register file.

3. **Liveness analysis is global.** Register allocation decisions are made over the whole kernel, not locally. A change in one section can ripple to register renames in other sections.

4. **Ptxas is deterministic.** Identical source produces byte-identical SASS. Prologue byte-identity is a reliable signal for diffing.

## Instruction count breakdown

| Role | Count | Δ from K01 |
|---|---|---|
| Setup + bounds | 7 | 0 |
| Pointer loads | 4 | 0 |
| Address + memory | 6 | 0 |
| Compute | 2 | +1 |
| Store + exit | 2 | 0 |
| **Total** | **21** | **+1** |

Exactly one extra instruction for one extra source-level operation. No hidden overhead, no optimization surprises. This is the cleanest possible controlled variation.

## What this kernel does not show

Same limitations as kernel 01. The minimal delta did not introduce any new infrastructure. No loops, no shared memory, no new control flow, no tensor cores.

## Takeaway for SASS reading

When diffing two SASS dumps:
- Most instructions will be identical. Focus on what changes.
- Register renaming may occur even in the identical-looking sections as a consequence of changes elsewhere. Track destinations, not just opcodes.
- Prologue byte-identity is a diffing tool in its own right: if prologues differ, the source changed somewhere you did not expect.
- The absence of fusion is as informative as its presence. If you see two arithmetic instructions where you expected one, it usually means the source-level pattern did not match FMA.