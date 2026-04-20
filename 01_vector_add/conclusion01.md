# Kernel 01 — vector_add

Baseline kernel for the SASS King project. The simplest possible CUDA kernel, used to establish what the "invisible infrastructure" of any SASS looks like before any algorithmic work is added.

## Source

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

## Compile

```
nvcc -arch=sm_120 -o 01_vector_add 01_vector_add.cu
cuobjdump --dump-sass 01_vector_add
```

## SASS dump (SM120)

```
/*0000*/  LDC       R1, c[0x0][0x37c]                   // SP init
/*0010*/  S2R       R0, SR_TID.X                         // threadIdx.x
/*0020*/  S2UR      UR4, SR_CTAID.X                      // blockIdx.x (uniform)
/*0030*/  LDCU      UR5, c[0x0][0x3a0]                   // n (uniform)
/*0040*/  LDC       R11, c[0x0][0x360]                   // blockDim.x
/*0050*/  IMAD      R11, R11, UR4, R0                    // i = blockDim*blockIdx + TID
/*0060*/  ISETP.GE.AND P0, PT, R11, UR5, PT              // P0 = (i >= n)
/*0070*/  @P0 EXIT                                        // out-of-bounds threads exit
/*0080*/  LDC.64    R2, c[0x0][0x380]                    // pointer a
/*0090*/  LDCU.64   UR4, c[0x0][0x358]                   // global descriptor
/*00a0*/  LDC.64    R4, c[0x0][0x388]                    // pointer b
/*00b0*/  LDC.64    R6, c[0x0][0x390]                    // pointer c
/*00c0*/  IMAD.WIDE R2, R11, 0x4, R2                     // &a[i]
/*00d0*/  LDG.E     R2, desc[UR4][R2.64]                 // load a[i]
/*00e0*/  IMAD.WIDE R4, R11, 0x4, R4                     // &b[i]
/*00f0*/  LDG.E     R5, desc[UR4][R4.64]                 // load b[i]
/*0100*/  IMAD.WIDE R6, R11, 0x4, R6                     // &c[i]
/*0110*/  FADD      R9, R2, R5                           // R9 = a[i] + b[i]
/*0120*/  STG.E     desc[UR4][R6.64], R9                 // store c[i] = R9
/*0130*/  EXIT
/*0140*/  BRA       .                                    // dead
/*0150*/  NOP                                            // padding
...
```

20 useful instructions. The rest is alignment padding.

## Structural decomposition

The kernel splits into six clearly identifiable sections, each with a distinct role.

### Prologue (0x00–0x50) — identity

Answers the question "who am I?" by computing the thread's global index `i`. Six instructions:

| Step | Destination | Source |
|------|-------------|--------|
| SP init | R1 | constant memory offset 0x37c (ABI convention) |
| TID read | R0 (per-thread) | special register SR_TID.X |
| CTAID read | UR4 (uniform) | special register SR_CTAID.X |
| n load | UR5 (uniform) | constant memory offset 0x3a0 |
| blockDim load | R11 (per-thread) | constant memory offset 0x360 |
| i computation | R11 | `R11 * UR4 + R0` = blockDim × blockIdx + threadIdx |

Two key observations about the prologue:

**Two register files in use.** Values shared across the warp (blockIdx, n) go into uniform registers (UR). Values unique per thread (threadIdx, i) go into normal registers (R). ptxas does this classification automatically based on data flow.

**Constant memory is the kernel argument channel.** The driver places all kernel arguments and launch parameters into constant bank 0 at fixed offsets. The kernel reads them via LDC (per-thread destination) or LDCU (uniform destination). The data itself of the input arrays is not in constant memory — only the pointers to the arrays.

### Bounds check (0x60–0x70) — predication, not branching

Two instructions. `ISETP.GE.AND P0, PT, R11, UR5, PT` computes `P0 = (i >= n)`. `@P0 EXIT` uses P0 to make out-of-bounds threads exit immediately.

This is predication, not branching. There is no BRA, no BSSY/BSYNC, no reconvergence mask to manage. In-bounds threads continue in straight-line code; out-of-bounds threads exit cleanly. Zero divergence overhead.

### Pointer loads (0x80–0xb0) — all addresses fetched in parallel

Four loads from constant memory, issued back-to-back with `stall=1`:

```
LDC.64    R2, c[0x0][0x380]      SBS=0    // a
LDCU.64   UR4, c[0x0][0x358]     SBS=1    // global descriptor
LDC.64    R4, c[0x0][0x388]      SBS=2    // b
LDC.64    R6, c[0x0][0x390]      SBS=3    // c
```

Each pointer gets its own scoreboard, so dependent calculations can proceed as soon as any one pointer is ready, independently of the others.

### Address computation and memory operations (0xc0–0x100) — interleaved

ptxas interleaves `IMAD.WIDE` (address computation) with `LDG.E` (memory load) so that multiple global loads are in flight at the same time:

```
IMAD.WIDE R2, R11, 4, R2     // &a[i]
LDG.E     R2, ...[R2.64]      SBS=4   → fire a[i]
IMAD.WIDE R4, R11, 4, R4     // &b[i]
LDG.E     R5, ...[R4.64]      SBS=4   → fire b[i]  (same SB)
IMAD.WIDE R6, R11, 4, R6     // &c[i]                (no LDG, this is for the store)
```

Both loads share **SB4**. The downstream consumer (FADD) will wait on SB4 once, which covers both loads simultaneously. This is ptxas's way of economizing scoreboards: the 6 available SBs are a hard budget.

### Compute (0x110) — one instruction

```
FADD R9, R2, R5   wait={SB4}
```

Single FADD. Waits on SB4, which means waiting for both `a[i]` and `b[i]` loads to complete.

### Store + exit (0x120–0x130)

```
STG.E desc[UR4][R6.64], R9
EXIT
```

Store is fire-and-forget — no scoreboard needed since no subsequent instruction reads back the stored value.

### Padding (0x140 onward)

One dead `BRA` plus NOPs. Used to align the kernel size to a 512-byte boundary. The hardware never executes these — the warp has already EXIT'd.

## Instruction count breakdown

| Role | Count | % |
|---|---|---|
| Thread/block ID setup | 5 | 25% |
| Bounds check | 2 | 10% |
| Pointer loads | 4 | 20% |
| Address arithmetic | 3 | 15% |
| Memory ops | 3 | 15% |
| Compute | **1** | **5%** |
| Exit | 1 | 5% |
| Alignment padding | (not counted) | — |

One useful compute instruction out of twenty. This is the signature of an elementwise memory-bound kernel: the hardware spends nearly all its time computing addresses and waiting for memory, not doing useful arithmetic.

## Patterns identified

This kernel establishes six reusable observations:

1. **Fixed prologue skeleton.** Every CUDA kernel starts with the same sequence: SP init → thread/block ID → argument loads → i computation → bounds check. This infrastructure is invariant across kernels and can be skipped when reading unfamiliar SASS.

2. **Per-thread vs uniform register split.** ptxas performs uniformity analysis automatically. Values that are the same across the warp (blockIdx, kernel args) end up in UR; values that vary per thread (threadIdx, computed indices, loaded data) end up in R.

3. **Predicated EXIT instead of branching.** Out-of-bounds handling is via `@P0 EXIT`, never via BRA around a body. Zero-cost divergence.

4. **Parallel pointer loads with distinct scoreboards.** Each independent long-latency load gets its own SB, allowing them to progress independently.

5. **Shared scoreboard for co-consumed loads.** When multiple loads are all consumed by the same downstream instruction (the FADD consumes both a[i] and b[i]), ptxas groups them onto a single SB. The consumer waits once, for all.

6. **Interleaved address/memory scheduling.** ptxas places IMAD.WIDE instructions between LDG emissions so that multiple memory operations are in flight simultaneously, maximizing memory-level parallelism.

## What this kernel does not show

The baseline is deliberately minimal. It does not show:
- Loops (no BRA backward, no ISETP loop condition)
- Divergent control flow (no BSSY/BSYNC)
- Shared memory (no LDS/STS/BAR.SYNC)
- Vectorized memory (no LDG.E.128)
- Register spilling (no STL/LDL)
- Tensor core operations
- Atomics
- Warp-level primitives

Each of these will be introduced as a minimal delta in subsequent kernels.

## Reference SASS reading template

From this kernel we derive a general method for reading any SASS dump:

1. **Skip the prologue.** Identify the ID setup by pattern matching; move past it.
2. **Find the bounds check or early exit.** Look for the first `@P EXIT`.
3. **Identify the loaded arguments.** Pointers come from constant memory in a block near the start of the body.
4. **Locate the compute section.** Look for the arithmetic instructions that correspond to the algorithm.
5. **Check the stores.** STG at the end writes the results.
6. **Everything else is plumbing** — address arithmetic, scoreboard bookkeeping, control flow around loops.

This template is used as a first-pass filter in every subsequent kernel audit.