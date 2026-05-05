# Chapter 17 — LDSM (ldmatrix) baseline on SM120

## Goal

This chapter establishes the LDSM opcode family on SM120 — the SASS-level realization of `ldmatrix.sync.aligned.*` from PTX. LDSM is the bridge between shared memory and tensor core fragments: every production GEMM or attention kernel on Turing and later uses ldmatrix to feed HMMA/QMMA.

Six kernel variants (17a-17f) cover the full ldmatrix family (x1/x2/x4, trans/no-trans), a production pattern combining LDSM with HMMA, and a serial latency microbenchmark.

The chapter answers four primary questions:

1. **What SASS opcode does ldmatrix emit?** A new family `LDSM`, distinct from every prior opcode in the corpus.
2. **How are the x1/x2/x4 width and `.trans` modifier encoded?** Width in 2 bits of the control code, trans in 1 bit, mnemonic suffix pattern decoded.
3. **How does LDSM interact with MMA in production?** No NOPs, scoreboard-only synchronization, register allocation packed so LDSM destinations are MMA sources.
4. **What is the cost?** Serial latency model: `total_cycles ≈ 33 × N`. Marginal cost comparable to MMA, but chain overhead essentially zero.

## Variants

| Variant | PTX | SASS | Purpose |
|---|---|---|---|
| 17a | `ldmatrix.x1.m8n8.shared.b16` | `LDSM.16.M88` | Baseline single tile, identify opcode family |
| 17b | `ldmatrix.x2.m8n8.shared.b16` | `LDSM.16.M88.2` | Identify width encoding in control code |
| 17c | `ldmatrix.x4.m8n8.shared.b16` | `LDSM.16.M88.4` | Production case, confirm 2-bit width field |
| 17d | `ldmatrix.x4.trans.m8n8.shared.b16` | `LDSM.16.MT88.4` | Isolate `.trans` encoding |
| 17e | LDSM.x4 + LDSM.x2 + HMMA.16816.F32 | production combo | Scoreboard pattern, NOP discipline, register allocation |
| 17f | N chained LDSM.x1 with clock64() | latency microbench | `total_cycles ≈ 33 × N` |

## Key SASS observations

### The LDSM opcode

```
LDSM.16.M[T]88[.<N>] R_dst, [R_addr(+UR_base)?]
```

Opcode bytes base: `0x000000000004783b` (with R_dst encoding in byte 3).

Fields in the mnemonic:
* `.16` = element size in bits (16 = half)
* `M88` (no trans) or `MT88` (trans) = 8×8 matrix tile shape, T inside the tag for transpose
* `.2` / `.4` suffix = multi-matrix width (x2 / x4); no suffix = x1 default
* Address: `[R]` or `[R+UR]` depending on register allocation context

### Full variant table

| PTX | SASS mnemonic | Tested |
|---|---|---|
| `ldmatrix.x1.m8n8.shared.b16` | `LDSM.16.M88` | 17a |
| `ldmatrix.x2.m8n8.shared.b16` | `LDSM.16.M88.2` | 17b |
| `ldmatrix.x4.m8n8.shared.b16` | `LDSM.16.M88.4` | 17c |
| `ldmatrix.x1.trans.m8n8.shared.b16` | `LDSM.16.MT88` | inferred |
| `ldmatrix.x2.trans.m8n8.shared.b16` | `LDSM.16.MT88.2` | inferred |
| `ldmatrix.x4.trans.m8n8.shared.b16` | `LDSM.16.MT88.4` | 17d |

### Control code topology — decoded

By comparing 17a/b/c byte-by-byte, then 17c vs 17d for the trans flag, 3 bits of the LDSM control code have been decoded:

| Bit(s) | Meaning | Evidence |
|---|---|---|
| 8, 9 | Width encoding: 00 = x1, 01 = x2, 10 = x4 | 17a/b/c |
| 14 | Transpose flag | 17d vs 17c |
| 32-39 | Scoreboard slot ID (variable per instance) | 17e, 17f |
| Others | Standard scheduling (stall/yield/wait) as elsewhere in the corpus | |

Width encoding reads literally as `log2(N)` for N in {1, 2, 4}:
* x1 → bits (9, 8) = (0, 0) → field = 0
* x2 → bits (9, 8) = (0, 1) → field = 1
* x4 → bits (9, 8) = (1, 0) → field = 2

One bit remains free in the width field (value 3 unused), possibly reserved for future extensions.

### Opcode bytes invariance

Opcode bytes are **invariant across width and trans variants** (when destination register base is the same). Everything configurable about LDSM lives in the control code. Same property as HMMA/QMMA.

### Addressing patterns

Two addressing forms observed:

* `[R + UR]` for x1, x2 (17a, 17b): R = per-lane row offset, UR = shared memory base. ptxas uses `[R+UR]` when the per-lane offsets are simple enough to fit in a compact LEA pattern.
* `[R]` for x4 (17c): the full address (shared base + tile offset + row offset) pre-computed into R0 by a single LEA instruction. ptxas chose register recycling (same R0 used by the preceding STS).

Both forms are syntactically valid for LDSM — the choice is a ptxas optimization, not a hardware constraint.

### Shared memory base pattern

Same as kernels 06-08: `UMOV UR4, 0x400` then `ULEA UR4, UR_cta, UR4, 0x18` to compute the per-CTA shared memory base. Unchanged between chapters.

## Production pattern (17e)

The combined LDSM + HMMA kernel reveals the exact GEMM/attention tile structure:

```
STS × N
BAR.SYNC.DEFER_BLOCKING 0x0
LDSM.16.M88.2 R10, [R7+UR5]     ← B fragment  (emitted first!)
LDSM.16.M88.4 R12, [R6+UR4]     ← A fragment
HMMA.16816.F32 R12, R12, R10, RZ
@!UPT UIADD3 URZ NOP
@!UPT UIADD3 URZ NOP
STG × N
```

### Observation 1: ptxas inverts the LDSM emission order

[OBS] Although the C++ source emits LDSM_A (x4) before LDSM_B (x2), the SASS shows `LDSM.x2` (B) before `LDSM.x4` (A). [HYP] Emitting the shorter x2 load first may give the longer x4 load more time to complete before the HMMA needs both. [INF] The ordering does not change the dependency requirement that HMMA wait for both LDSM results.

### Observation 2: no NOPs between LDSM and HMMA

Unlike HMMA→consumer patterns where 2 UIADD3 NOPs are inserted, LDSM→HMMA has **zero NOPs**. The scoreboard mechanism fully handles the synchronization.

LDSM sets a scoreboard slot in its control code (bits 32-39 of the high word). HMMA waits on the scoreboard(s) via the wait mask in the low byte of its control code: `0xff` = monitor all active scoreboards.

Compare with HMMA single-MMA from chapter 13b where the wait mask low byte was `0x14` (default), here it's `0xff` (full wait). This distinguishes "HMMA with LDSM sources" from "HMMA with LDG sources" at the control code level.

### Observation 3: zero-copy register allocation

```
LDSM.x2 destinations: R10, R11   (B fragment)
LDSM.x4 destinations: R12..R15   (A fragment)
HMMA operands:        R12 (D), R12 (A), R10 (B), RZ (C)
```

The LDSM destinations **are** the HMMA sources. ptxas has perfect register packing — no intermediate copies, no rename moves. This is the critical ILP-optimization for production GEMM: every cycle spent copying would be a cycle the tensor core sits idle.

### Observation 4: 2 NOPs after HMMA

Same as chapter 13b: HMMA → consumer (STG) dependency requires 2 UIADD3 NOPs because STG doesn't participate in the MMA scoreboard scheme.

## Latency (17f)

Three measurements at N = 16, 32, 64 chained LDSM.x1 with a pointer-chase trick (add the LDSM output — always zero in this setup — to the address, creating a true dependency without breaking the address):

| N | total_cycles | cycles per LDSM |
|---|---|---|
| 16 | 509 | 31.81 |
| 32 | 1038 | 32.44 |
| 64 | 2094 | 32.72 |

Linear regression on incremental costs:
* ΔN(16→32): 529 cycles for 16 added LDSMs → 33.06 cycles/LDSM
* ΔN(32→64): 1056 cycles for 32 added LDSMs → 33.00 cycles/LDSM

**Model**: `total_cycles ≈ 33 × N` with overhead essentially zero.

### Comparison with MMA latency

| Operation | Cycles per instance | Chain overhead | Model |
|---|---|---|---|
| HMMA.16816.F32 (13e) | ~35 | ~310 | `310 + 35×N` |
| QMMA.16832.F32 (14f) | ~35 | ~510 | `510 + 35×N` |
| **LDSM.16.M88 x1 (17f)** | **~33** | **~0** | **`33×N`** |

LDSM latency is comparable to MMA per-instance cost (~33 vs ~35 cycles), but has **no chain overhead**. Two interpretations:
* Hypothesis: LDSM and MMA share internal datapaths (tensor memory unit). Similar pipeline depth per instruction.
* LDSM doesn't need the ~310-510 cycle startup the MMA pipeline requires. No MAC initialization, just shared memory access + matrix assembly.

### Implications for production pipelining

In a GEMM k-loop, the typical latency chain per tile is:
* LDSM_A + LDSM_B (parallel) → ~33 cycles
* HMMA → ~35 cycles
* Total serial per tile: ~68 cycles (pessimistic, no overlap)

With software pipelining (load next tile while current tile MMAs), this overlaps down to max(LDSM, HMMA) ≈ 35 cycles per tile. This is what CUTLASS / FlashAttention achieve with their `cp.async` double-buffered pipelines.

## LDSM chain pattern (17f internals)

The chained latency kernel shows a clean scheduling pattern:

```
LDSM.16.M88 R3, [R3+UR4]    ctrl 0x000e240008000000   ← first: wait + SBS
IADD R4, R4, R3              ctrl 0x001fca00078e0000   ← wait on LDSM scoreboard
LDSM.16.M88 R5, [R4]         ctrl 0x000e240000000000   ← subsequent: SBS only
IADD R5, R4, R5              ctrl 0x001fca00078e0000
...
```

Observations:

* **No NOPs anywhere** — scoreboard handles all dependencies
* **Register renaming**: LDSM destinations alternate across ~8 registers (R3, R5, R2, R7, R6, R9, R8, ...) instead of overwriting the same register. This allows maximum ILP opportunity even though the chain is serial by construction.
* [OBS] **First LDSM has a wait bit** (byte 24-31 = `0x08`). [HYP] The bit may be related to the preceding BAR.SYNC synchronization. [OBS] Subsequent LDSMs in the chain omit that explicit wait bit and rely on the register-level dependency through the IADD.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| LDSM is a new opcode distinct from LDS/LDG | Confirmed (`0x3b` family) |
| Mnemonic follows pattern `LDSM.16.M[T]88[.N]` | Confirmed |
| Opcode bytes invariant across width/trans variants | Confirmed |
| Width encoded in 2 bits of control code | Confirmed (bits 8-9) |
| Trans encoded in 1 bit of control code | Confirmed (bit 14) |
| LDSM → HMMA requires no NOPs | Confirmed |
| Scoreboard sync between LDSM and MMA via control code | Confirmed (wait mask `0xff` on HMMA) |
| Register allocation: LDSM dst = MMA src (zero-copy) | Confirmed |
| LDSM latency comparable to MMA | Confirmed (~33 vs ~35 cycles) |
| Chain overhead negligible for LDSM | Confirmed (~0 vs ~310 for HMMA) |

## Open gaps

| Gap | Notes |
|---|---|
| `.trans` variants x1, x2 | Inferred from the model but not compiled. Predictions: `LDSM.16.MT88` for x1.trans, `LDSM.16.MT88.2` for x2.trans. |
| `stmatrix` (STSM?) | Not tested. PTX `stmatrix.sync.aligned.*` introduced in SM90. Might exist on SM120 as a separate opcode, or be equivalent to a sequence of STS instructions. Needs its own chapter. |
| LDSM width > 4 | The 2-bit width field allows value 3 unused. Probably reserved, no PTX variant maps to it. |
| LDSM with larger elements (32-bit?) | Only `.16` (half) tested. PTX `ldmatrix.*.b32` might exist. The `.16` in the mnemonic suggests an element size field that could hold other values. |
| LDSM in conjunction with cp.async | Production kernels often use cp.async (LDGSTS) to feed smem, then LDSM. Combined pattern not tested here — needs chapter 18 (pipelined tile). |
| scoreboard slot allocation algorithm | Which specific bits in bytes 32-39 correspond to which slot ID is partially observed but not formally decoded. |

## How to read LDSM in a production dump

When auditing a GEMM or attention kernel:

1. **Locate the LDSM instructions** by searching for `LDSM` in the dump. They cluster around BAR.SYNC barriers.
2. **Identify the width**: mnemonic has `.2` or `.4` suffix for x2/x4, otherwise x1.
3. **Identify the orientation**: `M88` vs `MT88` in the mnemonic (T for trans). B fragments are commonly trans in row-col GEMM layouts.
4. **Find the MMA that consumes** the LDSM outputs: should be within a few instructions, with same dst registers as sources.
5. **Check the scoreboard wait mask** on the consuming MMA: typically `0xff` when LDSM feeds it directly, distinct from `0x14` (HMMA with LDG sources) or other patterns.
6. **Count cycles**: `~33 per LDSM + ~35 per MMA`, pipelined across k-loop iterations.

## Summary

* 1 new SASS opcode family: `LDSM.16.M[T]88[.N]` (6 variants, 4 tested + 2 inferred)
* 3 control code bits decoded: width (8-9), trans (14)
* Production pattern LDSM→HMMA: no NOPs, scoreboard-only sync, zero-copy register allocation
* Latency model: `total_cycles ≈ 33 × N` (chain overhead ~0)
* LDSM latency ~33 cycles, comparable to MMA per-instance cost
* Pattern captured: `STS + BAR + LDSM(s) + MMA + NOPs + STG` — the building block of every SM120 GEMM/attention kernel
* Complete audit toolkit: kernels 01-14 + 17 cover basics + MMA + shared-to-fragment bridge
