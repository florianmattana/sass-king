# Chapter 18 — Pipelined MMA tile with cp.async on SM120

## Goal

This chapter establishes the production GEMM pipeline pattern on SM120: prefetching global memory to shared memory via `cp.async` (PTX) while computing MMA on the current tile, hiding memory latency. Three kernel variants (18a-18c) cover:

* The basic 2-stage pipeline fully unrolled (18a)
* A realistic K-loop with explicit software pipelining (18b)
* A 3-stage pipeline to isolate the `cp.async.wait_group N` encoding (18c)

The chapter answers four primary questions:

1. **What SASS opcodes does `cp.async` emit?** Three new opcodes: `LDGSTS.E.LTC128B.128` for the load itself, `LDGDEPBAR` for `commit_group`, and `DEPBAR.LE SB0, N` for `wait_group N`.
2. **How does ptxas arrange the pipeline in SASS?** Aggressively: it moves LDSM of the next tile before the MMA of the current tile whenever dependencies allow, hiding memory latency behind compute.
3. **How is the wait_group N argument encoded?** In 2 bits (positions 38-39) of the DEPBAR.LE control code, supporting N ∈ {0, 1, 2, 3}.
4. **What is the production GEMM pattern end-to-end?** Prologue (prefetch) → loop body (prefetch next + consume current) → tail (consume last) → epilogue (STG).

## Variants

| Variant | Structure | Purpose |
|---|---|---|
| 18a | 2 tiles fully unrolled, 2-stage pipeline | Discover LDGSTS, LDGDEPBAR, DEPBAR.LE opcodes |
| 18b | K-loop 4 tiles with `#pragma unroll 1`, 2-stage | Observe real SASS loop with BRA back-edge, register spill, double-buffer |
| 18c | 3 tiles fully unrolled, 3-stage pipeline | Decode N argument in DEPBAR.LE (N = 2, 1, 0) |

## Key SASS observations

### New opcode family: cp.async → LDGSTS

```
LDGSTS.E.LTC128B.128 [R_smem], desc[UR_desc][R_gmem.64]
```

Opcode bytes: low byte `0xae` family (distinct from LDG `0x81`, STS `0x88`, LDS `0x84`).

Mnemonic breakdown:
* `LDGSTS` = "Load Global Store Shared" — one SASS instruction that performs both operations asynchronously
* `.E` = Extended (64-bit addressing)
* `.LTC128B` = L2 cache hint, 128 Bytes cache line alignment
* `.128` = 128-bit transfer (16 bytes per thread)

The instruction starts a global-to-shared transfer without blocking the thread, completion tracked in scoreboard bank 0.

### Commit group: cp.async.commit_group → LDGDEPBAR

```
LDGDEPBAR ;
```

Opcode bytes: `0x00000000000079af` (no operands).

Mnemonic: "Load Global Dependency Barrier". Establishes ordering between previously issued LDGSTS instructions and subsequent DEPBAR.LE waits. Does not block.

### Wait group: cp.async.wait_group N → DEPBAR.LE SB0, N

```
DEPBAR.LE SB0, N   (N in 0..3)
```

Blocks until the number of in-flight cp.async commit groups is at most N.

#### N encoding (decoded from 18c)

Control codes observed:
```
N=0:  0x000080000000791a
N=1:  0x000080400000791a
N=2:  0x000080800000791a
```

Delta in byte 4 (bits 32-39):
* N=0: `0x00`
* N=1: `0x40` (bit 38 set)
* N=2: `0x80` (bit 39 set)

N is encoded as a 2-bit field in bits 38-39:
* `00` = 0
* `01` = 1
* `10` = 2
* `11` = 3 (not tested but predicted)

Max N = 3 (four in-flight groups). Beyond that, one must use `DEPBAR.LE SB0, 0` (wait for all).

### Scoreboard bank 0 for cp.async

LDGSTS completions are tracked in scoreboard bank 0 (SB0). All DEPBAR.LE waits observed target SB0.

Other scoreboard banks may exist for other async operations (TMA on SM90, tcgen05 on SM100) but are not observed on SM120.

## Production pipeline pattern (from 18a, 18b, 18c)

The canonical structure of a pipelined GEMM kernel on SM120:

```
Prologue:
  CS2R R_acc, SRZ                            # initialize accumulator to zero
  compute addresses
  @!PT LDS RZ × 3                            # scheduling hints (role not fully decoded)
  LDGSTS tile[0]
  LDGDEPBAR                                  # commit group 0

Main loop (k = 0 .. K-2):
  compute next tile address
  @!PT LDS RZ × 3
  LDGSTS tile[k+1]
  LDGDEPBAR                                  # commit group k+1
  DEPBAR.LE SB0, 0x1                         # wait until only 1 group in flight (tile k done)
  BAR.SYNC.DEFER_BLOCKING 0x0                # threadblock sync
  LDSM.x2 B tile[k]                          # ptxas emits smaller LDSM first
  LDSM.x4 A tile[k]
  HMMA (accumulate into R_acc)
  UIADD3 NOPs × 2
  counter++, compare, BRA back if not done

Tail (last tile):
  DEPBAR.LE SB0, 0x0                         # wait all
  BAR.SYNC
  LDSM.x2 B tile[K-1]
  LDSM.x4 A tile[K-1]
  HMMA (final, D renamed for STG)
  UIADD3 NOPs × 2

Epilogue:
  STG × N
  EXIT
```

### Observation: ptxas software pipelining is aggressive

In kernel 18a (2-stage fully unrolled), ptxas reordered the SASS so that **LDSM of tile 1 is emitted BEFORE the HMMA of tile 0**:

```
0x0280  LDSM.x2 (tile 1 B)       ← prefetch next tile's fragment
0x0290  LDSM.x4 (tile 1 A)
0x02a0  HMMA tile 0              ← compute current tile
0x02d0  HMMA tile 1 (chain)
```

This hides the LDSM latency (~33 cycles) behind the HMMA (~35 cycles). Optimization that production hand-written kernels try to achieve manually is already done automatically by ptxas when the source makes the dependencies clear.

### Observation: ptxas inverts LDSM emission order (A/B)

Consistently across all variants (17e, 18a, 18b, 18c), ptxas emits `LDSM.x2` (B fragment, smaller) **before** `LDSM.x4` (A fragment, larger). Likely to give the larger load more time to complete before the consuming HMMA needs both.

### Observation: Real K-loop in SASS (from 18b)

With `#pragma unroll 1` in C++, ptxas does emit a real SASS loop rather than unrolling:

```
0x0320  IADD R0, R0, 0x1                     # counter++
0x0330  ISETP.NE.U32.AND P1, PT, R0, 0x3, PT # P1 = (R0 != 3)
...
0x0470  @P1 BRA 0x2b0                        # loop back if P1
```

The loop body contains exactly one prefetch and one compute iteration. 3 iterations through the loop plus a tail gives the full 4 tiles.

### Observation: Double-buffer addressing via bit manipulation

Double-buffered shared memory selection uses `LOP3.LUT` to toggle the buffer parity:

```
0x02b0  LOP3.LUT R20, R0, 0x1, RZ, 0xc0, !PT    # R20 = R0 & 1 (current buffer parity)
0x02d0  LOP3.LUT R9,  R20, 0x1, RZ, 0x3c, !PT    # R9 = R20 XOR 1 (next buffer parity)
```

Pattern seen in all CUTLASS GEMM kernels.

### Observation: Register spill in pipelined GEMM (18b)

Kernel 18b contains `STL.64 [R1], R2` and corresponding `LDL` instructions — register spill to stack. The combination of accumulator + 2 buffer bases × 2 × 4 registers + loop control + LDSM destinations + HMMA sources exceeds the register budget and some addresses are spilled.

Potential optimization: use `__launch_bounds__` to give ptxas a tighter register target, or restructure to reduce live ranges. Not critical for this test kernel.

### Observation: CS2R for accumulator zero-initialization

```
CS2R R_acc, SRZ
```

`CS2R` = "Copy Special Register to Register". With SRZ source, this efficiently zeros a register. Used systematically at the start of GEMM-like kernels to initialize the MMA accumulator before the K-loop.

More efficient than individual `MOV R, RZ` instructions for 4-register clears.

### Observation: HMMA wait mask varies by context

```
17e HMMA (LDSMs directly):           wait mask 0xff
18b HMMA (LDSMs after LDGSTS chain): wait mask 0x04
```

The scoreboard wait mask adapts to which scoreboard slots are occupied at the time the HMMA executes. When the HMMA follows a complex dependency chain through LDGSTS → DEPBAR → LDSM, the active scoreboards are different from a simple LDSM → HMMA pattern.

Byte 0 of the MMA control code is the wait mask. Auditing a kernel, this byte indicates what the MMA is waiting on.

### Unresolved: the `@!PT LDS RZ, [RZ]` pattern

Before every LDGSTS instruction, ptxas emits 3 × `@!PT LDS RZ, [RZ]` — always-false predicated LDS with zero destination and zero source. These are effectively no-ops.

Pattern appears in all three variants. Always in groups of 3. Specific to cp.async context (not observed before cp.async-free LDSMs in chapter 17).

Hypothesis: scheduling slots for ptxas to reserve pipeline resources or align the LDGSTS instruction boundary. Not formally decoded. Marked as [GAP] for later investigation.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| cp.async maps to a new SASS opcode | Confirmed (`LDGSTS.E.LTC128B.128`) |
| commit_group emits a distinct opcode | Confirmed (`LDGDEPBAR`) |
| wait_group N emits DEPBAR.LE | Confirmed (`DEPBAR.LE SB0, N`) |
| N is encoded as a 2-bit field in control code | Confirmed (bits 38-39) |
| ptxas software-pipelines LDSM and MMA | Confirmed (18a shows LDSM next tile before MMA current tile) |
| `#pragma unroll 1` produces a real SASS loop | Confirmed (18b has explicit BRA back-edge) |
| 3-stage pipeline uses DEPBAR.LE N=2, 1, 0 | Confirmed (18c) |
| ptxas always emits LDSM.x2 before LDSM.x4 | Confirmed across 17e, 18a, 18b, 18c |

## Open gaps

| Gap | Notes |
|---|---|
| `@!PT LDS RZ, [RZ]` triplet before LDGSTS | Not decoded. Likely a scheduling hint or pipeline alignment marker specific to cp.async. Observed 3x per LDGSTS group in all variants. |
| Scoreboard banks other than SB0 | Only SB0 observed. SB1, SB2, etc. may exist for other async operations (TMA, tcgen05) but require SM90+/SM100+ features to test. |
| LDGSTS latency model | Not microbenchmarked. LDGSTS latency depends on L2 hit rate and global bandwidth, making a simple cycles-per-instruction model less informative than for MMA. |
| Other cp.async variants | Only `cp.async.ca.shared.global.L2::128B` tested. Other caching hints (cg, no-L2-hint) map to different SASS variants. |
| CUTLASS comparison | The patterns in 18b closely match CUTLASS GEMM mainloop structure. A direct comparison with a CUTLASS SM120 GEMM SASS dump would validate the decoded patterns against production code. |

## Toolkit completion

With chapters 13 (HMMA), 14 (QMMA), 17 (LDSM), and 18 (cp.async pipeline), the repo now has decoded every opcode needed to audit a production GEMM or attention kernel on SM120:

| Pattern | Opcode | Chapter |
|---|---|---|
| Global load | `LDG.E.*` | 08 |
| Shared store | `STS.*` | 06 |
| cp.async | `LDGSTS.E.LTC128B.128` | 18 |
| Commit async | `LDGDEPBAR` | 18 |
| Wait async | `DEPBAR.LE SB0, N` | 18 |
| Barrier | `BAR.SYNC.DEFER_BLOCKING 0x0` | 06 |
| Load matrix | `LDSM.16.M[T]88[.N]` | 17 |
| Tensor core FP16 | `HMMA.16816.F32` | 13 |
| Tensor core FP8/FP6/FP4 | `QMMA.16832.<acc>.<A>.<B>` | 14 |
| Global store | `STG.E.*` | 08 |

Any SM120 GEMM or attention kernel should now be readable end-to-end.

## Summary

* 3 new SASS opcodes: `LDGSTS.E.LTC128B.128`, `LDGDEPBAR`, `DEPBAR.LE SB0, N`
* N argument of DEPBAR.LE decoded as 2-bit field in control code bits 38-39
* Scoreboard bank 0 (SB0) is the async-copy bank on SM120
* ptxas software-pipelines LDSM and MMA automatically when dependencies allow
* ptxas emits a real SASS loop with BRA back-edge when `#pragma unroll 1` is specified
* Production GEMM pattern: prologue → loop (prefetch + compute) → tail → epilogue
* Complete opcode toolkit now in place for SM120 kernel audits
