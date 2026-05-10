# Chapter 15 — MMA narrow precision on SM120

## Goal

[OBS] Chapter 14 already established the `QMMA.16832.<acc>.<A>.<B>` opcode family for `kind::f8f6f4` at shape m16n8k32, including symmetric FP6 (`E3M2 × E3M2`, `E2M3 × E2M3`), FP4 (`E2M1 × E2M1`), and mixed FP8/FP4 (`E4M3 × E2M1`).

[OBS] Chapter 15 covers the remaining narrow-precision cases around that baseline: mixed FP6 operand order, FP16 accumulation with FP6 inputs, raw FP6/FP4 layout probes, and unscaled FP4 / mixed-FP6 latency-chain SASS.

[INF] The chapter is not a new opcode-family discovery chapter. It is a closure chapter for the narrow end of the already-discovered QMMA family.

## Toolchain note

[OBS] All new Chapter 15 kernels compile with `nvcc -arch=compute_120a -code=sm_120a`.

[OBS] SASS dumps were generated with `cuobjdump --dump-sass`.

[OBS] Runtime validation and `clock64()` latency measurements were not completed in this environment because the local NVIDIA driver was unavailable.

[INF] The SASS opcode and control-code observations are still valid because they come from successful compilation and cuobjdump disassembly, not from kernel execution.

## Variants

| Variant | Inputs | Acc | Purpose | Runtime status |
|---|---|---|---|---|
| 15a | E3M2 × E3M2 | F32 | Symmetric FP6 baseline | [OBS] Covered by 14g |
| 15b | E2M3 × E2M3 | F32 | Alternate symmetric FP6 baseline | [OBS] Covered by 14h |
| 15c | E3M2 × E2M3 | F32 | Missing mixed-FP6 direction | [OBS] SASS only |
| 15d | E2M1 × E2M1 | F32 | Unscaled FP4 k=32 baseline | [OBS] Covered by 14d |
| 15e | E4M3 × E2M1 | F32 | Mixed FP8/FP4 baseline | [OBS] Covered by 14i |
| 15f | E3M2 × E3M2 | F32 | Raw FP6 packing probe | [GAP] Runtime blocked |
| 15g | E2M1 × E2M1 | F32 | Raw FP4 packing probe | [GAP] Runtime blocked |
| 15h | E2M1 × E2M1 | F32 | Unscaled FP4 serial chain | [GAP] Runtime blocked |
| 15i | E3M2 × E2M3 | F32 | Mixed-FP6 serial chain | [GAP] Runtime blocked |
| 15j | E3M2 × E3M2 | F16 | FP16 accumulator with FP6 inputs | [OBS] SASS only |
| 15k | E2M3 × E3M2 | F32 | Reversed mixed-FP6 direction | [OBS] SASS only |

## Key SASS observations

### Mixed FP6 uses the existing QMMA family

[OBS] 15c emits:

```sass
QMMA.16832.F32.E3M2.E2M3 R12, R12, R16, R20
opcode: 0x000000100c0c727a
ctrl:   0x004ff60000246c14
```

[OBS] 15k emits:

```sass
QMMA.16832.F32.E2M3.E3M2 R12, R12, R16, R20
opcode: 0x000000100c0c727a
ctrl:   0x004ff6000018ac14
```

[RES] FP6 does not introduce a distinct SASS opcode in the tested shape. Both mixed FP6 directions use the same QMMA low-byte family (`0x7a`) as FP8 and FP4 `kind::f8f6f4`.

### Control-code dtype model survives both mixed FP6 directions

[OBS] Chapter 14 established the base E4M3/E4M3 F32 control code as `0x004ff60000002c14`.

[INF] For 15c, the chapter 14 model predicts `0x004ff60000246c14`: A=E3M2 sets A mantissa-not-3 and A exp=3 bits; B=E2M3 sets B exp=2. The observed control code is exactly `0x004ff60000246c14`.

[INF] For 15k, the chapter 14 model predicts `0x004ff6000018ac14`: A=E2M3 sets A exp=2; B=E3M2 sets B mantissa-not-3 and B exp=3. The observed control code is exactly `0x004ff6000018ac14`.

[RES] The QMMA input dtype encoding remains orthogonal by operand for the remaining FP6 mixed cases.

### FP16 accumulator encoding carries over to FP6 inputs

[OBS] 15j emits:

```sass
QMMA.16832.F16.E3M2.E3M2 R12, R12, R16, R18
opcode: 0x000000100c0c727a
ctrl:   0x004ff6000014cc12
```

[OBS] 14g emitted F32-accumulator `QMMA.16832.F32.E3M2.E3M2` with control code `0x004ff6000014ec14`.

[INF] The input dtype bits are preserved between 14g and 15j, while the low accumulator-control bits change from F32 form to F16 form.

[RES] The accumulator dtype encoding decoded in chapter 14 is not limited to E4M3 inputs; it also applies to E3M2 inputs in the tested shape.

### Chain forms compile for unscaled FP4 and mixed FP6

[OBS] 15h N=16/32/64 dumps contain chained `QMMA.16832.F32.E2M1.E2M1 R16, R4, R2, R16`.

[OBS] The first 15h chain QMMA has control code `0x084ff6000028ec10`.

[OBS] 15i N=16/32/64 dumps contain chained `QMMA.16832.F32.E3M2.E2M3 R16, R4, R2, R16`.

[OBS] The first 15i chain QMMA has control code `0x084ff60000246c10`.

[INF] The chain-control prefix `0x084ff6...` matches the chapter 14 QMMA chain-start pattern: bit 26 sets the MMA scoreboard and bit 27 waits in the chain context.

[GAP] The latency values for 15h and 15i are not measured because the runtime hardware path was unavailable.

### Raw layout probes are scaffolded, not resolved

[OBS] 15f compiles a raw-pattern FP6 probe and emits `QMMA.16832.F32.E3M2.E3M2` with control code `0x004ff6000014ec14`.

[OBS] 15g compiles a raw-nibble FP4 probe and emits `QMMA.16832.F32.E2M1.E2M1` with control code `0x004ff6000028ec14`.

[GAP] 15f and 15g do not resolve FP6 or FP4 fragment packing without runtime output.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| FP6 at m16n8k32 uses a distinct opcode from FP8 QMMA | [RES] Rejected. Mixed FP6 uses QMMA low byte `0x7a`. |
| Mixed FP6 uses the same per-operand dtype encoding model as chapter 14 | [RES] Confirmed by 15c and 15k. |
| Reversing E3M2/E2M3 mirrors A/B dtype control bits | [RES] Confirmed by 15k control code. |
| FP16 accumulator encoding is independent of narrow input dtype | [RES] Confirmed for E3M2 × E3M2 in 15j. |

## Open gaps

| Gap | Status |
|---|---|
| FP6 fragment value packing | [GAP] 15f source and SASS exist; runtime probe data still needed. |
| FP4 E2M1 fragment layout | [GAP] 15g source and SASS exist; runtime probe data still needed. |
| Unscaled FP4 QMMA latency | [GAP] 15h N=16/32/64 SASS exists; hardware timing still needed. |
| Mixed FP6 QMMA latency | [GAP] 15i N=16/32/64 SASS exists; hardware timing still needed. |
| k=32 FP4 throughput gap versus k=64 OMMA | [GAP] Structural difference is documented as QMMA.16832 vs OMMA.16864, but unscaled E2M1 timing is still missing. |

## Cross-references to FINDINGS.md

[OBS] Updated `FINDINGS.md` section `## Kernel 15 MMA narrow (FP6, FP4 at k=32)`.

[OBS] Cross-referenced chapter 14 findings for 14d, 14g, 14h, 14i, and 14j.

[OBS] Preserved GAP-14d-1 as unresolved because Chapter 15 did not produce runtime FP4 layout evidence.

## Summary

[OBS] Chapter 15 adds no new MMA opcode family.

[RES] Chapter 15 completes the SASS encoding matrix for the tested non-scaled narrow QMMA cases at m16n8k32.

[RES] The chapter 14 3-bit-per-operand dtype encoding model survives both remaining mixed-FP6 directions.

[RES] The chapter 14 accumulator dtype encoding carries over to FP6 inputs.

[GAP] The remaining Chapter 15 work is hardware-runtime work: FP6/FP4 value-layout probes and latency measurements for unscaled E2M1 and mixed FP6 chains.
