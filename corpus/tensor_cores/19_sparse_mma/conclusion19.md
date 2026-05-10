# Chapter 19 - Sparse MMA

## Goal

* [OBS] This chapter establishes the SASS form of warp-level `mma.sp::ordered_metadata` on SM120.
* [OBS] The chapter tests sparse non-scaled `kind::f8f6f4`, sparse block-scaled `kind::mxf8f6f4`, sparse block-scaled `kind::mxf4nvf4`, metadata value changes, selector rejection, F16 accumulator output, and a sparse dependency chain.
* [INF] The chapter is the sparse counterpart to chapters 14 and 16: chapter 14 establishes dense `QMMA.16832`, chapter 16 establishes dense block-scaled `QMMA.SF.16832` and `OMMA.SF.16864`, and chapter 19 shows the corresponding sparse forms.
* [GAP] Runtime numeric validation and latency timing are not established in this chapter because the local NVIDIA driver is unavailable.

## Variants

| Variant | Input form | Purpose | SASS result |
|---|---|---|---|
| 19a | [OBS] `kind::f8f6f4`, E4M3 x E4M3, F32 accumulator, metadata `0xaaaaaaaa`, selector `0` | [OBS] Baseline sparse FP8 path | [OBS] `QMMA.SP.16864.F32.E4M3.E4M3` |
| 19b | [OBS] `kind::f8f6f4`, E4M3 x E5M2, F32 accumulator | [OBS] B dtype variation | [OBS] `QMMA.SP.16864.F32.E4M3.E5M2` |
| 19c | [OBS] `kind::f8f6f4`, E3M2 x E2M3, F32 accumulator | [OBS] mixed FP6 dtype variation | [OBS] `QMMA.SP.16864.F32.E3M2.E2M3` |
| 19d | [OBS] `kind::f8f6f4`, E2M1 x E2M1, F32 accumulator | [OBS] sparse FP4 non-scaled path | [OBS] `QMMA.SP.16864.F32.E2M1.E2M1` |
| 19e | [OBS] 19a with metadata `0x55555555` | [OBS] metadata encoding variation | [OBS] same `QMMA.SP` encoding as 19a |
| 19f | [OBS] 19a with metadata `0xffffffff` | [OBS] metadata encoding variation | [OBS] same `QMMA.SP` encoding as 19a |
| 19g | [OBS] 19a with selector `1` | [OBS] selector admissibility test | [OBS] ptxas rejects the variant |
| 19h | [OBS] `kind::f8f6f4`, E3M2 x E2M1, F16 accumulator | [OBS] accumulator dtype variation | [OBS] `QMMA.SP.16864.F16.E3M2.E2M1` |
| 19i | [OBS] `kind::mxf8f6f4`, E3M2 x E2M1, ue8m0 scales | [OBS] sparse block-scaled QMMA path | [OBS] `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8` |
| 19j | [OBS] `kind::mxf4nvf4`, E2M1 x E2M1, 2X ue8m0 scales | [OBS] sparse block-scaled OMMA default path | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8` |
| 19k | [OBS] `kind::mxf4nvf4`, E2M1 x E2M1, 4X ue4m3 scales | [OBS] sparse FP4 fine-scale path | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` |
| 19l | [OBS] `kind::mxf4nvf4`, E2M1 x E2M1, 4X ue8m0 scales | [OBS] scale-mode disambiguation variant | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8.4X` |
| 19m | [OBS] 16 chained sparse E4M3 x E4M3 QMMAs | [OBS] dependency-chain scheduling probe | [OBS] 16 x `QMMA.SP.16864.F32.E4M3.E4M3` |

## Key SASS observations

### Sparse QMMA opcode

```sass
QMMA.SP.16864.F32.E4M3.E4M3 R4, R4, R16, R20, R0, 0x0
opcode: 0x000000100404727a
ctrl:   0x004ff60000013414
```

* [OBS] 19a emits `QMMA.SP.16864.F32.E4M3.E4M3`, not a new SASS opcode family.
* [OBS] The sparse non-scaled opcode low byte is `0x7a`, the same low byte used by dense `QMMA` in chapters 14 and 15.
* [OBS] The sparse mnemonic adds `.SP` and changes the displayed shape from dense `.16832` to sparse `.16864`.
* [INF] Sparse non-scaled `kind::f8f6f4` stays in the QMMA family because the opcode low byte remains `0x7a`.
* [INF] The displayed K dimension doubles for the sparse form because dense `kind::f8f6f4` was observed as `QMMA.16832` and sparse `kind::f8f6f4` is observed as `QMMA.SP.16864`.

### Metadata operand

```sass
MOV R0, 0xaaaaaaaa
QMMA.SP.16864.F32.E4M3.E4M3 R4, R4, R16, R20, R0, 0x0
```

* [OBS] 19a materializes metadata with `MOV R0, 0xaaaaaaaa`.
* [OBS] 19a consumes `R0` as the fifth SASS operand of `QMMA.SP`.
* [OBS] 19e changes the metadata producer to `MOV R0, 0x55555555` and leaves the `QMMA.SP` opcode bytes and control code unchanged.
* [OBS] 19f changes the metadata producer to `MOV R0, 0xffffffff` and leaves the `QMMA.SP` opcode bytes and control code unchanged.
* [INF] Metadata value is a runtime register operand, not an opcode or control-code field, because changing the metadata value changes only the producer `MOV`.
* [GAP] The semantic validity of each metadata bit pattern is not established because the kernels were not executed on hardware.

### Selector operand

* [OBS] The accepted sparse non-scaled SASS form prints selector `0x0` as the final immediate operand.
* [OBS] 19g with selector `1` is rejected by ptxas with `Argument 5 of instruction 'mma': unexpected value '1', expected to be 0`.
* [INF] Selector `0` is the only accepted selector for the tested `kind::f8f6f4` warp-level form because ptxas accepts selector `0` and rejects selector `1` in the controlled variant.
* [GAP] Selector semantics outside this tested warp-level `kind::f8f6f4` form remain unclassified.

### Dtype encoding

| Variant | SASS mnemonic | Opcode bytes | Control code |
|---|---|---|---|
| 19a | [OBS] `QMMA.SP.16864.F32.E4M3.E4M3` | [OBS] `0x000000100404727a` | [OBS] `0x004ff60000013414` |
| 19b | [OBS] `QMMA.SP.16864.F32.E4M3.E5M2` | [OBS] `0x000000100404727a` | [OBS] `0x004ff6000001b414` |
| 19c | [OBS] `QMMA.SP.16864.F32.E3M2.E2M3` | [OBS] `0x000000100404727a` | [OBS] `0x004ff60000257414` |
| 19d | [OBS] `QMMA.SP.16864.F32.E2M1.E2M1` | [OBS] `0x000000100404727a` | [OBS] `0x004ff6000029f414` |

* [OBS] 19a through 19d keep identical opcode bytes while dtype suffixes and control codes change.
* [INF] Sparse non-scaled dtype selection remains in the control code, matching the dense QMMA rule from chapter 14.
* [OBS] 19h emits `QMMA.SP.16864.F16.E3M2.E2M1` with control code `0x004ff6000025d414`.
* [OBS] 19h has 40 total instructions, 8 fewer than the F32 accumulator sparse non-scaled probes.
* [INF] The F16 accumulator variant reduces instruction count because C and D fragments use 2 registers each instead of 4, matching the accumulator-width pattern from dense QMMA in chapter 14.

### Sparse block-scaled QMMA

```sass
QMMA.SF.SP.16864.F32.E3M2.E2M1.E8 R4, R4, R8, R20, R18, R0, URZ, 0x0
opcode: 0x700012080404747a
ctrl:   0x004ff6000025fe14
```

* [OBS] 19i emits `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8`.
* [OBS] 19i keeps the QMMA low byte `0x7a`.
* [OBS] 19i materializes scale value `0x7f` in `R0` and metadata `0xaaaaaaaa` in `R18`.
* [OBS] 19i SASS operand order is D base, A base, B base, C base, metadata register, scale register, `URZ`, selector immediate.
* [INF] Sparse `kind::mxf8f6f4` composes the chapter 16 `.SF` scaling modifier with the chapter 19 `.SP` sparse modifier on the existing QMMA family.

### Sparse block-scaled OMMA

| Variant | SASS mnemonic | Opcode bytes | Control code |
|---|---|---|---|
| 19j | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8` | [OBS] `0x700012080404747f` | [OBS] `0x004ff60000093e14` |
| 19k | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` | [OBS] `0x700012080404747f` | [OBS] `0x004ff60000053e14` |
| 19l | [OBS] `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8.4X` | [OBS] `0x700012080404747f` | [OBS] `0x004ff60000013e14` |

* [OBS] 19j through 19l keep the OMMA low byte `0x7f`.
* [OBS] 19j through 19l keep identical opcode bytes while scale suffixes and control codes change.
* [OBS] Sparse OMMA displays `.168128`, while dense OMMA in chapter 16 displayed `.16864`.
* [INF] Sparse `kind::mxf4nvf4` keeps the chapter 16 OMMA family and doubles the displayed K field.
* [INF] Sparse OMMA scale mode remains in the control code, because `2X ue8m0`, `4X ue4m3`, and `4X ue8m0` keep identical opcode bytes while control codes differ.
* [INF] 19l partially constrains chapter 16's scale-mode gap: `scale_vec::4X` can combine with ue8m0 on the sparse `mxf4nvf4` path, but the dense `mxf4nvf4` 4X ue8m0 path remains untested.

### Chain scheduling

```sass
QMMA.SP.16864.F32.E4M3.E4M3 R12, R4, R8.reuse, RZ,  R0, 0x0
QMMA.SP.16864.F32.E4M3.E4M3 R12, R4, R8.reuse, R12, R0, 0x0
...
QMMA.SP.16864.F32.E4M3.E4M3 R12, R4, R8,       R12, R0, 0x0
```

* [OBS] 19m emits 16 `QMMA.SP.16864.F32.E4M3.E4M3` instructions.
* [OBS] 19m has 88 total instructions.
* [OBS] The first sparse chain instruction uses C=`RZ`, `.reuse` on B, and control code `0x084ff600000134ff`.
* [OBS] The middle sparse chain instructions use C=`R12`, `.reuse` on B, and control code `0x080ff6000001340c`.
* [OBS] The last sparse chain instruction uses C=`R12`, no `.reuse` on B, and control code `0x000fe2000001340c`.
* [INF] Sparse QMMA chain scheduling follows dense QMMA chain scheduling from chapter 14 because B reuse, D/C accumulator colocation, and chain-position control-code changes match the same pattern.
* [GAP] Sparse chain latency is not measured because runtime timing is blocked by the unavailable NVIDIA driver.

## Resolved hypotheses

| Hypothesis | Resolution |
|---|---|
| [HYP] Sparse non-scaled MMA may be a new opcode family. | [RES] Rejected for tested SM120 `kind::f8f6f4`; it emits `QMMA.SP` with low byte `0x7a`. |
| [HYP] Sparse block-scaled `mxf8f6f4` may be a new opcode family. | [RES] Rejected for tested SM120 form; it emits `QMMA.SF.SP` with low byte `0x7a`. |
| [HYP] Sparse block-scaled `mxf4nvf4` may be a new sparse-only opcode family. | [RES] Rejected for tested SM120 form; it emits `OMMA.SF.SP` with low byte `0x7f`. |
| [HYP] Sparsity metadata might be encoded inside the A fragment or opcode bits. | [RES] Rejected at SASS operand level; metadata is an explicit register operand. |
| [HYP] Sparse dtype changes might alter opcode bytes. | [RES] Rejected for tested non-scaled variants; dtype changes alter control code and mnemonic suffix, not opcode bytes. |

## Open gaps

| Gap | Notes |
|---|---|
| [GAP] Runtime metadata semantics | [GAP] SASS proves metadata is a register operand, but it does not prove which metadata constants represent valid 2:4 layouts. |
| [GAP] Sparse latency | [GAP] 19m compiles the intended dependency chain, but the local driver prevents `clock64()` measurement. |
| [GAP] `.SP` bit placement | [GAP] The mnemonic exposes sparse mode, but exact bit placement inside opcode or control fields is not fully decoded. |
| [GAP] Selector semantics | [GAP] Selector `0` compiles and selector `1` rejects for the tested form; broader selector behavior remains unclassified. |
| [GAP] Dense `mxf4nvf4` 4X ue8m0 | [GAP] Sparse 4X ue8m0 compiles in 19l; dense 4X ue8m0 remains untested. |

## How to read sparse MMA in production

* [OBS] `QMMA.SP.16864` identifies sparse non-scaled `kind::f8f6f4` on SM120.
* [OBS] `QMMA.SF.SP.16864` identifies sparse block-scaled `kind::mxf8f6f4` on SM120.
* [OBS] `OMMA.SF.SP.168128` identifies sparse block-scaled `kind::mxf4nvf4` on SM120.
* [OBS] The operand immediately before the selector immediate is the sparsity metadata register in non-scaled sparse QMMA.
* [OBS] In sparse block-scaled QMMA and OMMA, the operand order after C is metadata register, scale register, `URZ`, selector immediate.
* [INF] Sparse K-doubling can be recognized directly from the mnemonic: dense `QMMA.16832` becomes sparse `QMMA.SP.16864`, and dense `OMMA.SF.16864` becomes sparse `OMMA.SF.SP.168128`.

## Cross-references to FINDINGS.md

* [RES] Kernel 19 in `FINDINGS.md` records the sparse opcode-family resolution.
* [RES] Kernel 19 in `FINDINGS.md` records the metadata operand resolution.
* [OBS] Kernel 19 in `FINDINGS.md` records control codes for 19a through 19m.
* [INF] Kernel 19 in `FINDINGS.md` extends the chapter 14 dense QMMA dtype-control-code model to sparse non-scaled QMMA.
* [INF] Kernel 19 in `FINDINGS.md` extends the chapter 16 block-scaled OMMA model to sparse OMMA.

## What follows

* [INF] Kernel 20 should move to control flow because Kernel 19 resolves the planned sparse opcode and metadata questions at SASS level.
* [GAP] A later hardware run should measure sparse chain latency for `QMMA.SP` and `OMMA.SF.SP` once the NVIDIA driver is available.
