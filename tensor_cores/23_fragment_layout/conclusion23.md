# Kernel 23 FP4 / FP6 fragment layout

## Narrative

* [OBS] Chapter 23 extends Chapters 14, 15, 16, 17, and 19 by isolating fragment-layout use cases for FP4 and FP6.
* [OBS] The chapter compiles 21 accepted SASS probes and one negative invalid-format probe.
* [OBS] The accepted probes cover dense QMMA dtype layout, lane-pattern inputs, LDSM-fed QMMA, direct-register QMMA, block-scaled scale-vector behavior, sparse metadata separation, register/K boundaries, special bit patterns, and shared-memory alignment.
* [OBS] Runtime smoke execution succeeds on the host GPU for all accepted probes after fixing the LDSM B-fragment shared-memory stride to 16 bytes.
* [GAP] Full lane-to-value layout decode remains open.

## Dense FP4 / FP6 coverage

| Variant | PTX source family | SASS |
|---|---|---|
| 23a | [OBS] E2M1 x E2M1 | [OBS] `QMMA.16832.F32.E2M1.E2M1` |
| 23b | [OBS] E3M2 x E3M2 | [OBS] `QMMA.16832.F32.E3M2.E3M2` |
| 23c | [OBS] E2M3 x E2M3 | [OBS] `QMMA.16832.F32.E2M3.E2M3` |
| 23d | [OBS] E3M2 x E2M3 | [OBS] `QMMA.16832.F32.E3M2.E2M3` |
| 23e | [OBS] E2M3 x E3M2 | [OBS] `QMMA.16832.F32.E2M3.E3M2` |

* [RES] The tested dense FP4/FP6 variants remain in the `QMMA.16832` family.
* [OBS] The visible instruction word for 23a through 23e is `0x0000000e0408727a`; dtype selection is visible in the mnemonic and encoded outside the low opcode byte.

## Layout probes

* [OBS] 23f, 23g, and 23h compile lane-pattern variants for E2M1, E3M2, and E2M3.
* [OBS] 23j compiles `LDSM.16.M88.2` and `LDSM.16.M88.4` before `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23k compiles direct-register `QMMA.16832.F32.E2M1.E2M1` without LDSM.
* [OBS] 23s and 23t compile K-boundary and register-pair-boundary probes.
* [OBS] 23v compiles offset shared-memory LDSM addresses before QMMA.
* [OBS] 23j and 23v both run after the LDSM B-fragment address uses a 16-byte stride; both report first runtime output words `42a80000 42a80000 42a80000 42a80000`.
* [INF] The SASS separates shared-memory fragment loading from register-fragment consumption: LDSM presence is visible only in LDSM-fed probes, while direct-register probes feed QMMA without shared-memory matrix loads.
* [GAP] Runtime outputs are collected, but the full lane-to-value mapping still needs explicit decode.

## Scale and metadata probes

* [OBS] 23i emits `QMMA.SF.16832.F32.E4M3.E4M3.E8`.
* [OBS] 23p emits `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
* [OBS] 23q emits `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8`.
* [RES] Scale factors and sparse metadata remain explicit SASS operands in the tested block-scaled and sparse block-scaled forms.
* [INF] Fragment data, scale factors, and sparse metadata should be modeled as separate audit channels for pattern matching.

## Negative probe

* [OBS] 23m attempts `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.bf16.f32`.
* [OBS] ptxas rejects the probe with `Unexpected instruction types specified for 'mma'`.
* [RES] The tested `kind::f8f6f4` form does not accept the invalid E2M1 x BF16 combination.

## Open gaps

* [GAP] FP4/FP6 lane-to-value mapping remains undecoded from the runtime outputs.
* [GAP] E3M2 and E2M3 exact bit packing within each 32-bit source register remains unresolved without runtime decode.
* [GAP] Special-value interpretation for zero, NaN-like, Inf-like, and sign-bit patterns remains unresolved without runtime decode.
* [GAP] LDSM-fed QMMA correctness for packed FP4/FP6 values remains structural only until outputs are checked on hardware.
