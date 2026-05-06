# 23 FP4 / FP6 fragment layout

## Scope

* [OBS] This chapter studies FP4 / FP6 fragment-layout risk areas through 22 controlled probes, 23a through 23v.
* [OBS] The accepted probes compile with `nvcc -arch=compute_120a -code=sm_120a` and are dumped with `cuobjdump --dump-sass`.
* [OBS] The chapter covers dense QMMA, block-scaled QMMA/OMMA, sparse block-scaled QMMA, LDSM-fed QMMA, direct-register QMMA, boundary patterns, special bit patterns, and shared-memory alignment.
* [OBS] Runtime smoke execution succeeds on the host GPU for all accepted probes after fixing the LDSM B-fragment shared-memory stride to 16 bytes.
* [GAP] Full lane-to-value layout decode remains open.

## Sources

| File | Purpose |
|---|---|
| `23_fragment_layout.cu` | [OBS] Parameterized source for variants 23a through 23v and the opt-in invalid-format probe. |
| `compile.sh` | [OBS] Rebuilds all accepted SASS dumps and captures the negative 23m ptxas log. |

## Variants studied

| Variant | Probe | Main SASS result |
|---|---|---|
| 23a | E2M1 x E2M1 baseline | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23b | E3M2 x E3M2 baseline | [OBS] `QMMA.16832.F32.E3M2.E3M2`. |
| 23c | E2M3 x E2M3 baseline | [OBS] `QMMA.16832.F32.E2M3.E2M3`. |
| 23d | E3M2 x E2M3 mixed baseline | [OBS] `QMMA.16832.F32.E3M2.E2M3`. |
| 23e | E2M3 x E3M2 mixed baseline | [OBS] `QMMA.16832.F32.E2M3.E3M2`. |
| 23f | E2M1 lane-pattern probe | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23g | E3M2 lane-pattern probe | [OBS] `QMMA.16832.F32.E3M2.E3M2`. |
| 23h | E2M3 lane-pattern probe | [OBS] `QMMA.16832.F32.E2M3.E2M3`. |
| 23i | Scale-factor interaction | [OBS] `QMMA.SF.16832.F32.E4M3.E4M3.E8`. |
| 23j | LDSM-to-QMMA path | [OBS] `LDSM.16.M88.2`, `LDSM.16.M88.4`, then `QMMA.16832.F32.E2M1.E2M1`. |
| 23k | Direct-register QMMA path | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23l | Runtime decode probe | [OBS] Compiles as `QMMA.16832.F32.E2M1.E2M1`; first runtime output words are `c1d80000 c1d80000 c3220000 c3220000`. |
| 23m | Invalid-format negative probe | [OBS] ptxas rejects `kind::f8f6f4` with `bf16` input. |
| 23n | Chapter 14/15 cross-reference | [OBS] `QMMA.16832.F32.E3M2.E2M3`. |
| 23o | Unsigned/signed interpretation probe | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23p | Scale-vector layout probe | [OBS] `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. |
| 23q | Metadata independence probe | [OBS] `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8`. |
| 23r | Operand-order layout probe | [OBS] `QMMA.16832.F32.E3M2.E2M3`. |
| 23s | K-tile boundary probe | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23t | Register-pair boundary probe | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23u | Zero / NaN / Inf bit-pattern probe | [OBS] `QMMA.16832.F32.E2M1.E2M1`. |
| 23v | Shared-memory alignment probe | [OBS] Offset `LDSM.16.M88.2`, offset `LDSM.16.M88.4`, then `QMMA.16832.F32.E2M1.E2M1`. |

## Commands

```bash
cd tensor_cores/23_fragment_layout
bash compile.sh
```

## Key answers

* [RES] The tested dense FP4/FP6 baseline and mixed forms stay in the `QMMA.16832` family and expose the A/B dtypes directly in the mnemonic.
* [RES] LDSM-fed FP4 QMMA compiles structurally as `LDSM.16.M88.*` followed by `QMMA.16832.F32.E2M1.E2M1`.
* [RES] Direct-register FP4 QMMA compiles without LDSM, separating register-fragment construction from shared-memory fragment loading.
* [OBS] The scale-vector probe emits `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
* [OBS] The metadata-independence probe emits `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8`.
* [OBS] Runtime smoke execution confirms the accepted probes launch and produce deterministic first-word outputs on the host GPU.
* [GAP] Lane-to-value fragment layout for FP4/FP6 remains a decode task over the captured runtime outputs.
