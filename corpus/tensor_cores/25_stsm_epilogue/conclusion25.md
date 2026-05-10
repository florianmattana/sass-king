# Kernel 25 STSM epilogue layout and storeback semantics

## Narrative

* [OBS] Chapter 25 closes the first-pass STSM epilogue/storeback gap left after Chapters 22 and 24.
* [OBS] The chapter compiles 25 accepted executable SASS probes and one b8 compatibility probe with both a plain-`sm_120` rejection log and an `sm_120a` SASS dump.
* [OBS] Runtime smoke execution succeeds for all accepted executable probes on the host RTX 5070 Ti.
* [OBS] The accepted probes cover raw STSM layout, transposed layout, MMA accumulator paths, narrowing conversions, shared reload, global storeback, barrier/no-barrier contrast, and register-pressure behavior.
* [GAP] Full lane-to-value semantic decode remains open; the runtime words are captured for that follow-up work.

## Storeback patterns

| Pattern | Evidence |
|---|---|
| STSM layout baseline | [OBS] 25a-25f emit all tested `STSM.16.M88` and `STSM.16.MT88` width forms. |
| Shared reload after STSM | [OBS] Most accepted probes emit `BAR.SYNC.DEFER_BLOCKING` followed by `LDS.128`. |
| Global storeback | [OBS] Accepted storeback probes emit scalar `STG.E` stores after shared reload. |
| STS fallback | [OBS] 25j emits `STS.128`, not STSM. |
| No-barrier contrast | [OBS] 25n emits adjacent `STSM` and `LDS` without an intervening `BAR`. |
| Split accumulator | [OBS] 25o emits two `STSM.16.M88.4` stores and two `LDS.128` reloads. |
| Noncontiguous global stride | [OBS] 25w emits stores at offsets 0, 8, 16, and 24 bytes. |
| Register pressure | [OBS] 25y grows to 232 useful instructions and still preserves `STSM -> BAR -> LDS -> STG`. |

## Accumulator paths

* [OBS] 25g emits `HMMA.16816.F32` before `STSM.16.M88.4`.
* [OBS] 25h emits `QMMA.16832.F32.E2M1.E2M1` before `STSM.16.M88.4`.
* [OBS] 25u emits `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` before `STSM.16.M88.4`.
* [OBS] 25v emits `QMMA.SP.16864.F32.E4M3.E4M3` before `STSM.16.M88.4`.
* [RES] The tested dense, block-scaled, and sparse accumulator paths all preserve a visible STSM epilogue.

## Narrowing conversions

* [OBS] 25s emits `F2F.F16.F32` conversions before STSM.
* [OBS] 25t emits `F2F.BF16.F32` conversions before STSM.
* [RES] Accumulator narrowing is separable from matrix-store recognition: the conversion instructions appear before the `STSM` family.

## Architecture-target distinction

* [OBS] The `25q` compatibility probe compiles with plain `sm_120` and captures ptxas rejection of `.m16n8` and `stmatrix.b8`.
* [OBS] The same m16n8 b8 PTX forms compile for `sm_120a` and lower to `STSM.8.MT168`, `STSM.8.MT168.2`, and `STSM.8.MT168.4`.
* [RES] Claims about b8 STSM support must name the target: rejected for tested plain `sm_120`, accepted for tested `sm_120a`.

## Open gaps

* [GAP] Full lane-to-value STSM decode remains open over the captured runtime output words.
* [GAP] STSM latency and scoreboard/control-code decoding remain open.
* [GAP] The separate audit confidence framework remains open before Phase 3 pattern formalization.
