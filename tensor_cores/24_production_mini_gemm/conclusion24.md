# Kernel 24 Production mini-GEMM audit

## Narrative

* [OBS] Chapter 24 extends the isolated tensor-core chapters into production-like mini-GEMM audit structure.
* [OBS] The chapter compiles 30 accepted SASS probes, 24a through 24ad.
* [OBS] Runtime smoke execution succeeds for all 30 variants on the host RTX 5070 Ti.
* [OBS] The probes intentionally include structural baselines: 24i is a store-only epilogue baseline, while 24e/24g/24q can produce zeroed outputs because they validate load and instruction-order structure rather than numeric GEMM correctness.
* [GAP] Full numeric GEMM correctness, production library comparison, and full audit-confidence scoring remain future work.

## End-to-end regions

| Region | Evidence |
|---|---|
| Global to shared | [OBS] 24e, 24f, 24t, and 24x emit `LDGSTS.E.LTC128B.128`. |
| Async dependency wait | [OBS] 24e/24f/24t emit `LDGDEPBAR` and `DEPBAR.LE SB0, 0x0`; 24x emits waits for `SB0,2`, `SB0,1`, and `SB0,0`. |
| Shared matrix load | [OBS] 24e/24f/24g/24q/24t/24x emit `LDSM.16.M88.2` and `LDSM.16.M88.4`. |
| Tensor-core compute | [OBS] Dense variants emit `HMMA`, `QMMA`, or `OMMA`; sparse variants emit `QMMA.SP` or `OMMA.SF.SP`. |
| Shared epilogue | [OBS] 24j and 24k emit `STSM.16.M88.4` before synchronization and global output. |
| Global epilogue | [OBS] Most variants end in `STG.E.128`. |
| Split-K reduction | [OBS] 24z emits `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU`. |
| Cold error path | [OBS] 24ad emits `BSSY.RECONVERGENT`, `BSYNC.RECONVERGENT`, and `BPT.TRAP`. |

## MMA family coverage

* [OBS] 24a emits `HMMA.16816.F32`.
* [OBS] 24b emits `QMMA.16832.F32.E4M3.E4M3`.
* [OBS] 24c emits `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 24d and 24aa emit `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
* [OBS] 24r and 24ab emit `QMMA.SP.16864.F32.E4M3.E4M3`.
* [OBS] 24s emits `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X`.
* [RES] The mini-GEMM audit fixture covers the dense, block-scaled, sparse, and sparse block-scaled tensor-core families needed for first-pass SM120 production SASS recognition.

## Audit implications

* [RES] Production mini-GEMM SASS should be analyzed as a pipeline, not as isolated MMA opcodes: copy, wait, load, compute, epilogue, and reduction each have separate signatures.
* [RES] Scale and metadata operands must be traced from their load path into MMA operands. Chapter 24 confirms both channels survive as visible SASS dependencies in controlled probes.
* [RES] `STSM` is visible in shared-memory epilogues, but Chapter 25 is still needed to isolate lane-to-shared layout and storeback semantics without the noise of the full mini-GEMM fixture.
* [RES] On SM120, these warp-level mini-GEMM probes do not use `WGMMA` or `TCGEN`; those remain out of architectural scope for consumer Blackwell SM120.

## Open gaps

* [GAP] Full numeric GEMM correctness is not claimed by this chapter.
* [GAP] STSM lane-to-shared layout and accumulator storeback semantics remain Chapter 25 scope.
* [GAP] Production library comparison against CUTLASS, FlashAttention, or other real kernels remains Phase 4 scope after the pattern library is formalized.
