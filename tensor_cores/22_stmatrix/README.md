# 22 stmatrix / matrix store

## Scope

* [OBS] This chapter studies SM120 lowering of `stmatrix.sync.aligned` matrix stores and scalar shared-store fallback.
* [OBS] The accepted probes cover 12 variants: 22a through 22l.
* [OBS] The default probes compile with `nvcc -arch=sm_120` and are dumped with `cuobjdump --dump-sass`.
* [OBS] Runtime smoke execution succeeds on the host GPU for normal/transposed x4 STSM and barrier/no-barrier probes.
* [GAP] Full lane-to-shared layout decode remains open.

## Sources

| File | Purpose |
|---|---|
| `22_stmatrix.cu` | [OBS] Parameterized source for variants 22a through 22l and the opt-in unsupported b8 probe. |
| `compile.sh` | [OBS] Rebuilds all accepted binaries and SASS dumps for variants 22a through 22l. |

## Variants studied

| Variant | Source form | Main SASS result |
|---|---|---|
| 22a | `stmatrix.sync.aligned.x1.m8n8.shared.b16` | [OBS] Emits `STSM.16.M88`. |
| 22b | `stmatrix.sync.aligned.x2.m8n8.shared.b16` | [OBS] Emits `STSM.16.M88.2`. |
| 22c | `stmatrix.sync.aligned.x4.m8n8.shared.b16` | [OBS] Emits `STSM.16.M88.4`. |
| 22d | `stmatrix.sync.aligned.x1.trans.m8n8.shared.b16` | [OBS] Emits `STSM.16.MT88`. |
| 22e | `stmatrix.sync.aligned.x2.trans.m8n8.shared.b16` | [OBS] Emits `STSM.16.MT88.2`. |
| 22f | `stmatrix.sync.aligned.x4.trans.m8n8.shared.b16` | [OBS] Emits `STSM.16.MT88.4`. |
| 22g | Four scalar shared stores from the same register inputs | [OBS] Emits one `STS.128`, not STSM. |
| 22h | x4 non-transposed with full 128-bit readback per lane | [OBS] Emits `STSM.16.M88.4`, `BAR.SYNC.DEFER_BLOCKING`, and `LDS.128`. |
| 22i | x4 transposed with full 128-bit readback per lane | [OBS] Emits `STSM.16.MT88.4`, `BAR.SYNC.DEFER_BLOCKING`, and `LDS.128`. |
| 22j | x4 non-transposed plus cross-lane shared read after barrier | [OBS] Emits `STSM.16.M88.4`, `BAR.SYNC.DEFER_BLOCKING`, and `LDS R7, [R6+UR4]`. |
| 22k | x4 non-transposed followed by same-lane shared read without barrier | [OBS] Emits adjacent `STSM.16.M88.4` and `LDS`, with no `BAR.SYNC`. |
| 22l | HMMA-adjacent STSM path | [OBS] Emits `HMMA.16816.F32` followed later by `STSM.16.M88.4`. |

## Commands

```bash
cd tensor_cores/22_stmatrix
bash compile.sh
```

## Unsupported b8 probe

* [OBS] Compiling with `-DTEST_UNSUPPORTED_STMATRIX_B8` attempts the m16n8 b8 forms and fails during ptxas for `sm_120`.
* [OBS] ptxas reports `Feature '.m16n8' not supported on .target 'sm_120'`.
* [OBS] ptxas reports `Feature 'stmatrix.b8' not supported on .target 'sm_120'`.
* [INF] The tested SM120 target supports the m8n8 b16 STSM family but not the tested m16n8 b8 STSM family.

## Key answers

* [RES] `stmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16` is present on SM120 and lowers to `STSM.16.M88[.2|.4]`.
* [RES] `stmatrix.sync.aligned.x{1,2,4}.trans.m8n8.shared.b16` is present on SM120 and lowers to `STSM.16.MT88[.2|.4]`.
* [OBS] The scalar fallback probe emits `STS.128`, which is a different shared-store family from STSM.
* [OBS] The no-barrier probe emits `STSM` followed by `LDS` without an intervening `BAR.SYNC`.
* [OBS] Runtime smoke execution confirms normal and transposed x4 forms launch and produce deterministic first-word outputs on the host GPU.
* [GAP] Full lane-to-shared layout decode remains open.
