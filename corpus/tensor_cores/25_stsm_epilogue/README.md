# 25 STSM epilogue layout and storeback semantics

## Scope

* [OBS] This chapter studies STSM epilogue and storeback behavior through 26 probes, 25a through 25z.
* [OBS] The accepted probes compile with `nvcc -arch=compute_120a -code=sm_120a` and are dumped with `cuobjdump --dump-sass`.
* [OBS] The chapter covers x1/x2/x4 STSM layout, transposed STSM layout, MMA-to-STSM epilogues, STSM-to-STG storeback, STS fallback, predicated tails, alignment, barrier visibility, split accumulator storeback, FP32-to-F16/BF16 packing, OMMA, sparse QMMA, noncontiguous global stride, shared stride, register pressure, and runtime decode tables.
* [OBS] Runtime smoke execution succeeds on the host RTX 5070 Ti for all 25 accepted executable probes.
* [OBS] The `25q` compatibility probe confirms plain `sm_120` rejects the tested m16n8 b8 STSM forms, while `sm_120a` lowers them to `STSM.8.MT168`.

## Sources

| File | Purpose |
|---|---|
| `25_stsm_epilogue.cu` | [OBS] Parameterized source for variants 25a through 25z and the opt-in b8 compatibility probe. |
| `compile.sh` | [OBS] Rebuilds all accepted SASS dumps and captures the negative `25q` ptxas log. |

## Variants studied

| Variant | Probe | Main SASS result |
|---|---|---|
| 25a | x1 runtime layout | [OBS] `STSM.16.M88`, `BAR`, `LDS.128`, `STG`. |
| 25b | x2 runtime layout | [OBS] `STSM.16.M88.2`, `BAR`, `LDS.128`, `STG`. |
| 25c | x4 runtime layout | [OBS] `STSM.16.M88.4`, `BAR`, `LDS.128`, `STG`. |
| 25d | x1 transposed runtime layout | [OBS] `STSM.16.MT88`, `BAR`, `LDS.128`, `STG`. |
| 25e | x2 transposed runtime layout | [OBS] `STSM.16.MT88.2`, `BAR`, `LDS.128`, `STG`. |
| 25f | x4 transposed runtime layout | [OBS] `STSM.16.MT88.4`, `BAR`, `LDS.128`, `STG`. |
| 25g | HMMA to STSM epilogue | [OBS] `HMMA.16816.F32`, `STSM.16.M88.4`, `BAR`, `LDS.128`, `STG`. |
| 25h | QMMA to STSM epilogue | [OBS] `QMMA.16832.F32.E2M1.E2M1`, `STSM.16.M88.4`, `STG`. |
| 25i | STSM to STG storeback | [OBS] HMMA, STSM, shared reload, global storeback. |
| 25j | STS fallback comparison | [OBS] `STS.128`, not STSM. |
| 25k | Predicated tail storeback | [OBS] HMMA, STSM, BAR, predicated global storeback. |
| 25l | Alignment offset storeback | [OBS] Offset `STSM.16.M88.4 [R+0x20]` and offset `LDS.128`. |
| 25m | Barrier required visibility | [OBS] HMMA, STSM, `BAR.SYNC.DEFER_BLOCKING`, `LDS.128`. |
| 25n | No-barrier same-thread visibility | [OBS] HMMA, STSM, immediate `LDS` without `BAR`. |
| 25o | Split accumulator storeback | [OBS] Two `STSM.16.M88.4` stores, two `LDS.128` reloads, eight global stores. |
| 25p | Runtime decode table | [OBS] STSM, BAR, two `LDS.128` reloads, eight global stores. |
| 25q | b8 compatibility | [OBS] `sm_120` ptxas rejects `.m16n8` and `stmatrix.b8`; `sm_120a` emits `STSM.8.MT168[.2|.4]`. |
| 25r | Full epilogue checklist | [OBS] HMMA, STSM, BAR, LDS, STG. |
| 25s | FP32 accumulator to F16 pack | [OBS] `F2F.F16.F32` conversions before STSM. |
| 25t | FP32 accumulator to BF16 pack | [OBS] `F2F.BF16.F32` conversions before STSM. |
| 25u | OMMA to STSM epilogue | [OBS] `OMMA.SF.16864...UE4M3.4X`, STSM, STG. |
| 25v | Sparse QMMA to STSM epilogue | [OBS] `QMMA.SP.16864...`, STSM, STG. |
| 25w | Noncontiguous global stride | [OBS] STG offsets stride by 8 bytes between stored words. |
| 25x | Shared bank stride variant | [OBS] Strided shared addressing still emits `STSM.16.M88.4`. |
| 25y | Register-pressure epilogue | [OBS] 232 useful instructions; STSM/storeback survives after pressure arithmetic. |
| 25z | Full runtime layout decode | [OBS] HMMA, STSM, BAR, two `LDS.128` reloads, eight global stores. |

## Commands

```bash
cd corpus/tensor_cores/25_stsm_epilogue
bash compile.sh
for exe in build/25*; do "$exe"; done
```

`25q` is a negative compile log, not a runtime executable.

## Key answers

* [RES] The tested STSM epilogue path is structurally visible as `STSM -> BAR -> LDS -> STG` for cross-thread storeback.
* [RES] Same-thread no-barrier readback can compile as adjacent `STSM` and `LDS`, but this does not prove cross-thread visibility is safe without a barrier.
* [RES] STS fallback remains a separate shared-store family (`STS.128`) from matrix-store STSM.
* [RES] FP32 accumulator narrowing before STSM is visible as `F2F.F16.F32` or `F2F.BF16.F32`.
* [RES] HMMA, QMMA, OMMA, and sparse QMMA accumulator paths can feed STSM epilogues in the tested fixtures.
* [RES] Plain `sm_120` rejects the tested m16n8 b8 STSM forms; `sm_120a` lowers those forms to `STSM.8.MT168`, so b8 support is architecture-target-qualified.
* [GAP] The probes print runtime words for layout decode, but a full lane-to-value semantic table remains a follow-up decode task.
