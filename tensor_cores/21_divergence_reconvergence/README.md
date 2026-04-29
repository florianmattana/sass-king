# 21 Divergence and reconvergence

## Scope

* [OBS] This chapter studies SM120 lowering of lane-divergent branches, predication, BSSY/BSYNC reconvergence scopes, warp votes, early return, barrier-after-divergence, divergent memory, guarded MMA, lane-dependent trip counts, masked stores, divergent calls, and cold trap paths.
* [OBS] The planned probes cover 20 variants: 21a through 21t.
* [OBS] The probes compile with `nvcc -arch=sm_120` and are dumped with `cuobjdump --dump-sass`.
* [GAP] Runtime numeric validation is blocked in this environment until the NVIDIA driver is available.

## Sources

| File | Purpose |
|---|---|
| `21_divergence_reconvergence.cu` | [OBS] Parameterized source for variants 21a through 21t. |
| `compile.sh` | [OBS] Rebuilds all binaries and SASS dumps for variants 21a through 21t. |

## Variants studied

| Variant | Source form | Main SASS result |
|---|---|---|
| 21a | Simple lane-dependent short `if` | [OBS] Predicated FADD. No BSSY/BSYNC. |
| 21b | Uniform branch | [OBS] Uses `UFSEL`. No BSSY/BSYNC. |
| 21c | Lane-divergent `if` | [OBS] Emits one BSSY/BSYNC pair around the divergent arithmetic region. |
| 21d | Lane-divergent `if/else` | [OBS] Uses predicated arithmetic on both paths. No BSSY/BSYNC. |
| 21e | Nested divergence | [OBS] Emits one BSSY/BSYNC pair and forward branches. |
| 21f | Lane-dependent `break` | [OBS] Emits one BSSY/BSYNC pair and one useful back-edge. |
| 21g | Lane-dependent `continue` | [OBS] Emits one useful back-edge and no BSSY/BSYNC. |
| 21h | Lane-dependent early return | [OBS] Emits an additional predicated EXIT and no BSSY/BSYNC. |
| 21i | Divergence then `__syncthreads()` | [OBS] Uses FSEL plus `BAR.SYNC.DEFER_BLOCKING`. No BSSY/BSYNC. |
| 21j | `__ballot_sync` around divergence | [OBS] Emits `VOTE.ANY R5, PT, P0`. No BSSY/BSYNC. |
| 21k | Select vs branch vs mask | [OBS] Emits `FSEL` for select forms. No BSSY/BSYNC. |
| 21l | Short body vs long body | [OBS] Emits one BSSY/BSYNC pair around the long divergent body. |
| 21m | Divergent memory paths | [OBS] Emits forward branch plus separate path exits. No BSSY/BSYNC. |
| 21n | Guarded HMMA | [OBS] Emits predicated loads, `@P0 WARPSYNC.ALL`, and `@P0 HMMA.16816.F32`. No BSSY/BSYNC. |
| 21o | Lane-dependent trip count | [OBS] Emits one BSSY/BSYNC pair and one useful back-edge. |
| 21p | Bounds-check epilogue | [OBS] Emits additional predicated EXIT. No BSSY/BSYNC. |
| 21q | Masked store tail | [OBS] Emits predicated LDC/IMAD/STG and an additional predicated EXIT. No BSSY/BSYNC. |
| 21r | Divergent noinline call | [OBS] Emits BSSY/BSYNC around a local `CALL.REL.NOINC` plus `RET.REL.NODEC`. |
| 21s | Cold trap/error path | [OBS] Emits `BPT.TRAP 0x1` behind a predicated branch. No BSSY/BSYNC. |
| 21t | Vote-converged branch | [OBS] Emits `VOTE.ANY P0, P0`, predicated branch, and `SHFL.IDX`. No BSSY/BSYNC. |

## Commands

```bash
cd tensor_cores/21_divergence_reconvergence
bash compile.sh
```

## Key answers

* [OBS] Across the 20 dumps, BSSY/BSYNC appears in 21c, 21e, 21f, 21l, 21o, and 21r.
* [OBS] Lane divergence alone is not sufficient to force BSSY/BSYNC in every tested case: 21a, 21d, 21h, 21i, 21j, 21k, 21m, 21p, 21q, 21s, and 21t avoid it.
* [OBS] ptxas uses predicated arithmetic, `FSEL`, `UFSEL`, predicated EXIT, or forward branches for many divergent shapes instead of explicit reconvergence scopes.
* [OBS] The guarded HMMA variant emits `@P0 WARPSYNC.ALL` and `@P0 HMMA.16816.F32`, which is the first observed `WARPSYNC.ALL` in the SM120 chapter set.
* [INF] In the tested probes, BSSY/BSYNC is associated with divergent regions that ptxas keeps as real control regions or local calls, not with every lane-dependent predicate.
