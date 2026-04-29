# Chapter 21 - Divergence and reconvergence

## Goal

[OBS] Chapter 21 studies how ptxas lowers lane-divergent control flow into SM120 SASS. [OBS] The chapter covers 20 variants: short divergent predicates, uniform branch, divergent if/else, nested divergence, break, continue, early return, block barrier after divergence, warp vote, select-vs-branch, divergent memory, guarded HMMA, lane-dependent trip count, bounds-check epilogue, masked stores, divergent local call, cold trap path, and vote-converged branch.

## Why this chapter exists

[OBS] `FINDINGS.md` left GAP-audit-3 open after Chapters 09, 11, and 20 because BSSY/BSYNC had been observed in several contexts but not isolated across a full matrix of branch shapes. [OBS] Chapter 09 showed BSSY/BSYNC around SHFL after prior divergence. [OBS] Chapter 11 showed BSSY/BSYNC around local slowpath calls. [OBS] Chapter 20 showed BSSY/BSYNC for a `break` loop but not ordinary loops, conditional bodies, or `continue`.

[INF] Chapter 21 isolates the missing question: whether lane divergence by itself forces reconvergence machinery, or whether ptxas first tries predication, select, predicated exit, forward branch, warp vote, or explicit warp sync forms. [OBS] The answer from the 20 dumps is that lane divergence alone is not sufficient to force BSSY/BSYNC in every tested variant.

## Toolchain note

* [OBS] All variants compile with `nvcc -arch=sm_120`.
* [OBS] All dumps were produced with `cuobjdump --dump-sass`.
* [GAP] Runtime execution is not validated in this environment because the NVIDIA driver is unavailable.

## Survey

* [OBS] The chapter contains 20 SASS dumps from variants 21a through 21t.
* [OBS] Aggregate top mnemonics across the 20 dumps include 220 `NOP`, 45 `LDC.64`, 45 `EXIT`, 41 `IMAD.WIDE`, 39 `LDC`, 39 `FADD`, 31 `BRA`, 27 `LOP3.LUT`, 26 `LDG.E.CONSTANT`, 22 `STG.E`, 20 `FFMA`, 6 `BSSY.RECONVERGENT`, and 6 `BSYNC.RECONVERGENT`.
* [OBS] BSSY/BSYNC appears in exactly six variants: 21c, 21e, 21f, 21l, 21o, and 21r.
* [OBS] BSSY/BSYNC does not appear in 21a, 21b, 21d, 21g, 21h, 21i, 21j, 21k, 21m, 21n, 21p, 21q, 21s, or 21t.

## Variant results

| Variant | Source form | Instructions | BSSY/BSYNC | Key result |
|---|---:|---:|---:|---|
| 21a | Simple lane-dependent short `if` | 32 | 0/0 | [OBS] Predicated `FADD`. |
| 21b | Uniform branch | 32 | 0/0 | [OBS] `UFSEL` instead of branch region. |
| 21c | Lane-divergent `if` | 48 | 1/1 | [OBS] BSSY/BSYNC around divergent arithmetic. |
| 21d | Lane-divergent `if/else` | 32 | 0/0 | [OBS] Predicated arithmetic on both paths. |
| 21e | Nested divergence | 40 | 1/1 | [OBS] One reconvergence scope covers nested control. |
| 21f | Lane-dependent `break` | 40 | 1/1 | [OBS] Reconvergence scope plus useful back-edge. |
| 21g | Lane-dependent `continue` | 40 | 0/0 | [OBS] Useful back-edge but no reconvergence scope. |
| 21h | Lane-dependent early return | 32 | 0/0 | [OBS] Additional predicated `EXIT`. |
| 21i | Divergence then `__syncthreads()` | 40 | 0/0 | [OBS] `FSEL` plus `BAR.SYNC.DEFER_BLOCKING`. |
| 21j | `__ballot_sync` | 32 | 0/0 | [OBS] `VOTE.ANY R5, PT, P0`. |
| 21k | Select vs branch vs mask | 40 | 0/0 | [OBS] `FSEL` used for select forms. |
| 21l | Short body vs long body | 48 | 1/1 | [OBS] Long divergent body gets reconvergence scope. |
| 21m | Divergent memory paths | 48 | 0/0 | [OBS] Forward branch and separate path exits. |
| 21n | Guarded HMMA | 64 | 0/0 | [OBS] `@P0 WARPSYNC.ALL` and `@P0 HMMA.16816.F32`. |
| 21o | Lane-dependent trip count | 40 | 1/1 | [OBS] Reconvergence scope plus useful back-edge. |
| 21p | Bounds-check epilogue | 32 | 0/0 | [OBS] Additional predicated `EXIT`. |
| 21q | Masked store tail | 40 | 0/0 | [OBS] Predicated LDC/IMAD/STG and predicated EXIT. |
| 21r | Divergent noinline call | 48 | 1/1 | [OBS] BSSY/BSYNC around `CALL.REL.NOINC`. |
| 21s | Cold trap/error path | 32 | 0/0 | [OBS] `BPT.TRAP 0x1` behind branch. |
| 21t | Vote-converged branch | 32 | 0/0 | [OBS] `VOTE.ANY P0, P0` plus `SHFL.IDX`. |

## Key observations

### Predication and select avoid reconvergence scopes

* [OBS] 21a computes the lane predicate with `LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT` and applies `@P0 FADD`; no BSSY/BSYNC appears.
* [OBS] 21d emits `@P0 FFMA`, `@P0 FADD`, `@!P0 FFMA`, and `@!P0 FADD` for the two branch bodies; no BSSY/BSYNC appears.
* [OBS] 21k emits `FSEL` instructions for select-like expressions and no BSSY/BSYNC.
* [INF] In these variants, ptxas represents divergent choice as predicated datapath operations rather than as an explicit reconvergence region.

### BSSY/BSYNC appears for selected real control regions

* [OBS] 21c emits `BSSY.RECONVERGENT B0, 0x1e0`, `@!P0 BRA 0x1d0`, a divergent arithmetic body, and `BSYNC.RECONVERGENT B0`.
* [OBS] 21e emits one BSSY/BSYNC pair around nested divergent control and uses forward branches inside the region.
* [OBS] 21l emits one BSSY/BSYNC pair around the long divergent body.
* [INF] In the tested arithmetic-only probes, BSSY/BSYNC appears when ptxas keeps a divergent body as a real branch region instead of fully predicating/selecting it.

### Loop exits match Chapter 20 but lane-dependent trip counts add evidence

* [OBS] 21f lane-dependent `break` emits BSSY/BSYNC and a useful back-edge.
* [OBS] 21g lane-dependent `continue` emits a useful `BRA.U` back-edge and no BSSY/BSYNC.
* [OBS] 21o lane-dependent trip count emits BSSY/BSYNC and a useful back-edge.
* [INF] 21f confirms Chapter 20's `break` result under a lane-dependent exit condition. [INF] 21g confirms that `continue` can still avoid BSSY/BSYNC under the tested lane-dependent condition.

### Early exits and masked stores use predicated exits

* [OBS] 21h emits an additional `@P0 EXIT` for lane-dependent early return and no BSSY/BSYNC.
* [OBS] 21p emits an additional `@P0 EXIT` for the epilogue bounds check and no BSSY/BSYNC.
* [OBS] 21q emits predicated store setup and an additional `@P1 EXIT`; no BSSY/BSYNC appears.
* [INF] In the tested tail/early-return patterns, ptxas prefers predicated exits and predicated stores over explicit reconvergence scopes.

### Warp-level instructions have multiple lowering paths

* [OBS] 21j emits `VOTE.ANY R5, PT, P0` for `__ballot_sync`.
* [OBS] 21t emits `VOTE.ANY P0, P0`, a predicated branch, and `SHFL.IDX`, with no BSSY/BSYNC.
* [OBS] 21n emits `@P0 WARPSYNC.ALL` before `@P0 HMMA.16816.F32`.
* [INF] Chapter 21 extends Chapter 09: warp-synchronous operations do not always require visible BSSY/BSYNC in the same local pattern. ptxas can also use vote-converged control or `WARPSYNC.ALL`.

### Local calls keep the Chapter 11 pattern

* [OBS] 21r emits `BSSY.RECONVERGENT B0, 0x170`, a predicated branch, `CALL.REL.NOINC 0x1b0`, `BRA 0x160`, `BSYNC.RECONVERGENT B0`, and a local callee ending in `RET.REL.NODEC R2 0x0`.
* [INF] 21r matches the Chapter 11 local slowpath structure, but the branch condition is lane-derived rather than math-library slowpath derived.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| [HYP-21-1] Lane divergence always forces BSSY/BSYNC. | [RES] Rejected by 21a, 21d, 21h, 21i, 21j, 21k, 21m, 21n, 21p, 21q, 21s, and 21t. |
| [HYP-21-2] Short divergent bodies can be predicated without reconvergence scopes. | [RES] Confirmed by 21a and 21d. |
| [HYP-21-3] Divergent `break` still uses BSSY/BSYNC when the exit condition is lane-dependent. | [RES] Confirmed by 21f. |
| [HYP-21-4] Divergent `continue` necessarily uses BSSY/BSYNC when lane-dependent. | [RES] Rejected by 21g. |
| [HYP-21-5] A guarded HMMA must be surrounded by BSSY/BSYNC. | [RES] Rejected by 21n; the observed form is predicated loads, `@P0 WARPSYNC.ALL`, and `@P0 HMMA`. |

## Open gaps

| Gap | Follow-up required |
|---|---|
| [GAP] Runtime behavior is not validated. | Run 21a through 21t on SM120 hardware and record outputs. |
| [GAP] Exact threshold for predication/select versus BSSY branch region is not decoded. | Add body-size sweep variants with 1, 2, 4, 8, 16, and 32 dependent instructions. |
| [GAP] `WARPSYNC.ALL` semantics and encoding are newly observed but not decoded. | Add guarded HMMA/SHFL/LDSM variants with and without explicit `__syncwarp`. |
| [GAP] Control-code bits for BSSY/BSYNC and WARPSYNC are not decoded. | Diff variants with identical mnemonic regions but different reconvergence forms. |
| [GAP] Interaction with true runtime masks is still shallow. | Add variants using non-full masks for ballot, shfl, and syncwarp after divergence. |

## Cross-references to FINDINGS.md

* [OBS] Added Kernel 21 divergence/reconvergence observations to `FINDINGS.md`.
* [RES] Updated GAP-audit-3 in `FINDINGS.md` with the tested predication, BSSY/BSYNC, WARPSYNC, and local-call results.
* [GAP] Runtime validation and exact BSSY/WARPSYNC semantics remain open.

## Summary

* [RES] Lane-dependent divergence does not imply visible BSSY/BSYNC in every tested SM120 SASS form.
* [OBS] ptxas uses predication, FSEL/UFSEL, predicated EXIT, predicated stores, forward branches, BSSY/BSYNC, VOTE, SHFL, WARPSYNC.ALL, and local CALL depending on source shape.
* [OBS] BSSY/BSYNC appears in 6 of the 20 tested variants.
* [OBS] Guarded HMMA introduced the first observed `WARPSYNC.ALL` in this project.
* [GAP] The exact heuristic boundary between predication/select and explicit reconvergence remains unresolved.
