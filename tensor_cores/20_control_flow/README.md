# 20 Control flow

## Scope

* [OBS] This chapter studies SM120 loop lowering, unroll pragmas, back-edge `BRA` detection, predication, `break`, `continue`, template instantiation, and repeated-body collapse traps.
* [OBS] The accepted probes cover 22 variants: 20a through 20v.
* [OBS] The probes compile with `nvcc -arch=sm_120` and are dumped with `cuobjdump --dump-sass`.
* [GAP] Runtime numeric validation is blocked because `nvidia-smi` cannot communicate with the NVIDIA driver in this environment.

## Sources

| File | Purpose |
|---|---|
| `20_scalar_control_flow.cu` | [OBS] Parameterized scalar control-flow probe for variants 20a through 20f and 20i through 20v. |
| `20_hmma_control_flow.cu` | [OBS] Parameterized HMMA control-flow probe for variants 20g and 20h. |
| `compile.sh` | [OBS] Rebuilds all binaries and SASS dumps for variants 20a through 20v. |

## Variants studied

| Variant | Source form | Main SASS result |
|---|---|---|
| 20a | Constant loop N=4, scalar body | [OBS] Fully unrolled. No back-edge. 4 FFMA and 4 FADD. |
| 20b | Constant loop N=16, scalar body | [OBS] Fully unrolled. No back-edge. 16 FFMA and 16 FADD. |
| 20c | Dynamic loop, trip count from kernel arg | [OBS] Multi-path unroll cascade. 3 back-edges. 33 FFMA and 33 FADD. |
| 20d | Constant loop N=16 with `#pragma unroll 1` | [OBS] Preserved loop. 1 back-edge. 1 FFMA and 1 FADD body. |
| 20e | Nested constant scalar loop 4 x 2 | [OBS] Fully unrolled. No back-edge. 8 FFMA and 8 FADD. |
| 20f | Nested constant scalar loop 8 x 2 | [OBS] Fully unrolled. No back-edge. 16 FFMA and 16 FADD. |
| 20g | Nested constant HMMA loop 4 x 2 | [OBS] Fully unrolled. No back-edge. 8 HMMA. |
| 20h | Nested constant HMMA loop 8 x 2 | [OBS] Fully unrolled. No back-edge. 16 HMMA. |
| 20i | Dynamic loop with short `if` body | [OBS] Uses predication plus loop back-edges. No BSSY/BSYNC. |
| 20j | Dynamic loop with larger `if` body | [OBS] Uses predicated BRA and predicated arithmetic. No BSSY/BSYNC. |
| 20k | Dynamic loop with `break` | [OBS] Emits BSSY.RECONVERGENT and BSYNC.RECONVERGENT around the break-capable loop body. |
| 20l | Dynamic loop with `continue` | [OBS] Emits predicated arithmetic and back-edges. No BSSY/BSYNC. |
| 20m | Constant loop N=16 with explicit full unroll | [OBS] Byte-count equivalent to 20b. Fully unrolled. |
| 20n | Constant loop N=16 with `#pragma unroll 4` | [OBS] Preserved loop with 4-iteration unrolled body and 1 back-edge. |
| 20o | Dynamic loop with `#pragma unroll 4` | [OBS] Preserved loop/tail structure with 4-wide body and 2 back-edges. |
| 20p | Dynamic loop with accumulator dependency | [OBS] Multi-path loop structure remains. 3 back-edges. |
| 20q | Dynamic loop with four independent accumulators | [OBS] Multi-path loop structure remains. 3 back-edges. 132 FFMA. |
| 20r | Dynamic loop with volatile store | [OBS] Emits `STG.E.STRONG.SYS` and a preserved side-effect loop. |
| 20s | Template nested loop 8 x 2 | [OBS] One template instantiation emits one SASS function with 16 FFMA and 16 FADD. |
| 20t | Two template instantiations, 4 x 2 and 8 x 2 | [OBS] One dump contains two separate SASS functions, one per template instantiation. |
| 20u | Nested 8 x 2 with unique per-iteration stores | [OBS] Fully unrolled with 16 STG.E stores. |
| 20v | Nested 8 x 2 with identical repeated body | [OBS] Fully unrolled to 16 FADD. No multiplication collapse observed. |

## Commands

```bash
cd tensor_cores/20_control_flow
bash compile.sh
```

## Key answers

* [RES] Constant scalar loops with compile-time trip counts 4 and 16 fully unroll by default in the tested SM120 kernels.
* [RES] Nested constant loops with scalar or HMMA bodies fully unroll by default in the tested SM120 kernels.
* [RES] `#pragma unroll 1` preserves a constant-trip loop as a SASS loop with a backward `BRA.U`.
* [RES] `#pragma unroll 4` preserves a loop with a 4-iteration unrolled body and a backward `BRA.U`.
* [RES] In the tested variants, preserved loops always have at least one backward branch. No loop without a back-edge was observed.
* [RES] BSSY/BSYNC is not required for ordinary preserved loops in the tested variants. It appears for the `break` loop variant.
* [INF] The prior production-audit case with 8 x 2 expected MMA sites and only 2 observed QMMAs is not reproduced by a minimal nested HMMA loop, because 20h emits 16 HMMAs.

## Open gaps

* [GAP] Runtime behavior and numeric outputs are not validated in this environment.
* [GAP] The exact ptxas heuristic that creates the dynamic-loop multi-path cascade is not fully decoded.
* [GAP] The production-audit 2-QMMA case still requires the original binary/source pair to distinguish stale binary, specialization, or dead-code elimination.
* [GAP] Warp-divergent branch reconvergence is deferred to Chapter 21.
