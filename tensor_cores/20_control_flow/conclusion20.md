# Chapter 20 - Control flow

## Goal

[OBS] Chapter 20 studies how ptxas lowers loops and local control flow into SM120 SASS. [OBS] The chapter covers constant loops, dynamic loops, unroll pragmas, nested scalar loops, nested HMMA loops, conditional bodies, `break`, `continue`, volatile side effects, template instantiations, and repeated identical bodies.

## Why this chapter exists

[OBS] Chapters 13 through 19 decoded tensor-core and staging opcodes, but production audit remained blocked by control-flow interpretation. [OBS] `FINDINGS.md` recorded GAP-audit-1 and GAP-audit-2 after a production FP4 attention audit saw 2 QMMAs where a source-level 8 x 2 tile structure suggested 16 MMA sites. [INF] Chapter 20 isolates the loop-lowering part of that failure by compiling minimal scalar and HMMA loops whose static trip counts and unroll pragmas are controlled.

[OBS] The reader should use this folder to answer a concrete SASS-reading question: does the dump contain a real loop, a fully unrolled loop, a partial-unroll loop, or multiple template instantiations. [INF] In the tested variants, a real loop is identified by at least one branch whose target offset is lower than the branch instruction offset.

## Toolchain note

* [OBS] All variants compile with CUDA 13.2, `nvcc` V13.2.78.
* [OBS] All accepted probes use `nvcc -arch=sm_120`.
* [OBS] SASS dumps were produced with `cuobjdump --dump-sass`.
* [GAP] Runtime execution is blocked because `nvidia-smi` cannot communicate with the NVIDIA driver in this environment.

## Scope of the chapter

* [OBS] Chapter 20 covers loop lowering, unroll pragmas, back-edge detection, local predication, local branches, `break`, `continue`, volatile global stores, and template instantiation visibility.
* [OBS] Chapter 20 includes HMMA loops only to test whether tensor-core bodies affect loop lowering.
* [GAP] Chapter 20 does not resolve warp-divergent reconvergence semantics. BSSY/BSYNC pairing under real warp divergence remains Chapter 21 scope.

## Survey

* [OBS] The chapter contains 22 SASS dumps from variants 20a through 20v.
* [OBS] Aggregate top 20 mnemonics across the 22 dumps: FFMA 377, NOP 285, FADD 254, UIADD3 121, IMAD.WIDE 63, BRA.U 56, IADD 55, LDC.64 50, EXIT 47, UI2FP.F32.U32 47, LDC 46, I2FP.F32.U32 44, STG.E 43, BRA 35, LDG.E.CONSTANT 33, MOV 32, UISETP.NE.U32.AND 26, HMMA.16816.F32 24, IMAD 24, ISETP.GE.AND 24.
* [OBS] Prologues use the standard `LDC` stack load, `S2R SR_TID.X`, parameter loads, bounds `ISETP`, and early `@P0 EXIT` pattern.
* [OBS] Useful bodies are either fully unrolled straight-line arithmetic/MMA blocks, preserved loops with backward `BRA.U`, or dynamic-loop cascades with multiple backward branches.
* [OBS] Epilogues use `STG.E` or `STG.E.STRONG.SYS`, then `EXIT`, then terminal self-trap `BRA`.

## Variants

| Variant | Source form | Instructions | Back-edges | Key result |
|---|---:|---:|---:|---|
| 20a | Constant loop N=4, scalar body | 40 | 0 | [OBS] Fully unrolled to 4 FFMA and 4 FADD. |
| 20b | Constant loop N=16, scalar body | 64 | 0 | [OBS] Fully unrolled to 16 FFMA and 16 FADD. |
| 20c | Dynamic loop, trip count from kernel arg | 192 | 3 | [OBS] Multi-path unroll cascade with 33 FFMA and 33 FADD. |
| 20d | Constant loop N=16, `#pragma unroll 1` | 40 | 1 | [OBS] Preserved loop with one FFMA and one FADD body. |
| 20e | Nested scalar loop 4 x 2 | 48 | 0 | [OBS] Fully unrolled to 8 FFMA and 8 FADD. |
| 20f | Nested scalar loop 8 x 2 | 64 | 0 | [OBS] Fully unrolled to 16 FFMA and 16 FADD. |
| 20g | Nested HMMA loop 4 x 2 | 56 | 0 | [OBS] Fully unrolled to 8 HMMA. |
| 20h | Nested HMMA loop 8 x 2 | 80 | 0 | [OBS] Fully unrolled to 16 HMMA. |
| 20i | Dynamic loop with short `if` | 80 | 2 | [OBS] Uses predication and back-edges. No BSSY/BSYNC. |
| 20j | Dynamic loop with larger `if` | 136 | 1 | [OBS] Uses predicated BRA and predicated arithmetic. No BSSY/BSYNC. |
| 20k | Dynamic loop with `break` | 40 | 1 | [OBS] Emits BSSY.RECONVERGENT and BSYNC.RECONVERGENT. |
| 20l | Dynamic loop with `continue` | 168 | 2 | [OBS] Uses predicated arithmetic and back-edges. No BSSY/BSYNC. |
| 20m | Constant loop N=16, explicit full unroll | 64 | 0 | [OBS] Same loop shape as default full unroll. |
| 20n | Constant loop N=16, `#pragma unroll 4` | 48 | 1 | [OBS] Preserved loop with 4-wide body. |
| 20o | Dynamic loop, `#pragma unroll 4` | 64 | 2 | [OBS] Preserved loop/tail structure with 4-wide body. |
| 20p | Dynamic loop with dependency chain | 96 | 3 | [OBS] Multi-path structure remains. |
| 20q | Dynamic loop with independent accumulators | 208 | 3 | [OBS] Multi-path structure remains and body expands to 132 FFMA. |
| 20r | Dynamic loop with volatile store | 96 | 1 | [OBS] Emits `STG.E.STRONG.SYS`. |
| 20s | Template nested loop 8 x 2 | 64 | 0 | [OBS] One template instantiation emits one SASS function with 16 FFMA and 16 FADD. |
| 20t | Two template instantiations | 112 | 0 | [OBS] Dump contains two SASS functions, one 4 x 2 and one 8 x 2. |
| 20u | Nested 8 x 2 with unique stores | 88 | 0 | [OBS] Fully unrolled with 16 STG.E stores. |
| 20v | Nested 8 x 2 identical body | 48 | 0 | [OBS] Fully unrolled to 16 FADD. No multiply collapse observed. |

## Key SASS observations

### Constant loops

* [OBS] 20a has no back-edge and emits 4 FFMA plus 4 FADD for the N=4 loop body.
* [OBS] 20b has no back-edge and emits 16 FFMA plus 16 FADD for the N=16 loop body.
* [OBS] 20m, the explicit full-unroll variant, has the same instruction count and arithmetic counts as 20b.
* [RES] Default ptxas policy for the tested constant scalar loops is full unroll.

### Unroll pragmas

* [OBS] 20d emits a backward `BRA.U UP0, 0xe0` at offset `0x0130` for a constant N=16 loop with `#pragma unroll 1`.
* [OBS] 20d loop body contains one FFMA and one FADD.
* [OBS] 20n emits a backward `BRA.U UP0, 0xe0` at offset `0x01f0` for a constant N=16 loop with `#pragma unroll 4`.
* [OBS] 20n loop body contains four FFMA/FADD steps before the back-edge.
* [OBS] 20o emits a 4-wide dynamic-loop body plus tail handling and has 2 back-edges.
* [RES] `#pragma unroll 1` and `#pragma unroll 4` both affect SASS loop shape directly in the tested variants.

### Dynamic loops

* [OBS] 20c has 192 instructions, 11 BRA-family instructions including the trap BRA, and 3 backward branches.
* [OBS] 20c emits 33 FFMA and 33 FADD despite the source loop having one FFMA/FADD step per iteration.
* [INF] Dynamic trip count does not mean a single compact loop on SM120. In 20c, ptxas emits a multi-path unroll cascade analogous to the runtime-loop cascade documented in Kernel 04.
* [OBS] 20p and 20q also keep multi-path loop structures with 3 back-edges each.
* [OBS] 20q expands to 132 FFMA because the source carries four independent accumulators.

### Nested scalar and HMMA loops

* [OBS] 20e emits 8 FFMA and 8 FADD with no back-edge for a 4 x 2 nested scalar loop.
* [OBS] 20f emits 16 FFMA and 16 FADD with no back-edge for an 8 x 2 nested scalar loop.
* [OBS] 20g emits 8 `HMMA.16816.F32` instructions with no back-edge for a 4 x 2 nested HMMA loop.
* [OBS] 20h emits 16 `HMMA.16816.F32` instructions with no back-edge for an 8 x 2 nested HMMA loop.
* [RES] The minimal 8 x 2 HMMA reproduction does not produce the production-audit 2-MMA pattern. It emits all 16 static HMMA sites.

### Predication and branches

* [OBS] 20i uses predicated FADD for the short conditional body and emits no BSSY/BSYNC.
* [OBS] 20j uses predicated BRA and predicated arithmetic for the larger conditional body and emits no BSSY/BSYNC.
* [OBS] 20l uses many predicated arithmetic instructions for the `continue` form and emits no BSSY/BSYNC.
* [OBS] 20k emits `BSSY.RECONVERGENT B0, 0x1c0` at offset `0x0120` and `BSYNC.RECONVERGENT B0` at offset `0x01b0`.
* [INF] In the tested local-control variants, `break` is the only construct that forces BSSY/BSYNC.

### Template and side-effect traps

* [OBS] 20s emits one SASS function for `template_nested_kernel<8,2>` with 16 FFMA and 16 FADD.
* [OBS] 20t emits two SASS functions in one dump: `template_nested_kernel<4,2>` and `template_nested_kernel<8,2>`.
* [OBS] 20u emits 16 STG.E stores for the 16 unique per-iteration stores.
* [OBS] 20v emits 16 FADD for a repeated identical body and no back-edge.
* [OBS] 20r emits `STG.E.STRONG.SYS` for the volatile store loop.
* [INF] Template specialization can create multiple SASS functions in one dump, so production audits must count instructions per function, not per source file.

## Encoding / register / scheduling analysis

* [OBS] Back-edge detection is mechanical in these dumps: a `BRA` or `BRA.U` target offset lower than the branch instruction offset marks a preserved SASS loop.
* [OBS] Constant full-unroll variants retain only the terminal self-trap `BRA` after `EXIT`; they do not contain a branch target into the useful body.
* [OBS] The preserved-loop variants use uniform branches (`BRA.U`) when the loop predicate is warp-uniform.
* [OBS] 20k uses non-uniform reconvergence machinery for `break`, including BSSY/BSYNC around the loop body.
* [GAP] The exact control-code bits differentiating loop branches, trap branches, and break branches are not decoded in this chapter.

## Resolved hypotheses

| Hypothesis | Status |
|---|---|
| [HYP-20-1] Constant trip count at compile time always triggers full unroll on small bodies. | [RES] Confirmed for tested N=4, N=16, nested 4 x 2, nested 8 x 2, and HMMA nested loops without restrictive pragmas. |
| [HYP-20-2] `#pragma unroll 1` forces a real loop with BRA back-edge. | [RES] Confirmed by 20d. |
| [HYP-20-3] Dynamic trip count forces a real loop with BRA back-edge. | [RES] Confirmed in part: dynamic loops have back-edges, but ptxas often emits a multi-path unroll cascade rather than one compact loop. |
| [HYP-20-4] Nested constant loops may be partially unrolled under a heuristic threshold. | [RES] Rejected for tested 4 x 2 and 8 x 2 scalar/HMMA bodies. They fully unroll. |
| [HYP-20-5] BSSY/BSYNC on Blackwell is for divergence reconvergence only, not for ordinary loops. | [RES] Confirmed for ordinary loops. 20k shows break can also trigger reconvergence machinery. |
| [HYP-20-6] No BRA back-edge in a SASS dump means no SASS-level loop. | [RES] Confirmed for tested variants. Every preserved loop has a back-edge. |

## Open gaps

| Gap | Follow-up required |
|---|---|
| [GAP] Runtime validation blocked by unavailable driver. | Run all 20a through 20v binaries on SM120 hardware and record outputs. |
| [GAP] Dynamic-loop cascade heuristic not fully decoded. | Add larger and smaller runtime trip-count kernels with independent source variables and compare cascade thresholds. |
| [GAP] Original production 2-QMMA mystery not resolved end-to-end. | Rebuild the original FP4 attention binary from the exact source and compare function-level SASS to the stale dump. |
| [GAP] Warp-divergent reconvergence still not characterized. | Execute Chapter 21 with lane-dependent branches and warp-synchronous consumers. |

## Cross-references to FINDINGS.md

* [OBS] Added Kernel 20 control-flow observations to `FINDINGS.md`.
* [RES] Updated GAP-audit-1 and GAP-audit-2 with the tested loop-lowering results.
* [RES] Updated GAP-audit-4 in part by stating that SASS instruction counts are per function body and warp-level instructions such as HMMA still appear once per static site.
* [GAP] GAP-audit-3 remains open for Chapter 21.

## Summary

* [RES] Constant loops in the tested SM120 probes fully unroll unless an unroll pragma restricts them.
* [RES] Preserved loops in the tested SM120 probes always expose a backward branch.
* [RES] The 8 x 2 HMMA minimal reproduction emits 16 HMMAs, not 2.
* [OBS] Dynamic loops can become large multi-path unroll cascades with several back-edges.
* [OBS] `break` emits BSSY/BSYNC in the tested loop, while ordinary loops, conditional bodies, and `continue` do not.
* [GAP] Production-kernel audit still needs Chapter 21 for warp divergence and the original FP4 attention rebuild for the 2-QMMA case.
