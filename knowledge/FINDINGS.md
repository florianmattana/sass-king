# Findings and open hypotheses

Running log of observations and hypotheses, organized by kernel chapter.

Notation:
* **[OBS]** verified observation from a SASS dump
* **[INF]** inference from one or more observations; the evidence chain must be stated
* **[HYP]** open hypothesis, to be tested
* **[RES]** resolved hypothesis (rejected or confirmed)
* **[GAP]** open question not answered by the current evidence

---

## Kernel 01 vector_add (baseline)

### Observations

* [OBS] The observed elementwise baseline kernels use a 6 section prologue: stack pointer init, thread and block ID setup, argument loads, index computation, bounds check, global descriptor load.
* [OBS] `R1` is the stack pointer by ABI convention, loaded from `c[0x0][0x37c]`, initialized even when no spilling occurs.
* [OBS] Kernel arguments and launch parameters live in constant memory bank 0 at fixed offsets (`c[0x0][...]`).
* [OBS] `LDC` loads into a per thread register, `LDCU` into a uniform register. ptxas classifies automatically based on data flow.
* [OBS] Per thread registers R0 to R255 versus uniform registers UR0 to UR63 (or UR0 to UR255 on SM100+) are two physically separate register files.
* [OBS] Values shared across the warp (blockIdx, kernel args, n) go into UR. Values unique per thread (threadIdx, computed indices, loaded data) go into R.
* [OBS] Global memory accesses on SM120 use descriptor based addressing: `desc[UR][R.64]`. The descriptor is loaded once from `c[0x0][0x358]` and shared across all global accesses.
* [OBS] `IMAD.WIDE` produces a 64 bit result (register pair) from a 32 bit multiply. Used for pointer arithmetic.
* [OBS] Bounds check uses predication (`@P0 EXIT`), not an explicit branch-around-body sequence.
* [INF] The predicated early-exit form avoids an additional taken branch in the visible SASS control-flow graph; runtime divergence cost is not measured in kernel 01.
* [OBS] Each independent pointer load gets its own scoreboard, allowing dependent address computations to proceed independently.
* [OBS] Co-consumed loads share a single scoreboard. Two LDGs feeding one FADD both go on SB4, one wait covers both.
* [OBS] ptxas interleaves IMAD.WIDE (address computation) with LDG (memory load) so that multiple global loads are in flight simultaneously.
* [OBS] The store is fire and forget, no scoreboard set, since no downstream instruction reads it back.
* [OBS] NOP padding at the end of the kernel aligns the kernel size to a 512 byte boundary. Never executed.
* [OBS] On a minimal elementwise kernel, the useful compute ratio is 1 FADD out of 20 instructions = 5%. Signature of a memory bound kernel.
* [OBS] Only pointers to arrays live in constant memory. The data of the input arrays themselves lives in global memory and is accessed via LDG.
* [OBS] Patterns not yet observed in kernel 01 (to be introduced later): loops (backward BRA), divergent control flow (BSSY/BSYNC), shared memory (LDS/STS/BAR.SYNC), vectorized memory (LDG.E.128), register spilling (STL/LDL), tensor core, atomics, warp level primitives.

### Reference SASS reading template (established here)

Used as a first pass filter in every subsequent kernel audit:

1. Skip the prologue by pattern matching the ID setup sequence.
2. Find the bounds check or early exit. Look for the first `@P EXIT`.
3. Identify the loaded arguments. Pointers come from constant memory in a block near the start of the body.
4. Locate the compute section. Arithmetic instructions that correspond to the algorithm.
5. Check the stores. STG at the end writes the results.
6. Everything else is plumbing: address arithmetic, scoreboard bookkeeping, control flow around loops.

### Open hypotheses

* [HYP] Cross pipeline ALU to CBU transfer costs ~13 cycles versus ~5 for intra ALU. Observed on ISETP to `@P EXIT`. To microbenchmark.
* [HYP] Predicate to arithmetic consumer latency versus predicate to branch latency. Not measured.
* [HYP] Useful compute ratio thresholds (20% and 50%) for memory bound versus latency bound versus compute bound classification. Not calibrated against the actual roofline.

---

## Kernel 02 vector_add_plus1 (a + b + 1.0f)

### Observations

* [OBS] `a + b + c` does **not** fuse. Two separate FADD instructions emitted. FMA fusion is syntactically strict: the source must literally contain a multiply followed by an add.
* [OBS] ptxas tracks register liveness and recycles dead registers across semantic boundaries. R0 (holding threadIdx.x) becomes FP scratch for the first FADD once threadIdx.x is dead.
* [OBS] Adding one source level operation produces exactly one additional SASS instruction, no hidden overhead.
* [OBS] Non compute infrastructure (prologue, bounds check, memory ops) is byte for byte identical between kernels 01 and 02. Only the compute section grows.

### Diagnostic takeaways

* [OBS] When diffing two SASS dumps, most instructions will be identical. Focus on what changes.
* [OBS] Register renaming may occur even in identical looking sections as a consequence of changes elsewhere. Track destinations, not just opcodes.
* [OBS] The absence of fusion is as informative as its presence. Two arithmetic instructions where you expected one means the source level pattern did not match FMA.

---

## Kernel 03 vector_fma (a*b + c)

### Observations

* [OBS] FMA fusion is active: `a*b + c` compiles to a single FFMA.
* [OBS] Fusion is syntactic, the multiply and the add must be direct operands of each other.
* [OBS] Six scoreboards per warp (SB0 to SB5). Hard budget.
* [OBS] Scoreboards and stall count are independent mechanisms. Stall count is for instruction cadence, scoreboards are for data ready signaling.
* [OBS] Fixed latency operations (FFMA, IMAD, ISETP) use only stall count, no scoreboard. ptxas knows their latency at compile time.
* [OBS] Variable latency operations (LDG, LDS, S2R, MUFU, BAR) use scoreboards.
* [OBS] A warp stalls if at least one of two conditions holds: internal stall counter nonzero, or scoreboard wait mask on the next instruction not satisfied.
* [OBS] Scoreboard grouping for co-consumed producers. Three LDGs feeding one FFMA all go on SB4, one wait covers all three.
* [OBS] Yield flag is set on exactly those instructions that wait on a scoreboard. Tells the scheduler it can switch warps during the wait.
* [OBS] Uniform registers live in a distinct SRAM with a separate pipeline, not a mode of the main register file.
* [OBS] Uniform register file is smaller (64 registers on SM80 to SM89, expanded to 256 on SM100+) and the uniform pipeline has a narrower set of supported operations.
* [OBS] Adding one memory operand costs about 3 extra SASS instructions: LDC pointer, IMAD.WIDE address, LDG load.
* [OBS] In kernel 03, adding the `d` pointer grew plumbing by 3 instructions but compute stayed at 1 instruction thanks to FMA fusion. The fusion absorbed what would have been FMUL + FADD.
* [OBS] Uniform datapath benefits: no duplication of the value in the main register file, separate pipeline avoids contention with per thread arithmetic, lower energy per operation (one broadcast instead of 32 reads).
* [OBS] `ISETP` feeding `@P EXIT` has stall=13, much larger than typical 5 for ALU instructions.

### Open hypotheses

* [HYP] The 13 cycle stall on ISETP to `@P EXIT` is caused by cross pipeline transfer from ALU to CBU. To microbenchmark by comparing ISETP to arithmetic consumer versus ISETP to branch.
* [HYP] Throughput of LDG grouped on 1 SB versus split across N SB. Not measured.
* [HYP] Whether the 6 scoreboards per warp budget is effectively 6 or smaller due to hardware constraints. Not measured.

---

## Kernel 04 vector_loop (runtime trip count)

### Observations

* [OBS] Runtime trip count loops produce a cascade of unrolled paths. For this kernel: skip if K<1, scalar tail if K<4, unroll by 4 if 4≤K<16, unroll by 16 if K≥16.
* [OBS] Generalized Duff's device pattern. A 5 line loop produces ~80 SASS instructions across 4 execution paths.
* [OBS] ptxas inverts the loop counter: negates UR5 to count down toward 0 instead of up toward K. Saves a register and allows comparison against a small immediate.
* [OBS] First guard uses signed comparison (handles K≤0 correctly), subsequent guards use unsigned comparison once sign is known.
* [OBS] Remainder precomputed at dispatch: `UR4 = K AND 3` (= K mod 4) is extracted once and later consumed by the scalar tail loop.
* [OBS] Loop counter update (UIADD3) and termination test (UISETP) are interleaved between FFMAs in the loop body, not placed at the tail. Keeps arithmetic pipelines saturated.
* [OBS] HFMA2 used as a constant loader. `HFMA2 R2, -RZ, RZ, 1.875, 0.00931549` loads the FP32 value `1.001f` by exploiting the bit pattern equivalence between two packed FP16 values and one FP32.
* [OBS] HFMA2 is chosen when the kernel is compute heavy, MOV is chosen in lighter blocks. Evidence of pipeline balancing heuristic by ptxas.
* [OBS] Late pointer load for store destination when the store value has a long dependency chain. Minimizes register live range.
* [OBS] `UIADD3` at 0x360 has stall=12, larger than expected for a uniform datapath arithmetic instruction.
* [OBS] Source to SASS expansion ratio for this kernel is approximately 15x (5 source lines, ~77 useful instructions).
* [OBS] Methodological validation: a "smallest possible runtime loop" produced a multi path code expansion, an undocumented constant loading trick, and several optimization decisions that depend on kernel wide heuristics not visible at source level. Controlled variation is the only reliable way to map source patterns to SASS patterns because any a priori model of "what the SASS should look like" will be wrong in at least one significant way.
* [OBS] Henry Zhu (December 2025) documented that ptxas's decision between HFMA2, IMAD, and MOV for constant loading can be affected by surprising inputs including the name of the CUDA function, with performance impacts of ±20% in real kernels. Signal that ptxas scheduling heuristics for constant loading are not fully stable across compiler versions.

### Open hypotheses

* [HYP] Relative throughput of HFMA2 vs MOV vs IMAD for constant loading in compute heavy blocks. To microbenchmark by forcing each via PTX inline.
* [HYP] Per SM partition i-cache size on SM120 is ~8 KB. Unverified.
* [HYP] FMA, FMAHEAVY, FMALITE are distinct physical pipelines or submodes of the same pipeline. To microbenchmark by saturating one and observing the other.
* [HYP] `UPLOP3.LUT` with all UPT inputs and specific LUT values is an idiomatic "initialize UP to constant" pattern. Semantics not fully reverse engineered.
* [HYP] The anomalous UIADD3 stall=12 is caused by cross pipeline transfer. Not verified.

---

## Kernel 05 vector_loop_fixed (compile time loop count)

### Observations

* [OBS] Compile time loop bounds eliminate the entire unrolling cascade. ptxas emits exactly N copies of the body in straight line, no dispatch, no backward branch.
* [OBS] Refactoring runtime trip counts to compile time collapses SASS size drastically. Kernel 04 had ~80 instructions for runtime K, kernel 05 has 17 instructions for K=8.
* [OBS] FFMA on SM120 has one immediate slot, reserved for the addend (`src3`). The multiplier sources (`src1`, `src2`) must be registers.
* [OBS] Any non trivial FP32 constant used as a multiplier must be materialized in a register first. HFMA2 is the systematic idiom ptxas uses for this.
* [OBS] HFMA2 is used regardless of constant value (small, large, irrational, exact power of 2), number of uses (one or many), or kernel structure (loop or straight line).
* [OBS] Register allocation is context dependent. The same value can land in different registers across related kernels based on global liveness analysis.
* [OBS] Last operation in a chain may target a different register than the chain to break dependency with downstream instruction. Observed: last FFMA writes R3 to free R0 and avoid a tight dependency with STG.
* [OBS] Useful compute ratio reaches 47% in kernel 05 (8 FFMA out of 17 instructions), up from 5% in kernel 01. Direct effect of raising arithmetic intensity.

### Practical consequences for constants

Three optimization strategies derived from the FFMA immediate slot rule:

1. **Reuse constants.** Restructure expressions so the same constant is used multiple times. ptxas materializes each unique constant only once.
2. **Use constants as addends when possible.** Constants in `src3` position are encoded as immediates with no materialization cost. Structuring formulas so constants are added rather than multiplied saves instructions.
3. **Pass constants as kernel arguments.** Constants in kernel args live in constant memory and load via LDC, which uses a different pipeline. Can be advantageous if the FMA pipeline is the bottleneck and the LDC can be hidden behind other prologue work.

### Resolved hypotheses

* [RES] The cascade in kernel 04 was caused by runtime trip count. Fixing the count removes the cascade. **Confirmed.**

### Open hypotheses

* [HYP] Why HFMA2 specifically versus MOV or IMAD for constant materialization. Three candidate reasons (pipeline affinity with consuming FFMAs, compact encoding, hardcoded heuristic), none directly measured. **Chapter 11 evidence**: in kernel 11d (log2f) and 11e (log2f intrinsic), HFMA2 packs two distinct FP16 values `(1.443359375, -0.2030029296875)` into a single register which is then consumed as FP32 by a subsequent polynomial FFMA. This confirms HFMA2 is used beyond the "pipeline affinity" case and extends to "packed constant materialization for polynomial evaluation". The choice of HFMA2 over dual MOV appears to be driven by: (a) encoding one instruction instead of two, (b) the result is a bit-level-precise FP32 that the FFMA consumes directly.
* [HYP] Whether FFMA with immediate in `src1` or `src2` exists at all in any SM120 SASS dump. Would invalidate the addend only rule.
* [HYP] Whether the HFMA2 idiom persists on SM80 or SM89. Older architectures may use different defaults.

---

## Kernel 06 vector_smem (shared memory scalar)

### Observations

* [OBS] Shared memory base is constructed via `UMOV UR4, 0x400` then `ULEA UR4, UR5, UR4, 0x18` where UR5 is `SR_CgaCtaId`.
* [OBS] Shared addressing does not use a descriptor. Direct `[R]` or `[R+UR]`, unlike global memory which uses `desc[UR][R.64]`.
* [OBS] `BAR.SYNC.DEFER_BLOCKING 0x0` is the SASS form of `__syncthreads()` on SM120.
* [OBS] BAR.SYNC runs on the ADU pipeline, not CBU as initially intuited.
* [OBS] `__syncthreads()` always uses barrier 0. CUDA exposes 16 hardware barriers per block.
* [OBS] Modulo by a runtime variable generates a CALL to `__cuda_sm20_rem_u16`, a ~21 instruction function placed after EXIT.
* [OBS] The modulo function uses MUFU.RCP on the XU pipeline. First XU usage in our kernels.
* [OBS] I2F and F2I conversions in the modulo function also run on XU.
* [OBS] CALL.REL.NOINC requires the caller to write the return address into a register (MOV Rn, offset) before the call. RET.REL.NODEC reads it back.
* [OBS] u16 masking (`LOP3.LUT R, R, 0xffff, RZ, 0xc0`) before the modulo CALL is mandated by the u16 ABI of the helper function, not defensive.
* [OBS] The LOP3 mask in shared addressing encodes the modulo bound as `(size_in_floats - 1) * 4`. Verified for shared sizes 128, 256, 512.

### Variant 06b (`% 256`)

* [OBS] Modulo by a power of 2 constant fuses into LEA + LOP3 mask. Two instructions total. No CALL, no slowpath.

### Variant 06c (`% 255`)

* [OBS] Modulo by a non power of 2 constant inlines a reciprocal multiplication sequence. ~10 instructions inline, no CALL.
* [OBS] The inline sequence uses IMAD with magic number 0x809, SHF.R.U32.HI, and PRMT for byte manipulation.

### Variant 06d (`& 255`)

* [RES] `& 255` and `% 256` produce byte identical SASS. ptxas canonicalizes them to the same internal representation. **Confirmed.**

### Variant 06e (128 floats shared, 128 threads)

* [RES] `UMOV UR4, 0x400` does **not** encode shared memory size. Value unchanged at 0x400 despite half size. **Rejected.**
* [OBS] The LOP3 mask scales with shared size: 128 floats produces mask 0x1fc = (128-1) × 4.
* [OBS] Bounds check fusion: `ISETP.GT.U32.AND P0, ..., R6, 0x7f, PT` combined with `ISETP.GE.OR P0, ..., R7, UR5, P0`. ptxas fuses two conditions into one predicate via the `.OR` modifier.

### Variant 06f (512 threads, 512 floats shared)

* [RES] `UMOV UR4, 0x400` does **not** encode block size. Value unchanged at 0x400 despite doubled block. **Rejected.**
* [OBS] The LOP3 mask scales: 512 floats produces mask 0x7fc = (512-1) × 4.

### Variant 06g (two shared buffers)

* [OBS] For N shared buffers, ptxas generates one UMOV and derives the other N-1 bases via UIADD3. Not N separate UMOVs.
* [OBS] Shared buffers are laid out consecutively in memory. Offset between two 256 float buffers = 0x400 = 1024 bytes (size of the first in bytes). No automatic padding.
* [OBS] `[R+UR]` addressing mode lets a single per thread address register be reused with different uniform offsets to read multiple buffers at the same position.
* [OBS] Consecutive STS can be emitted without intermediate BAR.SYNC. One final `__syncthreads()` covers all stores.

### Variant 06h (integer division by runtime)

* [RES] Division and modulo use **separate functions**. Same reciprocal multiplication core, different final step. Division stops at the quotient. Modulo computes quotient then reconstructs remainder via IMAD. **Rejected that they share the function.**
* [OBS] Practical: `q = a / b; r = a % b;` generates two CALLs. `q = a / b; r = a - q * b;` generates one CALL plus one IMAD.

### Variant 06i (two modulos, two runtime divisors)

* [RES] Multiple modulo calls in the same kernel share a **single instance** of the helper function. Two CALLs target the same address. **Confirmed.**
* [OBS] Binary size stays constant regardless of the number of calls.
* [OBS] Modulo helper ABI: R8 u16 dividend input, R9 u16 divisor input and remainder output, Rn return address written by caller via MOV.
* [OBS] Runtime cost is not amortized. Each call executes the full function (~30 cycles minimum with MUFU.RCP).

### Open hypotheses

* [HYP] Meaning of the specific value `0x400` in `UMOV UR4, 0x400`. Confirmed inert across shared size (1, 128, 256, 512 floats), type (float vs int), and static vs dynamic shared. Present only when `__shared__` is declared. Possible interpretations not tested: architectural SM120 constant (base or offset for shared descriptor), bit field format for the following ULEA (where `0x18 = 24` is the shift), value tied to max concurrent blocks per SM or shared banks. To investigate by comparing with SM89 or SM90a dump, or by reading NVIDIA shared descriptor format documentation for SM90+.
* [HYP] `SR_CgaCtaId` falls back to a default value when the kernel does not declare `__cluster_dims__`. Unverified.
* [HYP] `.DEFER_BLOCKING` on BAR.SYNC is an SM90+ modifier. To verify by comparing with SM80 or SM89 dumps.
* [HYP] PRMT with `0x9910` and `0x7710` in the `% 255` function handles byte level manipulation for u16 truncation. Not analyzed.
* [HYP→PARTIAL] `HFMA2 R3, -RZ, RZ, 15, 0` and the subnormal FSETP in the modulo helper handle edge cases of the reciprocal multiplication. **Chapter 11 clarifies the FSETP part**: the subnormal handling pattern (`FSETP.GEU + FMUL by 2^24 + post-correction`) is a universal pattern used in every MUFU-based computation (log2f, rsqrt, fdividef). The FSETP in the modulo helper serves the same role: detect subnormal reciprocal estimates that would cause MUFU.RCP to behave poorly. The `HFMA2 R3, -RZ, RZ, 15, 0` role (constant 15 as FP16 bit pattern for some edge case correction) remains unexplained.
* [RES] The return address register choice depends on caller register pressure at the call site. Observed as R7 in kernel 06, R10 in kernel 06i, **R4 in kernel 11b, R9 in kernel 11h**. Four data points across three different kernels confirm the pattern. The callee reads the return address register as its first meaningful operation and uses it for `RET.REL.NODEC`.
* [HYP→RES PARTIAL] Why the modulo function has a symbolic label (`__cuda_sm20_rem_u16`) in the dump but the division function does not. **Partially resolved by chapter 11**: on CUDA 13.2, u32+ division and modulo are fully inlined (no external helper, no symbolic label). Only u16 operations retain external named helpers. The question reformulates to: why u16 is externally named while u32+ is inlined. [HYP] The threshold is probably driven by amortization: sub-word operations benefit from sharing a single helper body across call sites, while word+ operations inline efficiently.

---

## Kernel 07 UMOV 0x400 investigation

Chapter focused on isolating the meaning of the `UMOV UR4, 0x400` instruction carried over from kernel 06. Four variants tested, combined with kernel 01 as the no-shared control.

### Variant 07a (`float[1]`, minimal shared)

* [OBS] UMOV unchanged at 0x400 despite shared being reduced to 4 bytes (1 float).
* [OBS] `if (threadIdx.x == 0)` triggers predication on the shared write path. Instructions `LDC.64`, `IMAD.WIDE`, `LDG.E`, `STS` are all prefixed `@!P0` where P0 is the `tid != 0` test. Only thread 0 executes, the other 31 fetch but do not execute. Confirms the predication rule for short conditional blocks.
* [OBS] With only one shared element accessed by one thread, no LOP3 mask is needed. The shared address is `UR4` directly, used as `[UR4]` for the STS.

### Variant 07b (`int[256]`)

* [OBS] UMOV unchanged at 0x400. SASS is nearly identical to 06b byte for byte.
* [OBS] The type of the shared array does not affect the generated SASS for a simple read-write pattern, since both int and float occupy 4 bytes. The LOP3 mask (0x3fc), the LEA shifts (0x2), the IMAD.WIDE stride (0x4) all remain the same because they encode sizes, not types.

### Variant 07c (`extern __shared__`)

* [OBS] UMOV unchanged at 0x400 despite shared size being a runtime parameter.
* [OBS] ptxas does not know the size at compile time, so it falls back to the runtime modulo pattern seen in kernel 06: CALL to `__cuda_sm20_rem_u16` to compute `(tid + 1) % blockDim.x`.
* [OBS] Dynamic shared memory is treated identically to static shared at the addressing level. Only the size bound (source of the modulo) differs.

### Kernel 01 (no shared, control)

* [OBS] UMOV 0x400 is **absent** from the SASS. The S2UR for `SR_CgaCtaId` is also absent. Neither the UMOV nor the CgaCtaId read appears unless the kernel declares `__shared__`.

### Resolved hypotheses

* [RES] `UMOV UR4, 0x400` appears in every tested SM120 shared-memory variant and is absent from the no-shared control. The value 0x400 is inert across shared size, shared type, static vs dynamic declaration, and number of shared buffers in the tested corpus.
* [INF] The tested deltas constrain `0x400` to the shared addressing mechanism rather than source-level shared size, block size, type, or static/dynamic declaration.
* [GAP] Exact semantic meaning of the `0x400` value remains unknown.
* [RES] Predication is systematically used for short conditional bodies inside a kernel, not only for bounds check exits. Variant 07a confirms ptxas prefixes individual instructions with `@!P0` rather than branching around a 4-instruction body.

### Open hypotheses

* [HYP] Meaning of the specific value `0x400` in `UMOV UR4, 0x400`. Possible interpretations not tested: architectural SM120 constant (base or offset for shared descriptor), bit field format for the following ULEA (where `0x18 = 24` is the shift), value tied to max concurrent blocks per SM or shared banks. To investigate by comparing with SM89 or SM90a dump, or by reading NVIDIA shared descriptor format documentation for SM90+.

---

## Kernel 08 vector_load (vectorized global memory access)

Chapter focused on vectorized LDG/STG instructions. Seven variants tested to map the full spectrum of SM120 load widths and identify the rules.

### Observations

* [OBS] SM120 supports five distinct LDG widths: 32 bits (LDG.E), 64 bits (LDG.E.64), 128 bits (LDG.E.128), 256 bits (LDG.E.ENL2.256). 256 is the maximum.
* [OBS] At 256 bits the `.ENL2` modifier appears. The instruction takes two destination base registers (R16, R12) instead of one, because 8 consecutive registers cannot be referenced by a single register field in the opcode encoding.
* [OBS] The IMAD.WIDE stride immediate always equals sizeof(T) in bytes. Verified across float (0x4), float2 (0x8), float4 (0x10), float8 (0x20), float16 (0x40), double4_32a (0x20).
* [OBS] Vector arithmetic is not packed at SASS level. A source `float4 + float4` produces 4 separate FADD, one per component. No SIMD add instruction for FP32 on SM120.
* [OBS] FP64 compute is drastically slower than FP32. ptxas inserts 3 to 4 NOP between consecutive DADD instructions, something never seen with FADD in any earlier kernel.
* [OBS] Useful compute ratio scales with vector width: 5% for kernel 01 (scalar float), 18% for 08a (float4), 36% for 08d (float8). Plumbing cost stays constant, compute scales with elements per thread.

### Variant 08a (`float4`)

* [OBS] `LDG.E.128` loads 128 bits into 4 consecutive registers from a single base register.
* [OBS] `STG.E.128` stores 128 bits from 4 consecutive registers.
* [OBS] Canonical vectorization case. 22 total instructions for a 4x data volume compared to kernel 01.

### Variant 08b (`float*` with manual indexing)

* [RES] ptxas does NOT auto-vectorize scalar access patterns, even when they are trivially contiguous and provably aligned. **Confirmed.** 8 × LDG.E emitted instead of 2 × LDG.E.128.
* [OBS] ptxas does factor the address arithmetic (one IMAD.WIDE per array, then immediate offsets `+0x4`, `+0x8`, `+0xc` on the loads), but stops short of fusing the loads themselves.
* [OBS] `SHF.L.U32 R9, R0, 0x2, RZ` used for `i * 4`. ptxas recognizes multiplication by a power of 2 as a shift.
* [OBS] Vectorization on SM120 is syntactic. Must use vector types (`float4*`, `double4_32a*`) or `reinterpret_cast` to trigger wide LDG.

### Variant 08c (`float2`)

* [OBS] `LDG.E.64` exists and is used for 64-bit transfers. Destination is a pair of consecutive registers.
* [OBS] Fills the 64-bit slot in the width spectrum between scalar LDG.E and LDG.E.128.

### Variant 08d (`float8` custom struct, 32-byte aligned)

* [OBS] First `LDG.E.ENL2.256` observation. 256-bit transfer, 8 consecutive registers named via 2 base registers in the encoding.
* [OBS] `STG.E.ENL2.256` mirrors the load pattern for stores.

### Variant 08e (`float16` custom struct, 64-byte aligned)

* [OBS] 256 bits is the cap. 64-byte data produces 2 × LDG.E.ENL2.256 per array, not a hypothetical 512-bit instruction.
* [OBS] Address is computed once per array with IMAD.WIDE; the second half is addressed via immediate offset `+0x20` on the second LDG.

### Variant 08f (deprecated `double4`, 16-byte aligned)

* [OBS] 4 × LDG.E.128 emitted instead of 2 × LDG.E.ENL2.256, despite the same 32-byte total size as float8.
* [OBS] DADD compute shows consistent NOP padding (4 NOPs between each DADD).
* [OBS] Compiler warning: `double4` is deprecated, use `double4_16a` or `double4_32a`.

### Variant 08g (`double4_32a`)

* [RES] The limitation in 08f was **alignment**, not element size. **Confirmed.** With 32-byte alignment, ptxas emits `LDG.E.ENL2.256` exactly as for `float8`.
* [OBS] The width rule is: ptxas uses the widest LDG whose byte width does not exceed the alignment guarantee of the source type.
* [OBS] DADD still shows 3 to 4 NOP padding. FP64 pipeline latency is independent of the memory path used to feed it.

### Resolved hypotheses

* [RES] SM120 LDG width caps at 256 bits. **Confirmed** by float16 splitting into 2 × 256-bit loads.
* [RES] Auto-vectorization does not happen in ptxas for SM120. **Confirmed** by 08b (scalar float pointer with aligned contiguous access).
* [RES] Vector arithmetic is scalar at SASS. **Confirmed** by every variant in this chapter.
* [RES] Alignment of the source type determines usable LDG width. **Confirmed** by the double4 versus double4_32a comparison.

### Open hypotheses

* [HYP] Semantics of `.ENL2` beyond register encoding. Does it also change memory access behavior (latency, cache policy, memory order)? To microbenchmark.
* [HYP] Whether `LDG.E.ENL4.512` or wider modes exist. Not observed with float16. A `float16_64a` or similar 64-byte aligned struct might trigger it, if it exists.
* [HYP] Exact FP32 to FP64 ratio on SM120. NOP pattern between DADDs is consistent with the documented 64:1 consumer ratio but not a direct measurement.
* [HYP] `__restrict__` effect on auto-vectorization in scalar kernels. Not tested.
* [HYP] Whether `.ENL2.256` exists on SM89 and SM80, or whether those architectures cap LDG width at 128 bits. To verify by cross compiling.

---

## Kernel 09 warp shuffle and vote (warp-level communication primitives)

Chapter focused on warp-synchronous communication primitives. Twelve variants tested covering SHFL family, VOTE family, MATCH family, synchronization helpers, and edge cases (64-bit operands, partial masks, divergent predecessors).

### Observations (general)

* [OBS] SM120 has four distinct SHFL opcodes: SHFL.IDX, SHFL.BFLY, SHFL.UP, SHFL.DOWN. Each maps to one CUDA intrinsic.
* [OBS] SHFL format: `SHFL.{variant} Pdst, Rdst, Rsrc, mask_or_lane, segment_mask`. The segment mask is `0x1f` for IDX, BFLY, DOWN, but `RZ` for UP.
* [OBS] VOTE has two forms: predicate output (`VOTE.ANY Pdst, Psrc`) for `__any_sync` and `__all_sync`, register output (`VOTE.ANY Rdst, PT, Psrc`) for `__ballot_sync`.
* [OBS] MATCH has two opcodes: MATCH.ANY and MATCH.ALL. MATCH.ALL produces both a predicate and a register in a single instruction, the only such observed dual-output opcode so far.
* [OBS] `BSSY.RECONVERGENT` and `BSYNC.RECONVERGENT` wrap SHFL only when the SHFL is preceded by divergent code. No divergence means no reconvergence wrapper.
* [OBS] `__syncwarp()` has no dedicated SASS opcode on SM120. ptxas emits NOP padding when it must stall (first call before an unready SHFL) or eliminates it entirely (redundant calls).
* [OBS] `__syncwarp(partial_mask)` causes ptxas to emit a `MOV` of the mask into an unused register plus a single NOP. The mask does not affect the subsequent SHFL segment encoding.
* [OBS] `__activemask()` is the idiom `VOTE.ANY Rdst, PT, PT` (ballot with always-true predicate).
* [OBS] 64-bit SHFL is two consecutive 32-bit SHFL instructions, one per half. No SHFL.64 variant.
* [OBS] Division by 32 (for warp_id computation) compiles to `SHF.R.U32.HI R, RZ, 0x5, R`. Consistent with the power-of-2 division rule from kernel 06.
* [OBS] New SASS variant observed: `IMAD.WIDE.U32`. Used when ptxas can prove the index is non-negative (e.g., warp_id derived from tid/32).
* [OBS] `LOP3.LUT P, ..., 0xc0, !PT` fuses `x AND mask` with compare-to-zero, producing a predicate directly. Pattern repeated across most variants for the `(tid & 31) == 0` lane-zero test.
* [OBS] Initialization of `float val = 0.0f` before a possibly-divergent LDG uses `HFMA2 R, -RZ, RZ, 0, 0` (FP16 packed FMA trick, same as kernels 04 and 05 for arbitrary FP32 constants).

### Variant 09a (SHFL.BFLY with bounds check)

* [OBS] First observation of `BSSY.RECONVERGENT B0, 0xf0` and `BSYNC.RECONVERGENT B0` wrapping a divergent LDG. The SHFL sequence that follows requires all threads active.
* [OBS] 5 SHFL.BFLY + 5 FADD for a 32-thread butterfly reduction. Classic pattern.
* [OBS] HFMA2 used to initialize R2 = 0.0f before the divergent LDG, so out-of-bounds threads participate in SHFL with a defined value.

### Variant 09b (SHFL.BFLY without bounds check)

* [RES] BSSY/BSYNC.RECONVERGENT appears only when SHFL is preceded by divergent code. **Confirmed** by removal of the bounds check.
* [OBS] 21 useful instructions, down from 30 in 09a.

### Variant 09c (SHFL.IDX broadcast)

* [OBS] `SHFL.IDX PT, R9, R2, RZ, 0x1f`. Format: 4th operand is the source lane (RZ = lane 0), 5th operand is segment mask.

### Variant 09d (VOTE via `__ballot_sync`)

* [OBS] `FSETP.GT.AND P0, PT, R2, RZ, PT` then `VOTE.ANY R5, PT, P0`. Predicate input, register output (mask).

### Variant 09e (`__syncwarp()` full mask)

* [OBS] First `__syncwarp()` before a SHFL: 6 NOPs inserted. Subsequent `__syncwarp()` calls: eliminated completely.
* [OBS] `__syncwarp()` prevents ptxas from interleaving useful instructions between LDG and SHFL. The 6 NOPs replace what would otherwise be a LOP3 or similar filler.

### Variant 09f (SHFL.UP)

* [OBS] `SHFL.UP PT, R9, R2, 0x4, RZ`. 5th operand is `RZ`, not `0x1f` as in other SHFL variants.
* [HYP] The different encoding likely relates to UP's out-of-range semantics (lane N-delta does not exist for small N, so the lane keeps its original value). Requires a distinct mode bit.

### Variant 09g (SHFL.DOWN)

* [OBS] `SHFL.DOWN PT, R9, R2, 0x4, 0x1f`. Format matches BFLY and IDX, not UP.

### Variant 09h (VOTE.ALL vs VOTE.ANY)

* [RES] VOTE.ANY and VOTE.ALL are **two distinct SASS opcodes**, not the same opcode with a mode flag. Confirmed by side-by-side emission in the same kernel.
* [OBS] Both use predicate output form when invoked via `__any_sync` or `__all_sync`.

### Variant 09i (REDUX, moved to chapter 10)

* [OBS] Quick observation: `REDUX.SUM.S32 UR7, R2` replaces the entire 5 × SHFL.BFLY + 5 × FADD sequence of 09a with a single instruction. Details covered in chapter 10.

### Variant 09j (MATCH.ANY and MATCH.ALL)

* [OBS] `MATCH.ANY R0, R2` produces a register (mask of lanes with same value as mine).
* [OBS] `MATCH.ALL PT, R5, R2` produces a predicate (all lanes match) AND a register (the mask), in a single instruction.
* [OBS] First dual-output instruction (simultaneous predicate + register) in our SASS vocabulary.

### Variant 09k (`__syncwarp(partial_mask)`)

* [OBS] ptxas emitted `MOV R11, 0xffff` for the mask, but R11 is never read afterwards. Possibly dead code or a scheduler hint.
* [OBS] No `WARPSYNC` opcode was generated despite a partial mask. The SHFL still used segment mask `0x1f` (full warp).
* [HYP] On SM120, `__syncwarp` mask is effectively ignored at the SASS level. The intrinsic is a vestige of the Volta Independent Thread Scheduling transition requiring explicit participation declaration. To verify with deliberately divergent kernels.

### Variant 09l (`__activemask()`)

* [OBS] Compiles to exactly one instruction: `VOTE.ANY R5, PT, PT`. Ballot with always-true predicate returns the mask of active lanes.
* [OBS] No dedicated `ACTIVEMASK` opcode exists.

### Variant 09m (SHFL on 64-bit)

* [OBS] `__shfl_sync` on a `double` compiles to 2 × SHFL.IDX, one per 32-bit half.
* [OBS] The two SHFL are emitted in high-half-first order. Reason unclear (hypothesis: ptxas orders by encoding constraints or dependency distance).

### Resolved hypotheses

* [RES] BSSY/BSYNC.RECONVERGENT is conditional on preceding divergence. **Confirmed** (09b).
* [RES] VOTE.ALL exists as distinct opcode from VOTE.ANY. **Confirmed** (09h).
* [RES] `__syncwarp` has no dedicated SASS opcode. **Confirmed** (09e, 09k).
* [RES] MATCH.ALL produces simultaneous predicate + register output. **Confirmed** (09j).

### Open hypotheses

* [HYP] Semantic difference between SHFL.UP's `RZ` and other SHFL variants' `0x1f` in the 5th operand. Encoding-only, or sem difference in out-of-range behavior?
* [HYP] The MOV of partial mask in 09k: dead code or scheduler hint?
* [HYP] Pipeline assignment for SHFL (documentation says LSU but not confirmed by gpuasm).
* [HYP] Behavior of SHFL under genuine divergence (half-warp taking different paths). Not tested.
* [RES] `WARPSYNC` opcode exists on SM120. Chapter 21 variant 21n observes `@P0 WARPSYNC.ALL` before a predicated HMMA. The earlier Kernel 09 result remains narrower: `__syncwarp()` itself did not lower to WARPSYNC in 09e/09k.

### New instructions observed in this chapter

| Opcode | Usage |
|---|---|
| SHFL.BFLY | Butterfly shuffle (XOR pattern) |
| SHFL.IDX | Broadcast from specified lane |
| SHFL.UP | Lane N reads lane N-delta |
| SHFL.DOWN | Lane N reads lane N+delta |
| VOTE.ANY (predicate) | Predicate OR across warp |
| VOTE.ALL (predicate) | Predicate AND across warp |
| VOTE.ANY (register) | Ballot (active-thread mask with predicate filter) |
| MATCH.ANY | Register: mask of lanes with matching value |
| MATCH.ALL | Predicate + Register: all lanes match + mask |
| BSSY.RECONVERGENT | Declare reconvergence scope |
| BSYNC.RECONVERGENT | Close reconvergence scope |
| IMAD.WIDE.U32 | Unsigned-explicit 64-bit multiply-add |

---

## Kernel 10 REDUX (hardware warp reduction)

Chapter dedicated to the REDUX instruction family: hardware warp reduction added in Ampere (SM80), present on SM120. Eight variants tested covering SUM, MIN/MAX (signed and unsigned), AND, OR, XOR.

### Observations (general)

* [OBS] All 8 REDUX variants share the same opcode bytes `0x00000000020773c4`. Only the control code differs between variants. REDUX is a single opcode with eight semantic variants encoded in the control code bits.
* [OBS] Three control code bits encode the operation: bits 14-16 of the control code. Pattern: bit 16 selects between bitwise/arithmetic (0) and comparison (1) classes. Within each class, bits 14-15 select the specific operation.
* [OBS] Bit 9 (value 0x200) of the control code encodes signedness. Set to 1 for `.S32` variants, 0 for unsigned and bitwise variants.
* [OBS] cuobjdump display asymmetry: `.U32` is never shown (unsigned MIN/MAX appear as `REDUX.MIN` and `REDUX.MAX`). `.AND` is never shown (AND appears as plain `REDUX`). Only `.S32` is explicit.
* [OBS] REDUX is the first SASS instruction in our chapters that writes across register files: per-thread register input, uniform register output. `REDUX UR7, R2` takes a per-thread R register and writes to a uniform UR register.
* [OBS] Downstream per-thread consumers require a MOV R, UR to cross back to the per-thread file. The MOV waits on REDUX's scoreboard (REDUX is variable latency).
* [OBS] Identity load strategy depends on the identity value. HFMA2 when the 32-bit bit pattern can be expressed as concatenation of two half-float values (0x00000000, 0x80000000). MOV 32-bit immediate otherwise (0x7FFFFFFF, 0xFFFFFFFF).
* [OBS] `HFMA2 R, -RZ, RZ, -0.0, 0` loads 0x80000000 into R. The `-0.0` in the high half-float slot sets bit 31 of the concatenated result. First observation of HFMA2 used to load a specific integer bit pattern (INT_MIN) rather than an arbitrary FP32 constant.
* [OBS] 25 of 27 instructions are byte-identical across all 8 variants. Only the identity load (address 0x0050) and the REDUX itself (address 0x00f0) differ. The kernel skeleton is mechanically determined by the surrounding source (bounds check + warp-sync + per-warp output).
* [OBS] REDUX replaces the 10-instruction SHFL butterfly reduction (5 SHFL.BFLY + 5 FADD/IADD) with 1 instruction and 1 pipeline stage instead of 5 sequential stages.

### Resolved hypotheses

* [RES] REDUX is a single opcode with operation-in-control-code encoding, not a family of distinct opcodes. Confirmed by byte-identical opcode bytes across 8 variants.
* [RES] HFMA2 can load INT_MIN (0x80000000) via the `-0.0` half-float trick. Confirmed by kernel 10c.
* [RES] Ptxas selects HFMA2 vs MOV immediate based on whether the target constant is half-float encodable. Confirmed by the 8 identity loads.
* [RES] The MOV R, UR after REDUX is the scoreboard consumer. Confirmed by the stall/wait pattern visible in the control code of the MOV at address 0x0110.

### Open hypotheses

* [HYP] `__reduce_add_sync` on unsigned int produces the same SASS as on signed int. Not directly tested (would need a 10i variant).
* [HYP] Bits 15+16 of control code simultaneously set may correspond to reserved encodings. Not observed.
* [HYP] Cluster-scope CREDUX (mentioned in the herrmann SASS evolution gist for SM100a) may exist on SM120. Not tested, our kernels do not use clusters.
* [HYP] Full decomposition rules for "which integer constants are HFMA2-loadable" not formalized. **Chapter 11 extends the observed target space**: beyond 0x00000000 and 0x80000000 (integer-as-packed-FP16 loads), we observe HFMA2 loading arbitrary bit patterns that correspond to two valid FP16 constants concatenated (e.g., `HFMA2 R, -RZ, RZ, 1.443359375, -0.2030029296875` in log2f). The target space is: any 32-bit value whose high 16 bits are a valid FP16 constant that ptxas's constant table knows how to emit, and same for low 16 bits. Targets that cannot be split into two FP16 constants fall back to MOV immediate.

### Gaps

* [GAP] Full bit decoding of REDUX scheduling portion of the control code (stall, yield, SBS, wait mask) not validated against a parser. Inferred from context but not cross-checked.
* [GAP] REDUX cycle latency not measured. Microbenchmarking deferred per D009.
* [GAP] REDUX unsigned SUM not tested (no `__reduce_add_sync` on unsigned variant written).

### New instructions observed in this chapter

| Opcode | Usage |
|---|---|
| REDUX | Warp AND reduction, unsigned (default when no suffix) |
| REDUX.OR | Warp bitwise OR |
| REDUX.XOR | Warp bitwise XOR |
| REDUX.SUM.S32 | Warp integer sum, signed |
| REDUX.MIN | Warp min, unsigned (implicit) |
| REDUX.MIN.S32 | Warp min, signed |
| REDUX.MAX | Warp max, unsigned (implicit) |
| REDUX.MAX.S32 | Warp max, signed |

### New idioms observed in this chapter

* `HFMA2 R, -RZ, RZ, -0.0, 0` loads INT_MIN (0x80000000) into R.
* `MOV R, 0x7fffffff` loads INT_MAX.
* `MOV R, 0xffffffff` loads UINT_MAX or all-ones mask for AND.
* `MOV R, UR` cross-file transfer after REDUX, to feed per-thread consumers.

---

## Kernel 11 slowpath arithmetic (division, math library, hardware MUFU)

### Variants and compilation strategies

* [OBS] 11a `a[i] / d` u32 runtime: fully inline, no CALL. Twenty-one instructions implementing Granlund-Montgomery reciprocal multiplication.
* [OBS] 11b `a[i] / d` u64 runtime: hybrid. Fast path inline (u32 algorithm operating on 64-bit operands) + slowpath via `CALL.REL.NOINC 0x2e0` to a local subroutine after the main EXIT.
* [OBS] 11c `a[i] / d` s32 runtime: inline via IABS + unsigned algorithm + LOP3 0x3c sign XOR fix-up.
* [OBS] 11d `log2f(x)` standard: fully inline polynomial approximation with 10-coefficient Remez approximation evaluated via Horner's method. No CALL, no MUFU.LG2.
* [OBS] 11e `__log2f(x)` intrinsic: single MUFU.LG2 wrapped with subnormal handling. Four instructions total.
* [OBS] 11f `expf(x)` standard: inline range reduction + MUFU.EX2 + FMUL. No CALL despite being an IEEE-accurate standard library function.
* [OBS] 11g `sinf(x)` standard: dual-path. Fast path for `|x| < 105615` uses triple-FFMA range reduction. Slowpath via BRA (not CALL) implements Payne-Hanek reduction with LDG.E.CONSTANT from a 2/π table + FP64 DMUL + F2F.F32.F64.
* [OBS] 11h `sqrtf(x)` standard: fast path with MUFU.RSQ + 1 Newton-Raphson iteration + CALL.REL.NOINC to local subroutine at 0x1d0 for NaN/Inf/denormal cases.
* [OBS] 11i `rsqrtf(x)`: single MUFU.RSQ + subnormal handling. Four instructions.
* [OBS] 11j `__fdividef(a, b)`: MUFU.RCP + FMUL + subnormal handling on both operands.

### Major structural finding

* [OBS] On SM120 with CUDA 13.2, ptxas does NOT emit external named helpers (`__cuda_sm20_div_u32`, `__cuda_sm20_rem_u32`, etc.) for integer division at u32 width or above, nor for any math library function tested. Every operation compiles to one of three shapes: fully inline, inline with local BRA slowpath, or inline with CALL to local subroutine.
* [OBS] The only externally-named helper observed in the project remains `__cuda_sm20_rem_u16` from chapter 06. This is the u16-specific case; u32 and beyond are inlined.
* [HYP] This is either a CUDA 13+ inlining threshold change, or external helpers are reserved for sub-word types (u16) where calling overhead amortizes better.

### Integer division inner structure

* [OBS] All three integer division variants (11a u32, 11b u64 fast path, 11c s32) share the same 5-stage Granlund-Montgomery skeleton: UI2F.U32.RP (convert divisor, round +inf) → MUFU.RCP → IADD with magic `0xffffffe` adjustment → F2I.FTZ.U32.TRUNC.NTZ → IMAD.HI.U32 correction chain (one pass for u32, two for u64).
* [OBS] Double correction pair at the end of u32 division (two successive `ISETP.GE.U32.AND P0/P1 + @P IADD` pairs) handles the case where the reciprocal is underestimated by 1 or 2 units.
* [OBS] Zero-divisor sentinel via `LOP3.LUT R, RZ, divisor, RZ, 0x33, !PT` returns `~divisor` when divisor is 0 (UB guard, not IEEE-specified).
* [OBS] Signed division uses LOP3 0x3c (XOR truth table) on the MSBs of dividend and divisor to derive the result sign, which is then applied via `@pred IADD R, -R, RZ`.
* [HYP] The magic constant `0xffffffe = (2^28 - 2)` tunes F2I rounding behavior to minimize correction passes. Its derivation from algorithmic error analysis is not verified.

### 64-bit arithmetic primitives in slowpath

* [OBS] 11b slowpath uses multi-precision arithmetic: IMAD.WIDE.U32 (32×32→64), IADD3.X (three-input add with carry-in), IADD.X (two-input add with carry-in), IADD.64 (single-operation 64-bit add), ISETP.GE.U64.AND, ISETP.NE.S64.AND, SEL.64.
* [OBS] The `.X` suffix indicates carry-in from the condition code register (CC). IADD3.X reads CC, IADD3 doesn't.
* [OBS] The algorithm performs two Newton-Raphson iterations on an initial MUFU.RCP estimate, then computes remainder and quotient correction with multi-precision arithmetic.

### log2f polynomial structure

* [OBS] Range reduction: `IADD R4, R0, -0x3f3504f3` subtracts the bit pattern of `sqrt(2)/2 ≈ 0.7071` from the input's bit representation. This simultaneously extracts the unbiased exponent and shifts the mantissa into `[sqrt(2)/2, sqrt(2)]`.
* [OBS] Exponent extraction: `LOP3.LUT R7, R4, 0xff800000, RZ, 0xc0, !PT` masks the top 9 bits (sign + exponent). `IADD R4, R0, -R7` isolates the mantissa_scaled. `I2FP.F32.S32 R7, R7` converts the exponent to a float for the final reconstruction.
* [OBS] Polynomial body: 9 Horner-method FFMA + 2 FMUL + 1 FFMA with the log2(e) scale factor. Coefficients (in order): `-0.16845, 0.17169, -0.17901, 0.20512, -0.24047, 0.28857, -0.36067, 0.48090, -0.72135`, final multiplier `1.44270` = log2(e).
* [OBS] Edge case handling: `@P0 FFMA R4, R0, +INF, +INF` returns +INF for input > largest normal. `FSEL R5, R4, -INF, P1` returns -INF if input was zero.
* [OBS] HFMA2 at 0x00c0 packs `(1.44, -0.20)` as FP16 pair into R9 which is then consumed as FP32 in the first polynomial FFMA. [GAP] Why HFMA2 packed loading over MOV immediate is used remains unclear.

### expf range reduction structure

* [OBS] Decomposition: `exp(x) = 2^(x * log2(e)) = 2^(k + f) = 2^k * 2^f` where k is integer, f is fractional.
* [OBS] Magic number `12582913 = 0x00C00001` combined with `FFMA.RM` (round toward -inf) implements the classic "magic number" trick to extract the integer part of a float via controlled overflow.
* [OBS] `0x437c0000 = 252.0` is used as an intermediate scale factor.
* [OBS] Dual FFMA with coefficients `1.4426950216293334961` and `1.925963033500011079e-08` implements extended-precision multiplication by log2(e), giving ~50 bits of effective precision on the product.
* [OBS] After computing the fractional residual, `SHF.L.U32 R0, R0, 0x17, RZ` shifts the integer part into the FP32 exponent position (shift by 23). MUFU.EX2 handles the fractional part. Final `FMUL R5, R0, R9` combines them.
* [HYP] These magic constants match glibc expf patterns; verification against CUDA-specific documentation left as [GAP].

### Payne-Hanek range reduction for sinf

* [OBS] Triggered when `|x| >= 105615 ≈ 2^17`. For smaller inputs, triple-FFMA range reduction with coefficients of π/2 (at progressively smaller orders of magnitude) is sufficient.
* [OBS] The slowpath loads a 2/π table from constant memory (`LDCU.64 UR4, c[0x4][URZ]`) and iterates 6 times (counter UR4 from 0 to 6 via UIADD3).
* [OBS] Each iteration: LDG.E.CONSTANT loads 32 bits of the 2/π table, IMAD.WIDE.U32 accumulates partial products, predicate-gated MOV distributes results into R11..R16.
* [OBS] Loop uses `BRA.U UP0, <start>` with uniform predicate (loop is warp-uniform).
* [OBS] After the loop, FP64 DMUL (`UR4 = 0x3bf921fb54442d19 = π/2 in IEEE 754 double`) and F2F.F32.F64 complete the reduction.
* [OBS] The final polynomial evaluation uses 4 FFMA + FSEL pairs implementing `sin(x) ≈ x + c3*x^3 + c5*x^5 + c7*x^7` with coefficients selected based on quadrant (via R2P and predicate selection).
* [GAP] Exact per-iteration reconstruction of the Payne-Hanek accumulator from R11..R16 (combining via SHF.L.W.U32.HI at 0x05c0 and 0x05e0) is not fully traced.

### sqrtf Newton-Raphson structure

* [OBS] Fast path: one Newton-Raphson iteration starting from MUFU.RSQ. Sequence: `MUFU.RSQ y0`; `FMUL q0 = x*y0`; `FMUL y0/2`; `FFMA e = x - q0²`; `FFMA sqrt = q0 + e*y0/2`.
* [OBS] Range test `R4 > 0x727fffff` after `IADD R4, R0, -0xd000000` biases the input such that normal positive finite values fall below the threshold and denormals, NaN, and infinity fall above. Triggers BRA to slowpath.
* [OBS] Slowpath handles: zero (sqrt(0) = 0), negative (returns 0x7fffffff NaN), NaN (propagates), infinity (returns infinity), denormal (scale by 2^64, compute, rescale by 2^-32).
* [OBS] Slowpath uses `CALL.REL.NOINC 0x1d0` with return address in R9, and `RET.REL.NODEC R2 0x0` at the slowpath exit.

### Local CALL pattern (both 11b and 11h)

* [OBS] Structure: `BSSY.RECONVERGENT B0, <sync_address>` before divergent region, then `MOV Rn, <return_address>` + `CALL.REL.NOINC <slowpath_address>`, with `BSYNC.RECONVERGENT B0` at reconvergence, then main body completion.
* [OBS] Slowpath lives in the same `.text` section as the main kernel, placed after the main EXIT, unreachable from normal control flow.
* [OBS] Return register is chosen per-kernel based on local register pressure: R4 in 11b, R9 in 11h. This is a compiler decision, not an ABI.
* [OBS] `RET.REL.NODEC R? 0x0` reads the return address from the specified register. The `0x0` is the offset to subtract from the return address; always 0 in our observations.

### Subnormal handling pattern (universal)

* [OBS] Four kernels (11d, 11e, 11i, 11j) share identical subnormal handling skeleton: `FSETP.GEU.AND P0, PT, |R|, 1.175494350822287508e-38, PT` (test against FLT_MIN) → `@!P0 FMUL R, R, 16777216` (scale by 2^24 if subnormal) → `<MUFU or polynomial>` → `@!P0 <correction>` (adjust for the scaling).
* [OBS] Correction depends on operation: subtract 24 for log2, multiply by 2^12 = 4096 for rsqrt (since sqrt halves the exponent), no correction needed for fdividef (both operands scaled equally).
* [OBS] `1.175494350822287508e-38` is exactly FLT_MIN, the smallest positive normal FP32.

### New opcodes (29 total)

| Opcode | Where | Semantics |
|---|---|---|
| UI2F.U32.RP | 11a, 11b | Uniform u32 to float, round +inf |
| I2F.RP | 11c | Signed int32 to float, round +inf |
| I2F.U64.RP | 11b | u64 to float, round +inf |
| I2F.F64.S64 | 11g | s64 to double |
| I2FP.F32.S32 | 11d, 11g | s32 to float, FP32 output explicit |
| F2I.FTZ.U32.TRUNC.NTZ | 11a, 11b | Float to u32, FTZ, truncate, NTZ |
| F2I.U64.TRUNC | 11b | Float to u64, truncate |
| F2I.NTZ | 11g | Float to int, non-toward-zero rounding |
| F2F.F32.F64 | 11g | Double to float narrowing |
| IABS | 11c | Integer absolute value |
| FSEL | 11d, 11g, 11h | Float select based on predicate |
| SEL.64 | 11b | 64-bit select based on predicate |
| FFMA.SAT | 11f | FFMA with output saturated to [0, 1] |
| FFMA.RM | 11f | FFMA with round mode toward -inf |
| FMUL.FTZ | 11h | FMUL with flush-to-zero |
| FADD.FTZ | 11h | FADD with flush-to-zero |
| MUFU.LG2 | 11e | Hardware log2 approximation |
| MUFU.EX2 | 11f | Hardware 2^x approximation |
| MUFU.RSQ | 11h, 11i | Hardware reciprocal sqrt approximation |
| R2P | 11g | Register to predicates (unpack bits to P0..P7) |
| DMUL | 11g | FP64 multiply |
| IMAD.SHL.U32 | 11g | Integer multiply-add with implicit left shift, unsigned |
| IMAD.U32 | 11g | Integer multiply-add with explicit u32 semantics |
| IMAD.WIDE.U32 | 11b | 32×32→64 multiply unsigned explicit |
| SHF.L.W.U32.HI | 11g | Funnel shift left, with wrap, high half, unsigned |
| SHF.R.U32.HI | 11g | Funnel shift right, high half, unsigned |
| SHF.R.S32.HI | 11g | Funnel shift right, high half, signed (arithmetic) |
| LEA.HI | 11g | High-half LEA, returns top 32 bits of `(src1 << imm) + src2` |
| IADD3.X | 11b | Three-input add with carry-in |
| IADD.X | 11b | Two-input add with carry-in |
| IADD.64 | 11b | 64-bit integer add |
| ISETP.GE.U64.AND | 11b | 64-bit unsigned compare |
| ISETP.NE.S64.AND | 11b | 64-bit signed compare |
| MOV.64 | 11g | 64-bit move (register pair) |
| UMOV.64 | 11g | Uniform 64-bit move with 64-bit immediate support |
| BRA.U | 11g | Uniform conditional branch |
| UISETP.NE.U32.AND | 11g | Uniform integer compare (loop control) |
| CALL.REL.NOINC | 11b, 11h | Relative call without stack pointer increment |
| RET.REL.NODEC | 11b, 11h | Relative return without stack pointer decrement |
| BSSY.RECONVERGENT | 11b, 11g, 11h | Open reconvergence scope |
| BSYNC.RECONVERGENT | 11b, 11g, 11h | Close reconvergence scope |

### New modifiers

* [OBS] `.FTZ` (flush-to-zero) on FMUL/FADD/FSETP/F2I: subnormals treated as zero for input and output.
* [OBS] `.SAT` (saturate) on FFMA: result clamped to [0, 1].
* [OBS] `.RM` (round minus, toward -inf) on FFMA.
* [OBS] `.RP` (round plus, toward +inf) on I2F/UI2F.
* [OBS] `.NEU` (not equal unordered) on FSETP.
* [OBS] `.GTU` (greater than unordered) on FSETP.
* [OBS] `.NTZ` (non-toward-zero) on F2I: signed zero edge case resolution.
* [OBS] `.W` (with wrap) on SHF.L: funnel shift behavior.
* [OBS] `.CONSTANT` on LDG.E: load from constant cache path for read-only global memory access. First observation in project.

### Observations worth flagging

* [OBS] `R2P PR, R3, 0x3` in 11g: copies selected bits of R3 into predicate registers P0..P7. Rare opcode, useful when packed flags need to fan out into multiple predicated paths.
* [OBS] `.reuse` flag frequency correlates with register pressure: 6 occurrences in 11b, 7 in 11g, zero in simpler kernels (11a, 11c, 11e, 11f, 11i, 11j).
* [OBS] LDG.E.CONSTANT is distinct from LDC (constant memory bank) and LDCU (uniform constant). Used when a read-only table is stored in global memory but accessed via the constant cache path.
* [OBS] ISETP with `.S64` and `.U64` variants are first observed in 11b. Same structure as 32-bit variants but operate on register pairs.
* [OBS] **IABS as a single-instruction absolute value.** Kernel 11c uses `IABS R9, R11` to compute `|divisor|` in one instruction. Avoids the need for an ISETP + @pred IADD pair.
* [OBS] **SEL.64 assembles multi-precision results.** In 11b at 0x0690, `SEL.64 R2, R2, -0x1, P0` selects between a computed 64-bit quotient and -1 (all-ones 64-bit) based on a predicate. Used as the final assembly step of the u64 division result.
* [OBS] **LEA.HI emitted in 11g.** `LEA.HI R3, R9, R3, RZ, 0x1` at 0x0690 computes the high 32 bits of `(R9 << 1) + R3`. First observation of LEA.HI in the project; used in Payne-Hanek accumulation.
* [OBS] **SHF.R.S32.HI for sign-aware shift.** In 11g at 0x0640, `SHF.R.S32.HI R4, RZ, 0x1f, R9` sign-extends the high bit of R9 across all 32 bits of R4 (arithmetic shift right). Used to build a sign mask for later XOR operations. Distinct from SHF.R.U32.HI (logical shift).
* [OBS] **LOP3.LUT 0xfc is "A OR B" truth table.** In 11b at 0x00e0, `LOP3.LUT R4, R3, UR4, RZ, 0xfc, !PT` computes `R3 | UR4` (OR ignoring the third operand via RZ). Used to test if ANY bit of the u64 divisor's high word is set — if so, fast path is inapplicable and the slowpath CALL is taken.
* [OBS] **IMAD.U32 with negative immediate.** In 11g at 0x03b0, `IMAD.U32 R9, R6, -0x4, RZ` computes `R9 = R6 * -4 + 0`. The `.U32` variant is distinct from `.SHL.U32` — here the immediate is an arbitrary 32-bit value treated as the lower multiplier, not as a shift count.
* [OBS] **`.FTZ` ubiquitous in sqrtf slowpath.** Every FP instruction in the 11h slowpath (0x01d0-0x02f0) uses `.FTZ`: FMUL.FTZ, FADD.FTZ, FSETP.GEU.FTZ, FSETP.GTU.FTZ, FSETP.NEU.FTZ. This pattern isolates the subnormal handling logic from unexpected subnormal behavior in intermediate computations. [HYP] IEEE-precise sqrt may require explicit FTZ semantics for correct edge case handling.
* [OBS] **NaN propagation via FADD + 1.** In 11h slowpath at 0x0240, `@P0 FADD.FTZ R2, R0, 1` propagates NaN (the result of NaN + anything is NaN). Idiomatic way to copy a NaN while ensuring the result is a canonical NaN.
* [OBS] **FSEL with negation pattern for sign flipping.** In 11g at 0x0770, `FSEL R6, -R4, R4, !P1` selects between `-R4` and `R4` based on P1. Conditionally negates a value in one instruction without needing FADD.
* [OBS] **UMOV.64 with 64-bit immediate.** In 11g at 0x0630, `UMOV.64 UR4, 0x3bf921fb54442d19` loads the 64-bit FP64 representation of π/2 into a uniform register pair. First observation of a 64-bit immediate load in the project.
* [OBS] **Integer-bit-pattern FP32 constants via MOV.** In 11g at 0x07d0-0x07f0, `MOV R10, 0x3d2aaabb` and `MOV R9, 0x3effffff` load FP32 constants by their bit pattern as 32-bit integers. Values `0x3d2aaabb ≈ 0.04166` and `0x3effffff ≈ 0.5` are polynomial coefficients for the sin series.
* [OBS] **FFMA/FSEL accept +INF and -INF as immediates.** In 11d at 0x0280, `@P0 FFMA R4, R0, R7, +INF` uses +INF directly as the addend. In 11d at 0x0290, `FSEL R5, R4, -INF, P1` uses -INF as a selectable operand. These FP edge values are immediate-encodable in the instruction.
* [OBS] **FADD with large negative integer immediate.** In 11f at 0x0100, `FADD R5, R0, -12583039` uses `-12583039` which when interpreted as FP32 bit pattern is a specific float value used in the expf magic-number trick. Not an integer; the SASS disassembler shows the integer form but ptxas encoded an FP32 value.
* [OBS] **NOP padding around DMUL confirms FP64 throughput restriction.** In 11g at 0x06e0-0x0760, the DMUL at 0x0710 is surrounded by 4 NOPs before and 4 NOPs after. Consistent with the FP64 consumer-part throughput restriction first observed in kernel 08f. Confirms that FP64 operations on SM120 cannot issue back-to-back with normal arithmetic.
* [OBS] **Constant 0x3bf921fb54442d19 = π/2 in IEEE 754 FP64.** Loaded via UMOV.64 and consumed by DMUL. Standard mathematical constant, confirmed against IEEE 754 encoding.
* [OBS] **Constant 0x3f3504f3 = sqrt(2)/2 in IEEE 754 FP32.** Used as subtraction constant in log2f range reduction. Subtracting this bit pattern from an FP32 value shifts the mantissa by one binade and prepares the polynomial evaluation interval.
* [OBS] **Constant 2.3283064365386962891e-10 = 2^-32.** Used in 11h slowpath to rescale the sqrt result after the 2^64 input scaling (since sqrt(2^64) = 2^32, to undo the scaling we multiply by 2^-32).
* [OBS] **Constant 1.84467440737095516160e+19 = 2^64.** Used in 11h slowpath as the scale factor to bring denormal inputs into the normal range before MUFU.RSQ.
* [OBS] **Constant 0x727fffff is the exponent threshold for sqrtf fast path.** Inputs with biased exponent > 0xe4 (after the -0xd000000 shift) fall into the slowpath. Catches denormals, very-small-positive, very-large-positive, NaN, and Inf.
* [OBS] **Absolute value operand on FSETP.** In 11g at 0x00e0 and several other places, `FSETP.GE.AND P0, PT, |R0|, 105615, PT` takes the absolute value of R0 at read time. No separate IABS or FADD needed.

### Constants catalog (magic numbers observed)

Numerical constants appearing as SASS immediates in this chapter, with their mathematical meaning:

| Value (hex) | Value (decimal / interpretation) | Used in |
|---|---|---|
| `0xffffffe` | `2^28 - 2` | Granlund-Montgomery reciprocal adjustment |
| `0x3f3504f3` | `sqrt(2)/2` as FP32 bit pattern | log2f range reduction |
| `0x437c0000` | `252.0` as FP32 | expf scale factor |
| `12582913` = `0x00C00001` | Magic integer extraction constant | expf FFMA.RM |
| `-12583039` | FP32 bit pattern for residual | expf fractional extraction |
| `16777216` = `0x4B800000` | `2^24` as FP32 | Subnormal scale factor |
| `4096` = `0x45800000` | `2^12` as FP32 | Subnormal rsqrt correction |
| `1.84467440737095516160e+19` | `2^64` | sqrtf denormal scale |
| `2.3283064365386962891e-10` | `2^-32` | sqrtf post-scale correction |
| `1.4426950216293334961` | `log2(e)` high precision | expf, log2f |
| `1.925963033500011079e-08` | `log2(e)` low bits | expf extended precision |
| `0.63661974668502807617` | `2/π` | sinf range reduction rough |
| `-1.5707962512969970703` | `-π/2` high bits | sinf range reduction |
| `-7.5497894158615963534e-08` | `-π/2` middle bits | sinf range reduction |
| `-5.3903029534742383927e-15` | `-π/2` low bits | sinf range reduction |
| `0x3bf921fb54442d19` | `π/2` as FP64 bit pattern | sinf Payne-Hanek |
| `105615` | `≈ 2^17` threshold | sinf fast-path range boundary |
| `0x727fffff` | Biased exponent threshold | sqrtf fast-path range test |
| `0xd000000` | FP32 bias shift | sqrtf input biasing |
| `0x7f800000` | `+INF` bit pattern | log2f +INF return |
| `0x7fffffff` | NaN bit pattern (all 1s significand) | sqrtf NaN return |
| `0xff800000` | `-INF` / exponent mask | log2f exponent extraction |
| `1.175494350822287508e-38` | `FLT_MIN` | Universal subnormal threshold |
| `0x80000000` | INT_MIN / negative zero FP | Sign mask, REDUX MAX.S32 identity |

### Open questions

* [HYP] Does compiling chapter 11 on CUDA 11 or 12 produce `__cuda_sm20_div_u32` helpers? Would confirm compiler change.
* [HYP] Do chapter 06 u16 modulo helpers still inline under CUDA 13, or is u16 the threshold below which external helpers persist?
* [HYP] Is there a source-level complexity threshold above which ptxas falls back to external helpers?
* [HYP] For sinf large-argument, is the Payne-Hanek table accessed via constant memory (`c[0x4][URZ]`) or global memory (LDG.E.CONSTANT)? Both appear in 11g; need clarification of their respective roles.
* [HYP] Precision difference between `__fdividef` (11j) and `a/b` not measured at SASS level; both compile to MUFU.RCP + FMUL + subnormal handling. Is the C-level difference only at the compiler front end?

---

## Kernel 12 register spill (STL, LDL, stack frame, local memory)

### Variants and outcomes

* [OBS] 12a (10 FFMA accumulators, scalar, no flag): no spill. Baseline.
* [OBS] 12b (same source, `-maxrregcount=32`): SASS byte-identical to 12a. Flag has no effect.
* [OBS] 12c (same source, `-maxrregcount=24`): SASS byte-identical to 12a. Flag has no effect.
* [OBS] 12d (same source, `-maxrregcount=16`): ptxas emits `ptxas warning : For profile sm_120 adjusting per thread register count of 16 to lower bound of 24` and produces SASS byte-identical to 12c.
* [OBS] 12e (16 FFMA accumulators in loop, no flag): no spill. Ptxas uses up to R34 naturally.
* [OBS] 12f (same source, `-maxrregcount=24`): no spill. Ptxas restructures the loop body to fit within 24 registers, producing semantically equivalent but differently-scheduled SASS.
* [OBS] 12g (12 array pointers, no flag): no spill. Ptxas uses up to R30.
* [OBS] 12h (`__noinline__` function, 3 args, `-maxrregcount=24`): no spill. Local CALL pattern preserves register state without STL/LDL.
* [OBS] 12i (32 FFMA accumulators in loop, `-maxrregcount=24`): massive spill. 170+ STL/LDL pairs, kernel size ~550 instructions.
* [OBS] 12j (16 float arguments to `__noinline__` function, no flag): no spill. All 16 arguments passed in registers (R0, R2, R3, R4, R6-R11, R13, R15, R17, R19, R21, R23).
* [OBS] 12k (`int arr[64]` local array, no flag): spill. Stack frame of 0x100 = 256 bytes allocated, STL.128 used for vectorized init, LDL with runtime-indexed access.

### Major structural findings

* [OBS] **SM120 register floor = 24 per thread.** Ptxas refuses to allocate fewer than 24 registers per thread, silently adjusting any lower request. This is an SM120 profile constraint, not a kernel-specific limitation. [RES] This resolves the open hypothesis from the chapter 12 plan: "-maxrregcount=16 can be impossible for some kernels" — in fact, it is always impossible on SM120.
* [OBS] **Ptxas prefers restructuring over spilling.** When given a tight register budget, ptxas first tries to change the algorithm's scheduling (different live range overlap) before emitting STL/LDL. Observed in 12f where a 16-accumulator loop is recompiled with a different body shape instead of spilling.
* [OBS] **Argument passing through local CALL does not use the stack.** In 12j, 16 float arguments pass through registers (no ABI stack) because ptxas coordinates the register allocation globally between caller and callee. No caller-saved/callee-saved distinction; no spill at the CALL boundary.
* [HYP] The no-spill CALL behavior of 12j may break under `-rdc=true` (separate compilation). With caller and callee in different compilation units, ptxas cannot see both sides and must fall back to a classical ABI. Not tested.
* [OBS] **Static local arrays always allocate local memory.** An `int arr[64]` (256 bytes) immediately triggers stack frame allocation and STL/LDL instructions. This is deterministic regardless of other pressure considerations.
* [OBS] **Spill preserves the unroll.** In 12i, ptxas keeps the 4× loop unroll AND spills. Kernel size multiplied by ~10× compared to unspilled version, but ILP structure is preserved.

### Stack frame mechanics

* [OBS] **Stack pointer adjustment at prologue.** When local memory is needed, ptxas inserts `IADD R1, R1, -<frame_size>` immediately after the standard prologue. This is the signal to recognize that the kernel spills or has a local array.
* [OBS] **Frame size is exactly sized to requirements.** 0x100 for `int arr[64]` (256 bytes exact), 0x128 for 32 FFMA accumulators (296 bytes, includes scratch).
* [HYP] Frame alignment is on 8-byte boundaries. Both observed sizes (0x100 and 0x128) are multiples of 8. Not confirmed for smaller frames.
* [OBS] **R1 is the canonical spill base.** Every STL/STL.128 and LDL/LDL.LU in the project uses R1 (or UR7 when copied via R2UR) as the base register. No evidence of ptxas choosing a different base.

### Spill opcodes

* [OBS] **`STL [R1+offset], R`** — store one 32-bit register to local memory. Offset is a small immediate added to R1.
* [OBS] **`STL.128 [R1+offset], R`** — vectorized store of 4 consecutive 32-bit registers. Used when four values are computed together and spilled in a burst.
* [OBS] **`LDL R, [R1+offset]`** — load one 32-bit register from local memory.
* [OBS] **`LDL.LU R, [R1+offset]`** — load with last-use hint. [HYP] `.LU` tells the cache to evict the line after this read, since the value will not be reused. Observed systematically on loads whose destination is consumed immediately by the next FFMA.
* [OBS] **`R2UR UR, R`** — copy per-thread register to uniform register. Observed on R1 in 12k (`R2UR UR7, R1`) to enable `[R_offset + UR_base]` addressing for LDL.

### Rolling window spill pattern

* [OBS] **Canonical spill pattern in compute loops.** Ptxas interleaves one STL and one LDL per FFMA, maintaining a constant live set while rolling accumulators through local memory. The pattern is:
  ```
  STL  [R1+<old>], R_spilled    ; spill result of previous FFMA
  FFMA R_new, R_x, R_mul, R_acc ; compute
  LDL.LU R_reload, [R1+<new>]   ; reload next accumulator
  ```
  Each FFMA adds exactly 2 spill instructions. No batching observed.

* [OBS] **Frame regions have distinct purposes.** In 12i, the 0x128 frame is divided into: primary rotating accumulator slots (0x00-0x3c), secondary accumulator bank for unroll (0x40-0x68), persistent values like loop counter save (0x6c), and scratch for unroll rollover (0x70-0x124). Offsets are assigned by ptxas based on live range non-overlap.

### Register survival under pressure

* [OBS] **Pointers are never spilled.** In 12i under severe pressure, the pointer pair R2:R3 (global address for `a[i]`) remains in registers throughout the kernel. Same for R0 (thread index).
* [OBS] **Loop-invariant small values are kept live.** R17 (= `v = a[i]`) in 12i is loaded once and kept in a register, spilled only conceptually via the save at `[R1+0x6c]` for backup.
* [OBS] **Spill victims are the accumulators.** Floating-point accumulators are the preferred spill targets because they have predictable access patterns (one read, one write per iteration) that enable clean rolling.

### Vectorized spill (STL.128)

* [OBS] **STL.128 fires on burst initialization.** In 12k, the 64-element int array is initialized in chunks of 4: compute 4 values into R4..R7, `STL.128 [R1], R4`; compute next 4, `STL.128 [R1+0x10], R20`, etc. 16 STL.128 total for 64 ints.
* [GAP] **LDL.128 not observed.** Only STL has the vectorized variant in these dumps. [HYP] LDL.128 may exist but was not triggered because the reads in 12k are at runtime-computed indices, not sequential.

### Indexed local memory access

* [OBS] **Runtime-indexed LDL uses `[R_idx + UR_base]`.** Example from 12k: `LDL R, [R33+UR7]` where R33 holds a per-thread byte offset and UR7 holds the uniform stack base.
* [OBS] **Alignment mask via LOP3 0xfc.** The byte offset is masked with `0xfc = 11111100b` via `LOP3.LUT R, R_idx, 0xfc, RZ, 0xc0, !PT` before the LDL. This simultaneously aligns to 4-byte boundary and masks to the 64-element range (only low 6 bits effective after ×4 scaling).
* [OBS] **IADD3 with explicit PT predicates.** In 12k's final reduction: `IADD3 R3, PT, PT, R, R, R` — three-input add with always-true predicates in the first two slots. First observation of this form in integer context (chapter 11 had UIADD3 with explicit predicate slots in uniform form).

### Anti-spill strategies observed

Ptxas applies these techniques in order before resorting to spill:

* [OBS] **Recomputation.** Values cheap to recompute (e.g., `(float)k` via UI2FP.F32.U32) are recomputed rather than held in a register. Observed in 12e.
* [OBS] **Restructuring.** Change the loop body shape to reduce live range overlap without changing semantics. Observed in 12f.
* [OBS] **Register file reuse.** Dead registers recycled across semantic boundaries (established in earlier chapters).
* [OBS] **Global allocation across CALL.** No ABI-forced spill at function boundaries. Observed in 12j.
* [OBS] **Spill as last resort.** Only when no other option keeps the live set within budget.

### New opcodes (6 total)

| Opcode | Where | Semantics |
|---|---|---|
| STL | 12i, 12k | Store to local memory (32-bit) |
| STL.128 | 12k | Vectorized store to local memory (4× 32-bit) |
| LDL | 12i, 12k | Load from local memory (32-bit) |
| LDL.LU | 12i | Load from local memory with last-use hint |
| R2UR | 12k | Copy per-thread register to uniform register |
| IADD3 PT, PT, ... | 12k | Three-input integer add with predicate carry slots |

### New modifiers

* [OBS] `.LU` on LDL: last-use hint for cache eviction.

### Cost analysis

* [OBS] **Spill overhead: 2 instructions per value per iteration.** 1 STL + 1 LDL per accumulator that does not fit in registers.
* [OBS] **Kernel size multiplier: ~1.5× to 10× for spilled kernels.** Depends on whether the spill is rolling (multiplicative) or one-off (additive).
* [OBS] **Useful compute ratio drops by ~35%.** In 12i, compute ratio drops from ~37% to ~24% when spill is active.
* [HYP] LDL cycle latency: 20-30 cycles from L1-resident local memory. Not microbenchmarked.

### Observations worth flagging

* [OBS] **Kernel 12a-12d byte-identical.** Despite different `-maxrregcount` values (none, 32, 24, 16), the SASS is exactly the same. This confirms that ptxas' register allocation is not sensitive to `-maxrregcount` when the natural allocation fits within the requested budget.
* [OBS] **No STL in kernels 12b-12h.** Eight variants were designed to force spill; seven did not. Only extreme pressure (12i) and static local arrays (12k) triggered local memory.
* [OBS] **ptxas warning format.** The exact text `ptxas warning : For profile sm_120 adjusting per thread register count of N to lower bound of M` is the signal for the register floor constraint.
* [OBS] **LSU local memory path is distinct from global.** LDL/STL do not use descriptor-based addressing (`desc[UR][R.64]`). They use direct `[R+offset]` or `[R+UR]` modes.
* [OBS] **Spill is per-thread.** Each thread's R1 points to its own stack frame in per-thread local memory space. No shared local memory frame.
* [GAP] **Spill order heuristic not fully characterized.** Ptxas chooses accumulators over pointers and loop-invariants, but the exact priority function is not formalized.

### Open questions

* [HYP] Does LDL.128 exist? Not observed, but possibly triggered by sequential reads with compile-time-known indices.
* [HYP] Does `-rdc=true` break the no-spill CALL pattern of 12j? Expected yes, not tested.
* [HYP] Does ptxas ever choose a register other than R1 as spill base? No evidence against so far.
* [HYP] What is the exact threshold at which restructuring fails and spill begins? Somewhere between 16 and 32 FFMA accumulators under `-maxrregcount=24`.
* [HYP] How does `__launch_bounds__` interact with `-maxrregcount`? Both affect the register budget.
* [GAP] Control code fields for LSU-local pipeline (STL/LDL) not decoded bit-by-bit.

---

[OBS] Chapters 13 through 19 establish the observed SM120 tensor-core opcode families, load/staging instructions, sparse metadata operand placement, and chain scheduling behavior. [GAP] Remaining gaps are tracked explicitly under the relevant chapter and in the cross-chapter audit-gap section.

---

## Kernel 13 HMMA baseline (FP16, BF16, m16n8k16)
Chapter establishing the HMMA tensor core baseline on SM120. Five variants tested covering FP16 accumulator, FP32 accumulator, BF16 input, accumulator chaining, and serial latency microbenchmark.
### Variants and outcomes
* [OBS] 13a (FP16 input, FP16 accumulator, single MMA): baseline `HMMA.16816.F16`. 27 useful instructions. d[0] = 0x4c004c00 = 16.0.
* [OBS] 13b (FP16 input, FP32 accumulator, single MMA): `HMMA.16816.F32`. 31 useful instructions (+4 vs 13a: +2 LDG for C, +2 STG for D). d[0] = 16.0.
* [OBS] 13c (BF16 input, FP32 accumulator, single MMA): `HMMA.16816.F32.BF16`. 31 useful instructions. d[0] = 16.0. 30 of 31 instructions byte-identical to 13b; only the HMMA control code differs (bit 18).
* [OBS] 13d (FP32 accumulator, two chained MMAs where D1 feeds C2): 2 × `HMMA.16816.F32`. 37 useful instructions (+6 vs 13b: +1 HMMA, +2 NOPs, +3 scheduling adjustments). d[0] = 32.0.
* [OBS] 13e (N chained MMAs bracketed by clock64() for latency microbenchmark, N=16/32/64): unrolled chain. Measurements: 872/1439/2541 total cycles. Linear model: total_cycles ≈ 312 + 35 × N.
### Key observations
* [OBS] **HMMA opcode family** on SM120: `HMMA.16816.<acc_dtype>[.<input_dtype>]`. Shape modifier `.16816` is the M×N×K concatenation. Accumulator dtype suffix (`.F16` or `.F32`) is mandatory. Input dtype suffix is omitted for FP16 (implicit default) and explicit for BF16 (`.BF16`).
* [OBS] **MMA on SM120 is warp-level.** All 32 threads cooperate on one tile. No wgmma, no tcgen05.mma.
* [OBS] **HMMA takes 4 SASS operands: D base, A base, B base, C base.** Fragment spans (4 registers for A FP16, 2 for B FP16, 2 or 4 for C/D depending on dtype) are implicit in the opcode. The disassembler only shows the base register of each fragment.
* [OBS] **Opcode bytes are determined by the operand base registers only; dtype lives in the control code.** 13a, 13b, 13c with the same operand bases all share `0x000000100c0c723c` for the opcode+operand bytes. Only the 8-byte control code differs.
* [OBS] **Control code bits identified (partial):**
  * Bit 1 and 12 encode accumulator dtype (F16 vs F32 discrimination).
  * Bit 18 encodes input dtype (0 = FP16 implicit, 1 = BF16 explicit).
  * Bit 26 is the MMA scoreboard set flag (SBS), active on the first HMMA of a chain that produces a result consumed by a subsequent MMA.
  * Bit 27 is the MMA scoreboard wait flag, active on HMMAs whose C input depends on a prior HMMA's D output.
  * [GAP] The specific scoreboard slot number (SB0..SB5) is not decoded. Extraction requires a bit-level Blackwell ISA decoder; gpuasm.com is unavailable during this chapter.
### Fragment layout per thread
Matches CUTLASS atoms from `include/cute/arch/mma_sm80.hpp`:
| Variant | D registers | A registers | B registers | C registers |
|---|---|---|---|---|
| 13a FP16 acc | uint32[2] = 4 half | uint32[4] = 8 half | uint32[2] = 4 half | uint32[2] = 4 half |
| 13b FP32 acc, FP16 input | float[4] | uint32[4] = 8 half | uint32[2] = 4 half | float[4] |
| 13c FP32 acc, BF16 input | float[4] | uint32[4] = 8 bf16 | uint32[2] = 4 bf16 | float[4] |
### Register allocation patterns
* [OBS] **Single-MMA case, D smaller than A (13a FP16 acc):** D and A colocated with partial overlap. D occupies the first 2 of A's 4 registers. Example: `HMMA R12, R12, R16, R18` where D=R12:R13 and A=R12:R13:R14:R15.
* [OBS] **Single-MMA case, D equal to A in size (13b, 13c FP32 acc):** D and A colocated with complete overlap. D and A share exactly the same 4 registers. Example: `HMMA R12, R12, R16, R20` where D=R12:R13:R14:R15 and A=R12:R13:R14:R15.
* [OBS] **Chained-MMA case (13d, 13e):** D and C colocated (accumulator in-place). A keeps its own distinct register block because A is re-read by the next HMMA in the chain. Example from 13d: `HMMA R16, R12, R10, R16` where D=C=R16:R19 (in-place) and A=R12:R15 (separate).
* [RES] ptxas allocation rule: maximize register reuse subject to the data dependency graph. D/A colocation in single-MMA, D/C colocation in chained MMA. A and B always on distinct register bases.
### The .reuse modifier on MMA operands
* [OBS] **`.reuse` appears on the B operand of every HMMA in a chain except the last one.** Example from 13d: `HMMA.16816.F32 R16, R12, R10.reuse, R16` (HMMA1), `HMMA.16816.F32 R16, R12, R10, R16` (HMMA2, last in chain).
* [OBS] `.reuse` is not emitted on A even though A is also re-read by subsequent HMMAs. [HYP] The reuse cache may prioritize smaller operands (B is 2 registers, A is 4 registers) or the cache is positional and only the B slot is enabled for MMA.
* [OBS] `.reuse` is not emitted on C in the observed HMMA chain. [HYP] This may be because C is already colocated with D in the chaining case, which is a different form of reuse at the register file level.
### The NOP pad pattern
* [OBS] **The semantic NOP `@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ` has predicate `@!UPT` which is always false (UPT is always true, `!UPT` is always false), all operands are URZ, destination is URZ. The instruction has no observable effect.**
* [OBS] ptxas emits **2 NOPs after each HMMA** in variants 13a, 13b, 13c, 13d when the HMMA's D register is consumed by a subsequent dependent instruction (either another HMMA via D→C chain, or a STG).
* [OBS] 13e's last HMMA (N=16) has **zero NOPs** because its consumer is `CS2R R2, SR_CLOCKLO` which reads `SR_CLOCKLO`, not R16, and therefore does not depend on the HMMA result.
* [RES] The NOP pad is not intrinsic to HMMA. ptxas emits NOPs only when the next consumer has a data dependency on D. In a minimal kernel with no independent work to schedule between HMMA and the consumer, ptxas fills the gap with NOPs. In a production kernel (GEMM with ldmatrix, address arithmetic, index updates), these slots are filled with real work and uniform NOPs disappear or thin out.
* [HYP] The count of 2 NOPs is a fixed scheduling slot quantum that ptxas always tries to fill. The remainder of the ~35-cycle HMMA latency is covered by the scoreboard wait on bit 27.
### Scoreboard and variable latency
* [OBS] **HMMA is variable-latency and uses a scoreboard mechanism.** Evidence from control code transitions in 13d and 13e.
* [OBS] **Control code transitions in 13d:** HMMA1 has bit 27 set (produces result consumed by HMMA2 via D→C). HMMA2 has bit 27 clear (last in chain, no further MMA consumer).
* [OBS] **Control code transitions in 13e (N=16):** HMMA #1 has bits 26 and 27 both set (first in chain, establishes scoreboard). HMMAs #2-15 have bit 27 set but bit 26 clear (wait on scoreboard, do not re-set it). HMMA #16 has bit 27 clear (no wait, consumer is CS2R which is not scoreboard-coordinated).
* [HYP] Bit 26 encodes "set scoreboard slot". Set on the first HMMA of a chain to claim the scoreboard slot that subsequent HMMAs wait on.
* [HYP] Bit 27 encodes "wait on scoreboard". Set on HMMAs with a data dependency on a prior HMMA; clear when no MMA dependency exists.
* [GAP] The specific scoreboard slot number used (SB0..SB5) is not decoded.
### Unified HMMA control code table
| Kernel | HMMA position | Opcode+operand bytes | Control code |
|---|---|---|---|
| 13a | single | `0x000000100c0c723c` | `0x004ff60000000812` |
| 13b | single | `0x000000100c0c723c` | `0x004ff60000001814` |
| 13c | single | `0x000000100c0c723c` | `0x004ff60000041814` |
| 13d | HMMA #1 (feeds HMMA #2) | `0x0000000a0c10723c` | `0x084ff60000001810` |
| 13d | HMMA #2 (last in chain) | `0x0000000a0c10723c` | `0x000ff60000001810` |
| 13e | HMMA #1 (first of chain) | `0x000000020410723c` | `0x084ff60000001810` |
| 13e | HMMA #2-15 (mid-chain) | `0x000000020410723c` | `0x080ff60000001810` |
| 13e | HMMA #16 (last, feeds CS2R) | `0x000000020410723c` | `0x000ff00000001810` |
### Latency measurement
* [OBS] Chain of N serial HMMAs bracketed by `clock64()`, where the accumulator R16:R19 is both the D output and the C input of every HMMA, forces ptxas to keep the chain serial (no reordering possible due to the data dependency).
* Measurements:

| N | Total cycles | cycles/HMMA |
|---|---|---|
| 16 | 872 | 54.50 |
| 32 | 1439 | 44.97 |
| 64 | 2541 | 39.70 |

* [RES] Linear regression gives marginal cost per added HMMA: **~35 cycles/HMMA**. Fixed overhead: **~310 cycles**. Model: `total_cycles ≈ 310 + 35 × N`.
* [RES] HMMA.16816.F32 serial latency on SM120 is approximately 35 cycles per HMMA in a dependency-chained configuration.
* [IMPORTANT] 35 cycles is the **serial latency** (every HMMA waits on the previous). This is NOT the HMMA throughput. In a real GEMM kernel where many HMMAs are in flight simultaneously with independent accumulators, throughput is substantially lower than 35 cycles. Throughput measurement deferred to chapter 18 (pipelined tile).
* [HYP] The ~310-cycle overhead is composed of the latency of the very first HMMA (which includes TC pipeline startup), the CS2UR + CS2R latencies, the final HMMA result becoming visible to CS2R, and scoreboard system spin-up.
### New opcode
* **CS2UR** — special-register to uniform-register read. Observed in 13e for `CS2UR UR6, SR_CLOCKLO`. Uniform-register variant of CS2R. Writes the clock value to a uniform register shared across the warp (semantically correct because all threads sample the same clock at the same issue slot). Companion to the per-thread `CS2R` introduced in chapter 11.
### Scheduling observations
* [OBS] **Destination pointer address computation placed late.** In 13b, 13c, 13d the IMAD.WIDE.U32 that computes the store-destination address (`&d[tid*4]`) is emitted after all the input LDGs and just before HMMA. Rationale: the D address is not needed until STG, so delaying its computation minimizes its register live range.
* [OBS] **Register recycling across semantic boundaries.** In 13d, R4 is first used to hold `&b[tid*2]`. Once the two LDGs of b complete, R4 is free and ptxas overwrites R4 with `&d[tid*4]`. One physical register plays two unrelated roles. Consistent with chapter 02 register recycling pattern.
* [OBS] **Inverted LDG order for C registers** (observed in 13b, 13c, 13d, 13e): ptxas emits LDGs for c3, c2, c1, c0 in that order (offsets 0xc, 0x8, 0x4, 0x0). Stable across all variants. Not a performance concern; reflects ptxas allocation of R20-R23 in reverse.
* [OBS] **IADD.64 accepts uniform register as source operand.** `IADD.64 R2, R2, -UR6` in 13e. Mixed R/UR operand forms are valid on SM120.
### Compiler determinism
* [OBS] ptxas produces byte-identical SASS for the portion of the kernel unrelated to a local source change. Between 13b and 13c (FP16 input vs BF16 input), 30 of 31 instructions are byte-identical; only the HMMA control code differs (bit 18).
* [OBS] `LDG.E.CONSTANT` is emitted for any pointer marked `const __restrict__` in the kernel signature, not just for compile-time constant lookup tables. This extends the chapter 11 observation (Payne-Hanek table) to a general rule.
### Canonical kernel MMA minimal template
Stable across 13a, 13b, 13c, 13d (13e adds the CS2UR/CS2R bracket but otherwise follows the same template):
```
// Prologue (8 instructions)
LDC R1, c[0x0][0x37c]                   ; stack
S2R R_tid, SR_TID.X                      ; threadIdx

// Pointer loads (one LDC.64 per kernel arg, 8-byte stride in param space)
LDC.64 R_ptr_a, c[0x0][0x380]
LDC.64 R_ptr_b, c[0x0][0x388]
LDC.64 R_ptr_c, c[0x0][0x390]
LDC.64 R_ptr_d, c[0x0][0x398]
LDCU.64 UR_desc, c[0x0][0x358]           ; global descriptor

// Stride computation (SHF per per-thread stride multiplier)
SHF.L.U32 R_stride_ab, R_tid, K_ab, RZ
SHF.L.U32 R_stride_cd, R_tid, K_cd, RZ

// Address computation (IMAD.WIDE.U32 per pointer, destination placed late)
IMAD.WIDE.U32 R_addr_a, R_stride_ab, 0x4, R_ptr_a
IMAD.WIDE.U32 R_addr_b, ..., 0x4, R_ptr_b
IMAD.WIDE.U32 R_addr_c, R_stride_cd, 0x4, R_ptr_c

// Fragment loads
LDG.E.CONSTANT R_b0, desc[UR_desc][R_addr_b]
LDG.E.CONSTANT R_b1, desc[UR_desc][R_addr_b+0x4]
// ... for all A, B, C registers

IMAD.WIDE.U32 R_addr_d, R_stride_cd, 0x4, R_ptr_d   ; destination placed late

// MMA
HMMA.16816.<dtype> R_d, R_a, R_b, R_c

// NOP pad (per-HMMA, if consumer depends on D)
@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ
@!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ

// Stores
STG.E desc[UR_desc][R_addr_d], R_d_0
// ... for all D registers

// Epilogue
EXIT
BRA .                                    ; self-trap
NOP (padding)
```
### Canonical MMA accumulator chaining
From 13d, 13e:
```
HMMA1 D=R_acc, A, B.reuse, R_acc_in      ; .reuse on B if B reloaded by HMMA2
2 × UIADD3 NOPs
HMMA2 D=R_acc, A, B, R_acc               ; accumulator in-place: D registers = C registers
2 × UIADD3 NOPs
<next consumer>
```
Register allocation under MMA chaining:
* D and C colocated (in-place accumulator)
* A keeps its own distinct register block (re-read by each HMMA)
* B keeps its own distinct register block
* No D/A reuse (contrary to the single-MMA case)
### Resolved hypotheses
* [RES] HMMA opcode family confirmed as `HMMA.16816.<acc>[.<input>]` on SM120.
* [RES] Distinction FP16/BF16 input: explicit `.BF16` suffix in cuobjdump, encoded via bit 18 of HMMA control code.
* [RES] Register allocation rule: D/A colocated simple, D/C colocated chained.
* [RES] `.reuse` on B in chained MMA confirmed across 13d, 13e.
* [RES] 2 UIADD3 NOPs after HMMA emitted when consumer depends on D; absent when consumer is independent (e.g., CS2R in 13e's last HMMA).
* [RES] HMMA is variable-latency with scoreboard (bits 26 SBS and 27 wait).
* [RES] Serial HMMA.16816.F32 latency ≈ 35 cycles/HMMA on SM120. Model: `total_cycles ≈ 310 + 35 × N` for N-chain.
### Open questions
* [HYP] HMMA latency for FP16 accumulator (13a variant) not measured.
* [HYP] HMMA latency for BF16 input (13c variant) not measured.
* [HYP] HMMA at other shapes (m16n8k8 or smaller) not tested. Opcode may change suffix or stay the same.
* [HYP] Sensitivity of HMMA latency to input data (NaN, denormal, zero) not tested.
* [HYP] HMMA throughput in independent (non-chained) configuration deferred to chapter 18.
* [HYP] Why exactly 2 NOPs and not 1 or 3. Possibly related to a fixed scheduling slot quantum that remains between HMMA issue and scoreboard visibility.
* [GAP] Specific scoreboard slot ID used by HMMA (SB0..SB5) not decoded.
* [GAP] Low-order scheduling bits of HMMA control code (bits 0-4) not fully decoded.
### New instructions observed in this chapter
| Opcode | Usage |
|---|---|
| HMMA.16816.F16 | MMA warp-level, m16n8k16, FP16 input, FP16 accumulator |
| HMMA.16816.F32 | MMA warp-level, m16n8k16, FP16 input (implicit), FP32 accumulator |
| HMMA.16816.F32.BF16 | MMA warp-level, m16n8k16, BF16 input (explicit), FP32 accumulator |
| CS2UR | Special-register to uniform-register read (observed for SR_CLOCKLO) |
### New modifiers
* `.16816` on HMMA: shape m16n8k16 (MNK concatenation)
* `.F16` / `.F32` on HMMA: accumulator dtype
* `.BF16` on HMMA: input dtype override (default is FP16)
* `.reuse` on MMA B operand: reuse cache hint for next MMA that will re-read B

---

## Kernel 14 QMMA FP8/FP6/FP4 baseline (kind::f8f6f4, m16n8k32)
Chapter establishing the QMMA tensor core opcode family on SM120 — the SASS realization of `mma.sync.aligned.kind::f8f6f4.*` from PTX. Ten variants tested covering input dtypes (E4M3, E5M2, E3M2, E2M3, E2M1), accumulator dtypes (F32, F16), accumulator chaining, and serial latency microbenchmark. Encoding of the dtype field validated by Popper-style prediction.
### Variants and outcomes
* [OBS] 14a (E4M3.E4M3 input, F32 acc, single MMA): baseline `QMMA.16832.F32.E4M3.E4M3`. 31 useful instructions. d[0] = 32.0.
* [OBS] 14b (E4M3.E5M2 input, F32 acc): `QMMA.16832.F32.E4M3.E5M2`. d[0] = 32.0. Bytes opcode identical to 14a; control code differs by bit 15 (B "mantissa ≠ 3" flag).
* [OBS] 14c (E5M2.E5M2 input, F32 acc): `QMMA.16832.F32.E5M2.E5M2`. d[0] = 32.0. Adds bit 14 set (A symmetric to bit 15 for B).
* [OBS] 14d (E2M1.E2M1 input, F32 acc): `QMMA.16832.F32.E2M1.E2M1`. d[0] = 2.0 (expected 32.0 — see [GAP-14d-1]). Adds bits 19, 21 set for FP4 family.
* [OBS] 14e (E4M3.E4M3 input, F32 acc, two chained MMAs with D1 feeding C2): 2 × `QMMA.16832.F32.E4M3.E4M3`. 37 useful instructions (+6 vs 14a). d[0] = 64.0.
* [OBS] 14f (N chained MMAs with clock64() bracketing for latency, N=16/32/64): unrolled chain. Measurements: 1070/1637/2742 total cycles. Linear model: total_cycles ≈ 510 + 35 × N.
* [OBS] 14g (E3M2.E3M2 input, F32 acc): `QMMA.16832.F32.E3M2.E3M2`. d[0] = 0.0 (zero inputs). Adds bits 18, 20 set for FP6 family E3M2.
* [OBS] 14h (E2M3.E2M3 input, F32 acc): `QMMA.16832.F32.E2M3.E2M3`. d[0] = 0.0. Bits 14, 15 clear (E2M3 has mantissa = 3 like E4M3); bits 19, 21 set (E2M3 has exp = 2). Invalidates a naive "alt dtype = bit 14/15" model and reveals a 3-bit-per-operand encoding.
* [OBS] 14i (E4M3.E2M1 mixed input, F32 acc, Popper test): `QMMA.16832.F32.E4M3.E2M1`. d[0] = 0.0. Predicted control code from the model (`0x004ff6000020ac14`) matches observed exactly. Validates the encoding as scientifically robust.
* [OBS] 14j (E4M3.E4M3 input, F16 acc): `QMMA.16832.F16.E4M3.E4M3`. d[0] = 0x50005000 (two FP16 32.0 packed). Layout 2-4-2-2 (D and C are 2 registers each instead of 4). Identifies bits 1, 2, 13 as accumulator dtype encoding.
### Toolchain note
* [RES] PTX feature `.kind::f8f6f4` requires `-arch=compute_120a -code=sm_120a`. The shortcut `-arch=sm_120a` is not accepted by nvcc 13.2 (must use the explicit compute/code form). Plain `-arch=sm_120` rejects with `Feature '.kind::f8f6f4' not supported on .target 'sm_120'`. This applies to the full kind::f8f6f4 family, not only block-scaled variants. Resolves [HYP→?] from FINDINGS skeleton ("block-scaled MMA requires sm_120a").
* [OBS] Output binaries compiled for sm_120a carry `EF_CUDA_ACCELERATORS` in their elf header. Marker for any kernel using sm_120a-specific PTX features.
### Key observations
* [OBS] **QMMA opcode family** on SM120: `QMMA.16832.<acc>.<inputA>.<inputB>`. Shape modifier `.16832` is the M×N×K concatenation (m16n8k32). Accumulator dtype suffix mandatory. Both input dtypes explicit (no implicit default), unlike HMMA where FP16 is implicit.
* [OBS] **QMMA is a distinct opcode family from HMMA**, not an extension. Opcode bytes differ in bit pattern (HMMA `0x...723c`, QMMA `0x...727a`).
* [OBS] **QMMA covers FP8, FP6, and FP4** — the entire kind::f8f6f4 family is one opcode with dtype distinction in the control code.
* [OBS] **Opcode bytes invariant across input dtypes**: `0x000000100c0c727a` for variants 14a, b, c, d, g, h, i (when operand bases match). Dtype encoding lives entirely in the control code.
* [OBS] **Opcode bytes invariant across accumulator dtypes too**: 14j (F16 acc) shares the same opcode bytes as 14a (F32 acc) when operand bases match, despite the C base differing (R20 in 14a, R18 in 14j). Suggests opcode bytes encode {D, A, B} bases but not C base. C base lives in the control code. Not formally verified.
### Control code topology — fully decoded
* [OBS] By comparing 10 variants byte-by-byte, **13 bits of the QMMA control code have been decoded**:
| Bit | Meaning | Evidence |
|---|---|---|
| 1 | Acc dtype = F16 | 14j vs 14a |
| 2 | Acc dtype = F32 | 14j vs 14a |
| 4 | Always set on QMMA | invariant across all variants |
| 10, 11 | Always set on QMMA (MMA family marker) | invariant |
| 13 | F32 acc auxiliary bit | 14j vs 14a |
| 14 | A input mantissa ≠ 3 | 14b/c/d/g/h/i |
| 15 | B input mantissa ≠ 3 | 14b/c/d/g/h/i |
| 18 | A input exp = 3 (E3M2) | 14g |
| 19 | A input exp = 2 (E2M3, E2M1) | 14d/h |
| 20 | B input exp = 3 (E3M2) | 14g |
| 21 | B input exp = 2 (E2M3, E2M1) | 14d/h |
| 26 | MMA scoreboard set (SBS) | 14e/f |
| 27 | MMA scoreboard wait | 14e/f |
### Input dtype encoding model (3 bits per operand, validated)
* [OBS] Each input operand is encoded by **3 independent bits in the control code**:
  * For operand A: bits 14, 18, 19
  * For operand B: bits 15, 20, 21
  * Bit "mantissa ≠ 3" = 1 if mantissa is 1 or 2 (E5M2, E3M2, E2M1)
  * Bit "exp = 3" = 1 if exponent bits = 3 (E3M2 only)
  * Bit "exp = 2" = 1 if exponent bits = 2 (E2M3, E2M1)
* [OBS] **Encoding fully orthogonal A/B**, no inter-operand interaction. Hardware uses 5 of 8 possible 3-bit codepoints, leaving room for future format extensions.
* [OBS] **Codepoint mapping**:
| dtype | exp | mant | bit (mant≠3) | bit (exp=3) | bit (exp=2) | binary code |
|---|---|---|---|---|---|---|
| E4M3 | 4 | 3 | 0 | 0 | 0 | 000 (default) |
| E5M2 | 5 | 2 | 1 | 0 | 0 | 100 |
| E3M2 | 3 | 2 | 1 | 1 | 0 | 110 |
| E2M3 | 2 | 3 | 0 | 0 | 1 | 001 |
| E2M1 | 2 | 1 | 1 | 0 | 1 | 101 |
* [RES] **Validation by Popper-style prediction (14i)**: the model derived from 5 symmetric variants predicted ctrl = `0x004ff6000020ac14` for the asymmetric mixed case E4M3 × E2M1. Observed value matched **exactly**. The encoding is scientifically robust.
* [HYP] **Architectural interpretation**: the encoding suggests SM120 tensor core has MAC units organized by exponent class (4, 3, 2) with a sub-mode for mantissa-3 vs alternative mantissa widths. Consistent with a unified FP8/FP6/FP4 hardware that selects internal datapath via these 3 bits per operand.
### Accumulator dtype encoding (MMA-family wide)
* [OBS] By comparing 14j (F16 acc) to 14a (F32 acc), the accumulator dtype encoding emerges:
  * Bit 1 = 1 if accumulator is F16
  * Bit 2 = 1 if accumulator is F32
  * Bit 13 (QMMA) or bit 12 (HMMA) is an auxiliary bit set in F32 mode
* [RES] **This encoding is identical between HMMA and QMMA**. Comparing chapter 13a (HMMA F16, ctrl 0x004ff60000000812) to 13b (HMMA F32, ctrl 0x004ff60000001814) reveals the same bits 1, 2 transition. The accumulator dtype encoding is a 9th MMA-family wide invariant.
### Register allocation patterns (MMA-family wide)
* [OBS] **Single MMA, D smaller than A** (14j FP16 acc): D and A colocated with partial overlap. D = R12:R13 (2 regs), A = R12:R15 (4 regs). Mirrors HMMA pattern from 13a.
* [OBS] **Single MMA, D equal to A in size** (14a, 14b, 14c, 14d, 14g, 14h, 14i, all F32 acc): D and A colocated with complete overlap. Same 4 registers. Mirrors HMMA pattern from 13b.
* [OBS] **Chained MMA** (14e): D and C colocated (accumulator in-place). A keeps its own distinct register block because A is re-read by each QMMA. B keeps its own distinct register block. Mirrors HMMA pattern from 13d.
* [OBS] Fragment register counts per thread:
  * F32 acc (14a-i): D = float[4], A = uint32[4], B = uint32[2], C = float[4]
  * F16 acc (14j): D = uint32[2], A = uint32[4], B = uint32[2], C = uint32[2]
* [OBS] A and B register counts are independent of input dtype. Element packing within each uint32 changes (4 e4m3, 8 e2m1, etc.) but ptxas always uses the same number of registers.
### Chaining patterns (14e, MMA-family wide)
* [OBS] Two chained QMMAs produce SASS analogous to 13d for HMMA:
  ```
  QMMA #1: R16, R12, R10.reuse, R16    ctrl 0x084ff60000002c10
  2 × @!UPT UIADD3 NOPs
  QMMA #2: R16, R12, R10, R16          ctrl 0x000ff60000002c10
  2 × @!UPT UIADD3 NOPs
  ```
* [OBS] `.reuse` on B operand of every QMMA except the last in chain.
* [OBS] 2 UIADD3 NOPs after each QMMA when consumer depends on D.
* [OBS] D and C colocated (accumulator in-place).
* [OBS] A on a distinct register block, re-read by each QMMA in the chain.
### Scoreboard scheme (14e, 14f, MMA-family wide)
* [OBS] The scoreboard high-byte encoding is **byte-identical** between HMMA chains (chapter 13e) and QMMA chains (14e, 14f):
  * First MMA of chain (sets scoreboard): `0x084ff6...` — bits 26 (SBS) + 27 (wait) set
  * Mid-chain MMAs (wait on scoreboard): `0x080ff6...` — bit 27 only
  * Last MMA in chain (no MMA-consumer after): `0x000ff0...` — neither bit set
* [RES] **MMA scoreboard scheme is fully MMA-family wide**.
### Latency measurement (14f)
* [OBS] Three measurements at N = 16, 32, 64 chained QMMAs (each with accumulator in-place to force serial dependency):
| N | total_cycles | cycles/QMMA |
|---|---|---|
| 16 | 1070 | 66.88 |
| 32 | 1637 | 51.16 |
| 64 | 2742 | 42.84 |
* [OBS] Linear regression on incremental costs:
  * ΔN(16→32): 567 cycles for 16 added QMMAs → 35.4 cycles/QMMA
  * ΔN(32→64): 1105 cycles for 32 added QMMAs → 34.5 cycles/QMMA
* [RES] **Latency model**: `total_cycles ≈ 510 + 35 × N`.
* [RES] **Marginal cost identical to HMMA**: ~35 cycles/QMMA on SM120, identical to the ~35 cycles/HMMA observed in chapter 13e for HMMA.16816.F32.
* [OBS] **Striking observation**: QMMA m16n8k32 performs **2× more FMAs internally** than HMMA m16n8k16 (k=32 vs k=16), yet completes in the same serial latency. **Effective FMA throughput per cycle is 2× higher for FP8 QMMA than for FP16 HMMA**, consistent with Blackwell consumer FP8 throughput specs.
* [OBS] **Fixed chain overhead +200 cycles vs HMMA**: ~510 for QMMA vs ~310 for HMMA. Likely reflects deeper pipeline startup for FP8 MAC units. Amortized in production GEMM (N >> 100).
* [HYP] The +200 cycles overhead may include initialization of the FP8/FP6/FP4 unified MAC datapath, which has more configurability than the FP16-only HMMA path.
### Comparison HMMA vs QMMA
| | HMMA.16816.F32 | QMMA.16832.F32.E4M3.E4M3 |
|---|---|---|
| Shape | m16n8k16 | m16n8k32 |
| Operand bases | D, A, B, C | D, A, B, C |
| Internal FMA count | 256 (16×8×16 / 32 lanes × 2) | 512 (16×8×32 / 32 lanes × 2) — 2× higher |
| Serial latency / MMA | ~35 cycles | ~35 cycles |
| Effective FMA throughput per cycle | 7.3 | 14.6 — 2× higher |
| Chain fixed overhead | ~310 cycles | ~510 cycles |
| Scoreboard scheme | bits 26 SBS, 27 wait | identical |
| `.reuse` on B in chain | except last | identical |
| NOPs around MMA | 2 UIADD3 if D dependency | identical |
| Acc dtype encoding | bits 1, 2 | identical |
### MMA-family wide invariants (9 confirmed)
After chapter 14, the following patterns are confirmed common to HMMA and QMMA. They should be treated as MMA-family wide rather than opcode-specific:
1. **Single-MMA register allocation**: D and A colocated (with partial overlap when D < A in size).
2. **Chained-MMA register allocation**: D and C colocated. A and B on distinct bases.
3. **`.reuse` on B operand** of every MMA in a chain except the last.
4. **2 UIADD3 NOPs** after each MMA when the consumer depends on D.
5. **MMA scoreboard scheme**: bit 26 (SBS) on first MMA of chain, bit 27 (wait) on dependent MMAs.
6. **Inverted LDG order** for one operand (B or C depending on kernel).
7. **Late destination address**: IMAD.WIDE for store address placed just before the MMA.
8. **Template kernel structure**: prologue + LDG fragments + MMA + NOPs + STG, byte-identical between HMMA and QMMA when operand sizes match.
9. **Accumulator dtype encoding**: bit 1 = F16 acc, bit 2 = F32 acc.
### Resolved hypotheses
* [RES] QMMA is a new opcode family distinct from HMMA, not an extension. Confirmed by 14a vs HMMA dumps (different opcode bytes).
* [RES] QMMA opcode bytes are invariant across input dtypes (6 variants tested) and accumulator dtypes (when operand bases match).
* [RES] QMMA covers the full kind::f8f6f4 family (FP8, FP6, FP4) with one opcode.
* [RES] Input dtype encoding is per-operand symmetric and orthogonal A/B (Popper test 14i).
* [RES] QMMA serial latency ~35 cycles/QMMA, identical to HMMA. Model `total_cycles ≈ 510 + 35 × N`.
* [RES] Accumulator dtype encoding (bits 1 = F16, 2 = F32) is identical HMMA/QMMA.
* [RES] Chaining patterns identical HMMA/QMMA.
* [RES] Scoreboard scheme identical HMMA/QMMA.
* [RES] PTX `.kind::f8f6f4` requires sm_120a, not sm_120 standard.
### Open gaps
* [GAP-14d-1] **FP4 fragment layout unknown**. Variant 14d returned d[0] = 2.0 instead of expected 32.0 with naive `0x22222222` packing. The SASS observation (opcode, control code) is unaffected, but productive use of FP4 inputs requires the actual SM120 FP4 fragment layout. To investigate in a dedicated chapter for production FP4 attention.
* [GAP-14e-1] **Single-MMA vs chain-last control code delta**. Some bits beyond 26/27 differ between a single QMMA (14a) and a chain-last QMMA (14e #2). Possibly scoreboard slot ID encoding. Not decoded.
* [GAP] **Shape encoding bits**. The bits encoding m16n8k32 (vs hypothetical other shapes) cannot be isolated until another shape is tested. Currently mixed with acc dtype bits in the 10-13 zone.
* [GAP] **Block-scaled MMA (kind::mxf8f6f4)** not tested. Deferred to chapter 16 (block-scaled FP4 peak).
* [GAP] **Other QMMA shapes** not tested. m16n8k32 only.
* [GAP] **C base in opcode bytes**. Hypothesis [HYP-14j-A] that C base is not in opcode bytes (lives in control code) is consistent with 14a/14j observation but not directly tested.
### New instructions observed in this chapter
| Opcode | Usage |
|---|---|
| QMMA.16832.F32.E4M3.E4M3 | MMA m16n8k32, both inputs E4M3, FP32 accumulator |
| QMMA.16832.F32.E4M3.E5M2 | MMA m16n8k32, mixed E4M3/E5M2, FP32 acc |
| QMMA.16832.F32.E5M2.E5M2 | MMA m16n8k32, both E5M2, FP32 acc |
| QMMA.16832.F32.E2M1.E2M1 | MMA m16n8k32, both FP4 e2m1, FP32 acc |
| QMMA.16832.F32.E3M2.E3M2 | MMA m16n8k32, both FP6 e3m2, FP32 acc |
| QMMA.16832.F32.E2M3.E2M3 | MMA m16n8k32, both FP6 e2m3, FP32 acc |
| QMMA.16832.F32.E4M3.E2M1 | MMA m16n8k32, mixed E4M3/E2M1 (FP8/FP4), FP32 acc |
| QMMA.16832.F16.E4M3.E4M3 | MMA m16n8k32, both E4M3, FP16 accumulator |
### New modifiers
* `.16832` on QMMA: shape m16n8k32 (MNK concatenation, k doubled vs HMMA m16n8k16)
* `.F16` / `.F32` on QMMA: accumulator dtype
* `.E4M3`, `.E5M2`, `.E3M2`, `.E2M3`, `.E2M1` on QMMA: input dtype suffixes (both A and B explicit)
* `.reuse` on QMMA B operand: same as HMMA, reuse cache hint for chained MMAs
### Diagnostic workflow for QMMA
When you see a QMMA in a production kernel:
1. **Identify the dtype** from the mnemonic: `QMMA.16832.<acc>.<inputA>.<inputB>`. Both inputs are explicit.
2. **Verify the opcode bytes** start with `0x000000??0c0c727a` (or similar with operand-base substitutions). Confirms QMMA opcode family.
3. **Decode the input dtype bits** in the control code (bits 14-15 for "alt mantissa", 18-21 for exp class).
4. **Check the chaining context** via control code high bytes:
  * `0x084ff6...` = first of chain (sets scoreboard)
  * `0x080ff6...` = mid-chain (waits on scoreboard)
  * `0x000ff0...` = last (no wait, consumer is non-MMA)
5. **Look at the surrounding NOPs**: 2 UIADD3 NOPs after each QMMA when its D feeds another instruction.
6. **Check for `.reuse` on B**: signals chain context.

---

## Kernel 15 MMA narrow (FP6, FP4 at k=32)

Chapter 15 closes the remaining narrow-QMMA dtype combination that chapter 14 did not explicitly test. Most planned chapter 15 variants were already observed in chapter 14: 14g (E3M2 × E3M2), 14h (E2M3 × E2M3), 14d (E2M1 × E2M1), and 14i (E4M3 × E2M1).

### Variants and outcomes
* [OBS] 15c (E3M2 × E2M3, F32 accumulator, m16n8k32): compiled as `QMMA.16832.F32.E3M2.E2M3 R12, R12, R16, R20`.
* [OBS] 15c opcode bytes are `0x000000100c0c727a`, identical to chapter 14 QMMA variants with the same operand bases.
* [OBS] 15c control code is `0x004ff60000246c14`.
* [OBS] 15c dump contains 40 total instructions including padding and trap, and 31 instructions through `EXIT`.
* [OBS] 15c top mnemonics by frequency: 10 `LDG.E.CONSTANT`, 8 `NOP`, 4 `LDC.64`, 4 `IMAD.WIDE.U32`, 4 `STG.E`, 2 `SHF.L.U32`, 2 `@!UPT UIADD3` semantic NOPs, 1 `QMMA.16832.F32.E3M2.E2M3`.
* [OBS] 15f raw FP6 probe compiles to `QMMA.16832.F32.E3M2.E3M2 R12, R12, R16, R20` with control code `0x004ff6000014ec14`.
* [OBS] 15g raw FP4 probe compiles to `QMMA.16832.F32.E2M1.E2M1 R12, R12, R16, R20` with control code `0x004ff6000028ec14`.
* [OBS] 15h unscaled FP4 latency source compiles N=16/32/64 chains of `QMMA.16832.F32.E2M1.E2M1 R16, R4, R2, R16`; the first chained QMMA has control code `0x084ff6000028ec10`.
* [OBS] 15i mixed FP6 latency source compiles N=16/32/64 chains of `QMMA.16832.F32.E3M2.E2M3 R16, R4, R2, R16`; the first chained QMMA has control code `0x084ff60000246c10`.
* [OBS] 15j FP16-accumulator narrow-input variant compiles to `QMMA.16832.F16.E3M2.E3M2 R12, R12, R16, R18` with control code `0x004ff6000014cc12`.
* [OBS] 15k reversed mixed-FP6 variant compiles to `QMMA.16832.F32.E2M3.E3M2 R12, R12, R16, R20` with control code `0x004ff6000018ac14`.
* [OBS] Runtime execution and latency measurement were blocked in this environment because `nvidia-smi` could not communicate with the NVIDIA driver.

### Cross-reference to chapter 14
* [OBS] 14g already established `QMMA.16832.F32.E3M2.E3M2` with control code `0x004ff6000014ec14`.
* [OBS] 14h already established `QMMA.16832.F32.E2M3.E2M3` with control code `0x004ff60000282c14`.
* [OBS] 14d already established `QMMA.16832.F32.E2M1.E2M1` with control code `0x004ff6000028ec14`.
* [OBS] 14i already established mixed `QMMA.16832.F32.E4M3.E2M1` with control code `0x004ff6000020ac14`.
* [INF] 15c validates the chapter 14 per-operand dtype model for the remaining mixed FP6 case: base E4M3/E4M3 control code `0x004ff60000002c14` plus A=E3M2 bits 14 and 18 plus B=E2M3 bit 21 gives predicted `0x004ff60000246c14`, which matches the observed 15c control code exactly.
* [INF] 15k validates the mirror case: base E4M3/E4M3 control code `0x004ff60000002c14` plus A=E2M3 bit 19 plus B=E3M2 bits 15 and 20 gives predicted `0x004ff6000018ac14`, which matches the observed 15k control code exactly.
* [INF] 15j confirms the chapter 14 accumulator dtype encoding carries over to FP6 inputs: F16 accumulator changes the low control bits from F32 form while preserving E3M2 input bits.
* [RES] Chapter 15 question "Does FP6 have a distinct SASS opcode from FP8?" is resolved: no distinct opcode was observed. E3M2 and E2M3 use the same QMMA opcode family (`0x7a`) with dtype encoded in control-code bits, consistent with chapter 14.
* [RES] Chapter 15 question "Does mixed FP6 use orthogonal operand dtype encoding?" is resolved: the E3M2 × E2M3 control code equals the bitwise composition predicted from the symmetric FP6 cases.
* [RES] Chapter 15 operand-order asymmetry test resolved the control-code part: reversing E3M2/E2M3 mirrors the A/B dtype bits exactly. Value-layout asymmetry remains untested at runtime in this environment.

### Open gaps
* [GAP] FP6 element packing inside the A/B uint32 fragments remains decode-unresolved. Chapter 23 adds first-pass SASS coverage and runtime smoke outputs for FP6 baseline, mixed, lane-pattern, and operand-order probes, but the full lane-to-value map is not decoded yet.
* [GAP] FP4 E2M1 fragment layout remains decode-unresolved from GAP-14d-1. Chapter 23 adds first-pass SASS coverage and runtime smoke outputs for E2M1 baseline, lane-pattern, LDSM-fed, direct-register, K-boundary, register-boundary, and special-value probes, but the full lane-to-value map is not decoded yet.
* [OBS] Unscaled E2M1 QMMA latency probes run on the host GPU: N=16 total 1074 cycles, N=32 total 1639 cycles, N=64 total 2737 cycles.
* [INF] The unscaled E2M1 QMMA marginal latency from N=32 to N=64 is approximately 34.3 cycles per QMMA.
* [OBS] Mixed FP6 E3M2 x E2M3 latency probes run on the host GPU: N=16 total 1075 cycles, N=32 total 1636 cycles, N=64 total 2743 cycles.
* [INF] The mixed FP6 E3M2 x E2M3 marginal latency from N=32 to N=64 is approximately 34.6 cycles per QMMA.
* [GAP] The SASS-level cause of the k=32 FP4 throughput gap versus k=64 block-scaled OMMA remains chapter 16's shape/family distinction until 15h can be run on hardware.

### New instructions observed in this chapter
| Opcode | Usage |
|---|---|
| QMMA.16832.F32.E3M2.E2M3 | Mixed FP6 m16n8k32 MMA with FP32 accumulator |
| QMMA.16832.F32.E2M3.E3M2 | Reversed mixed FP6 m16n8k32 MMA with FP32 accumulator |
| QMMA.16832.F16.E3M2.E3M2 | FP6 m16n8k32 MMA with FP16 accumulator |

### Diagnostic update
* [INF] For `kind::f8f6f4` production audits, all observed FP8, FP6, FP4, and mixed narrow combinations remain under `QMMA.16832` for k=32. Narrow input dtype changes the mnemonic suffix and control-code dtype bits, not the opcode family.

---

## Kernel 16 FP4 peak block-scaled MMA on SM120
Chapter decoding the block-scaled MMA family on SM120: kind::mxf8f6f4 (k=32) and kind::mxf4nvf4 (k=64). Four variants: mxf8f6f4 baseline, mxf4nvf4 standard (2X ue8m0), mxf4nvf4 peak (4X ue4m3), and latency microbench on the peak path. Reveals a new SASS opcode family OMMA distinct from HMMA and QMMA, completing the SM120 tensor core landscape.
### Variants and outcomes
* [OBS] 16a (mxf8f6f4, scale_vec::1X, m16n8k32, e4m3 inputs, ue8m0 scales): d[0] = 32.0 (matches 14a baseline). Reveals `QMMA.SF.16832.F32.E4M3.E4M3.E8` — existing QMMA opcode with new `.SF` modifier. 123 lines SASS.
* [OBS] 16b (mxf4nvf4, scale_vec::2X, m16n8k64, e2m1 inputs, ue8m0 scales): d[0] = 0.0 (zero inputs). Reveals NEW opcode `OMMA.SF.16864.F32.E2M1.E2M1.E8` with low byte `0x7f`. 123 lines SASS.
* [OBS] 16c (mxf4nvf4, scale_vec::4X, m16n8k64, e2m1 inputs, ue4m3 scales): d[0] = 0.0. Reveals `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` — same OMMA opcode with new `.UE4M3.4X` modifier suffix. 123 lines SASS. This is the announced 900+ TFLOPS FP4 peak path.
* [OBS] 16d (OMMA 4X latency chain, N = 16/32/64): cycles/MMA measured. Linear fit total_cycles(N) = 330 + 29 × N. Asymptotic cycles/MMA = **~29**, lower than HMMA/QMMA (~35).
### New SASS opcode family: OMMA
* [OBS] `OMMA.SF.16864.F32.<A>.<B>.<scale_dtype>[.<scale_vec>]` is a new opcode family on SM120, low byte `0x7f`. Distinct from QMMA (`0x7a`) and HMMA (`0x3c`).
* [OBS] OMMA shape is always m16n8k64 (k=64). Used exclusively for `kind::mxf4nvf4` PTX.
* [HYP] The "O" in OMMA presumably stands for "Octal" (FP4 packs 8 values per uint32) or references the doubled k dimension.
### Complete dense SM120 MMA opcode landscape
After chapters 13, 14, 16:
| Family | Low byte | Shape | PTX kind | Scaled? |
|---|---|---|---|---|
| HMMA | 0x3c | m16n8k16 | mma.sync standard | No |
| QMMA | 0x7a | m16n8k32 | kind::f8f6f4, kind::mxf8f6f4 | Optional (.SF modifier) |
| **OMMA** | **0x7f** | m16n8k64 | kind::mxf4nvf4 | Always (implicit .SF) |
The low byte of the opcode is a reliable family identifier.
### Block-scaled SASS mnemonic structure
* [OBS] `<FAMILY>.SF.<shape>.<acc_dtype>.<A_dtype>.<B_dtype>.<scale_dtype>[.<scale_vec_tag>]`
* [OBS] `.SF` modifier = Scale Factor enabled (block scaling mode).
* [OBS] Scale dtype abbreviation:
  * `.E8` = ue8m0 (unsigned, 8-bit exp, 0 mantissa) — the common case, tagged short
  * `.UE4M3` = ue4m3 (unsigned, 4-bit exp, 3-bit mantissa) — explicit full notation
* [OBS] scale_vec convention:
  * `scale_vec::1X` is default for QMMA → silent (not tagged)
  * `scale_vec::2X` is default for OMMA → silent (not tagged)
  * `scale_vec::4X` is not default → explicit `.4X` tag required
### Operand layout for block-scaled MMA
* [OBS] 7 operands in the SASS mnemonic: D, A, B, C, SFA, SFB, URZ
  * SFA, SFB = scale factor registers (single uint32 each, with packed scale values for 2X and 4X modes)
  * URZ = uniform register zero, placeholder for the bidA/tidA/bidB/tidB parameters (constants = 0 in CUTLASS atoms)
* [OBS] When SFA and SFB have equal values, ptxas colocates them in the same register (register reuse optimization).
### Control code encoding of scaling mode
* [OBS] Block-scaled variants share identical opcode bytes `0x7000000a0c0c747f` for all OMMA modes. **The scaling configuration (scale_vec + scale_dtype) lives entirely in the control code, byte 2 (bits 16-23).**
* [OBS] Observed byte 2 values:
  * 16a (QMMA.SF 1X ue8m0): byte 2 = 0x00
  * 16b (OMMA.SF 2X ue8m0): byte 2 = 0x08 (bit 19 set)
  * 16c (OMMA.SF 4X ue4m3): byte 2 = 0x04 (bit 18 set)
* [HYP] Two candidate decodings:
  * Direct mapping: each (scale_vec, scale_dtype) pair has a unique byte 2 value
  * Orthogonal bits: bit 19 = scale_vec::2X indicator, bit 18 = ue4m3+4X fine-grained indicator
* [GAP] Cannot disambiguate without testing mxf4nvf4 4X with ue8m0 (not a CUTLASS atom). Both interpretations match the 3 observed variants.
### OMMA latency and FLOPs/cycle (16d)
* [OBS] OMMA.SF 4X ue4m3 serial chain latency measured at N = 16, 32, 64.
* [OBS] Linear model: total_cycles(N) = 330 + 29 × N.
* [OBS] Asymptotic cycles/MMA = ~29.
* [OBS] Cross-family comparison of chain-latency throughput:

| Opcode | Shape | FLOPs/MMA | Cycles/MMA | FLOPs/cycle |
|---|---|---|---|---|
| HMMA | m16n8k16 | 4096 | 35 | 117 |
| QMMA | m16n8k32 | 8192 | 35 | 234 |
| OMMA 4X | m16n8k64 | 16384 | **29** | **565** |

* [OBS] OMMA delivers 4.8× more FLOPs/cycle than HMMA and 2.4× more than QMMA at the latency floor.
* [OBS] Single-warp chain-bound throughput = ~254 TFLOPS on RTX 5070 Ti at 2.45 GHz × 46 SMs.
* [HYP] The announced 900+ TFLOPS peak is achievable only via pipelined throughput (multiple MMAs in flight simultaneously), not via a latency-bound chain. Chain bound provides ~3.5× headroom to 900 TFLOPS via pipelining.
### Chain dependency and register reuse patterns
* [OBS] OMMA chain pattern observed in 16d: first MMA has C=RZ (zero init), subsequent MMAs have C=R12 (previous D). Pattern identical to HMMA and QMMA chains (chap 13d, 14e).
* [OBS] `.reuse` flags on A (R2) and SFB (R8) in OMMA chain. ptxas keeps these operands in register cache between consecutive MMAs.
* [OBS] Scoreboard wait mask in control code byte 0:
  * First MMA in chain: byte 0 = 0xff (no dependency)
  * Subsequent MMAs: byte 0 = 0x0c (wait on previous D's scoreboard slot)
* [OBS] Same dependency-tracking mechanism across HMMA, QMMA, OMMA families.
### Resolved hypotheses
* [RES] Block-scaled mxf4nvf4 produces a new SASS opcode family (OMMA, low byte 0x7f)
* [RES] Block-scaled mxf8f6f4 uses existing QMMA with `.SF` modifier
* [RES] Scale factors add 2 new operands (SFA, SFB) plus URZ placeholder
* [RES] Shape m16n8k64 is specific to mxf4nvf4 (kind::mxf4nvf4)
* [RES] `.UE4M3.4X` suffix identifies the fine-grained FP4 peak path
* [RES] scale_vec and scale dtype encoded in control code byte 2 (not opcode bytes)
* [RES] OMMA chain latency ~29 cycles/MMA, faster than HMMA/QMMA (~35)
* [RES] OMMA gives 4.8× FLOPs/cycle vs HMMA at the latency floor
### Open gaps
* [GAP-16-1] Orthogonal vs direct-map encoding of dense OMMA control code byte 2 cannot be fully disambiguated without a dense mxf4nvf4 4X ue8m0 variant. **Chapter 19 partially constrains this gap**: sparse `mxf4nvf4` 4X ue8m0 compiles as `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8.4X` with control code `0x004ff60000013e14`, proving the sparse path admits 4X with ue8m0.
* [GAP-16-2] OMMA pipelined throughput not measured. Would require a multi-warp microbench with independent MMAs.
* [GAP-16-3] Scale factor values other than 1.0 not tested. Real block-scaled kernels use diverse scale values per block.
* [GAP-16-4] Direct comparison with production FP4 GEMM SASS (CUTLASS, Marlin FP4) not done. Would validate the decoded opcodes in production context.
* [GAP] FP4 (E2M1) register layout from chap 14d [GAP-14d-1] still unresolved. Independent of opcode decoding but blocks full audit of FP4 kernels.
### New instructions observed in this chapter
| Opcode | Usage |
|---|---|
| QMMA.SF.16832.F32.\<A\>.\<B\>.E8 | mma.sync kind::mxf8f6f4 with scale_vec::1X ue8m0 |
| OMMA.SF.16864.F32.\<A\>.\<B\>.E8 | mma.sync kind::mxf4nvf4 with scale_vec::2X ue8m0 (OMMA default) |
| OMMA.SF.16864.F32.\<A\>.\<B\>.UE4M3.4X | mma.sync kind::mxf4nvf4 with scale_vec::4X ue4m3 (FP4 peak) |
### New modifiers
* `.SF` on QMMA: Scale Factor enabled (block scaling mode)
* `.E8` suffix: scale dtype ue8m0 (short form for the common case)
* `.UE4M3` suffix: scale dtype ue4m3 (full notation for fine-grained mode)
* `.4X` suffix on OMMA: scale_vec::4X (finer granularity than default 2X)
### Implications for production kernel audit
With chapters 13 (HMMA), 14 (QMMA), 16 (dense block-scaled), 17 (LDSM), 18 (cp.async pipeline), 19 (sparse MMA), 20 (control flow), 21 (divergence/reconvergence), 22 (STSM), 23 (FP4/FP6 fragment-layout probes), and 24 (production mini-GEMM audit), the repo documents the observed warp-level tensor-core opcode families emitted by `mma.sync`, `mma.sp`, `ldmatrix`, and `stmatrix` on SM120, plus a first-pass end-to-end pipeline fixture. [INF] This is sufficient for first-pass identification of dense, sparse, block-scaled, matrix-load, matrix-store, async-copy, global-store, and reduction instructions in production dumps. [INF] It is still not sufficient for complete production epilogue auditing because full FP4/FP6 lane-to-value decode, STSM epilogue/storeback semantics, and audit-confidence qualification still have open gaps.
### Diagnostic workflow for block-scaled MMA in production
When auditing a block-scaled SASS dump:
1. Identify the MMA family via low byte: 0x3c = HMMA, 0x7a = QMMA, 0x7f = OMMA.
2. Check for `.SF` modifier in the mnemonic — indicates block scaling is active.
3. Read shape from mnemonic: 16816, 16832, or 16864.
4. Read input dtypes: E4M3, E5M2, E3M2, E2M3, E2M1 (see chap 14 for the full dtype encoding table).
5. Read scale dtype: `.E8` = ue8m0, `.UE4M3` = ue4m3.
6. Check for scale_vec tag: absent = default (1X for QMMA, 2X for OMMA), `.4X` = fine-grained.
7. Decode control code byte 2 for confirmation of scaling mode.
8. Look for `.reuse` flags on A and SFB operands in chains (register cache optimization).
9. Count cycles via scoreboard mask in byte 0 of control code.

---

## Kernel 17 LDSM (ldmatrix) baseline on SM120
Chapter establishing the LDSM SASS opcode family — the realization of `ldmatrix.sync.aligned.*` from PTX. LDSM bridges shared memory and tensor core fragments. Six variants tested: width grid (x1/x2/x4), transpose modifier, production combo (LDSM + HMMA), and serial latency microbenchmark.
### Variants and outcomes
* [OBS] 17a (x1, no trans): baseline `LDSM.16.M88 R5, [R5+UR4]`. 22 useful instructions. d[0] = 0x3c000000 (halves 0.0, 1.0).
* [OBS] 17b (x2, no trans): `LDSM.16.M88.2 R4, [R4+UR4]`. Mnemonic suffix `.2` and bit 8 of the control code differ from 17a. Thread 0 receives 4 halves from two tiles (0, 1, 64, 65).
* [OBS] 17c (x4, no trans): `LDSM.16.M88.4 R4, [R0]`. Mnemonic suffix `.4`, bit 9 of the control code. Production case: 4 tiles loaded simultaneously. Addressing uses `[R]` only (no UR base) because ptxas packs the full address in R0.
* [OBS] 17d (x4, trans): `LDSM.16.MT88.4 R4, [R0]`. Mnemonic changes `M88` to `MT88` for the transpose flag, bit 14 of the control code. Output verified transposed (column-ordered).
* [OBS] 17e (LDSM.x4 A + LDSM.x2 B + HMMA.16816.F32): production pattern. Both LDSMs feed directly into HMMA with no NOPs in between. Scoreboard sync only. Register allocation packed: LDSM destinations are HMMA sources.
* [OBS] 17f (N chained LDSM.x1 with clock64): total cycles 509, 1038, 2094 for N = 16, 32, 64. Linear model `total_cycles ≈ 33 × N`, chain overhead essentially zero.
### Key observations
* [OBS] **LDSM opcode family** on SM120: `LDSM.16.M[T]88[.<N>]` with opcode bytes `0x000000000004783b` (low byte `0x3b`, distinct from LDS `0x84` and LDG `0x81`). Base opcode invariant across all width and trans variants when destination register base is the same.
* [OBS] Mnemonic decoding:
  * `.16` = element size in bits (half)
  * `M88` (no trans) or `MT88` (trans) — the T for transpose lives **inside** the shape tag, not as a suffix
  * `.2` / `.4` suffix = multi-matrix width; no suffix = x1 (default)
* [OBS] Six variants of LDSM identified (x1/x2/x4 × trans/no-trans). Four tested (17a-d), two inferred from the model (`LDSM.16.MT88` for x1.trans, `LDSM.16.MT88.2` for x2.trans).
* [OBS] Addressing form varies by context:
  * `[R + UR]` used by x1, x2 (17a, 17b): R = per-lane row offset, UR = shared memory base
  * `[R]` used by x4 (17c): full address pre-computed into R0 (register recycling from STS)
  * Both forms are syntactically valid; the choice is a ptxas optimization, not a hardware constraint
* [OBS] Shared memory base uses the standard SM120 pattern: `UMOV UR4, 0x400` + `S2UR UR_cta` + `ULEA UR4, UR_cta, UR4, 0x18`. Identical to kernels 06-08.
### Control code topology
* [OBS] By comparing 17a/b/c byte-by-byte (for width) and 17c vs 17d (for trans), the following LDSM control code bits are decoded:
| Bit | Meaning | Evidence |
|---|---|---|
| 8 | Width bit 0 | 17a vs 17b |
| 9 | Width bit 1 | 17a vs 17c |
| 14 | Transpose flag | 17d vs 17c |
| 32-39 | Scoreboard slot ID (varies per instance) | 17e, 17f |
| Others | Standard scheduling (stall/yield/wait) | - |
* [OBS] Width field reads as `log2(N)` for N ∈ {1, 2, 4}:
  * x1: bits (9, 8) = (0, 0), field = 0
  * x2: bits (9, 8) = (0, 1), field = 1
  * x4: bits (9, 8) = (1, 0), field = 2
  * Value 3 unused, possibly reserved for future extensions
### Production pattern (17e)
* [OBS] Complete SASS sequence for `STS → BAR → LDSMs → HMMA → STG` on SM120:
  ```
  STS × N
  BAR.SYNC.DEFER_BLOCKING 0x0
  LDSM.16.M88.2 R10, [R7+UR5]     ← B fragment (emitted first by ptxas!)
  LDSM.16.M88.4 R12, [R6+UR4]     ← A fragment (emitted second)
  HMMA.16816.F32 R12, R12, R10, RZ
  @!UPT UIADD3 URZ NOP
  @!UPT UIADD3 URZ NOP
  STG × N
  ```
* [OBS] **ptxas inverts LDSM emission order**: source code emits A (x4) before B (x2), but SASS shows B (x2) first then A (x4). Likely because emitting the smaller LDSM first gives the larger one more time to complete before HMMA needs both.
* [OBS] **Zero NOPs between LDSM and HMMA**. Unlike MMA → consumer patterns (2 UIADD3 NOPs), the scoreboard mechanism fully handles LDSM → MMA sync.
* [OBS] **HMMA wait mask `0xff`**: in 17e, HMMA control code low byte is `0xff` (binary `1111 1111`), distinct from the `0x14` observed in chapter 13b (HMMA with LDG sources). The `0xff` mask monitors all active scoreboards, ensuring wait on both LDSMs.
* [OBS] **Zero-copy register allocation**: ptxas packs LDSM destinations exactly where HMMA reads its sources. No intermediate copies.
  * LDSM.x2 destinations: R10, R11 → HMMA.B source R10
  * LDSM.x4 destinations: R12..R15 → HMMA.A source R12, also colocated with HMMA.D
* [OBS] **2 UIADD3 NOPs after HMMA**: identical pattern to chapter 13b (HMMA → consumer dependency).
### Latency (17f)
* [OBS] Three measurements at N = 16, 32, 64 chained LDSM.x1 via pointer-chase trick (add LDSM output = 0 to the address, creating a real dependency without invalidating the address):
| N | total_cycles | cycles per LDSM |
|---|---|---|
| 16 | 509 | 31.81 |
| 32 | 1038 | 32.44 |
| 64 | 2094 | 32.72 |
* [OBS] Linear regression on incremental costs:
  * ΔN(16→32): 529 cycles for 16 added LDSMs → 33.06 cycles/LDSM
  * ΔN(32→64): 1056 cycles for 32 added LDSMs → 33.00 cycles/LDSM
* [RES] **Model**: `total_cycles ≈ 33 × N`. Chain overhead essentially zero (slightly negative, suggesting a small pipeline fill effect where the first LDSM is faster than steady-state).
* [OBS] **Comparison with MMA latency**:
| Operation | Cycles/instance | Chain overhead | Model |
|---|---|---|---|
| HMMA.16816.F32 (13e) | ~35 | ~310 | `310 + 35×N` |
| QMMA.16832.F32 (14f) | ~35 | ~510 | `510 + 35×N` |
| LDSM.16.M88 x1 (17f) | ~33 | ~0 | `33×N` |
* [OBS] LDSM per-instance cost comparable to MMA, but chain overhead is effectively zero. LDSM does not require the MAC initialization overhead of HMMA/QMMA.
* [HYP] LDSM and MMA likely share internal datapaths (tensor memory unit), explaining similar latencies. LDSM doesn't need the startup overhead because it's shared memory access + matrix assembly, no MAC unit init.
### LDSM chain pattern (17f internals)
* [OBS] Chain structure (16 LDSMs on SM120):
  ```
  LDSM.16.M88 R3, [R3+UR4]    ctrl 0x000e240008000000   (first: wait + SBS)
  IADD R4, R4, R3              ctrl 0x001fca00078e0000   (wait on LDSM scoreboard)
  LDSM.16.M88 R5, [R4]         ctrl 0x000e240000000000   (subsequent: SBS only)
  IADD R5, R4, R5
  ...
  ```
* [OBS] **No NOPs anywhere in the chain**. Scoreboard handles all dependencies.
* [OBS] **ptxas register renaming**: LDSM destinations alternate across ~8 registers (R3, R5, R2, R7, R6, R9, R8) instead of overwriting. Allows maximum ILP potential.
* [OBS] **First LDSM carries a wait bit** (byte 24-31 = `0x08`). [HYP] The bit may be related to the preceding BAR.SYNC synchronization. [OBS] Subsequent LDSMs in the chain omit that explicit wait bit and rely on the register dependency through IADD.
* [OBS] **Register pressure minimal**: 8 registers suffice for N=16 LDSMs in chain.
### Implications for production pipelining
* [HYP] In a GEMM k-loop, the typical latency chain per tile is:
  * LDSM_A + LDSM_B in parallel → ~33 cycles (critical path = max, not sum)
  * HMMA → ~35 cycles
  * Total serial per tile: ~68 cycles (pessimistic, no overlap)
* [HYP] With software pipelining (load next tile while current tile MMAs), this overlaps down to max(LDSM, HMMA) ≈ 35 cycles per tile. Confirmed pattern in CUTLASS / FlashAttention cp.async pipelines.
### Resolved hypotheses
* [RES] LDSM is a new opcode family distinct from LDS (`0x84`) and LDG (`0x81`). Family byte `0x3b`.
* [RES] Width (x1/x2/x4) encoded in 2 bits of control code (bits 8-9), not in opcode bytes.
* [RES] Trans encoded in 1 bit of control code (bit 14).
* [RES] Opcode bytes invariant across width and trans variants.
* [RES] LDSM → HMMA requires no NOPs. Scoreboard-only sync.
* [RES] HMMA wait mask differs between LDG-sourced (0x14) and LDSM-sourced (0xff).
* [RES] Register allocation perfectly packed: LDSM destinations = HMMA sources, zero-copy.
* [RES] LDSM serial latency ~33 cycles on SM120, comparable to MMA.
* [RES] Chain overhead ~0 for LDSM, vs ~310 for HMMA and ~510 for QMMA.
### Open gaps
* [GAP] `.trans` variants x1 and x2 not compiled/verified (inferred only). Predictions: `LDSM.16.MT88` and `LDSM.16.MT88.2`.
* [RES] `stmatrix` is covered by Chapter 22 for tested m8n8 b16 forms. Those forms lower to `STSM.16.M[T]88[.2|.4]`; scalar shared-store fallback remains `STS.128`.
* [GAP] LDSM width > 4 not tested. The 2-bit width field value 3 is unused — probably reserved.
* [GAP] LDSM with larger elements (e.g., `.b32` variant) not tested. The `.16` in the mnemonic suggests an element size field that could hold other values.
* [GAP] LDSM combined with cp.async (LDGSTS) in a full pipelined tile not tested. Needs chapter 18 (pipelined tile).
* [GAP] Exact bit allocation for scoreboard slot ID in bytes 32-39 not formally decoded.
### New instructions observed in this chapter
| Opcode | Usage |
|---|---|
| LDSM.16.M88 | ldmatrix.x1.m8n8.shared.b16 |
| LDSM.16.M88.2 | ldmatrix.x2.m8n8.shared.b16 |
| LDSM.16.M88.4 | ldmatrix.x4.m8n8.shared.b16 |
| LDSM.16.MT88.4 | ldmatrix.x4.trans.m8n8.shared.b16 |
### New modifiers
* `.16` on LDSM: element size (bits). Only 16 (half) tested.
* `M88` on LDSM: shape m8n8 in normal orientation
* `MT88` on LDSM: shape m8n8 in transposed orientation (T marker inside shape tag)
* `.2`, `.4` on LDSM: width suffix (x2, x4); absent for x1 default
### Diagnostic workflow for LDSM in production
When auditing a GEMM or attention kernel:
1. Locate LDSM instructions (search `LDSM`). They cluster around BAR.SYNC barriers.
2. Identify width: `.2` / `.4` suffix for x2 / x4; no suffix = x1.
3. Identify orientation: `M88` vs `MT88`. B fragments are commonly trans in row-col GEMM layouts.
4. Find the MMA that consumes LDSM outputs: should be within a few instructions, dst registers match MMA sources.
5. Check MMA wait mask: `0xff` when LDSM feeds directly, distinct from `0x14` (HMMA with LDG sources).
6. Estimate cycles: `~33 per LDSM + ~35 per MMA`, pipelined across k-loop iterations.

---

## Kernel 18 Pipelined MMA tile with cp.async on SM120
Chapter establishing the production GEMM pipeline pattern on SM120 via cp.async. Three variants: 2-stage full unroll, K-loop 4 tiles with 2-stage pipeline, 3-stage pipeline. Three new SASS opcodes decoded: LDGSTS (cp.async), LDGDEPBAR (commit_group), DEPBAR.LE SB0, N (wait_group N). Production pattern prologue → loop body → tail → epilogue fully captured.
### Variants and outcomes
* [OBS] 18a (2-stage unrolled, 2 tiles): full 2-stage pipeline, tile 0 + tile 1 prefetched then consumed. d[0] = 32.0. Reveals LDGSTS.E.LTC128B.128, LDGDEPBAR, DEPBAR.LE SB0, N opcodes. 164 lines SASS.
* [OBS] 18b (K-loop 4 tiles with `#pragma unroll 1`, 2-stage pipeline): real SASS loop with BRA back-edge at 0x0470, counter R0 with ISETP.NE.U32.AND comparison to K-1. d[0] = 64.0. 228 lines SASS. Register spill observed (STL.64/LDL).
* [OBS] 18c (3-stage pipeline, 3 tiles unrolled): 3 × LDGSTS + 3 × LDGDEPBAR, then 3 × DEPBAR.LE SB0 with N = 2, 1, 0. d[0] = 48.0. 196 lines SASS. Decodes the N argument encoding.
### New opcodes discovered
* [OBS] `LDGSTS.E.LTC128B.128` is the SASS realization of PTX `cp.async.ca.shared.global.L2::128B`. Low byte `0xae` family. Distinct from LDG (`0x81`), STS (`0x88`), LDS (`0x84`).
  * Mnemonic: "Load Global Store Shared"
  * `.E` = Extended (64-bit addressing)
  * `.LTC128B` = L2 cache hint, 128-Byte alignment
  * `.128` = 128-bit transfer (16 bytes per thread)
* [OBS] `LDGDEPBAR` is the SASS realization of PTX `cp.async.commit_group`. No operands. Opcode bytes `0x00000000000079af`. Mnemonic "Load Global Dependency Barrier". Establishes ordering w.r.t. previously issued LDGSTS.
* [OBS] `DEPBAR.LE SB0, N` is the SASS realization of PTX `cp.async.wait_group N`. Blocks until <= N commit groups remain in flight in scoreboard bank 0.
### DEPBAR.LE N encoding (decoded from 18c)
* [OBS] Three DEPBAR.LE instances observed with distinct N values:
  ```
  DEPBAR.LE SB0, 0x0    ctrl 0x000080000000791a
  DEPBAR.LE SB0, 0x1    ctrl 0x000080400000791a
  DEPBAR.LE SB0, 0x2    ctrl 0x000080800000791a
  ```
* [OBS] N is encoded as a **2-bit field in bits 38-39** of the DEPBAR.LE control code:
  * N=0: bits (38, 39) = (0, 0) → byte 4 = 0x00
  * N=1: bits (38, 39) = (1, 0) → byte 4 = 0x40
  * N=2: bits (38, 39) = (0, 1) → byte 4 = 0x80
  * N=3: bits (38, 39) = (1, 1) → byte 4 = 0xC0 (predicted)
* [HYP] Max N = 3 (four in-flight groups possible). Beyond this, must use DEPBAR.LE with N=0 (wait all).
### Scoreboard bank 0 for cp.async
* [OBS] All DEPBAR.LE instances observed target **SB0** (scoreboard bank 0). This is the async-copy dedicated bank on SM120.
* [HYP] Other scoreboard banks (SB1, SB2, etc.) may exist for other async operations (TMA on SM90+, tcgen05.mma on SM100+) but are not observed on SM120 with the tested kernels.
### Production pipeline pattern
* [OBS] The canonical production GEMM structure observed across 18a/b/c:
  ```
  Prologue:
    CS2R R_acc, SRZ                            # initialize accumulator to zero
    compute addresses
    @!PT LDS RZ × 3                            # scheduling hints
    LDGSTS tile[0]
    LDGDEPBAR                                  # commit group 0
  
  Main loop (k = 0 .. K-2):
    compute next tile address
    @!PT LDS RZ × 3
    LDGSTS tile[k+1]
    LDGDEPBAR
    DEPBAR.LE SB0, 0x1                         # wait tile k
    BAR.SYNC.DEFER_BLOCKING 0x0
    LDSM.x2 B tile[k]                          # ptxas emits B before A
    LDSM.x4 A tile[k]
    HMMA (accumulate into R_acc)
    UIADD3 NOPs × 2
    counter++, compare, BRA back
  
  Tail:
    DEPBAR.LE SB0, 0x0                         # wait all remaining
    BAR.SYNC
    LDSM.x2 B tile[K-1]
    LDSM.x4 A tile[K-1]
    HMMA (final, D renamed for STG)
    UIADD3 NOPs × 2
  
  Epilogue:
    STG × N
    EXIT
  ```
### Key patterns cristallized
* [OBS] **ptxas software pipelining** is aggressive: in 18a, LDSM of tile 1 is emitted BEFORE the HMMA of tile 0. This hides LDSM latency (~33 cycles) behind HMMA compute (~35 cycles). What CUTLASS tries to achieve manually is automatic when dependencies are clear.
* [OBS] **ptxas inverts LDSM emission order A/B**: systematically `LDSM.x2` (B fragment, smaller) before `LDSM.x4` (A fragment, larger). Observed in 17e, 18a, 18b, 18c. Likely to give the larger load more time to complete before the consuming HMMA needs both.
* [OBS] **Real SASS loop with `#pragma unroll 1`**: 18b has a counter register (R0), `ISETP.NE.U32.AND P1, PT, R0, 0x3, PT` comparison, and `@P1 BRA 0x2b0` back-edge. The loop iterates 3 times, each doing one prefetch + one compute, with a tail for the last tile.
* [OBS] **Double-buffer addressing via LOP3.LUT**: buffer parity toggle using bitwise operations. `LOP3.LUT R, R, 0x1, RZ, 0xc0, !PT` (AND with 1 for current buffer), `LOP3.LUT R, R, 0x1, RZ, 0x3c, !PT` (XOR with 1 for next buffer).
* [OBS] **Accumulator zero-initialization via CS2R**: `CS2R R_acc, SRZ` used systematically at start of GEMM kernels. More efficient than 4 × `MOV R, RZ` for a 4-register clear.
* [OBS] **Register spill in pipelined GEMM** (18b): `STL.64 [R1], R2` and corresponding `LDL` instructions. Register pressure is high in pipelined kernels with dual buffers + accumulator + LDSM/HMMA sources. Could potentially be mitigated with `__launch_bounds__` or restructuring.
* [OBS] **HMMA wait mask varies by context**:
  * 17e HMMA (LDSMs directly): wait mask 0xff (byte 0 of control code)
  * 18b HMMA (LDSMs after LDGSTS chain): wait mask 0x04
  * The mask adapts to active scoreboard slots at HMMA execution time.
### Unresolved pattern: `@!PT LDS RZ, [RZ]` triplet
* [OBS] Before every LDGSTS instruction, ptxas emits **3 × `@!PT LDS RZ, [RZ]`** (always-false predicated LDS with zero destination and zero source).
* [OBS] Pattern appears in all three variants (18a, 18b, 18c), always in groups of 3, always before a LDGSTS.
* [HYP] Likely a scheduling hint or pipeline alignment marker specific to cp.async context. Not observed in cp.async-free LDSM kernels (17a-17e).
* [GAP-18-1] The exact role and mechanism of the `@!PT LDS RZ` triplet remains undecoded.
### Resolved hypotheses
* [RES] cp.async maps to a new SASS opcode (LDGSTS.E.LTC128B.128)
* [RES] commit_group emits a distinct opcode (LDGDEPBAR, no operands)
* [RES] wait_group N emits DEPBAR.LE with N in control code
* [RES] N is encoded as a 2-bit field in DEPBAR.LE control code bits 38-39
* [RES] ptxas software-pipelines LDSM and MMA when dependencies allow
* [RES] `#pragma unroll 1` produces a real SASS loop with BRA back-edge
* [RES] 3-stage pipeline uses DEPBAR.LE with N = 2, 1, 0 successively
* [RES] ptxas emits LDSM.x2 before LDSM.x4 in the tested MMA-consuming contexts (17e, 18a, 18b, 18c).
### Open gaps
* [GAP-18-1] `@!PT LDS RZ, [RZ]` triplet role before LDGSTS not decoded.
* [GAP] Scoreboard banks other than SB0 not observed (SB1, SB2, etc. may be used by TMA on SM90+ or tcgen05 on SM100+).
* [GAP] LDGSTS latency not microbenchmarked (depends on L2 hit rate and bandwidth, less informative than MMA cycle count).
* [GAP] Other cp.async variants (`cp.async.cg.*`, non-L2 hint) not tested. Different caching modes may emit different LDGSTS variants.
* [GAP] Direct comparison with CUTLASS SM120 GEMM SASS not done. Would validate decoded patterns against production code.
### New instructions observed in this chapter
| Opcode | Usage |
|---|---|
| LDGSTS.E.LTC128B.128 | cp.async.ca.shared.global.L2::128B, 16-byte transfer |
| LDGDEPBAR | cp.async.commit_group |
| DEPBAR.LE SB0, N | cp.async.wait_group N (N in {0, 1, 2, 3}) |
| CS2R R, SRZ | Efficient accumulator zero-init (already observed in chap 13e for latency setup) |
### New modifiers
* `.E` on LDGSTS: Extended 64-bit addressing
* `.LTC128B` on LDGSTS: L2 cache hint at 128-Byte alignment
* `.128` on LDGSTS: 128-bit transfer size
* `SB0` on DEPBAR.LE: scoreboard bank 0 (cp.async dedicated bank)
* `.LE` on DEPBAR: less-or-equal comparison mode for wait count
### Toolkit completion after chapter 18
With chapters 13 (HMMA), 14 (QMMA), 17 (LDSM), and 18 (cp.async pipeline), the repo has decoded the core observed opcodes needed for SM120 GEMM-style audits:
| Pattern | Opcode | Chapter |
|---|---|---|
| Global load | LDG.E.* | 08 |
| Shared store | STS.* | 06 |
| cp.async | LDGSTS.E.LTC128B.128 | 18 |
| Commit async | LDGDEPBAR | 18 |
| Wait async | DEPBAR.LE SB0, N | 18 |
| Barrier | BAR.SYNC.DEFER_BLOCKING 0x0 | 06 |
| Load matrix | LDSM.16.M[T]88[.N] | 17 |
| Tensor core FP16 | HMMA.16816.F32 | 13 |
| Tensor core FP8/FP6/FP4 | QMMA.16832.<acc>.<A>.<B> | 14 |
| Global store | STG.E.* | 08 |
### Diagnostic workflow for pipelined GEMM in production
When auditing a production SASS dump:
1. Locate LDGSTS instructions — they cluster near the top of the kernel and inside the main loop.
2. Count LDGDEPBAR to identify pipeline stages (N stages = N commit groups per iteration setup).
3. Examine DEPBAR.LE byte 4: N value indicates how many groups are allowed to remain in flight.
4. Check for BRA back-edge: presence indicates a real SASS loop vs full unroll.
5. Count HMMA instructions: typically one per K-loop iteration, plus any tail.
6. Verify `.reuse` on B operand of each HMMA except the last (chain pattern).
7. LDGSTS to HMMA latency per tile = ~L2 access + ~33 (LDSM) + ~35 (HMMA). Pipeline hides this if next tile's LDGSTS is issued early enough.

---

## Kernel 19 Sparse MMA

* [OBS] Chapter 19 establishes warp-level sparse MMA encoding on SM120. Variants cover non-scaled `kind::f8f6f4`, metadata immediate changes, selector rejection, F16 accumulator, block-scaled `kind::mxf8f6f4`, block-scaled `kind::mxf4nvf4`, and a 16-instruction sparse dependency chain.

### Observations

* [OBS] 19a compiles `mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32` to `QMMA.SP.16864.F32.E4M3.E4M3 R4, R4, R16, R20, R0, 0x0` with opcode bytes `0x000000100404727a` and control code `0x004ff60000013414`.
* [OBS] 19a has 48 total instructions. Top mnemonics are 14 `NOP`, 12 `LDG.E.CONSTANT`, 4 `STG.E`, 4 `LDC.64`, and 4 `IMAD.WIDE.U32`.
* [OBS] 19a through 19f each have 48 total instructions.
* [OBS] Sparse non-scaled `kind::f8f6f4` uses the existing QMMA opcode low byte `0x7a`, matching dense QMMA from chapters 14 and 15, with a new `.SP` mnemonic modifier.
* [OBS] Sparse non-scaled shape is `.16864`, while dense non-scaled QMMA shape is `.16832` in chapters 14 and 15.
* [INF] Sparse non-scaled `kind::f8f6f4` doubles the SASS K field from 32 to 64 while staying in the QMMA family, because 19a emits `QMMA.SP.16864` and chapters 14 and 15 emit dense `QMMA.16832` for the same dtype family.
* [OBS] 19a uses 6 SASS operands: D base, A base, B base, C base, metadata register, and selector immediate.
* [OBS] 19a materializes metadata as `MOV R0, 0xaaaaaaaa`, and the sparse MMA consumes `R0` as the fifth operand.
* [OBS] 19e changes metadata to `0x55555555`; the SASS changes the metadata materialization to `MOV R0, 0x55555555`, while the `QMMA.SP` opcode bytes and control code remain identical to 19a.
* [OBS] 19f changes metadata to `0xffffffff`; the SASS changes the metadata materialization to `MOV R0, 0xffffffff`, while the `QMMA.SP` opcode bytes and control code remain identical to 19a.
* [INF] Metadata value is a runtime register operand, not an opcode or control-code field, because changing the metadata constant changes only the producer `MOV`, not the `QMMA.SP` instruction encoding.
* [OBS] 19b compiles E4M3 x E5M2 to `QMMA.SP.16864.F32.E4M3.E5M2` with the same opcode bytes as 19a and control code `0x004ff6000001b414`.
* [OBS] 19c compiles E3M2 x E2M3 to `QMMA.SP.16864.F32.E3M2.E2M3` with the same opcode bytes as 19a and control code `0x004ff60000257414`.
* [OBS] 19d compiles E2M1 x E2M1 to `QMMA.SP.16864.F32.E2M1.E2M1` with the same opcode bytes as 19a and control code `0x004ff6000029f414`.
* [INF] Sparse non-scaled dtype selection remains in the control code, not opcode bytes, because 19a through 19d keep opcode bytes `0x000000100404727a` while dtype suffixes and control codes change.
* [OBS] 19h compiles F16 accumulator E3M2 x E2M1 to `QMMA.SP.16864.F16.E3M2.E2M1 R12, R12, R16, R20, R0, 0x0` with opcode bytes `0x000000100c0c727a` and control code `0x004ff6000025d414`.
* [OBS] 19h has 40 total instructions, 8 fewer than the F32 accumulator sparse non-scaled probes, because F16 accumulator D and C fragments use 2 registers rather than 4.
* [OBS] Selector value `1` for the tested sparse `kind::f8f6f4` PTX form is rejected by ptxas with `Argument 5 of instruction 'mma': unexpected value '1', expected to be 0`.
* [OBS] 19i compiles sparse block-scaled `kind::mxf8f6f4` to `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8 R4, R4, R8, R20, R18, R0, URZ, 0x0` with opcode bytes `0x700012080404747a` and control code `0x004ff6000025fe14`.
* [OBS] 19i materializes scale value `0x7f` in `R0`, metadata `0xaaaaaaaa` in `R18`, and emits `R18` as the fifth MMA operand.
* [OBS] 19i through 19l each have 48 total instructions.
* [OBS] Sparse block-scaled SASS operand order is D base, A base, B base, C base, metadata register, scale register, `URZ`, selector immediate.
* [OBS] 19j compiles sparse block-scaled `kind::mxf4nvf4` 2X ue8m0 to `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8 R4, R4, R8, R20, R18, R0, URZ, 0x0` with opcode bytes `0x700012080404747f` and control code `0x004ff60000093e14`.
* [OBS] 19k compiles sparse block-scaled `kind::mxf4nvf4` 4X ue4m3 to `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` with the same opcode bytes as 19j and control code `0x004ff60000053e14`.
* [OBS] 19l compiles sparse block-scaled `kind::mxf4nvf4` 4X ue8m0 to `OMMA.SF.SP.168128.F32.E2M1.E2M1.E8.4X` with the same opcode bytes as 19j and control code `0x004ff60000013e14`.
* [INF] 19l partially constrains chapter 16's scale-mode gap: `scale_vec::4X` can combine with ue8m0 on the sparse `mxf4nvf4` path, but the dense `mxf4nvf4` 4X ue8m0 path remains untested.
* [INF] Sparse block-scaled `kind::mxf4nvf4` keeps chapter 16's OMMA low byte `0x7f`; sparse adds `.SP` and doubles the displayed K from `.16864` dense to `.168128` sparse.
* [INF] For sparse block-scaled `kind::mxf4nvf4`, scale mode remains encoded in control code, because 19j through 19l keep opcode bytes `0x700012080404747f` while scale suffixes and control codes change.
* [OBS] 19m emits 16 `QMMA.SP.16864.F32.E4M3.E4M3` instructions in a dependency chain.
* [OBS] 19m has 88 total instructions.
* [OBS] 19m first sparse chain instruction uses C=`RZ`, `.reuse` on B, and control code `0x084ff600000134ff`.
* [OBS] 19m middle sparse chain instructions use C=`R12`, `.reuse` on B, and control code `0x080ff6000001340c`.
* [OBS] 19m last sparse chain instruction uses C=`R12`, no `.reuse` on B, and control code `0x000fe2000001340c`.
* [INF] Sparse QMMA chain scheduling follows the chapter 14 QMMA chain pattern: B gets `.reuse` until the last instruction, D and C are colocated after the first instruction, and control code high bits track the dependency chain.

### Resolved hypotheses

* [RES] Sparse MMA SASS opcode question resolved for tested warp-level SM120 forms: non-scaled sparse `kind::f8f6f4` emits `QMMA.SP`, sparse `kind::mxf8f6f4` emits `QMMA.SF.SP`, and sparse `kind::mxf4nvf4` emits `OMMA.SF.SP`.
* [RES] Sparse metadata encoding question resolved at SASS operand level: metadata is an explicit register operand before the selector immediate.

### Open gaps

* [GAP] Runtime validity of metadata patterns `0xaaaaaaaa`, `0x55555555`, and `0xffffffff` is not measured because the local NVIDIA driver is unavailable.
* [GAP] Sparse QMMA and sparse OMMA latency are not measured. Chapter 19 establishes SASS forms, but a dedicated sparse latency run is still needed.
* [GAP] Exact bit-level meaning of the sparse `.SP` modifier inside QMMA and OMMA opcode/control fields is not fully decoded.
* [GAP] Selector semantics for all warp-level sparse families are not fully mapped. Selector `1` is rejected for the tested `kind::f8f6f4` form, but other shapes and legacy `mma.sp` forms were not exhaustively compiled.

---

## Kernel 20 Control flow

* [OBS] Chapter 20 establishes SM120 SASS loop-lowering behavior for 22 controlled variants covering constant loops, dynamic loops, unroll pragmas, nested scalar loops, nested HMMA loops, conditional bodies, `break`, `continue`, volatile stores, template instantiation, unique per-iteration stores, and identical repeated bodies.

### Variants and outcomes

* [OBS] 20a constant loop N=4 emits 40 instructions, no back-edge, 4 FFMA, and 4 FADD.
* [OBS] 20b constant loop N=16 emits 64 instructions, no back-edge, 16 FFMA, and 16 FADD.
* [OBS] 20c dynamic loop emits 192 instructions, 11 BRA-family instructions including the trap branch, 3 back-edges, 33 FFMA, and 33 FADD.
* [OBS] 20d constant loop N=16 with `#pragma unroll 1` emits 40 instructions and one backward `BRA.U UP0, 0xe0` at offset `0x0130`.
* [OBS] 20e nested scalar loop 4 x 2 emits 48 instructions, no back-edge, 8 FFMA, and 8 FADD.
* [OBS] 20f nested scalar loop 8 x 2 emits 64 instructions, no back-edge, 16 FFMA, and 16 FADD.
* [OBS] 20g nested HMMA loop 4 x 2 emits 56 instructions, no back-edge, and 8 `HMMA.16816.F32`.
* [OBS] 20h nested HMMA loop 8 x 2 emits 80 instructions, no back-edge, and 16 `HMMA.16816.F32`.
* [OBS] 20i dynamic loop with short `if` emits 80 instructions, 2 back-edges, predicated FADD, and no BSSY/BSYNC.
* [OBS] 20j dynamic loop with larger `if` emits 136 instructions, 1 back-edge, predicated BRA, predicated arithmetic, and no BSSY/BSYNC.
* [OBS] 20k dynamic loop with `break` emits 40 instructions, 1 back-edge, `BSSY.RECONVERGENT B0, 0x1c0` at offset `0x0120`, and `BSYNC.RECONVERGENT B0` at offset `0x01b0`.
* [OBS] 20l dynamic loop with `continue` emits 168 instructions, 2 back-edges, many predicated arithmetic instructions, and no BSSY/BSYNC.
* [OBS] 20m constant loop N=16 with explicit full unroll emits the same instruction count and arithmetic counts as 20b: 64 instructions, 16 FFMA, 16 FADD, and no back-edge.
* [OBS] 20n constant loop N=16 with `#pragma unroll 4` emits 48 instructions, 1 back-edge, and a 4-iteration unrolled body.
* [OBS] 20o dynamic loop with `#pragma unroll 4` emits 64 instructions, 2 back-edges, and a 4-wide loop/tail structure.
* [OBS] 20p dynamic loop with an accumulator dependency chain emits 96 instructions, 3 back-edges, and 33 FFMA.
* [OBS] 20q dynamic loop with four independent accumulators emits 208 instructions, 3 back-edges, and 132 FFMA.
* [OBS] 20r dynamic loop with volatile store emits 96 instructions, 1 back-edge, and `STG.E.STRONG.SYS` for the volatile global stores.
* [OBS] 20s template nested loop 8 x 2 emits one SASS function with 64 instructions, no back-edge, 16 FFMA, and 16 FADD.
* [OBS] 20t emits two SASS functions in one dump: `template_nested_kernel<4,2>` and `template_nested_kernel<8,2>`.
* [OBS] 20u nested 8 x 2 loop with unique per-iteration stores emits 88 instructions, no back-edge, 16 FFMA, 16 FADD, and 16 STG.E.
* [OBS] 20v nested 8 x 2 identical repeated body emits 48 instructions, no back-edge, and 16 FADD. No multiply-by-16 collapse is observed.

### Loop-lowering rules established

* [RES] Constant scalar loops with compile-time trip counts 4 and 16 fully unroll by default in the tested SM120 kernels.
* [RES] Nested constant loops with scalar bodies fully unroll by default for tested 4 x 2 and 8 x 2 shapes.
* [RES] Nested constant loops with HMMA bodies fully unroll by default for tested 4 x 2 and 8 x 2 shapes.
* [RES] `#pragma unroll 1` preserves a constant-trip loop as a SASS loop with a backward `BRA.U`.
* [RES] `#pragma unroll 4` preserves a loop with a 4-iteration unrolled body and a backward `BRA.U`.
* [OBS] Dynamic loops in 20c, 20p, and 20q do not lower to a single compact loop. They lower to multi-path unroll cascades with multiple back-edges.
* [INF] Dynamic-loop lowering in Chapter 20 matches the runtime-loop cascade pattern from Kernel 04: ptxas creates specialized paths for trip-count ranges and uses back-edges inside those paths.
* [RES] In the tested variants, a preserved SASS loop always has at least one back-edge branch. No loop without a back-edge was observed.

### Control-flow constructs

* [OBS] Ordinary preserved loops use `BRA.U` when the controlling predicate is uniform.
* [OBS] Short conditional loop bodies can lower to predicated arithmetic without BSSY/BSYNC.
* [OBS] Larger conditional loop bodies can lower to predicated BRA plus predicated arithmetic without BSSY/BSYNC.
* [OBS] `continue` can lower to predicated arithmetic and back-edges without BSSY/BSYNC.
* [OBS] `break` in 20k emits BSSY/BSYNC around the break-capable loop body.
* [INF] BSSY/BSYNC is not required for ordinary loops in the tested variants, but can be introduced by non-structured loop exit through `break`.

### Production audit implications

* [RES] The minimal 8 x 2 HMMA reproduction does not reproduce the earlier production-audit 2-MMA observation. 20h emits 16 static HMMAs.
* [INF] The prior FP4 attention 2-QMMA observation is therefore not explained by normal ptxas lowering of an 8 x 2 nested MMA loop. Remaining explanations are stale binary, different template specialization, source not matching the dump, dead-code elimination around unused MMA results, or a production-specific transformation not reproduced by the minimal kernel.
* [OBS] Template instantiations appear as separate SASS functions in a single dump, as shown by 20t.
* [INF] Production audits must count instructions per SASS function, not per source file, because a source file can contribute multiple template-specialized kernel bodies.
* [OBS] Unique per-iteration stores in 20u survive as 16 STG.E instructions.
* [OBS] Identical repeated scalar bodies in 20v remain 16 FADD instructions in the tested source. No repeated-body collapse into `x * 16` is observed.

### Resolved hypotheses

* [RES] HYP-20-1 constant trip count at compile time fully unrolls for the tested small and moderate constant loops.
* [RES] HYP-20-2 `#pragma unroll 1` forces a real loop with a back-edge for the tested N=16 loop.
* [RES] HYP-20-3 dynamic trip count produces back-edges, but the result is often a multi-path cascade rather than one simple loop.
* [RES] HYP-20-4 nested constant loops are not partially unrolled in the tested 4 x 2 and 8 x 2 scalar/HMMA cases. They fully unroll.
* [RES] HYP-20-5 BSSY/BSYNC is not used for ordinary loops in the tested variants. It is used for the tested `break` loop.
* [RES] HYP-20-6 no SASS-level loop without a back-edge was observed in the tested variants.

### Open gaps

* [GAP] Runtime numeric validation of 20a through 20v is blocked by the unavailable NVIDIA driver.
* [GAP] Exact ptxas thresholds and heuristics for the dynamic-loop cascade are not fully decoded.
* [GAP] Original production FP4 attention 2-QMMA case remains unresolved end-to-end because the original source/binary pair was not rebuilt and compared in this chapter.
* [RES] First-pass warp-divergent reconvergence patterns are covered by Chapter 21. Remaining gaps are threshold heuristics, `WARPSYNC.ALL` semantics, non-full masks after genuine divergence, and deeper BSSY/BSYNC stack behavior.

### New diagnostic rules

* [OBS] A back-edge is a `BRA` or `BRA.U` whose target offset is lower than the branch instruction offset.
* [OBS] A dump with no useful-body back-edge and repeated arithmetic/MMA sites is a fully unrolled loop in the tested variants.
* [OBS] The terminal self-trap `BRA` after `EXIT` must not be counted as a loop back-edge because its target is the same offset.
* [OBS] Multiple `Function :` sections in one SASS dump can indicate template specializations or multiple kernels, as shown by 20t.

---

## Kernel 21 Divergence and reconvergence

* [OBS] Chapter 21 establishes SM120 lowering behavior for 20 controlled variants covering lane-dependent short if, uniform branch, divergent if, if/else, nested divergence, break, continue, early return, barrier after divergence, ballot, select-vs-branch, body-size contrast, divergent memory, guarded HMMA, lane-dependent trip count, bounds-check epilogue, masked stores, divergent local call, cold trap path, and vote-converged branch.

### Variants and outcomes

* [OBS] 21a simple lane-dependent short `if` emits 32 instructions, a predicated `FADD`, and no BSSY/BSYNC.
* [OBS] 21b uniform branch emits 32 instructions, `UFSEL`, and no BSSY/BSYNC.
* [OBS] 21c lane-divergent `if` emits 48 instructions, `BSSY.RECONVERGENT B0, 0x1e0`, a predicated forward `BRA`, and `BSYNC.RECONVERGENT B0`.
* [OBS] 21d lane-divergent `if/else` emits 32 instructions, predicated arithmetic for both paths, and no BSSY/BSYNC.
* [OBS] 21e nested divergence emits 40 instructions, one BSSY/BSYNC pair, and forward branches inside the reconvergence region.
* [OBS] 21f lane-dependent `break` emits 40 instructions, one BSSY/BSYNC pair, and one useful back-edge.
* [OBS] 21g lane-dependent `continue` emits 40 instructions, one useful `BRA.U` back-edge, and no BSSY/BSYNC.
* [OBS] 21h lane-dependent early return emits 32 instructions, an additional predicated `EXIT`, and no BSSY/BSYNC.
* [OBS] 21i divergent arithmetic followed by `__syncthreads()` emits 40 instructions, `FSEL`, `STS`, `BAR.SYNC.DEFER_BLOCKING`, `LDS`, and no BSSY/BSYNC.
* [OBS] 21j `__ballot_sync` around a lane-dependent predicate emits 32 instructions, `VOTE.ANY R5, PT, P0`, and no BSSY/BSYNC.
* [OBS] 21k select-vs-branch-vs-mask emits 40 instructions, `FSEL` instructions, and no BSSY/BSYNC.
* [OBS] 21l short body followed by long divergent body emits 48 instructions and one BSSY/BSYNC pair around the long divergent body.
* [OBS] 21m divergent memory paths emits 48 instructions, a predicated forward branch, separate path exits, and no BSSY/BSYNC.
* [OBS] 21n guarded HMMA emits 64 instructions, many predicated loads, `@P0 WARPSYNC.ALL`, `@P0 HMMA.16816.F32`, predicated FADD reductions, and no BSSY/BSYNC.
* [OBS] 21o lane-dependent trip count emits 40 instructions, one BSSY/BSYNC pair, and one useful back-edge.
* [OBS] 21p bounds-check epilogue emits 32 instructions, an additional predicated `EXIT`, and no BSSY/BSYNC.
* [OBS] 21q masked store tail emits 40 instructions, predicated `LDC.64`, predicated `IMAD.WIDE`, predicated `STG.E`, an additional predicated `EXIT`, and no BSSY/BSYNC.
* [OBS] 21r divergent noinline call emits 48 instructions, `BSSY.RECONVERGENT B0, 0x170`, `CALL.REL.NOINC 0x1b0`, `BSYNC.RECONVERGENT B0`, a local callee loop with `BRA.U`, and `RET.REL.NODEC R2 0x0`.
* [OBS] 21s cold trap/error path emits 32 instructions, a predicated forward branch, `BPT.TRAP 0x1`, and no BSSY/BSYNC.
* [OBS] 21t vote-converged branch emits 32 instructions, `VOTE.ANY P0, P0`, predicated branch, `SHFL.IDX`, and no BSSY/BSYNC.

### Divergence and reconvergence rules established

* [OBS] Across the 20 dumps, BSSY/BSYNC appears in exactly six variants: 21c, 21e, 21f, 21l, 21o, and 21r.
* [OBS] Lane-dependent control appears without BSSY/BSYNC in 21a, 21d, 21h, 21i, 21j, 21k, 21m, 21n, 21p, 21q, 21s, and 21t.
* [RES] Lane divergence alone does not force visible BSSY/BSYNC in every tested SM120 SASS form.
* [OBS] ptxas uses predicated arithmetic in 21a and 21d to avoid explicit reconvergence scopes.
* [OBS] ptxas uses `FSEL` in 21i and 21k, and `UFSEL` in 21b, to represent selected values without explicit reconvergence scopes.
* [OBS] ptxas uses predicated `EXIT` for lane-dependent early return and epilogue/tail checks in 21h, 21p, and 21q.
* [OBS] ptxas uses predicated store setup in 21q for masked writeback.
* [OBS] ptxas uses BSSY/BSYNC for the tested branch-kept divergent arithmetic regions 21c, 21e, and 21l.
* [OBS] ptxas uses BSSY/BSYNC for lane-dependent `break` in 21f and lane-dependent trip-count loop in 21o.
* [OBS] ptxas uses BSSY/BSYNC around the divergent local call in 21r.
* [INF] In the tested variants, BSSY/BSYNC is associated with divergent regions that ptxas keeps as explicit control regions or local calls, not with every lane-derived predicate.

### Warp-level and tensor-core implications

* [OBS] 21j confirms the register-output ballot form `VOTE.ANY Rdst, PT, Psrc` already seen in Kernel 09.
* [OBS] 21t uses predicate-output `VOTE.ANY P0, P0` before a branch guarding `SHFL.IDX`, and no BSSY/BSYNC appears.
* [OBS] 21n introduces `WARPSYNC.ALL` in the project corpus: `@P0 WARPSYNC.ALL` appears before `@P0 HMMA.16816.F32`.
* [OBS] 21n shows a lane-predicated HMMA can be emitted directly as `@P0 HMMA.16816.F32` in the tested source.
* [GAP] `WARPSYNC.ALL` semantics and encoding are not decoded.
* [GAP] Runtime behavior of predicated HMMA under partial-lane participation is not validated.

### Resolved hypotheses

* [RES] HYP-21-1 lane divergence always forces BSSY/BSYNC. Rejected by 21a, 21d, 21h, 21i, 21j, 21k, 21m, 21n, 21p, 21q, 21s, and 21t.
* [RES] HYP-21-2 short divergent bodies can be predicated without BSSY/BSYNC. Confirmed by 21a and 21d.
* [RES] HYP-21-3 lane-dependent `break` uses BSSY/BSYNC in the tested loop. Confirmed by 21f.
* [RES] HYP-21-4 lane-dependent `continue` necessarily uses BSSY/BSYNC. Rejected by 21g.
* [RES] HYP-21-5 guarded HMMA must be surrounded by BSSY/BSYNC. Rejected by 21n; the observed form is predicated setup, `@P0 WARPSYNC.ALL`, and `@P0 HMMA`.

### Open gaps

* [GAP] Runtime numeric validation of 21a through 21t is blocked by the unavailable NVIDIA driver.
* [GAP] Exact ptxas threshold for predication/select versus explicit BSSY branch region remains unresolved.
* [GAP] `WARPSYNC.ALL` control-code bits, predicate behavior, and interaction with HMMA/LDSM/SHFL remain unresolved.
* [GAP] Non-full warp masks after genuine divergence remain under-tested.
* [GAP] Exact BSSY/BSYNC stack semantics and barrier register allocation beyond B0 remain unresolved.

---

## Kernel 22 stmatrix / matrix store

* [OBS] Chapter 22 establishes SM120 matrix-store behavior for 12 controlled variants covering m8n8 b16 STSM x1/x2/x4, transposed STSM x1/x2/x4, scalar STS fallback, layout readback shapes, barrier/no-barrier dependency shapes, and HMMA-adjacent STSM.

### Variants and outcomes

* [OBS] 22a `stmatrix.sync.aligned.x1.m8n8.shared.b16` emits 32 instructions and `STSM.16.M88 [R7], R2`.
* [OBS] 22b `stmatrix.sync.aligned.x2.m8n8.shared.b16` emits 32 instructions and `STSM.16.M88.2 [R0], R6`.
* [OBS] 22c `stmatrix.sync.aligned.x4.m8n8.shared.b16` emits 32 instructions and `STSM.16.M88.4 [R0], R8`.
* [OBS] 22d `stmatrix.sync.aligned.x1.trans.m8n8.shared.b16` emits 32 instructions and `STSM.16.MT88 [R7], R2`.
* [OBS] 22e `stmatrix.sync.aligned.x2.trans.m8n8.shared.b16` emits 32 instructions and `STSM.16.MT88.2 [R0], R6`.
* [OBS] 22f `stmatrix.sync.aligned.x4.trans.m8n8.shared.b16` emits 32 instructions and `STSM.16.MT88.4 [R0], R8`.
* [OBS] 22g scalar shared-store fallback emits 32 instructions and `STS.128 [R0], R8`.
* [OBS] 22h x4 layout readback emits 40 instructions, `STSM.16.M88.4`, `BAR.SYNC.DEFER_BLOCKING 0x0`, and `LDS.128`.
* [OBS] 22i x4 transposed layout readback emits 40 instructions, `STSM.16.MT88.4`, `BAR.SYNC.DEFER_BLOCKING 0x0`, and `LDS.128`.
* [OBS] 22j barrier visibility probe emits 32 instructions, `STSM.16.M88.4`, `BAR.SYNC.DEFER_BLOCKING 0x0`, and `LDS R7, [R6+UR4]`.
* [OBS] 22k no-barrier same-thread probe emits 32 instructions with adjacent `STSM.16.M88.4 [R0], R8` and `LDS R7, [R0]`.
* [OBS] 22l HMMA-adjacent STSM emits 40 instructions, `HMMA.16816.F32 R8, R4, R8, RZ`, and later `STSM.16.M88.4 [R0], R8`.

### STSM family rules established

* [RES] `stmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16` lowers to the SM120 SASS family `STSM.16.M88[.2|.4]`.
* [RES] `stmatrix.sync.aligned.x{1,2,4}.trans.m8n8.shared.b16` lowers to the SM120 SASS family `STSM.16.MT88[.2|.4]`.
* [OBS] The tested STSM instruction low opcode byte is `0x44`.
* [OBS] Normal x1 and transposed x1 share the visible instruction word `0x0000000207007844`, while their printed second line differs: normal `0x010fe20000000000`, transposed `0x010fe20000004000`.
* [OBS] Normal x4 and transposed x4 share the visible instruction word `0x0000000800007844`, while their printed second line differs: normal `0x010fe20000000200`, transposed `0x010fe20000004200`.
* [INF] For the tested pairs, the transposed modifier is represented by a `0x4000` delta in the printed second line.
* [OBS] The scalar fallback variant emits `STS.128`, not `STSM`.
* [INF] Scalar shared stores may vectorize into `STS.128`, but they remain a different store family from matrix-store STSM.

### Unsupported forms and runtime gaps

* [OBS] The opt-in b8 probe attempts `stmatrix.sync.aligned.m16n8.x1.trans.shared.b8`, `.x2`, and `.x4`.
* [OBS] ptxas rejects the opt-in b8 probe for `sm_120` with `Feature '.m16n8' not supported on .target 'sm_120'`.
* [OBS] ptxas rejects the opt-in b8 probe for `sm_120` with `Feature 'stmatrix.b8' not supported on .target 'sm_120'`.
* [RES] The tested SM120 target does not support the tested m16n8 b8 STSM PTX forms.
* [OBS] Runtime smoke probes for normal and transposed x4 STSM run on the host GPU.
* [OBS] 22h normal x4 first output words are `00001000 00001001 00001002 00001003 00001004 00001005 00001006 00001007`.
* [OBS] 22i transposed x4 first output words are `10041000 100c1008 10141010 101c1018 00000000 00000000 00000000 00000000`.
* [OBS] 22j barrier visibility first output words are `00001004 00001008 0000100c 00001010 00001014 00001018 0000101c 00001020`.
* [OBS] 22k no-barrier same-thread first output words are `00001000 00001004 00001008 0000100c 00001010 00001014 00001018 0000101c`.
* [GAP] Full STSM lane-to-shared layout semantics remain undecoded from the runtime outputs.
* [GAP] STSM latency is not measured.
* [GAP] STSM scoreboard and control-code fields are not decoded beyond the printed SASS/control annotations.
* [GAP] A full production accumulator-to-shared epilogue remains unvalidated.

---

## Kernel 23 FP4 / FP6 fragment layout

* [OBS] Chapter 23 compiles 22 controlled probes, 23a through 23v, for FP4 / FP6 fragment-layout use cases on SM120.
* [OBS] The accepted probes require `nvcc -arch=compute_120a -code=sm_120a`, matching the `kind::f8f6f4`, `kind::mxf8f6f4`, and `kind::mxf4nvf4` requirements established in Chapters 14, 16, and 19.

### Variants and outcomes

* [OBS] 23a E2M1 x E2M1 emits 48 instructions and `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23b E3M2 x E3M2 emits 48 instructions and `QMMA.16832.F32.E3M2.E3M2`.
* [OBS] 23c E2M3 x E2M3 emits 48 instructions and `QMMA.16832.F32.E2M3.E2M3`.
* [OBS] 23d E3M2 x E2M3 emits 48 instructions and `QMMA.16832.F32.E3M2.E2M3`.
* [OBS] 23e E2M3 x E3M2 emits 48 instructions and `QMMA.16832.F32.E2M3.E3M2`.
* [OBS] 23f, 23g, and 23h lane-pattern probes emit the expected dense `QMMA.16832` dtype mnemonics for E2M1, E3M2, and E2M3.
* [OBS] 23i scale-factor interaction emits `QMMA.SF.16832.F32.E4M3.E4M3.E8`.
* [OBS] 23j LDSM-to-QMMA emits `LDSM.16.M88.2`, `LDSM.16.M88.4`, and later `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23k direct-register path emits `QMMA.16832.F32.E2M1.E2M1` without LDSM.
* [OBS] 23l runtime-decode probe emits `QMMA.16832.F32.E2M1.E2M1` and runs on the host GPU with first output words `c1d80000 c1d80000 c3220000 c3220000`.
* [OBS] 23m invalid-format probe attempts `kind::f8f6f4` with E2M1 x BF16 and is rejected by ptxas with `Unexpected instruction types specified for 'mma'`.
* [OBS] 23n cross-reference probe emits `QMMA.16832.F32.E3M2.E2M3`, matching the Chapter 15 mixed-FP6 family.
* [OBS] 23o unsigned/signed interpretation probe emits `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23p scale-vector probe emits `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
* [OBS] 23q metadata-independence probe emits `QMMA.SF.SP.16864.F32.E3M2.E2M1.E8`.
* [OBS] 23r operand-order probe emits `QMMA.16832.F32.E3M2.E2M3`.
* [OBS] 23s K-tile boundary probe emits 56 instructions and `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23t register-pair boundary probe emits `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23u zero / NaN / Inf bit-pattern probe emits `QMMA.16832.F32.E2M1.E2M1`.
* [OBS] 23v shared-memory alignment probe emits offset `LDSM.16.M88.2`, offset `LDSM.16.M88.4`, and later `QMMA.16832.F32.E2M1.E2M1`.

### Fragment-layout rules established

* [RES] The tested dense FP4/FP6 source forms stay in the `QMMA.16832` family; the A and B dtypes are explicit in the SASS mnemonic.
* [RES] Dense E2M1, E3M2, E2M3, and mixed E3M2/E2M3 forms share the tested low opcode byte `0x7a`.
* [RES] LDSM-fed FP4 QMMA is structurally visible as an LDSM stage followed by QMMA, while direct-register FP4 QMMA has no LDSM stage.
* [RES] Scale factors and sparse metadata remain explicit SASS operands in the tested `QMMA.SF`, `OMMA.SF`, and `QMMA.SF.SP` probes.
* [INF] A production audit model should treat fragment data, scale factors, sparse metadata, and shared-memory fragment loads as separate channels.

### Runtime gaps

* [OBS] All accepted Chapter 23 runtime smoke probes execute on the host GPU after fixing the LDSM B-fragment shared-memory stride to 16 bytes.
* [GAP] FP4/FP6 full lane-to-value mapping remains undecoded from the runtime outputs.
* [GAP] E3M2 and E2M3 exact bit packing within each 32-bit source register remains unresolved without runtime decode.
* [GAP] Special-value interpretation for zero, sign-bit, NaN-like, and Inf-like patterns remains unresolved without runtime decode.
* [GAP] LDSM-fed QMMA correctness for packed FP4/FP6 values remains structural only until outputs are checked on hardware.

---

## Kernel 24 Production mini-GEMM audit

* [OBS] Chapter 24 compiles 30 controlled probes, 24a through 24ad, for production-like mini-GEMM audit structure on SM120.
* [OBS] The probes require `nvcc -arch=compute_120a -code=sm_120a`, matching the arch-conditional tensor-core forms established in Chapters 14, 16, 19, and 23.
* [OBS] Runtime smoke execution succeeds on the host RTX 5070 Ti for all 30 variants.
* [GAP] The chapter validates structural SASS signatures and launchability, not full numeric GEMM correctness.

### Variants and outcomes

* [OBS] 24a emits `HMMA.16816.F32` and `STG.E.128`.
* [OBS] 24b emits `QMMA.16832.F32.E4M3.E4M3` and `STG.E.128`.
* [OBS] 24c emits `QMMA.16832.F32.E2M1.E2M1` and `STG.E.128`.
* [OBS] 24d emits `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` and `STG.E.128`.
* [OBS] 24e, 24f, 24t, and 24x emit production-like async staging: `LDGSTS`, `LDGDEPBAR`, `DEPBAR.LE`, `BAR`, `LDSM`, `HMMA`, and `STG`.
* [OBS] 24g and 24q isolate LDSM order and shared-memory alignment before `HMMA`.
* [OBS] 24h emits four chained `QMMA.16832.F32.E4M3.E4M3` instructions.
* [OBS] 24i is a store-only epilogue baseline and emits `STG.E.128` with no MMA.
* [OBS] 24j and 24k emit `STSM.16.M88.4` epilogue structure after QMMA.
* [OBS] 24l and 24y cover bounded/vectorized `STG.E.128` epilogues after `HMMA`.
* [OBS] 24m and 24n cover preserved and unrolled tile-loop HMMA structure.
* [OBS] 24o emits predicated guarded HMMA plus `WARPSYNC.ALL`.
* [OBS] 24p increases instruction count to 112 useful instructions while retaining `HMMA` and `STG`.
* [OBS] 24r emits `QMMA.SP.16864.F32.E4M3.E4M3`.
* [OBS] 24s emits `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X`.
* [OBS] 24u emits warp-level `HMMA` and no `WGMMA` or `TCGEN` mnemonics.
* [OBS] 24v and 24w expose uniform-register and descriptor/address arithmetic paths feeding tensor-core work.
* [OBS] 24z emits `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU`, separating split-K style reduction from normal `STG.E.128`.
* [OBS] 24aa loads scale operands with `LDG.E.CONSTANT` before `OMMA.SF...UE4M3.4X`.
* [OBS] 24ab loads sparse metadata with `LDG.E.CONSTANT` before `QMMA.SP`.
* [OBS] 24ac uses nontrivial stride parameters before `HMMA`.
* [OBS] 24ad emits `BSSY.RECONVERGENT`, `BSYNC.RECONVERGENT`, and `BPT.TRAP` around the cold path.

### Production audit rules established

* [RES] A first-pass SM120 mini-GEMM audit should segment the dump into global-to-shared copy, async dependency wait, shared matrix load, tensor-core compute, epilogue, optional reduction, and cold/error paths.
* [RES] Scale values and sparse metadata should be tracked as separate dependency channels from fragment data; Chapter 24 shows both channels feeding their MMA operands through visible loads.
* [RES] Split-K or multi-CTA accumulation can be recognized separately from vectorized stores by looking for `REDG.E.ADD.F32...`.
* [RES] The tested production-like SM120 path remains warp-level: `HMMA`, `QMMA`, `OMMA`, sparse modifiers, `LDSM`, `STSM`, `LDGSTS`, `DEPBAR`, and `STG`; no `WGMMA` or `TCGEN` appears.
* [GAP] STSM lane-to-shared layout and accumulator storeback semantics remain Chapter 25 scope.

---

## Kernel 25 STSM epilogue layout and storeback semantics

* [OBS] Chapter 25 compiles 25 accepted executable SASS probes and one b8 compatibility probe, 25a through 25z, for STSM epilogue/storeback behavior on SM120/SM120a.
* [OBS] The accepted probes require `nvcc -arch=compute_120a -code=sm_120a` and runtime smoke execution succeeds on the host RTX 5070 Ti.
* [OBS] The 25q compatibility probe compiles with plain `sm_120` and captures ptxas rejection for `.m16n8` and `stmatrix.b8`; the same source compiles for `sm_120a` and emits `STSM.8.MT168`, `STSM.8.MT168.2`, and `STSM.8.MT168.4`.
* [GAP] Full lane-to-value semantic decode remains open over the captured runtime output words.

### Variants and outcomes

* [OBS] 25a, 25b, and 25c emit `STSM.16.M88`, `STSM.16.M88.2`, and `STSM.16.M88.4`.
* [OBS] 25d, 25e, and 25f emit `STSM.16.MT88`, `STSM.16.MT88.2`, and `STSM.16.MT88.4`.
* [OBS] 25g emits `HMMA.16816.F32` before `STSM.16.M88.4`.
* [OBS] 25h emits `QMMA.16832.F32.E2M1.E2M1` before `STSM.16.M88.4`.
* [OBS] 25i emits HMMA, STSM, `BAR.SYNC.DEFER_BLOCKING`, `LDS.128`, and `STG.E`.
* [OBS] 25j emits `STS.128` rather than STSM, preserving the scalar/vector shared-store fallback distinction.
* [OBS] 25k covers predicated tail storeback after STSM.
* [OBS] 25l emits offset `STSM.16.M88.4 [R+0x20]` and offset `LDS.128`.
* [OBS] 25m emits the normal barrier-visible sequence: STSM, `BAR.SYNC.DEFER_BLOCKING`, `LDS.128`.
* [OBS] 25n emits adjacent STSM and `LDS` without `BAR.SYNC`, useful only as a same-thread/no-barrier contrast.
* [OBS] 25o emits two `STSM.16.M88.4` stores and two `LDS.128` reloads for split accumulator storeback.
* [OBS] 25p and 25z emit two `LDS.128` reloads and eight global stores for runtime layout decode tables.
* [OBS] 25s emits `F2F.F16.F32` conversions before STSM.
* [OBS] 25t emits `F2F.BF16.F32` conversions before STSM.
* [OBS] 25u emits `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` before STSM.
* [OBS] 25v emits `QMMA.SP.16864.F32.E4M3.E4M3` before STSM.
* [OBS] 25w emits noncontiguous global stores at 8-byte-spaced offsets.
* [OBS] 25x keeps `STSM.16.M88.4` under strided shared addressing.
* [OBS] 25y grows to 232 useful instructions under register-pressure arithmetic and still preserves STSM, barrier, shared reload, and global storeback.

### Epilogue rules established

* [RES] Cross-thread STSM storeback should be modeled as `STSM -> BAR -> LDS -> STG` in first-pass production audits.
* [RES] Same-thread no-barrier STSM readback can compile, but Chapter 25 does not prove cross-thread visibility is safe without a barrier.
* [RES] STS fallback is distinct from STSM: the tested scalar fallback emits `STS.128`.
* [RES] Accumulator narrowing before STSM is visible as `F2F.F16.F32` or `F2F.BF16.F32`, separate from the STSM opcode itself.
* [RES] HMMA, QMMA, OMMA, and sparse QMMA accumulator paths can feed visible STSM epilogues in the tested SM120a probes.
* [RES] b8 STSM support is target-qualified: rejected for tested plain `sm_120`, accepted for tested `sm_120a` as `STSM.8.MT168`.

---

# Additions to the Cross chapter summary section

## Tensor-core cross-chapter summary

### Pipelines observed so far

| Pipeline / role | SASS families observed | Evidence |
|---|---|---|
| Tensor core MMA | [OBS] `HMMA`, `QMMA`, `QMMA.SF`, `OMMA.SF`, `QMMA.SP`, `QMMA.SF.SP`, `OMMA.SF.SP` | chapters 13, 14, 16, 19, 23, 24 |
| Matrix load | [OBS] `LDSM.16.M[T]88[.N]` | chapters 17, 24 |
| Matrix store | [OBS] `STSM.16.M[T]88[.2|.4]`, `STSM.8.MT168` on tested `sm_120a` b8 form | chapters 22, 24, 25 |
| Async global-to-shared copy | [OBS] `LDGSTS`, `LDGDEPBAR`, `DEPBAR.LE SB0, N` | chapters 18, 24 |
| Global epilogue/reduction | [OBS] `STG.E.128`, `REDG.E.ADD.F32...` | chapter 24 |

### Architectural invariants

* [OBS] Dense HMMA, dense QMMA, dense OMMA, and tested sparse QMMA/OMMA forms are warp-level instructions on SM120. The SASS contains one instruction executed by the warp, with fragment operands distributed across the 32 participating lanes.
* [OBS] No `wgmma.mma_async` or `tcgen05.mma` appears in the SM120 dumps studied through chapter 24.
* [OBS] Tensor-core family selection is visible in the mnemonic and low opcode byte: `HMMA` low byte `0x3c`, `QMMA` low byte `0x7a`, and `OMMA` low byte `0x7f`.
* [INF] Sparse tensor-core forms are modifiers on the dense QMMA/OMMA families for the tested SM120 forms, because `QMMA.SP` and `QMMA.SF.SP` keep low byte `0x7a`, while `OMMA.SF.SP` keeps low byte `0x7f`.
* [GAP] Non-TN MMA layouts are not systematically tested across all SM120 MMA families.
* [RES] Matrix-store behavior for tested m8n8 b16 forms is covered by Chapter 22.

### Canonical tensor-core tile pattern

```text
LDGSTS tile[N+1]                     [OBS] chapters 18, 24, optional for pipelined kernels
LDGDEPBAR / DEPBAR.LE SB0, N         [OBS] chapters 18, 24
BAR.SYNC                             [OBS] chapters 18, 24
LDSM B fragment                      [OBS] chapters 17, 18, 24
LDSM A fragment                      [OBS] chapters 17, 18, 24
HMMA / QMMA / OMMA / sparse variant  [OBS] chapters 13, 14, 16, 19, 24
accumulator chain via D == C         [OBS] chapters 13, 14, 16, 19, 24
STSM optional shared epilogue        [OBS] chapters 22, 24
STG / REDG epilogue                  [OBS] chapter 24
```

* [OBS] Chapters 17, 18, and 24 show ptxas emits LDSM B before LDSM A in tested tensor-core tiles.
* [OBS] Chapters 13, 14, 16, and 19 show chained MMA forms colocate D and C registers after the first MMA in the chain.
* [OBS] Chapter 22 adds the tested matrix-store side through `STSM.16.M[T]88[.2|.4]`.
* [OBS] Chapter 24 adds a first-pass full production-like pipeline fixture with async copy, LDSM, MMA, optional STSM, STG, and reduction.
* [OBS] Chapter 25 adds isolated STSM epilogue/storeback probes, including narrowing conversions, shared reload, STG storeback, and fallback comparison.
* [GAP] Full lane-to-value semantic decode for STSM remains open over the captured runtime output words.

### Arithmetic operator compilation rules

| Source / PTX family | SASS strategy |
|---|---|
| [OBS] `mma.sync.aligned.m16n8k16...` | [OBS] `HMMA.16816.<acc>[.<input>]` |
| [OBS] `mma.sync.aligned.kind::f8f6f4.m16n8k32...` | [OBS] `QMMA.16832.<acc>.<A>.<B>` |
| [OBS] `mma.sync.aligned.kind::mxf8f6f4.block_scale...` | [OBS] `QMMA.SF.16832.<acc>.<A>.<B>.<scale>` |
| [OBS] `mma.sync.aligned.kind::mxf4nvf4.block_scale...` | [OBS] `OMMA.SF.16864.<acc>.<A>.<B>.<scale>[.<scale_vec>]` |
| [OBS] `mma.sp::ordered_metadata...kind::f8f6f4...` | [OBS] `QMMA.SP.16864.<acc>.<A>.<B>` |
| [OBS] `mma.sp::ordered_metadata...kind::mxf8f6f4.block_scale...` | [OBS] `QMMA.SF.SP.16864.<acc>.<A>.<B>.<scale>` |
| [OBS] `mma.sp::ordered_metadata...kind::mxf4nvf4.block_scale...` | [OBS] `OMMA.SF.SP.168128.<acc>.<A>.<B>.<scale>[.<scale_vec>]` |
| [OBS] `ldmatrix.sync.aligned.x{1,2,4}.shared.b16` | [OBS] `LDSM.16.M88[.N]` |
| [OBS] `stmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16` | [OBS] `STSM.16.M88[.2|.4]` |
| [OBS] `stmatrix.sync.aligned.x{1,2,4}.trans.m8n8.shared.b16` | [OBS] `STSM.16.MT88[.2|.4]` |

### Cost rules established so far

* [RES] HMMA.16816.F32 serial dependency-chain latency is approximately 35 cycles per MMA on SM120.
* [RES] QMMA.16832.F32 serial dependency-chain latency is approximately 35 cycles per MMA on SM120.
* [RES] OMMA.SF.16864.F32 serial dependency-chain latency is approximately 29 cycles per MMA on SM120.
* [RES] LDSM serial latency is approximately 33 cycles on SM120.
* [GAP] STSM latency is not measured.
* [GAP] Sparse QMMA/OMMA serial latency is not measured.
* [GAP] Scale factor value effects are not measured beyond unity-scale probes.
* [GAP] Sparse metadata load overhead at runtime is not measured.

### Compiler artifacts and diagnostic signals

* [OBS] `HMMA` in a dump identifies FP16/BF16 tensor-core MMA on SM120.
* [OBS] `QMMA.16832` identifies dense `kind::f8f6f4` non-scaled MMA on SM120.
* [OBS] `.SF` on QMMA or OMMA identifies block scaling on SM120.
* [OBS] `.SP` on QMMA or OMMA identifies sparse `mma.sp::ordered_metadata` on SM120.
* [OBS] For tested Chapter 23 forms, FP4/FP6 A and B dtypes are explicit in the `QMMA`, `QMMA.SF`, `OMMA.SF`, and `QMMA.SF.SP` mnemonics.
* [OBS] `LDSM` identifies matrix-fragment loads from shared memory.
* [OBS] `STSM` identifies matrix-fragment stores to shared memory.
* [OBS] `LDGSTS` plus `LDGDEPBAR` plus `DEPBAR.LE SB0, N` identifies cp.async staging.

## Opcode operand semantics

* [OBS] HMMA/QMMA/OMMA dense SASS operand order is D base, A base, B base, C base.
* [OBS] Dense block-scaled QMMA/OMMA adds scale operands after C: SFA, SFB, and `URZ`.
* [OBS] Sparse non-scaled QMMA adds metadata register and selector immediate after C.
* [OBS] Sparse block-scaled QMMA/OMMA operand order after C is metadata register, scale register, `URZ`, selector immediate.
* [OBS] LDSM operand order is destination register base and shared-memory address operand.
* [OBS] STSM operand order is shared-memory address operand followed by source register base for the tested m8n8 b16 forms.
* [GAP] Full lane-to-shared layout semantics for STSM are not runtime-validated.

## Global diagnostic workflow for tensor-core dumps

* [OBS] Grep for `HMMA`, `QMMA`, `OMMA`, `QMMA.SP`, `QMMA.SF.SP`, and `OMMA.SF.SP` to locate tensor-core compute.
* [OBS] Grep for `LDSM` to locate shared-memory fragment loading.
* [OBS] Grep for `STSM` to locate shared-memory fragment stores.
* [OBS] Grep for `LDGSTS`, `LDGDEPBAR`, and `DEPBAR.LE` to locate cp.async staging.
* [OBS] Read shape and dtype directly from the MMA mnemonic for observed SM120 tensor-core families.
* [OBS] Check accumulator chaining by looking for D and C register colocation across consecutive MMAs.
* [OBS] Check block scaling by looking for `.SF` and post-C scale operands.
* [OBS] Check sparse MMA by looking for `.SP`, metadata register operand, and selector immediate.
* [OBS] For FP4/FP6 layout audits, distinguish LDSM-fed fragments from direct-register fragments before making lane-layout claims.
* [RES] Grep for `STSM` is now a valid first-pass matrix-store locator for tested m8n8 b16 SM120 SASS.

## Cross chapter summary

### Pipelines observed so far

| Pipeline | Instructions observed |
|---|---|
| FMA | FFMA, FFMA.SAT, FFMA.RM, FADD, FADD.FTZ, FMUL, FMUL.FTZ, IMAD, IMAD.U32, IMAD.WIDE, IMAD.WIDE.U32, IMAD.HI.U32, IMAD.SHL.U32, HFMA2 |
| ALU | ISETP, ISETP.GE.U64.AND, ISETP.NE.S64.AND, FSETP, FSETP.GEU, FSETP.NEU, FSETP.GTU, FSETP.GEU.FTZ, FSETP.GTU.FTZ, FSETP.NEU.FTZ, MOV, MOV.64, LEA, LEA.HI, LOP3, FSEL, SHF, SHF.L.W.U32.HI, SHF.R.U32.HI, SHF.R.S32.HI, SEL, SEL.64, IABS, IADD, IADD.X, IADD3, IADD3.X, IADD.64, R2P, R2UR |
| LSU | LDG, LDG.E.64, LDG.E.128, LDG.E.ENL2.256, LDG.E.CONSTANT, STG, STG.E.64, STG.E.128, STG.E.ENL2.256, LDS, STS, STL, STL.128, LDL, LDL.LU, SHFL.BFLY, SHFL.IDX, SHFL.UP, SHFL.DOWN, MATCH.ANY, MATCH.ALL |
| ADU | LDC, LDC.64, S2R, BAR.SYNC |
| DCC | LDCU, LDCU.64 |
| UNIFORM | S2UR, UMOV, UMOV.64, ULEA, UIADD3, UISETP, UISETP.NE.U32.AND, UPLOP3, ULOP3 |
| XU | MUFU.RCP, MUFU.LG2, MUFU.EX2, MUFU.RSQ, I2F, I2F.F64.S64, I2F.U64.RP, I2F.RP, I2FP.F32.S32, UI2F.U32.RP, UI2FP.F32.U32, F2I, F2I.FTZ.U32.TRUNC.NTZ, F2I.U64.TRUNC, F2I.NTZ, F2F.F32.F64, CS2R |
| CBU | EXIT, BRA, BRA.U, CALL, CALL.REL.NOINC, RET.REL.NODEC, BSSY, BSSY.RECONVERGENT, BSYNC, BSYNC.RECONVERGENT |
| FP64 | DADD (first observed in kernel 08f), DMUL (observed in kernel 11g) |
| VOTE | VOTE.ANY, VOTE.ALL |
| REDUX | REDUX, REDUX.OR, REDUX.XOR, REDUX.SUM, REDUX.MIN, REDUX.MAX |
| TC | HMMA, QMMA, QMMA.SF, OMMA.SF, QMMA.SP, QMMA.SF.SP, OMMA.SF.SP, LDSM, STSM, LDGSTS, LDGDEPBAR, DEPBAR.LE |

### Architectural invariants

Rules that hold across every SM120 dump we have examined. These are reliable enough to use as sanity checks when reading an unfamiliar dump.

* Every SASS instruction is 128 bits (16 bytes): 64 bits opcode + 64 bits control code. Fixed since Volta.
* Stall count field in the control code is 4 bits (values 0 to 15). Stall=15 repeated on consecutive instructions signals an ILP shortage or pipeline saturation.
* Six scoreboards per warp (SB0 to SB5). Hard budget.
* Yield flag is set on every instruction with a scoreboard wait. Tells the scheduler it can switch to another warp during the wait.
* Fixed-latency operations (FFMA, IMAD, ISETP, FADD, FMUL) use only stall count, no scoreboard.
* Variable-latency operations (LDG, LDS, LDC, S2R, MUFU, BAR, SHFL, MATCH, VOTE) use scoreboards.
* Per-thread register file R0-R255 and uniform register file UR0-UR63 (UR0-UR255 on SM100+) are physically separate with distinct datapaths.
* ptxas classifies values as uniform or per-thread automatically. Uniformity means "identical across all 32 threads of the warp".
* NOP padding at the end of every kernel aligns total size to a memory boundary (typically 128 or 512 bytes). Never executed.
* ptxas emits a BRA-to-self after the final EXIT as a safety trap. Never reached by a correct control path.
* **Ptxas is deterministic.** Identical source produces byte-identical SASS. Prologue byte-identity between two kernels is a reliable signal that their prologues have not drifted. Conversely, if prologue bytes differ when they should not, the source changed somewhere the reader did not expect.
* **Register allocation is global.** Ptxas performs liveness analysis over the entire kernel, not locally. Dead registers are recycled across semantic boundaries (threadIdx's R0 becomes an FP scratch register in kernel 02, pointer high half R3 becomes an FFMA target in kernel 05). A source change in one section can cause register renames in another section as a ripple effect of the global pass.
* **Ptxas prefers inlining over external helpers on SM120 / CUDA 13.2.** Integer division at u32 width and above, math library functions (log2f, expf, sinf, sqrtf), and all intrinsics are inlined rather than compiled to calls of named external helpers. The only externally-named helper observed is `__cuda_sm20_rem_u16` (sub-word modulo). For operations with a rare slowpath (sqrtf NaN handling, u64 division when high bits are non-zero), ptxas emits a local in-kernel subroutine via CALL.REL.NOINC rather than a separate symbol. Implication for reading production SASS: a long kernel is more likely to be one big inline body than many external calls; search for local BRA and local CALL patterns rather than expecting external symbols.
* **SM120 register floor = 24 per thread.** Ptxas refuses any `-maxrregcount` below 24 on sm_120 profile, silently adjusting to 24 and emitting `ptxas warning : For profile sm_120 adjusting per thread register count of N to lower bound of 24`. The absolute minimum register budget per thread is 24.
* **Ptxas prefers restructuring over spilling.** When register pressure is tight, ptxas first tries recomputation (cheap values are recomputed rather than held), then loop restructuring (change body scheduling to reduce live range overlap), then global allocation across CALL (no caller-saved/callee-saved ABI). Spill (STL/LDL) is the last resort. Consequence: the absence of STL/LDL in a complex kernel does not imply low register pressure — it implies ptxas found a non-spill solution.
* **No stack-based ABI across local CALL.** Caller and callee within the same compilation unit share the register namespace. Ptxas coordinates allocation globally. There is no caller-saved/callee-saved distinction. [HYP] This breaks under `-rdc=true` separate compilation.
* **Static local arrays always spill.** Any `int arr[N]`, `float arr[N]`, etc. declared in device code with size exceeding the register-holdable limit is allocated in local memory. Stack pointer R1 is decremented by the array size at the prologue, and access uses LDL/STL/STL.128.

### Canonical prologue (every kernel)

Every CUDA kernel on SM120 starts with the same 8-instruction skeleton:

```
LDC R1, c[0x0][0x37c]                  // stack pointer init (ABI)
S2R R_tid, SR_TID.X                    // threadIdx.x
S2UR UR_ctaid, SR_CTAID.X              // blockIdx.x (uniform)
LDCU UR_n, c[0x0][0x...]               // kernel scalar arg (n)
LDC R_bdim, c[0x0][0x360]              // blockDim.x
IMAD R_i, R_bdim, UR_ctaid, R_tid      // i = blockDim.x * blockIdx.x + threadIdx.x
ISETP.GE.AND P0, PT, R_i, UR_n, PT     // P0 = (i >= n)
@P0 EXIT                                // out-of-bounds threads exit
```

Pattern-matched by:
* Stack pointer init at c[0x0][0x37c].
* blockDim.x at c[0x0][0x360].
* Kernel arguments starting at c[0x0][0x380] (first), 0x388 (second), and so on.
* Kernel scalar arguments (like n) at c[0x0][0x3a0] or similar.
* The global memory descriptor at c[0x0][0x358].

The ISETP that feeds `@P0 EXIT` consistently has stall=13 across all dumps. Hypothesis (unresolved): cross-pipeline transfer from ALU to CBU.

### Canonical bounds check

Two-instruction pattern:

```
ISETP.GE.AND P0, PT, R_i, UR_n, PT     // stall=13
@P0 EXIT
```

Always predication, never BRA-around. Reason: predication requires no reconvergence slot, so divergence cost is zero for out-of-bounds exit.

### Canonical lane-zero test (per-warp output)

Used in every reduction or warp-wide operation kernel:

```
LOP3.LUT P0, RZ, R_tid, 0x1f, RZ, 0xc0, !PT
@P0 EXIT
```

Single instruction computes `(tid AND 0x1f) == 0` and produces the predicate directly. Fuses AND with compare-to-zero. The LUT value `0xc0` with the `!PT` input encodes the fusion.

### Canonical warp-id derivation

Division by 32 for `warp_id = threadIdx.x / 32` is always a right shift, never a modulo CALL:

```
SHF.R.U32.HI R_warp, RZ, 0x5, R_tid
```

More generally: division by 2^k is `SHF.R.U32.HI R, RZ, k, R`, multiplication by 2^k is `SHF.L.U32 R, R, k, RZ`. Applies to any power-of-2 constant.

### Canonical pointer-and-address pattern

For a kernel argument that is a pointer:

```
LDC.64 R_ptr, c[0x0][0x3??]            // pointer from constant memory
IMAD.WIDE R_addr, R_i, sizeof(T), R_ptr  // &arr[i]
LDG.E R_val, desc[UR_desc][R_addr.64]  // load
```

The IMAD.WIDE immediate is always sizeof(T) in bytes. The LDG uses descriptor-based addressing with the descriptor loaded via LDCU.64 from c[0x0][0x358] once per kernel.

### Canonical kernel epilogue

```
STG.E desc[UR_desc][R_addr.64], R_val   // store (fire and forget, no SB)
EXIT
BRA . (self)                             // safety trap, never executed
NOP
NOP
...                                      // padding to 128 or 512 byte alignment
```

### Canonical FP constant materialization

Any non-trivial FP32 constant used as an FFMA multiplier is loaded via HFMA2:

```
HFMA2 R, -RZ, RZ, FP16_high, FP16_low
```

The bit pattern of R becomes `FP16_high | FP16_low`, interpreted as FP32. Rule: FFMA has one immediate slot (src3, the addend). Multiplier sources must be registers. HFMA2 is the systematic materialization idiom.

**Three variants observed:**
* **HFMA2 with two distinct FP16 values**: `HFMA2 R, -RZ, RZ, 1.44, -0.20` loads a specific FP32 bit pattern by concatenating the two FP16 halves. Used for polynomial coefficients in log2f (kernel 11d).
* **HFMA2 with two zero/special values**: `HFMA2 R, -RZ, RZ, 0, 0` loads 0x00000000. `HFMA2 R, -RZ, RZ, -0.0, 0` loads 0x80000000 (INT_MIN / sign bit). Used as identity element for REDUX (kernel 10).
* **MOV R, 0x<immediate> as fallback**: when the target bit pattern cannot be split into two valid encodable FP16 constants, ptxas falls back to `MOV R, 0x<32-bit>` which is slightly larger encoding but handles arbitrary values. Observed for 0x7FFFFFFF, 0xFFFFFFFF (kernel 10), 0x3d2aaabb, 0x3effffff (kernel 11g sinf coefficients), 0x437c0000 (kernel 11f expf), 0x7f800000 (+INF).

Exception: the addend of an FFMA can be an immediate directly (e.g., `FFMA R, R, R, 0.5`). This includes special values like `+INF` and `-INF` observed in log2f edge case handling.

### Canonical subnormal handling (FP32 MUFU input guard)

Every kernel that uses MUFU.LG2, MUFU.EX2, MUFU.RSQ, or MUFU.RCP on a user-provided FP32 input wraps it with subnormal handling:

```
FSETP.GEU.AND P0, PT, |R|, 1.175494350822287508e-38, PT  ; test |x| >= FLT_MIN
@!P0 FMUL R, R, 16777216                                   ; scale by 2^24 if subnormal
<MUFU operation>
@!P0 <correction>                                          ; post-correct for the scaling
```

`1.175494350822287508e-38` is FLT_MIN (smallest positive normal FP32). `16777216 = 2^24 = 0x4B800000`.

Correction depends on the operation:
* **log2**: `FADD R, R, -24` (subtract 24 from result since `log2(2^24 * x) = log2(x) + 24`)
* **rsqrt**: `FMUL R, R, 4096` (multiply by `2^12` since rsqrt halves the exponent, correction is `2^(scale/2)`)
* **fdividef (both operands scaled)**: no correction needed (ratio preserves)

Recognize this pattern when reading any SASS dump that uses MUFU on floats: if MUFU is preceded by `FSETP.GEU` against FLT_MIN and a scale-by-2^24, subnormal handling is active.

### Canonical inline integer division (Granlund-Montgomery reciprocal)

All u32 and u64 (fast path) and s32 integer divisions by a runtime variable on SM120/CUDA 13 compile to the same 5-stage inline pattern:

```
UI2F.U32.RP UR_or_R, divisor               ; convert divisor to float, round +inf
MUFU.RCP R, UR_or_R                         ; float reciprocal
IADD R, R, 0xffffffe                        ; magic adjustment constant
F2I.FTZ.U32.TRUNC.NTZ R_q, R                ; back to int
IADD R_neg, RZ, -R_q                        ; prepare error computation
IMAD R_tmp, R_neg, divisor, RZ              ; error = -quotient * divisor
IMAD.HI.U32 R_q_corrected, R_q, R_tmp, R_base   ; correction via high-multiply
IMAD.HI.U32 R_final, R_q_corrected, dividend, RZ  ; final quotient
IADD R_neg, -R_final, RZ
IMAD R_r, divisor, R_neg, dividend          ; remainder = dividend - quotient * divisor
ISETP.GE.U32.AND P0, ..., R_r, divisor, PT  ; double-correction pair
@P0 IADD R_r, R_r, -divisor
@P0 IADD R_final, R_final, 0x1
ISETP.GE.U32.AND P1, ..., R_r, divisor, PT
@P1 IADD R_final, R_final, 0x1
@!P_zerodiv LOP3.LUT R_final, RZ, divisor, RZ, 0x33, !PT  ; zero-divisor sentinel
```

Twenty-one instructions for u32, ~30 for u64 with multi-precision correction, slightly more for s32 with IABS wrapping and LOP3 0x3c sign fix-up.

Recognize: any sequence starting with `UI2F.U32.RP` followed by `MUFU.RCP` and ending with `IMAD.HI.U32` is an inline integer division or modulo.

### Canonical local CALL subroutine (in-kernel slowpath)

Rare slowpaths for operations that are mostly-fast-but-sometimes-complex use a local CALL pattern rather than full inlining or external helpers:

```
<main kernel body>
BSSY.RECONVERGENT B0, <sync_address>         ; open reconvergence scope
<test for slowpath condition>
@!P BRA <fast_path>                          ; if fast path applicable, skip CALL
MOV Rn, <return_address>                     ; set up return in arbitrary register
CALL.REL.NOINC <local_slowpath_addr>         ; call local subroutine
<return_address>:
BRA <post>                                   ; skip fast path
<fast_path>:
<inline fast path computation>
<post>:
BSYNC.RECONVERGENT B0                        ; reconverge
<main kernel continuation>
EXIT

<local_slowpath_addr>:
<handle edge cases / slow algorithm>
RET.REL.NODEC Rn 0x0                         ; return via Rn
<safety trap BRA-to-self>
<NOP padding>
```

Observed in kernels 11b (u64 division slowpath) and 11h (sqrtf NaN/Inf/denormal slowpath). The return register `Rn` is chosen per-kernel based on local register pressure (R4, R9, and other values observed).

Distinguish from external helper CALL: local CALL targets a hex offset within the kernel, external CALL targets a named symbol like `__cuda_sm20_rem_u16`.

### Canonical stack frame allocation (when local memory is needed)

When a kernel needs local memory (for spill or static local array), the standard prologue is immediately followed by an R1 adjustment:

```
LDC R1, c[0x0][0x37c]             ; standard prologue: load stack pointer
S2R R3, SR_TID.X                   ; standard prologue: thread ID
S2UR UR4, SR_CTAID.X               ; standard prologue: block ID
LDCU UR5, c[0x0][0x<n_addr>]       ; standard prologue: load n
IADD R1, R1, -<frame_size>         ; ADDITIONAL: carve local frame on stack
... (bounds check continues)
```

The `IADD R1, R1, -<frame_size>` instruction is the signal that the kernel uses local memory. `<frame_size>` is a negative immediate, sized exactly to the memory requirement rounded up to 8-byte alignment.

Frame sizes observed:
* 0x100 (256 bytes) for `int arr[64]` (exact size, no padding)
* 0x128 (296 bytes) for 32 FFMA accumulators + scratch

[HYP] Alignment is on 8 bytes minimum. Smaller frames not tested.

### Canonical register spill (rolling window in compute loop)

Under register pressure, ptxas interleaves one STL and one LDL per FFMA, maintaining the live set within budget while keeping the unroll intact:

```
STL  [R1+<old_slot>], R_spilled    ; persist result of previous FFMA
FFMA R_result, R_x, R_mul, R_acc   ; compute
LDL.LU R_reload, [R1+<new_slot>]   ; load next accumulator (last-use hint)
STL  [R1+<old_slot>], R_result     ; persist current result
FFMA R_next, R_x, R_mul, R_acc     ; next FFMA in the chain
LDL.LU R_reload, [R1+<new_slot+4>]
...
```

Pattern observed in kernel 12i. Each FFMA is flanked by exactly one spill (old value out) and one reload (new value in). The `.LU` modifier on LDL tells the cache to evict the line after read, since the value will not be reused in the same form.

### Canonical vectorized spill for local array init

When initializing a static local array with values computed in a burst of 4 registers, ptxas uses STL.128:

```
<compute 4 values into R4, R5, R6, R7>
STL.128 [R1], R4              ; writes R4:R7 in one transaction
<compute next 4 values into R20, R21, R22, R23>
STL.128 [R1+0x10], R20        ; writes R20:R23 at next 16-byte slot
...
```

Observed in kernel 12k for `int arr[64]` initialization. Reduces LSU transaction count by 4× compared to scalar STL.

### Canonical indexed local memory read

When reading a local array at runtime-computed indices, the pattern combines R2UR conversion of the stack pointer with a LOP3 alignment mask:

```
R2UR UR_base, R1                                     ; copy stack pointer to uniform
LOP3.LUT R_offset, R_idx, 0xfc, RZ, 0xc0, !PT        ; mask: align + range limit
LDL R_result, [R_offset + UR_base]                    ; indexed local load
```

The `LOP3 0xfc` truth table (masking with `0xfc = 11111100b`) simultaneously aligns the offset to 4 bytes (for 32-bit access) and constrains it to the local array range (low 6 bits × 4 = 64 entries for `int arr[64]`).

[HYP] R2UR is required because LDL addressing mode accepts `[R_offset + UR_base]` but not `[R_offset + R_base]`. Not confirmed by direct testing.

### Canonical warp reduction (butterfly)

```
SHFL.BFLY PT, R_tmp, R_val, 0x10, 0x1f ; FADD R_val, R_val, R_tmp
SHFL.BFLY PT, R_tmp, R_val, 0x08, 0x1f ; FADD R_val, R_val, R_tmp
SHFL.BFLY PT, R_tmp, R_val, 0x04, 0x1f ; FADD R_val, R_val, R_tmp
SHFL.BFLY PT, R_tmp, R_val, 0x02, 0x1f ; FADD R_val, R_val, R_tmp
SHFL.BFLY PT, R_tmp, R_val, 0x01, 0x1f ; FADD R_val, R_val, R_tmp
```

5 stages for a 32-lane warp. If the source has divergent code before the reduction, ptxas wraps the whole thing in BSSY.RECONVERGENT / BSYNC.RECONVERGENT.

For int32 only, a single `REDUX.SUM.S32 UR, R` replaces all 10 instructions.

### Canonical shared memory access

```
S2UR UR_cga, SR_CgaCtaId
UMOV UR_base, 0x400
ULEA UR_base, UR_cga, UR_base, 0x18     // per-block shared base
LEA R_addr, R_tid, UR_base, 0x2         // &smem[tid]
STS [R_addr], R_val
BAR.SYNC.DEFER_BLOCKING 0x0             // __syncthreads()
LDS R_val, [R_addr]
```

The `UMOV UR, 0x400` + `ULEA ..., 0x18` pair appears if and only if the kernel uses `__shared__` memory.

**Decoded formula.** The ULEA computes `UR_base = (UR_cga << 0x18) + 0x400`, which expands to:

```
shared_base = (SR_CgaCtaId << 24) + 0x400
```

where `SR_CgaCtaId` is the block's unique identifier (per-cluster on SM90+, fallback to block ID on non-cluster kernels). Interpretation:

* **`0x400`** is an architectural offset, independent of the kernel's parameters. Observed constant at 0x400 across shared sizes 512B, 1024B, 2048B, block sizes 128, 256, 512. Hypothesis: it is the entry point of shared memory within the SM-wide memory-mapped pool, or a descriptor field encoded into the base.
* **Shift by 24** creates a per-block stride of 16 MiB in the virtual address space. This is far larger than actual shared memory per SM (48 KB to 228 KB), so the high bits of CgaCtaId never represent real addresses. Hypothesis: the shift is a pattern for the shared memory descriptor encoding rather than a direct address, and the hardware masks/maps the high bits internally.

**For multiple shared buffers,** ptxas emits a single UMOV + ULEA and derives each subsequent buffer's base by adding the cumulative size of the preceding buffers:

```
UMOV UR4, 0x400
ULEA UR4, UR_cga, UR4, 0x18              // base of smem_a
UIADD3 UR5, UPT, UPT, UR4, 0x400, URZ    // base of smem_b (smem_a is 0x400 bytes)
```

Shared buffers are placed consecutively without automatic padding. Alignment only from the programmer's struct declarations or explicit `__align__`.

**Shared memory addressing does not use a descriptor.** Unlike LDG/STG which use `desc[UR][R.64]`, LDS/STS use `[R]` direct or `[R+UR]`. The base is encoded in the register itself. The `[R+UR]` form is useful when multiple buffers share the same per-thread offset: one R for `&smem[tid]`, multiple UR for the different buffer bases.

### Arithmetic operator compilation rules

| Source | ptxas strategy |
|---|---|
| `%` u16 by runtime variable | CALL `__cuda_sm20_rem_u16` (external named helper, sole case observed) |
| `%` by non-power-of-2 constant | Inline reciprocal multiplication with magic number |
| `%` by power-of-2 constant | Fused into LEA + LOP3 mask (cheap) |
| `/` u32 by runtime variable | Inline Granlund-Montgomery reciprocal multiplication (21 instructions) |
| `/` u64 by runtime variable | Inline u32-fast-path + local CALL to in-kernel subroutine for slowpath |
| `/` s32 by runtime variable | Inline via IABS + unsigned algorithm + LOP3 0x3c sign fix-up |
| `/` by power-of-2 constant | SHF.R.U32.HI (shift right) |
| `*` by power-of-2 constant | SHF.L.U32 (shift left) |
| `+`, `*+` in FP | FFMA when `a*b+c` pattern present, else separate FMUL / FADD |
| FP constant as multiplier | HFMA2 materialization into register |
| FP constant as addend | Immediate in FFMA src3 slot |
| `log2f(x)` standard | Inline 10-coefficient Remez polynomial + range reduction |
| `__log2f(x)` intrinsic | Single MUFU.LG2 + subnormal handling (4 instructions) |
| `expf(x)` standard | Inline range reduction + MUFU.EX2 + FMUL (no CALL) |
| `sinf(x)` standard | Inline fast path + BRA slowpath with Payne-Hanek for large args |
| `sqrtf(x)` standard | Fast path MUFU.RSQ + 1 NR iteration + CALL to local slowpath for edge cases |
| `rsqrtf(x)` | Single MUFU.RSQ + subnormal handling (4 instructions) |
| `__fdividef(a,b)` | MUFU.RCP + FMUL + subnormal handling |
| FP subnormal on MUFU input | Pre-scale by 2^24, operate, post-correct (universal pattern) |
| Static local array `T arr[N]` | Stack frame allocation via `IADD R1, R1, -size` + STL/LDL for access |
| Register pressure above budget | Rolling STL/LDL pattern around compute instructions, unroll preserved |
| 16+ arguments to local CALL | All passed in registers (no ABI stack); caller/callee coordinate allocation globally |

### Scheduling patterns

ptxas systematically places independent instructions between a producer and its consumer to hide latency. Observed patterns:

* **IMAD.WIDE interleaved with LDG.** Address computations are placed between consecutive LDG emissions so that multiple loads are in flight simultaneously.
* **Pointer address of the store placed early.** When the value is the bottleneck, the store address is computed while the value is still being produced.
* **Pointer address of the store placed late.** When the value chain is long (many sequential FFMAs), the store address is deferred to minimize register live range.
* **Constant materialization (HFMA2) placed at the pipeline choice point.** HFMA2 is used in compute-heavy blocks where the FMA pipeline is loaded, MOV is used otherwise.
* **VOTE.ANY hoisted to kernel start** when its input is `PT` (as in `__activemask()`). No data dependency, so it can run in parallel with pointer loads.
* **Counter update interleaved with compute in unrolled loops.** UIADD3 and UISETP for loop control are placed between FFMAs, not at the tail.
* **Polynomial FFMA chain with `.reuse` on multiplicand.** In Horner-method polynomial evaluation (log2f, sin, etc.), the variable `x` is reused across every FFMA. Ptxas emits `.reuse` on the `x` operand so the hardware caches it in the reuse cache, avoiding register file reads. Recognizable by a long monotonic chain like `FFMA R, R_x.reuse, R_acc, imm1; FFMA R, R_x.reuse, R, imm2; ...` with ~8-12 iterations.
* **Independent work between REDUX/MUFU/SHFL and their consumer.** Variable-latency producers (REDUX, MUFU, SHFL, LDG) are followed by independent instructions that do not depend on them, hiding the latency. The consumer's scoreboard wait comes last.
* **LOP3.LUT or ISETP for lane-zero test placed after REDUX.** The lane-zero predicate for conditional store is computed after REDUX issues but before its result is consumed, using REDUX's latency window for useful work.

### Scoreboard assignment rules

ptxas chooses between grouping producers onto a single scoreboard or assigning distinct scoreboards based on the downstream consumer topology:

* **Co-consumed producers share a scoreboard.** Multiple LDGs all feeding the same downstream FFMA go onto the same SB so the consumer waits once for all of them. Observed in kernel 01 (two LDGs on SB4) and kernel 03 (three LDGs on SB4).
* **Parallelizable consumers get distinct scoreboards.** Multiple LDCs each feeding their own independent IMAD.WIDE get distinct SBs (SB0, SB1, SB2, SB3) so the IMADs can proceed in parallel. Observed in kernel 03's four pointer loads.
* **Trade-off.** The SB budget is 6 per warp. Grouping economizes SBs for later use; distinct SBs preserve parallelism. ptxas chooses based on whether the consumers depend on each other or not.

### Pipeline characteristics

Observed per-pipeline latency behavior:

* **LDC stall=1 as cadence.** Constant cache responds fast, so ptxas emits LDCs back-to-back with stall=1, relying on scoreboard grouping or distinct SBs for correctness. LDC latency is short enough that the stall count does not need to be inflated.
* **LDG requires large effective latency.** Global memory loads have variable latency on the order of hundreds of cycles. ptxas relies on scoreboards entirely, emitting stall=1 or stall=2 on the LDG itself and placing the consumer's wait much later.
* **ISETP → @P EXIT stall=13 is universal.** The predicate producer for a bounds check always has stall=13 regardless of the surrounding code. Recurs in every kernel. Hypothesis: cross-pipeline transfer from ALU to CBU.
* **NOP padding on FP64 operations.** The FP64 pipeline on consumer SM120 has restricted throughput. Consecutive DADDs (kernel 08f) and isolated DMULs (kernel 11g) both show NOP gaps of 3-4 NOPs around the operation as an instruction-level manifestation. Confirmed across two distinct operations, so this is a pipeline-wide characteristic, not operation-specific.
* **MUFU variable latency.** MUFU.RCP, MUFU.RSQ, MUFU.LG2, MUFU.EX2 all use scoreboards for synchronization. Consumers wait on the scoreboard, ptxas interleaves unrelated work between producer and consumer. Estimated latency ~20-30 cycles based on the typical distance between MUFU and its consumer in dumps, but not microbenchmarked.
* **CALL latency.** CALL.REL.NOINC + RET.REL.NODEC adds at least two instructions of overhead (the MOV for return address + the CALL itself, plus the RET). Additional cycles for the jump, not measured.

### Cost rules (quantified overhead per source feature)

Quantified overhead per source-level construct, useful for budgeting kernel cost:

* **Pointer tax: +3 instructions per array argument.** Each additional pointer in the kernel signature costs approximately three SASS instructions: one LDC.64 to load the pointer from constant memory, one IMAD.WIDE to compute the per-thread address, and potentially one more memory op (LDG or STG) depending on usage. Independent of how the array is consumed by compute.
* **Constant materialization amortization.** A distinct FP32 multiplier constant costs one HFMA2 materialization. If the constant is used N times, the amortized cost per use is 1/N instructions. Kernels with many distinct multiplier constants (FIR filters, polynomial evaluation) pay the materialization cost for each.
* **Modulo runtime u16: CALL to external helper.** A modulo by a runtime u16 variable triggers a CALL to `__cuda_sm20_rem_u16` costing ~40-60 cycles in throughput. The only external helper observed in the project.
* **Modulo non-power-of-2 constant: ~10 instructions inline.** A constant that is not a power of 2 but is known at compile time compiles to an inline reciprocal multiplication with a magic number, around 10 instructions, 3-4× more expensive than power-of-2 but far cheaper than runtime CALL.
* **Modulo power-of-2 constant: 1-2 instructions.** Fused into LEA + LOP3 mask. Effectively free.
* **Integer division u32 runtime: 21 instructions inline.** Granlund-Montgomery reciprocal multiplication. No CALL. Roughly 1 MUFU.RCP + 4-5 IMAD chain + correction pair.
* **Integer division u64 runtime: ~60 instructions (fast path + local CALL slowpath).** Fast path 30+ instructions inline, slowpath CALL adds another 30+ when triggered. Budget worst-case ~60 instructions per division.
* **Integer division s32 runtime: ~25 instructions inline.** u32 algorithm + IABS wrapping + LOP3 0x3c sign fix-up.
* **log2f standard: ~15 instructions polynomial + subnormal handling.** 10 FFMA polynomial + 2 FMUL + 1 FFMA with log2(e) + range reduction + edge case handling.
* **`__log2f` intrinsic: 3-4 instructions.** MUFU.LG2 + subnormal handling only.
* **expf standard: ~10 instructions.** Range reduction + MUFU.EX2 + FMUL. Much cheaper than log2f standard because it uses MUFU.EX2 directly instead of polynomial.
* **sinf standard (small args): ~10 instructions fast path.** Triple FFMA range reduction + 4-FFMA polynomial.
* **sinf standard (large args |x| >= 2^17): ~80+ instructions slowpath.** Payne-Hanek with 6-iteration table loop + FP64 DMUL. Significant cost.
* **sqrtf standard: 5-6 instructions fast path, ~20 additional if slowpath triggered.** Newton-Raphson 1 iteration on MUFU.RSQ. Slowpath (NaN/Inf/denormal) rare, adds CALL overhead.
* **rsqrtf: 3-4 instructions.** MUFU.RSQ + subnormal handling only.
* **`__fdividef`: 3-4 instructions.** MUFU.RCP + FMUL + subnormal handling.
* **Subnormal handling cost: +3 instructions per MUFU-using kernel.** FSETP.GEU + @!P FMUL scale + @!P correction. Fixed overhead any time MUFU is applied to user FP input.
* **Register spill overhead: +2 instructions per spilled value per use.** One STL to persist + one LDL to reload = 2 additional instructions wrapped around each compute instruction using a spilled value. Kernel size multiplied by 1.5× to 10× depending on rolling-window depth.
* **Static local array overhead: +N × 0.25 STL.128 for init.** For an array of N 32-bit elements initialized sequentially, ptxas emits N/4 STL.128 stores. Plus one `IADD R1, R1, -(4N)` at the prologue. Reads add LDL per access.
* **Stack frame allocation: +1 instruction (IADD R1, R1, -frame).** Fixed cost when any local memory is needed.
* **Indexed local read: +1 LOP3 per access.** The `LOP3 0xfc` alignment mask is emitted before each LDL with runtime-computed index.
* **R2UR for stack addressing: +1 instruction per kernel.** Single R2UR needed once per kernel to copy R1 to UR for indexed LDL base.

### Compiler artifacts to watch for

* **STL / LDL in kernel body** indicates register spilling to local memory. Signal of a kernel too wide for the register file or containing a static local array. Observed in kernels 12i (pressure-induced spill) and 12k (array-induced spill). The rolling pattern `STL ... FFMA ... LDL` around each compute instruction is the visual signature of pressure-induced spill.
* **STL.128** indicates vectorized spill, typically for array initialization. Four consecutive registers stored in one transaction.
* **LDL.LU** indicates last-use hint for cache eviction. Paired with rolling-window spill pattern where the loaded value is consumed once.
* **`IADD R1, R1, -<frame>` immediately after prologue** is the unambiguous signal that the kernel uses local memory. Frame size = memory requirement rounded to 8-byte alignment.
* **R2UR UR, R1** indicates preparation for indexed local memory reads with per-thread offsets and uniform base.
* **ptxas warning `adjusting per thread register count of N to lower bound of 24`** confirms SM120 register floor. Any request below 24 is silently raised.
* **CALL.REL.NOINC to a named symbol** indicates an external out-of-line helper. On SM120/CUDA 13, only observed for `__cuda_sm20_rem_u16` and `__cuda_sm20_div_u16` (sub-word integer modulo/division).
* **CALL.REL.NOINC to a local address** (hex offset within same kernel) indicates an inline-but-out-of-hot-path subroutine. The body is placed after the main EXIT. Used for rare slowpaths (sqrtf NaN handling, u64 division high-bits).
* **BRA to a forward address within the main body** indicates an inline slowpath that doesn't warrant a subroutine (sinf Payne-Hanek, expf non-fast-path). Control returns naturally via fall-through.
* **Kernel size significantly larger than expected** often means cascade unrolling (runtime trip count) OR full inlining of math library functions (log2f, sinf) OR register spill (STL/LDL multiply kernel size by up to 10×).
* **BSSY / BSYNC.RECONVERGENT** wrapping a short section means ptxas kept an explicit reconvergence region. [OBS] Observed causes include divergence before a warp-synchronous operation (Kernel 09), local slowpath/call regions (Kernel 11 and 21r), `break` or lane-dependent loop exits (Kernel 20 and 21), and branch-kept divergent arithmetic bodies (Kernel 21c/21e/21l).
* **Consecutive NOPs between arithmetic instructions** mean the pipeline cannot keep up (FP64 on consumer parts, or ILP shortage).
* **Long chain of FFMA with monotonic `.reuse` pattern** indicates inline polynomial evaluation via Horner's method. Look at the immediate constants to identify which function (log2, sin, cos, exp).
* **UI2F.U32.RP followed by MUFU.RCP** is the signature of inline integer division/modulo via Granlund-Montgomery.
* **LDG.E.CONSTANT in a loop with UIADD3 counter** indicates a read-only table access, most likely Payne-Hanek or similar table-based algorithm.
* **Magic constants to watch for in SASS immediates:**
  * `0xffffffe` (268435454): IADD adjustment in Granlund-Montgomery reciprocal multiplication
  * `0x3f3504f3` (sqrt(2)/2 as FP32): log2f range reduction
  * `0x437c0000` (252.0): expf intermediate scale
  * `12582913 = 0x00C00001`: expf integer extraction via FFMA.RM
  * `16777216 = 0x4B800000` (2^24): subnormal handling scale factor
  * `4096 = 0x45800000` (2^12): subnormal rsqrt correction
  * `0x7fffffff` (NaN bit pattern): FP32 NaN for edge case returns
  * `0x7f800000` (+INF bit pattern): FP32 infinity for edge case returns
  * `0x80000000` (INT_MIN): HFMA2 `-RZ, RZ, -0.0, 0` loads this value

### Reading control code annotations

Every SASS instruction on SM120 is 128 bits: 64 bits opcode + 64 bits control code. The control code encodes scheduling information that cuobjdump and gpuasm surface as text annotations. You rarely need to decode the bits yourself because the tools do it for you.

The annotations you will encounter in our analyses:

* **`stall=N`** where N is 0 to 15. Number of cycles the warp must wait after issuing this instruction before it can issue the next one from the same warp. Other warps can run during this stall. Typical values: 1 to 5 for normal arithmetic, 13 for ISETP→@P EXIT (cross-pipeline transfer), 12 to 15 require the yield flag.
* **`yield`** (flag). When set, the scheduler is allowed to switch to another warp. Always set on every instruction with a scoreboard wait. A reliable heuristic: if you see a scoreboard wait without yield, something is unusual.
* **`SBS=N`** where N is 0 to 5. This instruction signals scoreboard N when its variable-latency result is ready. Set on LDG, LDS, LDC, S2R, MUFU, BAR, SHFL, VOTE, MATCH, REDUX. Not set on FFMA, FADD, IMAD, ISETP (fixed latency).
* **`wait={SBN, SBM, ...}`** Set of scoreboards that must be clear before this instruction can issue. Mask of 6 bits. Used on consumers of variable-latency producers.
* **`.reuse`** suffix on a source operand (not on the opcode). Tells the hardware to cache the register value in the reuse cache to save a register file read next cycle.

What to look for in practice:

* Many consecutive `stall=15` signals a pipeline saturation or ILP shortage.
* Many consecutive `stall=1` with useful work is a healthy pipelined stream.
* Long `wait={SB_something}` chains on a single scoreboard mean a memory-bound stretch.
* Multiple LDGs sharing an SBS is a compiler optimization (co-consumed producers).
* A scoreboard wait without yield is unusual and worth investigating.

The exact bit layout follows the Volta format (Jia et al. 2018) with likely extensions in the upper bits for Blackwell-specific features. See DECISIONS.md D006 for the gaps we have consciously left open (bit-level decoding not implemented).

### Reading opcode modifiers

SASS opcodes carry suffixes that change their semantics. Most follow intuitive conventions. Only a few are SM120-specific and worth explicit attention.

Intuitive suffixes (do not require a glossary):

* **`.64`, `.128`, `.256`**: memory transaction width in bits.
* **`.U32`, `.S32`**: unsigned or signed 32-bit interpretation.
* **`.HI`, `.LO`**: high or low half of a wider result.
* **`.WIDE`**: result is 64-bit (register pair) from a 32-bit multiply.
* **`.E`**: external, used for global memory to distinguish from LDS/LDC.
* **`.GE`, `.LT`, `.EQ`, `.NE`, `.GT`, `.LE`**: comparison operators.
* **`.AND`, `.OR`, `.XOR`**: combines the result with a predicate input.
* **`.LUT`**: 3-input lookup table (LOP3 and its uniform cousins).

SM120-specific or non-obvious suffixes worth knowing:

* **`.ENL2`** (appears on LDG/STG at 256 bits): enlarged encoding using two register base fields instead of one. Needed because 8 consecutive registers cannot fit a single 5-bit register field in the opcode. Observed only at 256-bit width.
* **`.DEFER_BLOCKING`** (appears on BAR.SYNC): a variant of block-level barrier that allows the warp to defer blocking until necessary. Default form of `__syncthreads()` on SM120.
* **`.RECONVERGENT`** (appears on BSSY and BSYNC): Independent Thread Scheduling reconvergence scope. [OBS] It can wrap warp-synchronous instructions that follow divergent code, divergent local calls, `break`/lane-dependent loop exits, and branch-kept divergent arithmetic regions in the tested dumps.
* **`.GEU`** (appears on FSETP): greater equal unordered, FP semantics where NaN comparisons return true.
* **`.NEU`, `.GTU`, `.LTU`, `.EQU`, `.LEU`** (appear on FSETP): unordered variants of the comparison predicates. NaN comparisons return true for the unordered variants.
* **`.TRUNC.NTZ`** (appears on F2I): truncation with non-toward-zero rounding for negative zero edge case.
* **`.FTZ`** (appears on FMUL, FADD, FSETP, F2I): flush-to-zero. Subnormals are treated as zero for both input and output. Improves throughput on some paths at the cost of precision for very small values.
* **`.SAT`** (appears on FFMA): saturate. Output is clamped to [0, 1]. Used for probability-like computations.
* **`.RM`** (appears on FFMA): round mode toward -inf (round minus). Used in expf for controlled integer extraction via overflow.
* **`.RP`** (appears on I2F, UI2F): round mode toward +inf (round plus). Used in integer division to guarantee the reciprocal estimate is an overestimate.
* **`.W`** (appears on SHF.L): with wrap. Changes the shift to a funnel shift (combining two source registers).
* **`.CONSTANT`** (appears on LDG.E): load via constant cache path for read-only global memory. Benefits from read-only caching behavior. Distinct from LDC (constant memory bank) and LDCU (uniform constant).
* **`.NOINC`** (appears on CALL.REL): no-increment of the stack pointer. Return address is passed via register.
* **`.NODEC`** (appears on RET.REL): no-decrement of the stack pointer. Mirror of CALL.REL.NOINC.
* **`.LU`** (appears on LDL): last use. Hint that the loaded line will not be reused, telling the local memory cache to evict after read. Observed systematically in rolling-window spill patterns where the loaded value is consumed once by the next FFMA then overwritten.
* **`.128`** (appears on STL): vectorized 128-bit transaction. Stores 4 consecutive 32-bit registers in one operation. Used for sequential local array initialization.

Operand-level flags (not opcode suffixes but worth distinguishing):

* **`.reuse`** on a source operand: hint to the hardware to cache that value in the reuse cache. Frequency correlates with register pressure — dense kernels use `.reuse` more than simple kernels.
* **`.64`** on a register name (e.g., `R2.64`): treats R2:R3 as a 64-bit register pair.
* **`.H0`, `.H1`** on a register name: selects the low or high 16 bits of a packed half-precision register.
* **`|R|`** (absolute value on FSETP, FFMA src, etc.): take absolute value of the operand at read time.
* **`-R`** (negation on source operand): negate at read time. Combined with `|R|` allows sign manipulation without separate instructions.
* **`~R`** (bitwise NOT on integer source): complement at read time. Observed in IADD.X for multi-precision arithmetic.

Modifiers we have not yet encountered in our dumps but that appear in the Blackwell ISA reference are not covered here.

### Opcode operand semantics

Syntax reminders for the operand orders of opcodes that are easy to misread.

* **FFMA `dst, srcA, srcB, srcC`.** Computes `dst = srcA * srcB + srcC`. srcC is the addend, the only position that can hold an FP32 immediate. srcA and srcB must be registers. Same order as PTX `fma.f32`.
* **FFMA.SAT `dst, srcA, srcB, srcC`.** Same as FFMA but output is clamped to [0, 1]. Used for probability/fraction computations (observed in expf).
* **FFMA.RM `dst, srcA, srcB, srcC`.** Same as FFMA but with round mode toward -inf (round minus). Used in expf for controlled integer extraction.
* **IMAD `dst, srcA, srcB, srcC`.** Computes `dst = srcA * srcB + srcC`. Like FFMA but integer. Ubiquitous in address arithmetic.
* **IMAD.WIDE `dst, srcA, imm, srcC`.** Computes a 64-bit result `dst = srcA * imm + srcC` where `dst` is a register pair `R:R+1` and srcC is also a pair. The `imm` is typically `sizeof(element)` in bytes. Used for `&array[i]` computation.
* **IMAD.WIDE.U32 `dst, srcA, srcB, srcC`.** Explicit unsigned 32×32→64 multiply-accumulate. Used for multi-precision arithmetic in u64 division slowpath.
* **IMAD.HI.U32 `dst, srcA, srcB, srcC`.** Returns the high 32 bits of `srcA * srcB + srcC`. Key primitive in Granlund-Montgomery reciprocal multiplication.
* **IMAD.SHL.U32 `dst, srcA, imm, srcC`.** Integer multiply-add with explicit left shift. Semantics: `dst = (srcA << log2(imm)) + srcC` when `imm` is a power of 2. Observed in sinf Payne-Hanek preparation.
* **IABS `dst, src`.** Integer absolute value. Used in signed division via unsigned algorithm.
* **IADD `dst, srcA, srcB`.** Two-input integer add (no carry-in).
* **IADD.X `dst, srcA, srcB`.** Two-input add with carry-in from CC. Used in multi-precision arithmetic.
* **IADD3 `dst, srcA, srcB, srcC`.** Three-input add: `dst = srcA + srcB + srcC`. Common in index arithmetic for loop counters and address computation.
* **IADD3.X `dst, PT, PT, srcA, srcB, srcC, Px, Py`.** Three-input add with carry-in and predicate-gated output. The `P` operands are carry-chain predicates. Used in multi-precision accumulation.
* **IADD.64 `dst, srcA, srcB`.** 64-bit integer add (single operation, no carry chain). Operates on register pairs.
* **UIADD3 `UR_dst, UPsrcA, UPsrcB, UR_1, UR_2, UR_3`.** Uniform three-input add with predicate masks. `UPsrcA` and `UPsrcB` are predicates that can conditionally negate the corresponding addend. Used to flip signs of inputs on the uniform path.
* **LEA `R_dst, R_src1, UR_base, imm`.** Computes `R_dst = (R_src1 << imm) + UR_base`. Mixed per-thread and uniform: per-thread index scaled by shift, added to uniform base. Canonical shared memory address derivation on SM120.
* **LEA.HI `R_dst, R_src1, R_src2, R_src3, imm`.** High-half LEA, returns high 32 bits of `(src1 << imm) + src2`.
* **ULEA `UR_dst, UR_src1, UR_src2, imm`.** Uniform LEA: `UR_dst = (UR_src1 << imm) + UR_src2`. Purely uniform, used in shared memory base construction.
* **PRMT `R_dst, R_src1, imm, R_src2`.** Byte permutation. Each nibble of the 16-bit immediate selects one byte from the 8 available bytes (4 from src1, 4 from src2) to place at the corresponding destination byte. Used in the integer division helper for byte-level manipulation of the magic number.
* **LOP3.LUT `R_dst, srcA, srcB, srcC, lut_imm, predicate`.** Applies a 3-input lookup table (256 possible functions) over the three source operands. The `lut_imm` byte encodes the truth table. Canonical values:
  * `0xc0`: `A AND B` (used for lane-zero tests, power-of-2 masks)
  * `0x33`: `NOT B` (used for zero-divisor sentinel in division)
  * `0x3c`: `A XOR B` (used for sign fix-up in signed division)
  * `0xfc`: `A OR B` (used for combining predicate bits)
* **SHF.L.W.U32.HI `R_dst, R_src1, R_src2, R_src3`.** Funnel shift left with wrap, high half, unsigned. Computes the high 32 bits of `(R_src3 : R_src1) << R_src2`. Used in Payne-Hanek accumulation.
* **SHF.R.U32.HI `R_dst, R_src1, R_src2, R_src3`.** Funnel shift right, high half, unsigned. Complement of SHF.L.W.U32.HI.
* **SHF.R.S32.HI `R_dst, R_src1, R_src2, R_src3`.** Same as SHF.R.U32.HI but signed (arithmetic shift right).
* **FSEL `dst, srcA, srcB, predicate`.** Float select: `dst = srcA` if predicate is true, else `srcB`. The negation `-` can be applied to operands (e.g., `FSEL R, -A, A, !P`).
* **SEL.64 `dst, srcA, srcB, predicate`.** 64-bit select. Operands are register pairs.
* **R2P `PR, R_src, imm`.** Register-to-predicates: copies bits of `R_src` into predicate registers P0..P7, with `imm` selecting which bit positions. Rare; used to unpack packed flags.
* **FSETP.* `Pd, Ps, srcA, srcB, Ps_in`.** Float set predicate. Comparison types: `.GE`, `.LT`, `.EQ`, `.NE`, `.GT`, `.LE` plus unordered variants `.GEU`, `.LTU`, `.EQU`, `.NEU`, `.GTU`, `.LEU` where NaN comparisons return true. The `.FTZ` suffix flushes subnormals to zero for comparison.
* **ISETP.*.U64 / .S64 `Pd, Ps, srcA, srcB, Ps_in`.** 64-bit integer compare. Operates on register pairs.
* **UI2F.U32.RP `UR_dst, UR_src`.** Convert uniform u32 to float with round toward +inf. Guarantees the float is ≥ the original integer.
* **I2F.RP `R_dst, R_src`.** Per-thread signed int32 to float, round toward +inf.
* **I2F.U64.RP `R_dst, R_src`.** u64 to float, round toward +inf.
* **I2F.F64.S64 `R_dst, R_src`.** Signed int64 to double (FP64).
* **I2FP.F32.S32 `R_dst, R_src`.** Signed int32 to float (FP32 output explicit).
* **F2I.FTZ.U32.TRUNC.NTZ `R_dst, R_src`.** Float to uint32, flush subnormals, truncate, non-toward-zero rounding. The `.NTZ` disambiguates signed zero edge case.
* **F2I.U64.TRUNC `R_dst, R_src`.** Float to uint64 with truncation.
* **F2I.NTZ `R_dst, R_src`.** Float to int32 with non-toward-zero rounding.
* **F2F.F32.F64 `R_dst, R_src`.** Double to float narrowing conversion.
* **DMUL `R_dst, R_srcA, R_srcB`.** FP64 multiply. Operates on register pairs.
* **MUFU.LG2 `R_dst, R_src`.** Hardware log2 approximation. Single-cycle throughput on XU pipeline.
* **MUFU.EX2 `R_dst, R_src`.** Hardware 2^x approximation.
* **MUFU.RSQ `R_dst, R_src`.** Hardware reciprocal sqrt approximation.
* **MUFU.RCP `R_dst, R_src`.** Hardware reciprocal approximation (previously seen in chapter 06).
* **CALL.REL.NOINC `<target>`.** Relative call, no increment of stack pointer. Return address is set via `MOV Rn, <return_addr>` beforehand.
* **RET.REL.NODEC `Rn`, `<offset>`.** Relative return, no decrement of stack pointer. Return address comes from Rn plus offset.
* **BSSY.RECONVERGENT `Bn`, `<sync_address>`.** Set synchronization barrier for Independent Thread Scheduling reconvergence. Typically at the start of a divergent region.
* **BSYNC.RECONVERGENT `Bn`.** Wait for barrier `Bn` reconvergence. Typically at the end of a divergent region.
* **STL `[R_base+offset], R_src`.** Store 32-bit register to local memory. R_base is always R1 (stack pointer) in our observations. Offset is a small immediate (up to 0x124 observed).
* **STL.128 `[R_base+offset], R_src`.** Vectorized store of 4 consecutive 32-bit registers (R_src, R_src+1, R_src+2, R_src+3) to local memory. Offset is 16-byte aligned. Reduces LSU transaction count by 4× compared to scalar STL.
* **LDL `R_dst, [R_base+offset]`.** Load 32-bit value from local memory into register. Inverse of STL.
* **LDL.LU `R_dst, [R_base+offset]`.** Load with last-use hint. Cache line is marked for eviction after the read. Used in rolling-window spill patterns where the value is consumed once.
* **LDL `R_dst, [R_offset+UR_base]`.** Indexed local load with per-thread offset (R_offset) and uniform base (UR_base, typically UR7 holding a copy of R1). Used for runtime-indexed access to local arrays.
* **R2UR `UR_dst, R_src`.** Copy per-thread register to uniform register. Observed for R1 (stack pointer) when preparing indexed LDL addressing. Conceptually treats the per-thread value as uniform within the warp.
* **IADD3 `R_dst, PT, PT, R_a, R_b, R_c`.** Three-input integer add with explicit carry predicate slots. The first two PT (predicate true) operands occupy carry-in slots and make the operation a pure 3-input sum without carry propagation. Observed in local array reductions.

### Arithmetic helper patterns

Ptxas handles expensive arithmetic operations via three distinct strategies on SM120 + CUDA 13.2:

1. **External named helper** (rare, observed only for u16 modulo): CALL to a named function like `__cuda_sm20_rem_u16`, body placed elsewhere in the binary.
2. **Local in-kernel subroutine** (for rare slowpaths): `CALL.REL.NOINC` to an address within the same kernel, body placed after the main EXIT.
3. **Full inlining** (dominant strategy): the entire algorithm expanded into the main kernel body, possibly with a BRA branch for an inline slowpath.

**`__cuda_sm20_rem_u16` (runtime modulo, external helper).** 21 instructions implementing Newton-Raphson-like reciprocal multiplication for u16 integer modulo. Structure:

```
Stage 1: integer -> float conversion (I2F.U16 on XU)
Stage 2: reciprocal via MUFU.RCP (XU pipeline)
Stage 3: floating-point quotient via FMUL
Stage 4: float -> int with F2I.U32.TRUNC.NTZ
Stage 5: quotient correction + remainder = a - q * b via IMAD
```

ABI: R8 = dividend, R9 = divisor on entry, R9 = remainder on exit. Return address passed via separate register (R7 or R10 depending on caller's pressure).

**`__cuda_sm20_div_u16` (runtime division, external helper).** Similar structure, same reciprocal primitive, stops at Stage 4 (does not reconstruct remainder).

**Calling convention for external helpers:**
* Caller executes `MOV Rn, return_address` to set up the return address register.
* Caller issues `CALL.REL.NOINC` to the helper.
* Callee reads the return address register early and does `RET.REL.NODEC` at the end.
* u16 ABI requires the caller to mask arguments: `LOP3.LUT R, R, 0xffff, RZ, 0xc0, !PT` truncates to 16 bits before the CALL.

**Inline Granlund-Montgomery for integer division (u32, u64 fast path, s32).** 5-stage reciprocal multiplication, 21 instructions for u32, 30+ for u64 fast path with correction, slightly larger for s32 with sign handling:

```
Stage 1: UI2F.U32.RP (or I2F.RP for signed) - convert divisor to float, round +inf
Stage 2: MUFU.RCP - float reciprocal
Stage 3: IADD with magic constant 0xffffffe - tune F2I rounding
Stage 4: F2I.FTZ.U32.TRUNC.NTZ - back to integer
Stage 5: IMAD.HI.U32 chain for quotient correction; predicated IADD pair for double correction
```

For signed variant: wrapped with IABS (both operands) at entry, LOP3 0x3c (sign XOR) applied to result.

**Inline polynomial math library helpers (log2f, expf, sinf fast path).** Pattern: range reduction (bit manipulation + FFMA) → polynomial evaluation via Horner's method (chain of FFMAs with `.reuse` on the multiplicand) → reconstruction (FADD with exponent, bit manipulation).

Recognizable by:
* Long chains of FFMA with monotonic structure (`.reuse` flag on multiplicand)
* Specific mathematical constants matching known polynomial approximations
* `IADD R, R0, -<bit pattern>` where the bit pattern is that of a mathematical constant (e.g., `-0x3f3504f3 = -sqrt(2)/2`)
* For expf: magic constant `12582913 = 0x00C00001` with FFMA.RM for integer extraction

**Local CALL subroutine (sqrtf slowpath, u64 div slowpath).** Pattern:

```
BSSY.RECONVERGENT B0, <sync_address>       ; establish reconvergence point
<test for slowpath condition>
@!P BRA <fast_path>                         ; if not needed, skip
MOV Rn, <return_address>                    ; set up return (arbitrary register per kernel)
CALL.REL.NOINC <local_slowpath>             ; call
<return_address>: (continuation)
BSYNC.RECONVERGENT B0                       ; reconverge

<main kernel EXIT>

<local_slowpath>:
<handle edge cases>
RET.REL.NODEC Rn 0x0                         ; return via Rn, offset 0
```

The local subroutine is unreachable from normal control flow (placed after EXIT), called only when the slowpath condition is met.

**Subnormal handling pattern (universal for FP MUFU).** Used in 11d, 11e, 11i, 11j and likely in every kernel that uses MUFU on FP32:

```
FSETP.GEU.AND P0, PT, |R|, 1.175494350822287508e-38, PT    ; test |x| >= FLT_MIN
@!P0 FMUL R, R, 16777216                                     ; scale up by 2^24 if subnormal
<MUFU operation>
@!P0 <correction>                                            ; correct for the scaling
```

Correction depends on operation:
* log2: subtract 24 (since log2(2^24 * x) = log2(x) + 24)
* rsqrt: multiply by 2^12 = 4096 (since rsqrt halves the exponent, scale factor is 2^12)
* fdividef: no correction if both operands scaled equally (ratio preserved)

**Payne-Hanek range reduction (sinf, cosf with large arguments).** Triggered when `|x| >= 2^17`. Pattern:

```
LDCU.64 UR, c[0x4][URZ]                    ; load 2/π table base pointer
IMAD.SHL.U32 R, input, 0x100, RZ           ; shift input mantissa for alignment
<6-iteration loop with UIADD3 counter>:
  LDG.E.CONSTANT R, desc[UR][R.64]         ; load chunk of 2/π table
  IMAD.WIDE.U32 R, chunk, mantissa, acc    ; multi-precision accumulate
  @predN MOV R11..R16, partial              ; save partial products
<post-loop combine via SHF.L.W.U32.HI>
I2F.F64.S64 R, reduced                     ; convert to double
DMUL R, R, π/2_FP64                         ; multiply in FP64
F2F.F32.F64 R, R                           ; narrow back to FP32
```

**Signature of different helper categories (updated):**
* **Div/mod u16 external helper**: CALL to `__cuda_sm20_rem_u16` or `__cuda_sm20_div_u16`, 20-30 instructions, uses MUFU.RCP, caller masks with `0xffff` pre-CALL.
* **Div u32/u64/s32 inline**: 21+ instructions with UI2F.U32.RP → MUFU.RCP → magic IADD → F2I → IMAD.HI.U32 chain. No CALL.
* **Math library inline polynomial**: long FFMA chain with `.reuse` on multiplicand, specific coefficients matching known approximations (Remez, Taylor).
* **Transcendental with MUFU direct (intrinsics)**: single MUFU.LG2/EX2/RSQ + subnormal handling. 4-5 instructions.
* **sqrtf/sinf slowpath via local CALL**: BSSY + MOV link + CALL.REL.NOINC + RET.REL.NODEC, body after main EXIT.
* **Payne-Hanek large-argument trig**: LDG.E.CONSTANT table access + UIADD3 loop + IMAD.WIDE.U32 accumulation + FP64 DMUL + F2F.F32.F64.

### Global diagnostic workflow

When opening any SASS dump for performance work:

1. **Skip the prologue** by pattern matching the 8-instruction skeleton.
2. **Check for stack frame allocation**: look for `IADD R1, R1, -<frame>` immediately after prologue. Its presence means the kernel uses local memory; its absence means the kernel runs entirely in registers. Frame size tells you how much local memory is claimed.
3. **Find the bounds check** `@P0 EXIT` and the body section that follows.
4. **Locate the hot region** (backward BRA for a loop body, or the dense compute block).
5. **Count the useful compute ratio** (FFMA + FADD + FMUL + MMA divided by total body instructions).
6. **Grep for artifact signals**:
   * STL / LDL / LDL.LU → register spill (kernel too wide) OR static local array
   * STL.128 → vectorized spill, typically array initialization
   * R2UR on R1 → preparation for indexed local memory reads
   * `LOP3 0xfc` before LDL → alignment mask for indexed local array access
   * CALL to named symbol → external helper (currently only `__cuda_sm20_rem_u16`)
   * CALL to hex offset within kernel → local slowpath subroutine
   * BRA forward followed by BRA back → inline BRA slowpath
   * abnormal size → cascade unrolling, full math library inlining, OR spill-induced bloat (kernel size ×10 possible)
7. **If spill detected, analyze spill pattern**:
   * Rolling window (STL + FFMA + LDL triplet repeated) → pressure-induced spill around compute
   * Burst STL.128 sequence at kernel start → static local array initialization
   * LDL with `[R_idx + UR_base]` → runtime-indexed local array access
8. **Trace scoreboards**: which SB is used, who produces, who waits.
9. **Check stall counts**. Stall=15 repeated signals a problem.
10. **Verify fusion**. Count FFMAs vs separate FMUL+FADD chains.
11. **Look for NOP padding within the body** (FP64 bottleneck on consumer SM120).
12. **Identify arithmetic patterns** via their signatures:
    * UI2F.U32.RP + MUFU.RCP → inline integer division/modulo
    * FSETP.GEU vs FLT_MIN + FMUL 2^24 → subnormal handling on MUFU input
    * Long FFMA chain with `.reuse` on multiplicand → polynomial evaluation (check coefficients)
    * LDG.E.CONSTANT in a counter loop → table-based algorithm (Payne-Hanek or similar)
    * IADD with magic `0x3f3504f3` or similar FP32 bit pattern → mathematical constant subtraction
    * BSSY/BSYNC with short body → branch-kept divergent region, divergence before warp-synchronous op, OR slowpath/local-call branch
13. **Check ptxas warnings** in the compile output. `adjusting per thread register count of N to lower bound of 24` confirms SM120 register floor was hit.
14. **Correlate with NCU**. SASS identifies the "who", NCU quantifies the "how much". For spilled kernels, check `stall_lg_throttle` and local memory metrics in NCU.

## Audit gaps identified from production kernel attempt

During an attempt to audit a production fused FP4 attention kernel on SM120, several gaps in the current findings were identified that prevent reliable end-to-end kernel audit. Listed here for future chapter planning.

### [GAP-audit-1] C++ loop to SASS mapping is not understood

* [OBS] A kernel with constant-trip-count nested loops (N_TILES=8, K_TILES=2) should produce 16 static MMAs if fully unrolled.
* [OBS] The observed SASS had 2 QMMAs only, with zero BRA back-edges.
* [OBS] This outcome is inconsistent with both "fully unrolled" and "loop preserved" interpretations.
* [HYP] Either the binary was stale (compiled from an earlier version of source), or ptxas applied an optimization we do not recognize, or our model of unroll behavior is wrong.
* [RES] Chapter 20 resolves the minimal-loop part of this gap for tested scalar and HMMA loops: constant 8 x 2 nested HMMA loops fully unroll and emit 16 static HMMAs.
* [GAP] The original production 2-QMMA observation remains unresolved until the exact source/binary pair is rebuilt and compared. The remaining explanations are stale binary, different template specialization, dead-code elimination around unused MMA results, or a production-specific transformation not reproduced by Chapter 20.

### [GAP-audit-2] Loop detection without BRA back-edge is not documented

* [OBS] All chapters so far assumed loops always emit a BRA back-edge (target < source) at the bottom of the loop body.
* [OBS] In the FP4 attention SASS, no back-edge was found in a kernel that must have dynamic loops at the C++ level (`seq_tile` depends on runtime `seq_k`).
* [RES] The tested form of "Blackwell uses BSSY/BSYNC for ordinary loop structures" is rejected by Chapter 20. [OBS] Chapter 21 shows BSSY/BSYNC for lane-dependent trip-count and break loops, so the refined rule is: ordinary loops do not require BSSY/BSYNC in tested variants, while non-uniform loop exits/trip counts can introduce it.
* [HYP] Or the compiler specialized the binary with constant folding of dynamic parameters.
* [RES] Chapter 20 finds no SASS-level loop without a back-edge in the tested variants. Preserved constant loops, dynamic-loop cascades, `break`, `continue`, and unroll-pragmas all expose at least one backward `BRA` or `BRA.U`.
* [RES] Chapter 20 rejects the tested form of "BSSY/BSYNC as ordinary loop encoding": ordinary loops do not need BSSY/BSYNC. The `break` variant can emit BSSY/BSYNC because it has a non-structured loop exit.
* [GAP] This remains open only for untested production-specific transformations beyond the Chapter 21 divergence matrix.

### [RES-audit-3] Divergence and predication patterns now have first-pass coverage

* [OBS] 269 forward BRAs observed in the FP4 attention kernel, organized in groups of 4 pointing to the same target, with stride 0x30.
* [HYP] These correspond to unrolled `if/else if/else if/...` chains from FP4 encoding (8 levels), processed in parallel.
* [RES] Chapter 21 covers first-pass predication vs branching, divergence signatures, and reconvergence point identification across 20 controlled variants.
* [OBS] Chapter 21 shows lane divergence alone does not force visible BSSY/BSYNC in every tested SM120 SASS form.
* [OBS] Chapter 21 shows ptxas can use predicated arithmetic, FSEL/UFSEL, predicated EXIT, predicated stores, forward branches, BSSY/BSYNC, VOTE, SHFL, WARPSYNC.ALL, and local CALL depending on source shape.
* [GAP] Production FP4 attention branch groups still require source/binary-specific audit; Chapter 21 provides the grammar but does not classify that stale production dump end-to-end.

### [GAP-audit-4] Thread vs warp vs block scope is not explicit in any chapter

* [OBS] When counting QMMAs in a SASS dump, the interpretation depends on scope: is this the code executed by 1 thread, 1 warp, or 1 block?
* [HYP] The answer is "1 thread executing warp-level instructions," meaning the same instruction is executed simultaneously by all 32 threads in the warp.
* [RES] Chapter 20 partially resolves the audit-counting part: SASS instruction counts are per static function body, and warp-level instructions such as HMMA appear once per static MMA site in that function body. A nested 8 x 2 HMMA source emits 16 HMMA instructions, not 16 x 32 thread-local instruction sites.
* [RES] Chapter 21 adds first-pass coverage for divergent warp-level instructions and guarded HMMA. [GAP] Full scope teaching remains incomplete for `WARPSYNC.ALL`, non-full masks, and runtime behavior of predicated HMMA.

### [GAP-audit-5] Template specialization effects not covered

* [OBS] The kernel was template `<int HEAD_DIM>` with two instantiations (64, 128) visible in the SASS dump.
* [OBS] Other constants (N_TILES, K_TILES) are computed from HEAD_DIM but not template parameters.
* [RES] Chapter 20 partially covers specialization visibility: 20t emits two separate SASS functions for two template instantiations in one dump.
* [GAP] The effect of production template constants on complete attention/GEMM loop structure remains open until a production-style templated kernel is rebuilt and audited end-to-end.

### [GAP-audit-6] No documented audit methodology

* [OBS] Chapters 01-19 document opcodes and patterns in isolation.
* [GAP] There is no written methodology for auditing a production kernel end-to-end: how to identify kernel sections, how to map C++ source to SASS regions, how to interpret instruction densities, how to validate hypotheses about kernel behavior.

### [GAP-audit-7] No confidence qualification framework

* [OBS] During the FP4 attention audit attempt, hypotheses produced were inconsistently confident, and some turned out to be wrong without this being flagged.
* [GAP] The [OBS]/[HYP]/[RES]/[GAP] tagging is at the claim level, but there is no framework for qualifying confidence at the audit level (e.g. "this audit conclusion depends on N untested assumptions").

### Plan to close the gaps

* [RES] Chapter 20 is complete for control flow, loops, unroll behavior, back-edge detection, and local predication vs branching. It closes the minimal-loop portion of GAP-audit-1, resolves tested loop-detection cases in GAP-audit-2, and partially resolves GAP-audit-4 and GAP-audit-5.
* [RES] Chapter 21 is complete for first-pass divergence and reconvergence coverage, including BSSY/BSYNC, warp-divergent branches, predicated arithmetic, predicated exits, VOTE, SHFL, local CALL, and guarded HMMA/WARPSYNC.ALL.
* [RES] Chapter 22 is complete for first-pass stmatrix / matrix-store SASS coverage and runtime smoke execution. It closes the matrix-store gap left by chapters 17 and 18 for tested m8n8 b16 forms.
* [GAP] Chapter 22 full STSM lane-to-shared layout decode remains open.
* [RES] Chapter 23 is complete for first-pass FP4 / FP6 fragment-layout SASS coverage and runtime smoke execution, including dense QMMA, LDSM-fed QMMA, direct-register QMMA, scale-vector OMMA, and sparse metadata separation.
* [GAP] Chapter 23 full runtime value-layout decode remains open; GAP-14d-1 and the FP6/FP4 packing gaps from Chapter 15 are reduced but not fully closed.
* [RES] Chapter 24 is complete for first-pass production mini-GEMM SASS coverage and runtime smoke execution, including LDGSTS, LDSM, HMMA/QMMA/OMMA, sparse MMA, STSM, STG, REDG, scale-load, metadata-load, and cold-path probes. It reduces GAP-audit-6 by documenting an end-to-end dump segmentation workflow.
* [GAP] Chapter 24 does not fully close GAP-audit-7 because audit-confidence scoring still needs a written framework beyond the structural probe set.
* [RES] Chapter 25 is complete for first-pass STSM epilogue/storeback SASS coverage and runtime smoke execution, including STSM layout variants, STS fallback, MMA-to-STSM paths, F16/BF16 narrowing, barrier/no-barrier contrast, split accumulator storeback, noncontiguous global stores, and register-pressure behavior.
* [GAP] Chapter 25 full lane-to-value STSM semantic decode remains open over the captured runtime words.

### Decision: Phase 3 gated

* [RES] Required Phase 3 gates from Chapters 20, 21, and 22 are complete.
* [RES] The strongly recommended Chapter 23, 24, and 25 structural chapters are complete for first-pass SASS coverage and runtime smoke execution.
* [INF] An audit confidence framework remains strongly recommended before Phase 3 pattern formalization because the structural chapters do not define how to score confidence for production-kernel conclusions.
