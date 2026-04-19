# Findings and open hypotheses

Running log of observations and hypotheses, organized by kernel chapter.

Notation:
* **[OBS]** verified observation from a SASS dump
* **[HYP]** open hypothesis, to be tested
* **[RES]** resolved hypothesis (rejected or confirmed)

---

## Kernel 01 vector_add (baseline)

### Observations

* [OBS] Every kernel has a 6 section prologue: stack pointer init, thread and block ID setup, argument loads, index computation, bounds check, global descriptor load.
* [OBS] `R1` is the stack pointer by ABI convention, loaded from `c[0x0][0x37c]`, initialized even when no spilling occurs.
* [OBS] Kernel arguments and launch parameters live in constant memory bank 0 at fixed offsets (`c[0x0][...]`).
* [OBS] `LDC` loads into a per thread register, `LDCU` into a uniform register. ptxas classifies automatically based on data flow.
* [OBS] Per thread registers R0 to R255 versus uniform registers UR0 to UR63 (or UR0 to UR255 on SM100+) are two physically separate register files.
* [OBS] Values shared across the warp (blockIdx, kernel args, n) go into UR. Values unique per thread (threadIdx, computed indices, loaded data) go into R.
* [OBS] Global memory accesses on SM120 use descriptor based addressing: `desc[UR][R.64]`. The descriptor is loaded once from `c[0x0][0x358]` and shared across all global accesses.
* [OBS] `IMAD.WIDE` produces a 64 bit result (register pair) from a 32 bit multiply. Used for pointer arithmetic.
* [OBS] Bounds check uses predication (`@P0 EXIT`), not a branch. Zero divergence cost.
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

* [RES] `UMOV UR4, 0x400` appears if and only if the kernel uses shared memory. The value 0x400 is inert across shared size, shared type, static vs dynamic declaration, and number of shared buffers. **It is an architectural constant tied to the shared addressing mechanism on SM120.** Exact semantic meaning of the value itself is still unknown.
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
* [HYP] Does `WARPSYNC` opcode exist at all on SM120, or is it fully replaced by NOP padding for all mask variants?

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

## Cross chapter summary

### Pipelines observed so far

| Pipeline | Instructions observed |
|---|---|
| FMA | FFMA, FFMA.SAT, FFMA.RM, FADD, FADD.FTZ, FMUL, FMUL.FTZ, IMAD, IMAD.U32, IMAD.WIDE, IMAD.WIDE.U32, IMAD.HI.U32, IMAD.SHL.U32, HFMA2 |
| ALU | ISETP, ISETP.GE.U64.AND, ISETP.NE.S64.AND, FSETP, FSETP.GEU, FSETP.NEU, FSETP.GTU, FSETP.GEU.FTZ, FSETP.GTU.FTZ, FSETP.NEU.FTZ, MOV, MOV.64, LEA, LEA.HI, LOP3, FSEL, SHF, SHF.L.W.U32.HI, SHF.R.U32.HI, SHF.R.S32.HI, SEL, SEL.64, IABS, IADD, IADD.X, IADD3, IADD3.X, IADD.64, R2P |
| LSU | LDG, LDG.E.64, LDG.E.128, LDG.E.ENL2.256, LDG.E.CONSTANT, STG, STG.E.64, STG.E.128, STG.E.ENL2.256, LDS, STS, SHFL.BFLY, SHFL.IDX, SHFL.UP, SHFL.DOWN, MATCH.ANY, MATCH.ALL |
| ADU | LDC, LDC.64, S2R, BAR.SYNC |
| DCC | LDCU, LDCU.64 |
| UNIFORM | S2UR, UMOV, UMOV.64, ULEA, UIADD3, UISETP, UISETP.NE.U32.AND, UPLOP3, ULOP3 |
| XU | MUFU.RCP, MUFU.LG2, MUFU.EX2, MUFU.RSQ, I2F, I2F.F64.S64, I2F.U64.RP, I2F.RP, I2FP.F32.S32, UI2F.U32.RP, F2I, F2I.FTZ.U32.TRUNC.NTZ, F2I.U64.TRUNC, F2I.NTZ, F2F.F32.F64 |
| CBU | EXIT, BRA, BRA.U, CALL, CALL.REL.NOINC, RET.REL.NODEC, BSSY, BSSY.RECONVERGENT, BSYNC, BSYNC.RECONVERGENT |
| FP64 | DADD (first observed in kernel 08f), DMUL (observed in kernel 11g) |
| VOTE | VOTE.ANY, VOTE.ALL |
| REDUX | REDUX, REDUX.OR, REDUX.XOR, REDUX.SUM, REDUX.MIN, REDUX.MAX |

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

### Compiler artifacts to watch for

* **STL / LDL** (not yet observed in our kernels) would indicate register spilling. Signal of a kernel too wide for the register file.
* **CALL.REL.NOINC to a named symbol** indicates an external out-of-line helper. On SM120/CUDA 13, only observed for `__cuda_sm20_rem_u16` and `__cuda_sm20_div_u16` (sub-word integer modulo/division).
* **CALL.REL.NOINC to a local address** (hex offset within same kernel) indicates an inline-but-out-of-hot-path subroutine. The body is placed after the main EXIT. Used for rare slowpaths (sqrtf NaN handling, u64 division high-bits).
* **BRA to a forward address within the main body** indicates an inline slowpath that doesn't warrant a subroutine (sinf Payne-Hanek, expf non-fast-path). Control returns naturally via fall-through.
* **Kernel size significantly larger than expected** often means cascade unrolling (runtime trip count) OR full inlining of math library functions (log2f, sinf).
* **BSSY / BSYNC.RECONVERGENT** wrapping a short section means ptxas detected divergence before a warp-synchronous operation OR a divergent slowpath.
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
* **`.RECONVERGENT`** (appears on BSSY and BSYNC): Independent Thread Scheduling reconvergence scope. Wraps warp-synchronous instructions that follow divergent code.
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
2. **Find the bounds check** `@P0 EXIT` and the body section that follows.
3. **Locate the hot region** (backward BRA for a loop body, or the dense compute block).
4. **Count the useful compute ratio** (FFMA + FADD + FMUL + MMA divided by total body instructions).
5. **Grep for artifact signals**:
   * STL / LDL → register spill (kernel too wide)
   * CALL to named symbol → external helper (currently only `__cuda_sm20_rem_u16`)
   * CALL to hex offset within kernel → local slowpath subroutine
   * BRA forward followed by BRA back → inline BRA slowpath
   * abnormal size → cascade unrolling or full math library inlining
6. **Trace scoreboards**: which SB is used, who produces, who waits.
7. **Check stall counts**. Stall=15 repeated signals a problem.
8. **Verify fusion**. Count FFMAs vs separate FMUL+FADD chains.
9. **Look for NOP padding within the body** (FP64 bottleneck on consumer SM120).
10. **Identify arithmetic patterns** via their signatures:
    * UI2F.U32.RP + MUFU.RCP → inline integer division/modulo
    * FSETP.GEU vs FLT_MIN + FMUL 2^24 → subnormal handling on MUFU input
    * Long FFMA chain with `.reuse` on multiplicand → polynomial evaluation (check coefficients)
    * LDG.E.CONSTANT in a counter loop → table-based algorithm (Payne-Hanek or similar)
    * IADD with magic `0x3f3504f3` or similar FP32 bit pattern → mathematical constant subtraction
    * BSSY/BSYNC with short body → divergence before warp-synchronous op OR slowpath branch
11. **Correlate with NCU**. SASS identifies the "who", NCU quantifies the "how much".
