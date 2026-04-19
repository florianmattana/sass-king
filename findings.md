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

* [HYP] Why HFMA2 specifically versus MOV or IMAD for constant materialization. Three candidate reasons (pipeline affinity with consuming FFMAs, compact encoding, hardcoded heuristic), none directly measured.
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
* [HYP] `HFMA2 R3, -RZ, RZ, 15, 0` and the subnormal FSETP in the modulo helper handle edge cases of the reciprocal multiplication. Role not fully worked out.
* [HYP] The return address register choice (R7 in kernel 06, R10 in kernel 06i) depends on caller register pressure at the call site. To test with a caller under low pressure.
* [HYP] Why the modulo function has a symbolic label (`__cuda_sm20_rem_u16`) in the dump but the division function does not.

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

## Cross chapter summary

### Pipelines observed so far

| Pipeline | Instructions observed |
|---|---|
| FMA | FFMA, FADD, FMUL, IMAD, HFMA2 |
| ALU | ISETP, FSETP, MOV, LEA, LOP3, FSEL, SHF, SEL |
| LSU | LDG, LDG.E.64, LDG.E.128, LDG.E.ENL2.256, STG, STG.E.64, STG.E.128, STG.E.ENL2.256, LDS, STS, SHFL.BFLY, SHFL.IDX, SHFL.UP, SHFL.DOWN, MATCH.ANY, MATCH.ALL |
| ADU | LDC, S2R, BAR.SYNC |
| DCC | LDCU |
| UNIFORM | S2UR, UMOV, ULEA, UIADD3, UISETP, UPLOP3, ULOP3 |
| XU | MUFU.RCP, I2F, F2I (first observed in kernel 06 modulo helper) |
| CBU | EXIT, BRA, CALL, RET, BSSY, BSYNC |
| FP64 | DADD (first observed in kernel 08f) |
| VOTE | VOTE.ANY, VOTE.ALL |

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

Exception: the addend of an FFMA can be an immediate directly (e.g., `FFMA R, R, R, 0.5`).

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

The `UMOV UR, 0x400` + `ULEA ..., 0x18` pair appears if and only if the kernel uses `__shared__` memory. The `0x18 = 24` in ULEA is a shift amount, producing a per-block offset within the SM-wide shared pool.

### Arithmetic operator compilation rules

| Source | ptxas strategy |
|---|---|
| `%` by runtime variable | CALL `__cuda_sm20_rem_u16` (expensive) |
| `%` by non-power-of-2 constant | Inline reciprocal multiplication with magic number |
| `%` by power-of-2 constant | Fused into LEA + LOP3 mask (cheap) |
| `/` by runtime variable | CALL to division helper |
| `/` by power-of-2 constant | SHF.R.U32.HI (shift right) |
| `*` by power-of-2 constant | SHF.L.U32 (shift left) |
| `+`, `*+` in FP | FFMA when `a*b+c` pattern present, else separate FMUL / FADD |
| FP constant as multiplier | HFMA2 materialization into register |
| FP constant as addend | Immediate in FFMA src3 slot |

### Scheduling patterns

ptxas systematically places independent instructions between a producer and its consumer to hide latency. Observed patterns:

* **IMAD.WIDE interleaved with LDG.** Address computations are placed between consecutive LDG emissions so that multiple loads are in flight simultaneously.
* **Pointer address of the store placed early.** When the value is the bottleneck, the store address is computed while the value is still being produced.
* **Pointer address of the store placed late.** When the value chain is long (many sequential FFMAs), the store address is deferred to minimize register live range.
* **Constant materialization (HFMA2) placed at the pipeline choice point.** HFMA2 is used in compute-heavy blocks where the FMA pipeline is loaded, MOV is used otherwise.
* **VOTE.ANY hoisted to kernel start** when its input is `PT` (as in `__activemask()`). No data dependency, so it can run in parallel with pointer loads.
* **Counter update interleaved with compute in unrolled loops.** UIADD3 and UISETP for loop control are placed between FFMAs, not at the tail.

### Compiler artifacts to watch for

* **STL / LDL** (not yet observed in our kernels) would indicate register spilling. Signal of a kernel too wide for the register file.
* **CALL** indicates an out-of-line function: division/modulo slowpath, transcendental math, non-inlined helper.
* **Kernel size significantly larger than expected** often means cascade unrolling (runtime trip count).
* **BSSY / BSYNC.RECONVERGENT** wrapping a short section means ptxas detected divergence before a warp-synchronous operation.
* **Consecutive NOPs between arithmetic instructions** mean the pipeline cannot keep up (FP64 on consumer parts, or ILP shortage).

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
* **`.TRUNC.NTZ`** (appears on F2I): truncation with non-toward-zero rounding for negative zero edge case.

Operand-level flags (not opcode suffixes but worth distinguishing):

* **`.reuse`** on a source operand: hint to the hardware to cache that value in the reuse cache.
* **`.64`** on a register name (e.g., `R2.64`): treats R2:R3 as a 64-bit register pair.
* **`.H0`, `.H1`** on a register name: selects the low or high 16 bits of a packed half-precision register.

Modifiers we have not yet encountered in our dumps but that appear in the Blackwell ISA reference are not covered here.

### Global diagnostic workflow

When opening any SASS dump for performance work:

1. Skip the prologue by pattern matching the 8-instruction skeleton.
2. Find the bounds check `@P0 EXIT` and the body section that follows.
3. Locate the hot region (backward BRA for a loop body, or the dense compute block).
4. Count the useful compute ratio (FFMA + FADD + FMUL + MMA divided by total body instructions).
5. Grep for artifact signals: STL/LDL (spill), CALL (slowpath), abnormal size (cascade).
6. Trace scoreboards: which SB is used, who produces, who waits.
7. Check stall counts. Stall=15 repeated signals a problem.
8. Verify fusion. Count FFMAs vs separate FMUL+FADD chains.
9. Look for NOP padding within the body (FP64 bottleneck).
10. Correlate with NCU. SASS identifies the "who", NCU quantifies the "how much".
