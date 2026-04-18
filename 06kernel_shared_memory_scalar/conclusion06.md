# Kernel 06 — vector_smem (shared memory scalar)

First kernel using shared memory. Source performs a round trip `global to shared to shared to global` with an index shift to prevent the compiler from optimizing the shared path away.

## Source

```cuda
__global__ void vector_smem(const float* a, float* c, int n) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % blockDim.x;
        c[i] = smem[src];
    }
}
```

Launch: `<<<4, 256>>>` with n=1024.

## SASS main body structure (30 instructions)

Follows the 6 section skeleton from earlier kernels, then adds:

```
[Prologue: 8 instructions]
  Stack pointer, ID setup, index computation, bounds check.
  Identical to kernels 01-05.

[Global load a[i]: 4 instructions]
  LDC.64 pointer a, LDCU.64 global descriptor, IMAD.WIDE address, LDG.E load.
  Identical to earlier kernels.

[New: shared base and descriptor setup: 4 instructions]
  S2UR UR5, SR_CgaCtaId
  UMOV UR4, 0x400
  ULEA UR4, UR5, UR4, 0x18

  UR4 becomes the shared memory base for the kernel. Construction not fully
  understood yet (see open hypotheses).

[New: shared write path: 3 instructions]
  LEA R7, R0, UR4, 0x2        ; &smem[tid] = UR4 + tid*4
  STS [R7], R2                 ; smem[tid] = a[i]
  BAR.SYNC.DEFER_BLOCKING 0x0  ; __syncthreads()

[Modulo slowpath: 1 CALL into a 21 instruction function]
  MOV R7, 0x170
  CALL.REL.NOINC __cuda_sm20_rem_u16

[Shared read path + global store: 4 instructions]
  LEA R0, R0, UR4, 0x2        ; &smem[src]
  LDS R7, [R0]                 ; smem[src]
  IMAD.WIDE R2, R5, 0x4, R2    ; &c[i]
  STG.E desc[UR6][R2.64], R7   ; c[i] = smem[src]

[Exit + modulo function placed after: ~25 instructions]
```

## Solid observations (verified across variants 06b-06i)

### 1. Shared memory: addressing architecture

**One UMOV plus derivations by addition for N shared buffers.**
Verified in kernel 06g with two `__shared__` arrays: a single `UMOV UR4, 0x400` followed by `UIADD3 UR5, UPT, UPT, UR4, 0x400, URZ` to derive the base of the second buffer.

**Shared buffers are placed consecutively in memory.**
The offset between two buffers `smem_a[256]` and `smem_b[256]` is exactly `0x400 = 1024 bytes`, which is the size of `smem_a` in bytes. No automatic padding inserted between them.

**Shared addressing does not use a descriptor.**
Unlike LDG/STG which use `desc[UR][R.64]`, shared accesses use direct `[R]` or `[R+UR]`. The shared base is encoded in the address register itself.

**`[R+UR]` mode for reading multiple buffers at the same position.**
Observed in kernel 06g: `LDS R8, [R6+UR4]` and `LDS R11, [R6+UR5]`. A single per-thread address computation (R6 after modulo mask), two uniform offsets for the two buffers. Optimal pattern for multi buffer reads.

**Consecutive STS can be emitted without intermediate BAR.**
Observed in kernel 06g: two STS back to back, a single BAR.SYNC after both. One `__syncthreads()` covers all preceding stores in the block.

### 2. Integer modulo: three distinct modes based on the divisor

| Source | ptxas strategy | Instructions for modulo |
|---|---|---|
| `% blockDim.x` (runtime) | CALL `__cuda_sm20_rem_u16` + 21 instruction function | 1 CALL + body |
| `% 255` (non power of 2) | Inline reciprocal multiplication (magic number) | ~10 inline |
| `% 256` (power of 2) | Fused into LEA + LOP3 mask | 2 instructions |
| `& 255` | **Identical to `% 256`** (byte exact SASS) | 2 instructions |

ptxas recognizes three distinct patterns at compile time and chooses the adapted strategy. The byte exact match proves that ptxas canonicalizes `% 2^k` and `& (2^k - 1)` to the same internal representation.

### 3. Modulo function: characteristics

**Single instance of the function shared across N calls (kernel 06i).**
Two modulos with different runtime divisors in the same kernel produce two CALLs targeting the same function address. Binary size stays constant regardless of the number of calls.

**Observed ABI for `__cuda_sm20_rem_u16`.**
- R8: u16 dividend input
- R9: u16 divisor input, remainder output
- Rn (R7 or R10 depending on kernel): return address, written by the caller via MOV before CALL

**Runtime cost is not amortized.**
Every call executes the full function (~30 cycles minimum, including MUFU.RCP on the XU pipeline). N calls mean N times this cost.

**Integer division: separate function with similar body (kernel 06h).**
Division and modulo have two distinct functions sharing the same primitive (reciprocal multiplication with magic number), differing only in the final step. Division stops at the quotient. Modulo computes the quotient then reconstructs the remainder via IMAD.

**Practical consequence.**
Using `/` and `%` in the same kernel on the same operands produces two separate CALLs. To save one CALL, write `q = a / b; r = a - q * b;` instead of `q = a / b; r = a % b;`.

### 4. CALL/RET mechanism: manual link register

**CALL.REL.NOINC and RET.REL.NODEC.**
The hardware does not auto manage the link register. The caller writes the return address into a register (MOV Rn, 0x170 before CALL) and the callee copies this value to its return register (MOV R2, Rn) before RET.

This mechanism lets ptxas handle multiple calls to the same function with different return addresses without a runtime library or automatic stack pointer.

### 5. u16 masking around CALL

**Before the CALL to `__cuda_sm20_rem_u16`.**
```
LOP3.LUT Rn, Rn, 0xffff, RZ, 0xc0, !PT
```
Truncates the operand to 16 bits. The function has a u16 ABI (indicated by `_u16` in its name), so arguments must respect this format.

The mask is not defensive or arbitrary, it is **mandatory** by the ABI.

### 6. BAR.SYNC on SM120

**`.DEFER_BLOCKING` modifier observed.**
```
BAR.SYNC.DEFER_BLOCKING 0x0
```

**Pipeline: ADU** (observed in gpuasm, contrary to the CBU intuition).

**Argument 0x0 is the barrier number.**
CUDA exposes 16 hardware barriers per block. `__syncthreads()` always uses barrier 0.

### 7. First XU pipeline usage (modulo function)

The modulo function uses MUFU.RCP (reciprocal), which runs on the **XU** pipeline (Transcendental Unit). This is the first XU usage in our kernels.

I2F and F2I conversions observed in the same function, also on XU.

### 8. New special register: SR_CgaCtaId

Accessible via S2UR, used in the shared base construction. CGA stands for Cooperative Grid Array (SM90+ cluster terminology).

## UMOV UR4, 0x400 test: three hypotheses rejected

Three variants tested to identify what `0x400` encodes:

| Variant | Shared size | Block size | Modulo | UMOV observed |
|---|---|---|---|---|
| 06b | 256 floats (1024 B) | 256 | 256 | **0x400** |
| 06e | 128 floats (512 B) | 128 | 128 | **0x400** |
| 06f | 512 floats (2048 B) | 512 | 512 | **0x400** |

**Rejected.** `0x400` does not encode shared size, block size, or launch config.

**What does change with size.** The LOP3 mask:
- 256 floats → 0x3fc = 1020 = (256-1) × 4
- 128 floats → 0x1fc = 508 = (128-1) × 4
- 512 floats → 0x7fc = 2044 = (512-1) × 4

**Conclusion.** `0x400` appears to be an architectural constant or a parameter of the shared descriptor format, independent of kernel parameters. To be revisited in kernel 07.

## Open hypotheses at the end of chapter 06

1. **Meaning of `UMOV UR4, 0x400`.** Likely architectural constant, exact role unknown.
2. **Division function lacks a symbolic label in the dump.** Why modulo is named and division is not.
3. **SR_CgaCtaId in a kernel without clusters.** Fallback on a default ID or implicit block ID.
4. **`.DEFER_BLOCKING` on BAR.SYNC.** New SM90+ modifier or inherited.
5. **PRMT with `0x9910` and `0x7710` in the `% 255` function.** Byte level manipulation not analyzed.
6. **`HFMA2 R3, -RZ, RZ, 15, 0` sequence and subnormal FSETP.** Exact role in the reciprocal multiplication algorithm.
7. **Return address register choice** in the calling convention (R7 vs R10 across kernels).

## New instructions observed in this chapter

Opcodes appearing for the first time in our dumps:

| Opcode | Usage |
|---|---|
| STS | Store to Shared |
| LDS | Load from Shared |
| BAR.SYNC.DEFER_BLOCKING | Block level barrier |
| UMOV | Uniform MOV (immediate materialization into UR) |
| ULEA | Uniform Load Effective Address |
| UIADD3 | Uniform Integer 3 input Add |
| LOP3.LUT | 3 input Logic Operation with LUT |
| LEA | Load Effective Address (per thread) |
| CALL.REL.NOINC | Function call with manual link register |
| RET.REL.NODEC | Function return with manual link register |
| MUFU.RCP | Reciprocal (XU pipeline) |
| I2F.U16 | Integer to Float (conversion, XU pipeline) |
| F2I.U32.TRUNC.NTZ | Float to Integer (conversion, XU pipeline) |
| FSETP.GEU.AND | FP compare predicate with combination |
| FSEL | FP select (ternary) |
| PRMT | Byte permutation |
| SHF.R.U32.HI | Funnel shift right, high half |
| S2R / S2UR for SR_CgaCtaId | New special register |

## What this chapter adds to SASS reading skills

**Recognize shared memory at a glance.**
STS, LDS, BAR.SYNC, and the characteristic `UMOV + ULEA` construction form an identifiable visual pattern.

**Diagnose an arithmetic slowpath.**
A CALL in a kernel that should have none signals runtime division/modulo, math library, or a non inlinable function. Looking for the MOV setup right before and the function placed after EXIT confirms the pattern.

**Choose the right source form.**
- Modulo/division by a power of 2 constant: optimal, no hidden cost.
- Modulo/division by a non power of 2 constant: moderate inline cost.
- Modulo/division by a runtime variable: expensive, external CALL.

**Predict shared layout.**
Multiple `__shared__` arrays stack in declaration order. No automatic padding. No linear setup overhead for multiple buffers.

**Group `__syncthreads()` calls.**
Several consecutive shared stores only need one final BAR. ptxas does not fuse barriers, the programmer has to write them efficiently.
