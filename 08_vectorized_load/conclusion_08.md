# Kernel 08 vector_load (vectorized global memory access)

First kernel chapter using vectorized load and store instructions. The natural continuation from kernel 01 (scalar float) to the wider memory transaction widths available on SM120. Seven variants tested to map the full LDG width spectrum and identify the rules that govern width selection.

## Source (baseline 08a)

```cuda
__global__ void vector4_add(const float4* a, const float4* b, float4* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 va = a[i];
        float4 vb = b[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;
    }
}
```

Delta from kernel 01: scalar `float*` replaced by `float4*`. Each thread now processes 16 bytes instead of 4.

## SASS structure (08a, 22 instructions)

```
[Prologue: 8 instructions]
  Identical to kernels 01 to 06.

[Pointer loads: 3 instructions]
  LDC.64 R2, c[0x0][0x380]         ; pointer a
  LDC.64 R4, c[0x0][0x388]         ; pointer b
  LDC.64 R6, c[0x0][0x390]         ; pointer c
  plus LDCU.64 UR4, c[0x0][0x358]  ; global descriptor

[Address arithmetic and loads: 5 instructions]
  IMAD.WIDE R2, R9, 0x10, R2       ; stride 16 = sizeof(float4)
  LDG.E.128 R12, desc[UR4][R2.64]  ; 128-bit load into R12:R13:R14:R15
  IMAD.WIDE R4, R9, 0x10, R4
  LDG.E.128 R16, desc[UR4][R4.64]  ; into R16:R17:R18:R19
  IMAD.WIDE R6, R9, 0x10, R6

[Compute: 4 instructions]
  FADD R12, R12, R16   ; x
  FADD R13, R13, R17   ; y
  FADD R14, R14, R18   ; z
  FADD R15, R15, R19   ; w

[Store and exit: 2 instructions]
  STG.E.128 desc[UR4][R6.64], R12
  EXIT
```

## Solid observations (verified across variants 08a to 08g)

### 1. Load width spectrum on SM120

Five distinct load widths confirmed, all gated by the source-level type:

| Type | Bytes per thread | Alignment | Load instruction | Width |
|---|---|---|---|---|
| float | 4 | 4 | LDG.E | 32 bits |
| float2 | 8 | 8 | LDG.E.64 | 64 bits |
| float4 | 16 | 16 | LDG.E.128 | 128 bits |
| float8 | 32 | 32 | LDG.E.ENL2.256 | 256 bits |
| float16 | 64 | 64 | 2 × LDG.E.ENL2.256 | 256 bits × 2 |

256 bits is the maximum width of a single LDG on SM120. Above 32 bytes, ptxas splits into multiple 256-bit loads, never into wider instructions.

### 2. The `.ENL2` modifier appears at 256 bits

At 32, 64, and 128 bit widths, one destination register names the load target and ptxas uses the following N consecutive registers implicitly.

At 256 bits, the instruction encoding requires two destination registers:

```
LDG.E.ENL2.256 R16, R12, desc[UR4][R2.64]
               ^^^  ^^^
               base of second half, base of first half
```

The 8 target registers are split into two blocks of 4, each named by one of the two destination operands. Hypothesis: the `.ENL2` suffix signals "enlarged encoding using 2 register base fields" and exists because a single 4-bit or 5-bit register field cannot reference 8 consecutive registers in a single opcode slot.

### 3. IMAD.WIDE stride encodes sizeof(T)

The immediate operand of `IMAD.WIDE` matches the size of the element type in bytes:

| Type | sizeof(T) | IMAD.WIDE stride |
|---|---|---|
| float | 4 | 0x4 |
| float2 | 8 | 0x8 |
| float4 | 16 | 0x10 |
| float8 | 32 | 0x20 |
| float16 | 64 | 0x40 |
| double4_32a | 32 | 0x20 |

This observation holds across all variants and across kernels 01 to 08.

### 4. Alignment of the source type determines usable load width

Variant 08f used the deprecated `double4` (which has 16-byte alignment by default) and produced 4 × LDG.E.128 instead of 2 × LDG.E.ENL2.256, despite matching the 32-byte size of float8. Variant 08g used `double4_32a` (explicit 32-byte alignment) and produced the expected 1 × LDG.E.ENL2.256.

Conclusion: the load width is not determined by element type width (32 vs 64 bit) but by the alignment guaranteed by the type. For 256-bit loads, ptxas requires 32-byte alignment. For 128-bit loads, 16-byte alignment.

Practical consequence for kernel writers: when using doubles or other 64-bit elements, always use the `_16a` or `_32a` suffixed types (introduced in CUDA 13.x). The legacy unsized vector types have insufficient alignment guarantees and block the maximum vectorization.

### 5. ptxas does not auto-vectorize scalar access patterns

Variant 08b used `const float*` with manual indexing `a[base + 0]`, `a[base + 1]`, ..., `a[base + 3]` where `base = i * 4`. The source guarantees 16-byte alignment through the multiplication, and the accesses are trivially contiguous. ptxas emitted 8 separate LDG.E instructions (4 for each array) and 4 separate STG.E, not the 2 × LDG.E.128 and 1 × STG.E.128 that would match variant 08a.

ptxas did factor the address arithmetic (one IMAD.WIDE per array, with immediate offsets `+0x4`, `+0x8`, `+0xc` on the LDG instructions), but it did not fuse the scalar loads into a single wide transaction.

Conclusion: vectorization on SM120 is syntactic. Writing `float*` with contiguous access produces scalar SASS regardless of alignment or access pattern. To obtain wide LDG instructions, the source must use the vector type (`float4*`, `double4_32a*`, etc.) or an explicit `reinterpret_cast` at the call site.

### 6. Vector arithmetic is not vector at SASS level

A `float4 + float4` in source produces 4 separate FADD instructions in SASS, one per component:

```
FADD R12, R12, R16   ; x
FADD R13, R13, R17   ; y
FADD R14, R14, R18   ; z
FADD R15, R15, R19   ; w
```

No packed SIMD add (like AVX or NEON) exists for FP32 on SM120 in the observed patterns. The "vector" in vector types is about memory transaction width, not compute width.

HFMA2 remains the exception, but it is FP16-only and only appears in our observations as a constant-loading trick (kernels 04 and 05).

### 7. FP64 compute is drastically slower than FP32 on SM120

Variants 08f and 08g both use DADD for double precision addition. ptxas consistently inserts 3 to 4 NOP instructions between consecutive DADDs:

```
DADD R8, R8, R16
NOP
NOP
NOP
NOP
DADD R10, R10, R18
```

This padding is absent in all FP32 kernels (01 to 08e), where ptxas always finds an IMAD, a LDC, or another useful instruction to interleave. The presence of NOP indicates that DADD has either a very high fixed latency, a required minimum cadence on the FP64 pipeline, or insufficient FP64 throughput for ptxas to keep the pipeline busy.

Consistent with the SM120 hardware profile: consumer Blackwell has a small FP64 unit with a typical ratio of 64:1 FP32 to FP64 throughput.

Practical consequence: avoid FP64 in performance-sensitive kernels on SM120. Use FP32 or lower precision wherever numerical requirements allow.

### 8. Useful compute ratio scales with vectorization

Kernel 01 had 1 FADD out of 20 useful instructions (5% useful compute).
Kernel 08a has 4 FADD out of 22 useful instructions (18% useful compute).
Kernel 08d has 8 FADD out of 22 useful instructions (36% useful compute).

Each doubling of vector width roughly doubles the useful compute ratio because the plumbing cost (prologue, pointer loads, address arithmetic) stays constant while compute scales linearly with element count.

This is the arithmetic intensity lever: wider memory transactions reduce the ratio of overhead to work done per thread, making the kernel approach the roofline compute ceiling instead of bottlenecking on memory bandwidth.

## Variants summary

| Variant | Source type | Bytes / thread | Load observed | Stride | Alignment | Notes |
|---|---|---|---|---|---|---|
| 08a | float4 | 16 | LDG.E.128 | 0x10 | 16 | Canonical case |
| 08b | float (scalar indexing) | 16 | 4 × LDG.E | 0x4 | 4 | No auto-vectorization |
| 08c | float2 | 8 | LDG.E.64 | 0x8 | 8 | Confirms 64-bit exists |
| 08d | float8 (custom struct) | 32 | LDG.E.ENL2.256 | 0x20 | 32 | First `.ENL2` observed |
| 08e | float16 (custom struct) | 64 | 2 × LDG.E.ENL2.256 | 0x40 | 64 | 256 is the cap |
| 08f | double4 (legacy) | 32 | 2 × LDG.E.128 | 0x20 | **16** | Alignment blocks 256 |
| 08g | double4_32a | 32 | LDG.E.ENL2.256 | 0x20 | 32 | Alignment hypothesis confirmed |

## What this chapter adds to SASS reading skills

**Identify memory transaction width at a glance.** The suffix on LDG and STG (nothing, `.64`, `.128`, `.ENL2.256`) tells you how many bytes cross the memory subsystem per instruction. Count transactions in a loop body to reason about bandwidth utilization.

**Verify that source-level vectorization actually took effect.** If your source uses `float4*` but the SASS shows scalar LDG.E, something is blocking the vectorization: alignment, pointer aliasing, or an intermediate operation that prevents ptxas from tracking the vector.

**Diagnose FP64 bottlenecks without a profiler.** NOP padding between DADD (or DFMA) indicates the FP64 pipeline is the bottleneck. Reorganizing to FP32 or reducing FP64 dependency chains will directly improve throughput.

**Choose the right vector type for the element size.** Always prefer the `_16a` or `_32a` suffix variants for 64-bit elements. Prefer the `float4`, `float8`, `double2` types over scalar indexing for contiguous memory access patterns.

## Open hypotheses at the end of chapter 08

1. **Semantics of `.ENL2` modifier.** Confirmed it affects register encoding (two base registers instead of one). Unclear whether it also affects memory access behavior (latency, cache allocation, memory order). To verify by microbenchmark or by checking SM90+ documentation.
2. **Does LDG.E.ENL4.512 or a wider mode exist?** Not observed with 64-byte `float16` (ptxas split into 2 × 256). A kernel with `float16_64a` or a larger aligned struct would trigger the test. Currently assumed the 256-bit cap is a hard hardware limit.
3. **Exact FP32 to FP64 ratio on SM120.** NVIDIA documents 64:1 for consumer Blackwell. The NOP pattern between DADDs is consistent but not a direct measurement. To quantify by microbenchmark with a long DADD chain and clock() instrumentation.
4. **Behavior of vectorization when the pointer type is `float*` but the access pattern suggests vectorization.** Variant 08b showed no auto-vectorization with 4 contiguous scalar accesses. What about 2 or 8? What about with `__restrict__`? Not tested.
5. **SM89 and SM80 load width spectrum.** Do they support `.ENL2.256`? Or does the maximum cap at 128 bits for earlier architectures? To verify by recompiling 08d and 08g for `sm_89` and `sm_80`.

## New instructions observed in this chapter

| Opcode | Usage |
|---|---|
| LDG.E.64 | 64-bit load (float2) |
| LDG.E.128 | 128-bit load (float4, double2) |
| LDG.E.ENL2.256 | 256-bit load with enlarged register encoding (float8, double4_32a) |
| STG.E.64 | 64-bit store |
| STG.E.128 | 128-bit store |
| STG.E.ENL2.256 | 256-bit store with enlarged register encoding |
| SHF.L.U32 | Shift left unsigned (used in 08b for `i * 4`) |
| DADD | Double precision add |