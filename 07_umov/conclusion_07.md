# Kernel 07 UMOV 0x400 investigation

Chapter dedicated to isolating the meaning of `UMOV UR4, 0x400`, an instruction first observed in kernel 06 as part of the shared memory base construction. Four variants tested against kernel 01 (no shared, control) to determine what the value `0x400` encodes.

## Context

In kernel 06 the shared memory addressing starts with two instructions:

```
UMOV UR4, 0x400
ULEA UR4, UR5, UR4, 0x18
```

where UR5 holds `SR_CgaCtaId`. Three hypotheses were tested and rejected in kernel 06:

* UMOV 0x400 does not encode shared memory size. Variant 06e with `float[128]` kept the value at 0x400.
* UMOV 0x400 does not encode block size. Variant 06f with 512 threads kept the value at 0x400.
* UMOV 0x400 is not proportional to the launch configuration. Multiple variants confirmed.

At the end of chapter 06, the hypothesis narrowed to: `0x400` might be an architectural constant tied to the shared memory addressing mechanism itself, independent of kernel parameters. Kernel 07 is the targeted investigation of that hypothesis.

## Methodology

Four variants, each isolating one property of the shared declaration:

| Variant | Test | Expected finding |
|---|---|---|
| 07a | `__shared__ float[1]` | Effect of minimal shared size |
| 07b | `__shared__ int[256]` | Effect of element type |
| 07c | `extern __shared__ float[]` | Effect of static vs dynamic declaration |
| 01 (control) | No `__shared__` at all | Is the UMOV related to shared at all |

## Results

### Variant 07a: `__shared__ float[1]`

```
UMOV UR4, 0x400
```

Unchanged despite the shared array being reduced to 4 bytes (1 float). The LOP3 modulo mask disappears because there is only one element; the address is `UR4` directly.

Secondary observation: the source contains `if (threadIdx.x == 0)` and ptxas emitted predicated instructions throughout the shared write path:

```
@!P0 LDC.64 R2, c[0x0][0x380]
@!P0 IMAD.WIDE R2, R7, 0x4, R2
@!P0 LDG.E R2, desc[UR6][R2.64]
@!P0 STS [UR4], R2
```

where P0 is the `tid != 0` test. Thread 0 executes the write, the other 31 threads fetch the instructions but do not execute them. Confirms the predication rule observed in earlier kernels: short conditional blocks are predicated, not branched.

### Variant 07b: `__shared__ int[256]`

```
UMOV UR4, 0x400
```

Unchanged. The SASS is nearly byte-for-byte identical to variant 06b (`__shared__ float[256]`). The type of the shared array does not affect the generated code for a simple read-write pattern, because both `int` and `float` occupy 4 bytes. The LOP3 mask (0x3fc), the LEA shifts (0x2), and the IMAD.WIDE stride (0x4) all remain the same because they encode sizes, not types.

### Variant 07c: `extern __shared__ float smem[]`

```
UMOV UR4, 0x400
```

Unchanged despite the shared size being a runtime parameter of the kernel launch (`<<<grid, block, smem_bytes>>>`). 

Secondary observation: ptxas cannot know the size at compile time, so the modulo operation `(tid + 1) % blockDim.x` falls back to the runtime CALL pattern seen in kernel 06 (the `__cuda_sm20_rem_u16` helper placed after EXIT).

Dynamic shared memory produces identical addressing SASS to static shared. Only the source of the modulo bound differs (constant at compile time for static, runtime for dynamic).

### Kernel 01 (control, no shared)

No `UMOV UR4, 0x400` and no `S2UR UR5, SR_CgaCtaId` in the dump. The two instructions that build the shared base are both absent when the kernel does not declare any shared memory.

## Solid observations

### 1. UMOV 0x400 is triggered exclusively by the presence of shared memory

* Absent if no `__shared__` is declared (kernel 01).
* Present as soon as any `__shared__` is declared (kernels 06 and 07).
* Value 0x400 is invariant across:
  * Shared size (1, 128, 256, 512 floats)
  * Shared element type (float, int)
  * Static vs dynamic declaration
  * Number of shared buffers in the kernel (06g had two)

This is now a resolved observation: `UMOV UR4, 0x400` is the SM120 architectural constant for initializing the uniform register used in shared memory addressing. The value is not parameterized by the kernel.

### 2. SR_CgaCtaId is read only when shared is used

The `S2UR UR5, SR_CgaCtaId` instruction paired with UMOV 0x400 also disappears in the no-shared control. Both instructions form the characteristic two-line shared base setup. Seeing either one in a SASS dump signals that the kernel uses shared memory.

### 3. Predication extends to multi-instruction bodies

Variant 07a produced 4 predicated instructions (`@!P0` prefix on LDC, IMAD.WIDE, LDG, STS) instead of branching around them with BRA. ptxas consistently chooses predication for short conditional bodies, even when the body touches memory (LDG, STS).

The cost of predication here is that all 32 threads of the warp fetch and decode the 4 instructions, even though only thread 0 executes. For a 4-instruction body this is cheaper than the BRA alternative because the branch itself plus divergence reconvergence would add more overhead.

### 4. Element type does not change load/store SASS for same-size types

Variants 06b (`float[256]`) and 07b (`int[256]`) are byte-identical in the SASS body except for the kernel function name. 4-byte int and 4-byte float share the same LDG.E, STG.E, IMAD.WIDE, LEA encodings because these instructions encode element size, not element type. Type distinction matters only for arithmetic (FADD versus IADD, FMUL versus IMUL), never for memory transactions.

## Resolved hypotheses

The chapter 06 hypothesis is now refined into one resolved and one open part.

Resolved:

* `UMOV UR4, 0x400` appears if and only if the kernel uses shared memory. It is architecturally constant (not parameterized by size, type, or declaration style) and is always paired with `S2UR UR5, SR_CgaCtaId`.

Still open:

* What does the specific value `0x400 = 1024` mean? It is clearly not the shared memory size in bytes (shown by 07a with 4 bytes of shared still producing 0x400). Candidate interpretations not yet tested:
  * Architectural constant representing the maximum block size on SM120 (1024 threads), used in the descriptor format to size the shared region per block.
  * A field in a bit-packed descriptor format where 0x400 places a specific bit in a specific position.
  * A shift amount or mask interacting with the `0x18` immediate in the following `ULEA`.

## Diagnostic rule derived

When reading a SASS dump, the two-line sequence

```
S2UR UR*, SR_CgaCtaId
UMOV UR*, 0x400
```

(or with the operations reordered by the scheduler) is a reliable visual marker that the kernel uses shared memory. If either of these is absent in the SASS, no shared memory is in use regardless of what the source might suggest after macro expansion or template instantiation.

## What this chapter adds to SASS reading skills

**Reject eliminative hypotheses systematically.** The chapter demonstrates the value of variation tests in closing hypothesis space. Three plausible interpretations of `0x400` were rejected in kernel 06, and this chapter closed two more. The remaining candidates for the meaning of `0x400` are narrower and more testable.

**Control kernel is as informative as variation kernels.** Using kernel 01 as the no-shared baseline produced the sharpest single finding of the chapter: UMOV 0x400 and SR_CgaCtaId are conditional on the presence of shared memory. Without an explicit control, this would have been inferred but not proved.

**Identical SASS across type variations confirms ptxas internal canonicalization.** Variants 06b and 07b being byte-identical except for the function name confirms that ptxas treats 4-byte integer and 4-byte floating point identically at the memory layer. The same canonicalization was observed earlier between `% 256` and `& 255` in kernel 06d.

## Open hypotheses at the end of chapter 07

1. Exact numerical meaning of `0x400`. Narrowed but not resolved. Next tests: compare with dumps from SM89 (RTX 4090) and SM90a (H100) to see if the value changes with architecture. Cross-reference with NVIDIA documentation on shared descriptor format for SM90+.
2. Whether `SR_CgaCtaId` value differs between kernels with and without `__cluster_dims__`. Not tested.
3. Whether the predication threshold changes with block body size. Variant 07a predicated a 4-instruction body. Somewhere between 4 and N instructions, ptxas would prefer BRA. The threshold is not measured.

## New instructions observed in this chapter

No new opcodes. The chapter reused instructions already cataloged in kernels 01 to 06. The contribution is in the rules and conditional triggers for existing opcodes, not in new SASS vocabulary.