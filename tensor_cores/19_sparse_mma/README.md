# 19 Sparse MMA (2:4 structured sparsity)

`mma.sp::ordered_metadata` variants on SM120. Sparse MMA operates on a 2:4 structured-sparse A matrix (2 nonzero elements per group of 4), which doubles effective K and produces 2× throughput compared to dense.

SM120 advertised peak for sparse FP4: 1801 TOPS, reached by `SM120_SPARSE_16x8x128_TN_VS<e2m1, e2m1, f32, ue8m0, 64>`.

## Variants planned

| # | PTX atom | Notes |
|---|---|---|
| 19a | `mma.sp::ordered_metadata.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e4m3.e4m3.f32` | Sparse FP8 non-scaled |
| 19b | `mma.sp::ordered_metadata.sync.aligned.kind::f8f6f4.m16n8k64.row.col.f32.e2m1.e2m1.f32` | Sparse FP4 non-scaled (doubled K from dense 32) |
| 19c | `mma.sp::ordered_metadata.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0` | Sparse FP4 block-scaled peak (1801 TOPS) |
| 19d | `...kind::mxf8f6f4.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3` | Alternate SF |
| 19e | Dense vs sparse same atom side-by-side | Delta analysis |

## Key questions

1. What SASS modifier or opcode distinguishes sparse MMA from dense?
2. How is sparsity metadata (which of the 4 elements are nonzero) encoded? Separate register, packed into A, or operand to the instruction?
3. What is the `sparsitySelector` qualifier and how does it appear at SASS?
4. Does `ordered_metadata` mean the metadata has a specific layout?
5. Is the K doubling (m16n8k64 sparse = m16n8k32 dense effective) an artifact of counting, or a hardware-level expansion?
6. Can sparse and dense MMA coexist in the same kernel with consistent scoreboarding?

## Context from FINDINGS.md

**From chapters 13-16**: dense MMA opcode family established. Sparse is an extension but may have its own opcode prefix or modifier.

**From chapter 16**: FP4 peak at k=64 with block scaling = 933 TFLOPS. Sparse FP4 peak at k=128 with block scaling = 1866 TOPS (measured by Lei Mao). The 2× ratio must correspond to the structural doubling.

**Cost rule extension**: sparse metadata load adds overhead. Quantify in this chapter.

## Status

* [ ] 19a sparse FP8
* [ ] 19b sparse FP4 non-scaled
* [ ] 19c sparse FP4 block-scaled peak (ue8m0)
* [ ] 19d sparse FP4 block-scaled (ue4m3)
* [ ] 19e dense vs sparse side-by-side
* [ ] conclusion19.md

## Dependencies

Chapter 16 provides the dense FP4 peak baseline. Sparse is an extension, conceptually and at SASS.