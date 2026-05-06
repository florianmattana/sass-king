# Kernel 22 stmatrix / matrix store

## Narrative

* [OBS] Chapter 22 closes the matrix-store gap left by chapters 17 and 18 by testing `stmatrix.sync.aligned` on SM120.
* [OBS] Chapter 17 established the load side of matrix fragments as `LDSM.16.M[T]88[.N]`.
* [OBS] Chapter 22 establishes the store side for the tested m8n8 b16 forms as `STSM.16.M[T]88[.2|.4]`.
* [INF] For the tested shared-memory matrix-copy path, SM120 has a real STSM opcode family rather than lowering every matrix store to scalar `STS`.
* [OBS] Runtime smoke execution succeeds on the host GPU for normal/transposed x4 STSM and barrier/no-barrier probes.
* [GAP] Full lane-to-shared layout decode remains open.

## Source structure

* [OBS] The source is a single parameterized CUDA file, `22_stmatrix.cu`, with variants selected by `-DVARIANT`.
* [OBS] Variants 22a through 22f isolate x1/x2/x4 and non-transposed/transposed m8n8 b16 STSM forms.
* [OBS] Variant 22g is a scalar shared-store fallback using ordinary C++ stores to the same shared-memory region.
* [OBS] Variants 22h and 22i keep the x4 normal/transposed stores and read back 128 bits per lane through `LDS.128`.
* [OBS] Variant 22j reads another lane-shaped shared location after a barrier.
* [OBS] Variant 22k omits `__syncthreads()` and reads the same lane-shaped shared location immediately after STSM.
* [OBS] Variant 22l places STSM after an HMMA-shaped register path.

## Opcode forms

| Variant | PTX source form | SASS |
|---|---|---|
| 22a | [OBS] `stmatrix.sync.aligned.x1.m8n8.shared.b16` | [OBS] `STSM.16.M88 [R7], R2` |
| 22b | [OBS] `stmatrix.sync.aligned.x2.m8n8.shared.b16` | [OBS] `STSM.16.M88.2 [R0], R6` |
| 22c | [OBS] `stmatrix.sync.aligned.x4.m8n8.shared.b16` | [OBS] `STSM.16.M88.4 [R0], R8` |
| 22d | [OBS] `stmatrix.sync.aligned.x1.trans.m8n8.shared.b16` | [OBS] `STSM.16.MT88 [R7], R2` |
| 22e | [OBS] `stmatrix.sync.aligned.x2.trans.m8n8.shared.b16` | [OBS] `STSM.16.MT88.2 [R0], R6` |
| 22f | [OBS] `stmatrix.sync.aligned.x4.trans.m8n8.shared.b16` | [OBS] `STSM.16.MT88.4 [R0], R8` |

* [OBS] The visible low opcode word for the tested STSM forms ends in low byte `0x44`.
* [OBS] The printed second control/encoding line differs between normal and transposed forms: x1 normal shows `0x010fe20000000000`, x1 transposed shows `0x010fe20000004000`, x4 normal shows `0x010fe20000000200`, and x4 transposed shows `0x010fe20000004200`.
* [INF] The `.trans` distinction is encoded in bits reflected by the `0x4000` delta in the printed second line for the tested x1 and x4 pairs.

## Fallback comparison

* [OBS] Variant 22g emits `STS.128 [R0], R8`.
* [OBS] Variant 22g does not emit `STSM`.
* [INF] Ordinary scalar shared stores can combine into vectorized `STS.128`, but this is not the same instruction family as matrix-store STSM.

## Synchronization and dependency probes

* [OBS] Variants 22a through 22j emit `BAR.SYNC.DEFER_BLOCKING 0x0` between STSM or STS and downstream shared-memory readback.
* [OBS] Variant 22k emits `STSM.16.M88.4 [R0], R8` followed directly by `LDS R7, [R0]` with no `BAR.SYNC`.
* [GAP] Runtime correctness of reading STSM output without a block barrier is not validated.
* [INF] The no-barrier SASS form is useful as a dependency signature, but it is not evidence that cross-thread shared visibility is safe without synchronization.

## HMMA-adjacent probe

* [OBS] Variant 22l emits `HMMA.16816.F32 R8, R4, R8, RZ`.
* [OBS] Variant 22l later emits `STSM.16.M88.4 [R0], R8`.
* [INF] STSM can appear in a tensor-core-adjacent instruction stream on SM120.
* [GAP] Variant 22l is not a full production epilogue and does not validate accumulator layout conversion.

## Unsupported b8 forms

* [OBS] The opt-in unsupported probe attempts `stmatrix.sync.aligned.m16n8.x1.trans.shared.b8`, `.x2`, and `.x4`.
* [OBS] ptxas rejects the probe for `sm_120` with `Feature '.m16n8' not supported on .target 'sm_120'`.
* [OBS] ptxas rejects the probe for `sm_120` with `Feature 'stmatrix.b8' not supported on .target 'sm_120'`.
* [RES] The tested SM120 target does not support the tested m16n8 b8 STSM PTX forms.

## Open gaps

* [OBS] Runtime smoke execution for `STSM.16.M88.4` and `STSM.16.MT88.4` succeeds on the host GPU.
* [GAP] Full lane-to-shared layout semantics remain undecoded from the runtime outputs.
* [GAP] STSM latency is not measured.
* [GAP] STSM scoreboard and pipeline control fields are not decoded beyond the printed SASS/control annotations.
* [GAP] Production accumulator-to-shared epilogue layout remains for Kernel 24 or a dedicated production mini-GEMM variant.
