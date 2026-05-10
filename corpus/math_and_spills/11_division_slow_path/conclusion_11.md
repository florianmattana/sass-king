# Kernel 11: Slowpath arithmetic (division, math library, hardware MUFU)

## Scope

Ten variants covering integer division by runtime variables (u32, u64, s32), floating-point math library functions (log2f, expf, sinf, sqrtf), and their hardware-accelerated counterparts (__log2f, rsqrtf, __fdividef). The chapter maps how ptxas compiles each operation and establishes the frontier between inline polynomial approximation, hardware MUFU direct calls, and subroutine-style CALL patterns.

## Variants

| Variant | Source | SASS observed |
|---|---|---|
| 11a | `a[i] / d` u32 runtime | Inline Granlund-Montgomery reciprocal multiplication |
| 11b | `a[i] / d` u64 runtime | Hybrid: inline u32-fast-path + local CALL for u64 slowpath |
| 11c | `a[i] / d` s32 runtime | Inline via IABS + unsigned helper + LOP3 0x3c sign fix-up |
| 11d | `log2f(x)` standard | Fully inline polynomial approximation (Remez, 10 coefficients) |
| 11e | `__log2f(x)` intrinsic | Single MUFU.LG2 + subnormal handling |
| 11f | `expf(x)` standard | Inline range reduction + MUFU.EX2 (no CALL) |
| 11g | `sinf(x)` standard | Inline fast path + BRA slowpath with Payne-Hanek reduction (FP64) |
| 11h | `sqrtf(x)` standard | Inline fast path (MUFU.RSQ + 2 NR) + local CALL for NaN/Inf/denormal |
| 11i | `rsqrtf(x)` | Single MUFU.RSQ + subnormal handling |
| 11j | `__fdividef(a, b)` | MUFU.RCP + FMUL + subnormal handling |

## Structural finding: inlining over external helpers

The chapter's most important result contradicts the working hypothesis. On SM120 with CUDA 13.2, ptxas does **not** emit external named helpers (`__cuda_sm20_div_u32`, `__cuda_sm20_rem_u32`, etc.) for integer division at u32 width or above, nor for any math library function tested. Every operation compiles to one of three shapes:

* **Fully inline**, no CALL (11a, 11c, 11d, 11e, 11f, 11i, 11j)
* **Inline with local BRA slowpath** (11g): both paths live in the main kernel body
* **Inline with CALL to a local subroutine** (11b, 11h): slowpath placed after the main EXIT, called via CALL.REL.NOINC

The only externally-named helper observed in the entire project remains `__cuda_sm20_rem_u16` from chapter 06. [HYP] This is either a CUDA 13+ inlining threshold change, or the external helpers are reserved for sub-word types (u16) where the calling overhead amortizes better.

Practical consequence for reading production SASS: the absence of a CALL does not mean "no slowpath" — it means the slowpath is inlined. Look for BRA skipping over arithmetic blocks rather than CALL signatures.

## Integer division patterns

### 11a: u32 division, fully inline

Twenty-one instructions implement Granlund-Montgomery reciprocal multiplication for a u32 dividend by a u32 runtime divisor. Structure:

```
UI2F.U32.RP UR4, UR5                    ; divisor float approximation, round to +inf
MUFU.RCP R0, UR4                         ; 1/divisor_float
IADD R0, R0, 0xffffffe                   ; adjustment magic constant
F2I.FTZ.U32.TRUNC.NTZ R5, R0             ; round back to int (the reciprocal estimate)
IADD R9, RZ, -R5                         ; -R5 for error computation
IMAD R9, R9, UR5, RZ                     ; -R5 * divisor = -(quotient * divisor)
IMAD.HI.U32 R5, R5, R9, R4               ; error correction via high-multiply
IMAD.HI.U32 R5, R5, R2, RZ               ; final quotient via high-multiply with dividend
IADD R3, -R5, RZ                         ; -quotient for remainder computation
IMAD R4, R3, UR5, R2                     ; remainder = dividend - quotient * divisor
ISETP.GE.U32.AND P0, ..., R4, UR5        ; double correction pair
@P0 IADD R4, R4, -UR5                    ; (quotient could be underestimated by 1 or 2)
@P0 IADD R5, R5, 0x1
ISETP.GE.U32.AND P1, ..., R4, UR5        ; second correction check
@P1 IADD R5, R5, 0x1
@!P2 LOP3.LUT R5, RZ, UR5, RZ, 0x33, !PT  ; zero-divisor returns 0xFFFFFFFF (UB guard)
```

The final LOP3 with LUT `0x33` computes `~divisor` when the divisor is zero, producing a sentinel value rather than crashing.

### 11b: u64 division, hybrid inline + local CALL

The fast path tests whether the high 32 bits of the divisor are zero (`LOP3.LUT R4, R3, UR4, RZ, 0xfc` which computes `R3 | UR4`). If so, the same u32 Granlund-Montgomery algorithm runs with 64-bit dividend accumulation, reconstructing the 64-bit quotient. If not, `CALL.REL.NOINC 0x2e0` branches to a ~60-instruction subroutine placed after the main EXIT.

The slowpath uses multi-precision arithmetic primitives:
* **IMAD.WIDE.U32**: 32×32→64 multiply, result in register pair
* **IADD3.X**: three-input add with carry-in (`.X` means extended, reads CC flag)
* **IADD.X**: two-input add with carry-in
* **IADD.64**: explicit 64-bit add (single operation, no carry chain)
* **ISETP.GE.U64.AND**, **ISETP.NE.S64.AND**: 64-bit compares
* **SEL.64**: 64-bit select based on predicate

The algorithm performs two Newton-Raphson iterations on an initial MUFU.RCP estimate, then computes the remainder and quotient correction with multi-precision arithmetic. Return via `RET.REL.NODEC R4 0x0` where R4 contains the return address (set to `0x290` before the CALL via `MOV R4, 0x290`).

### 11c: s32 division, inline with sign handling

Signed division is implemented as unsigned division on absolute values plus sign reconstruction:

```
IABS R9, R11                             ; |divisor|
[unsigned Granlund-Montgomery on |dividend| / |divisor|]
IABS R2, R0                              ; |dividend|
LOP3.LUT R0, R0, R11, RZ, 0x3c, !PT      ; XOR of sign bits (0x3c = A^B)
...
@!P0 IADD R5, -R5, RZ                    ; negate quotient if signs differ
```

LUT `0x3c` is the truth table for XOR over the three inputs treating the first two symbolically — specifically, computing `A ^ B` with C ignored. The MSBs of `dividend ^ divisor` give the sign of the result.

Zero-divisor handling uses `ISETP.NE.AND P1, PT, R11, RZ` and the same LUT `0x33` sentinel pattern as u32.

### Invariant: reciprocal multiplication structure

All three integer division variants (11a u32, 11b u64 fast path, 11c s32) share the same five-stage skeleton:

1. Convert divisor to float with `.RP` (round toward +inf, guarantees overestimate)
2. MUFU.RCP to get float reciprocal
3. Adjust with magic constant (0xffffffe, visible in 11a and 11b)
4. Convert back to int with F2I.FTZ.*.TRUNC.NTZ
5. Correct via IMAD.HI.U32 chain (one pass for u32, two for u64) plus predicated IADD correction

The `0xffffffe` adjustment is `(1 << 28) - 2`, a bias constant that tunes the reciprocal estimate to avoid double-correction in the common case. [HYP] The choice of this exact value is related to the bit representation of the F2I rounding boundary.

## Floating-point math library patterns

### 11d: log2f standard, fully inline polynomial

No CALL, no MUFU.LG2. Ptxas emits a ten-term Remez polynomial evaluated via Horner's method.

Range reduction: `IADD R4, R0, -0x3f3504f3` subtracts the bit pattern of `sqrt(2)/2 ≈ 0.7071067811865475` from the input's bit representation. This simultaneously extracts the unbiased exponent (high bits of the difference) and shifts the mantissa into `[sqrt(2)/2, sqrt(2)]`.

```
LOP3.LUT R7, R4, 0xff800000, RZ, 0xc0, !PT   ; mask exponent bits (0xff800000 = sign+exp)
IADD R4, R0, -R7                              ; mantissa_scaled = x - exponent_mask
I2FP.F32.S32 R7, R7                           ; convert exponent to float
FADD R6, R4, -1                               ; fractional part of mantissa
FSEL R4, RZ, -23, P0                          ; subnormal correction: add -23 to exponent
```

The polynomial evaluation:

```
FFMA R9, R6, R9, -0.16845393180847167969
FFMA R9, R6, R9,  0.17168870568275451660
FFMA R9, R6, R9, -0.17900948226451873779
FFMA R9, R6, R9,  0.20512372255325317383
FFMA R9, R6, R9, -0.24046532809734344482
FFMA R9, R6, R9,  0.28857114911079406738
FFMA R9, R6, R9, -0.36067417263984680176
FFMA R9, R6, R9,  0.48089820146560668945
FFMA R9, R6, R9, -0.72134751081466674805
FMUL R9, R6, R9
FMUL R9, R6, R9
FFMA R9, R6, 1.4426950216293334961, R9        ; multiply by 1/ln(2) (= log2(e))
```

Ten FFMA + two FMUL + one final FFMA = 13 polynomial instructions. The final FFMA multiplies by `log2(e) ≈ 1.4427` to convert from the natural logarithm computed by the polynomial to log base 2.

Edge cases handled after the polynomial:
* `@P0 FFMA R4, R0, R7, +INF` where R7 = 0x7f800000 = +INF bit pattern: returns +INF for input > largest normal
* `FSEL R5, R4, -INF, P1`: returns -INF if the input was zero (detected by FSETP.NEU early)

The initial `HFMA2 R9, -RZ, RZ, 1.443359375, -0.2030029296875` loads the FP16-packed pair `(1.44..., -0.20...)` into R9 as a 32-bit register. This is then consumed as FP32 in the first FFMA of the polynomial. [GAP] The exact bit pattern and why ptxas packs these specific values via HFMA2 rather than MOV immediate is unclear — probably the result of constant materialization fusion during scheduling.

### 11e: __log2f intrinsic, single MUFU.LG2

```
FSETP.GEU.AND P0, PT, |R0|, 1.175494350822287508e-38, PT   ; test subnormal
@!P0 FMUL R0, R0, 16777216                                   ; scale by 2^24 if subnormal
MUFU.LG2 R9, R0                                              ; hardware log2
@!P0 FADD R9, R9, -24                                        ; correct exponent if scaled
```

Four instructions total (plus the pointer arithmetic). The subnormal handling scales the input by 2^24 because MUFU.LG2 doesn't handle subnormals directly. The correction `-24` is subtracted from the result.

### 11f: expf standard, inline with MUFU.EX2

```
HFMA2 R5, -RZ, RZ, 0.96630859375, -0.0022525787353515625     ; pack constants
MOV R9, 0x437c0000                                            ; = 252.0 FP32
FFMA.SAT R0, R2, R5, 0.5                                      ; saturate to [0, 1]
FFMA.RM R0, R0, R9, 12582913                                  ; integer extraction via overflow
FADD R5, R0, -12583039                                        ; fractional residual
SHF.L.U32 R0, R0, 0x17, RZ                                    ; shift integer to exponent bits
FFMA R9, R2, 1.4426950216293334961, -R5                       ; residual with 1/ln(2)
FFMA R9, R2, 1.925963033500011079e-08, R9                     ; refinement for ln(2) high bits
MUFU.EX2 R9, R9                                               ; 2^residual
FMUL R5, R0, R9                                               ; 2^k * 2^residual
```

The decomposition is `exp(x) = 2^(x * log2(e)) = 2^(k + f) = 2^k * 2^f` where `k` is the integer part of `x * log2(e)` and `f` is the fractional part.

The magic number `12582913 = 0x00C00001` combined with FFMA.RM (round toward -inf) is the classic "magic number" trick to extract the integer part of a float via controlled overflow. `0x437c0000 = 252.0` is used as a scale factor. [HYP] These specific constants are standard library idioms used in glibc's expf; they match the algorithmic structure of float-to-int conversion via bit-level manipulation.

The dual FFMA with `1.4426950216293334961` and `1.925963033500011079e-08` implements an extended-precision multiplication by `log2(e)` where the second coefficient captures the lower 32 bits of the double-precision value, giving about 50 bits of effective precision on the product.

### 11g: sinf standard, dual-path with Payne-Hanek slowpath

Fast path for `|x| < 105615 ≈ 2^17`:

```
FMUL R3, R0, 0.63661974668502807617             ; x * 2/π (rough scale)
F2I.NTZ R3, R3                                   ; k = round-to-nearest
I2FP.F32.S32 R7, R3                              ; back to float
FFMA R6, R7, -1.5707962512969970703, R0          ; x - k * π/2 (high bits of π/2)
FFMA R6, R7, -7.5497894158615963534e-08, R6      ; extended precision refinement
FFMA R6, R7, -5.3903029534742383927e-15, R6      ; further refinement
```

Three FFMA with progressively smaller coefficients of π/2 achieve high precision reduction. The three coefficients are the triple-double decomposition of π/2.

For `|x| >= 105615`, the Payne-Hanek reduction kicks in:

```
BRA 0x140                                        ; skip fast-path end
LDCU.64 UR4, c[0x4][URZ]                         ; load base pointer to 2/π table
IMAD.SHL.U32 R3, R0, 0x100, RZ                   ; shift x to expose mantissa
MOV.64 R4, RZ                                    ; clear accumulator high
MOV.64 R8, RZ                                    ; counter / accumulator low
LOP3.LUT R3, R3, 0x80000000, RZ, 0xfc, !PT       ; set implicit leading 1 of mantissa

<loop 6 iterations>
  LDG.E.CONSTANT R10, desc[UR6][R6.64]           ; load chunk of 2/π table
  ISETP.EQ.AND P0..P5, PT, R8, 0x0/0x4/0x8/...  ; per-iteration selector
  UIADD3 UR4, ..., UR4, 0x1, URZ                 ; increment loop counter
  IADD.64 R8, R8, 0x4                             ; advance table index
  IADD.64 R6, R6, 0x4                             ; advance pointer
  IMAD.WIDE.U32 R4, R10, R3, R4                   ; accumulate bits via 32×32→64
  @P0..P5 MOV R11..R16, R4                        ; store partial products
  UISETP.NE.U32.AND UP0, UPT, UR4, 0x6, UPT       ; continue until 6 iterations done
  BRA.U UP0, <loop_start>
```

The loop multiplies the input's mantissa by successive 32-bit chunks of `2/π` stored in a read-only table accessed via `LDG.E.CONSTANT`. This is the Payne-Hanek algorithm that achieves exact range reduction for arbitrarily large arguments.

After the loop, the reduction completes in FP64:

```
I2F.F64.S64 R4, R6                               ; convert reduced integer to double
DMUL R4, R4, UR4                                 ; multiply by π/2 in FP64
F2F.F32.F64 R4, R4                               ; narrow back to FP32
```

The FP64 DMUL with `UR4 = 0x3bf921fb54442d19` (= π/2 in IEEE 754 double, packed into a uniform register) preserves accuracy across the reduction.

Both paths converge on a polynomial evaluation using 4 FFMA + FSEL pairs implementing `sin(x) ≈ x + c3*x^3 + c5*x^5 + c7*x^7` with coefficients selected based on whether sin or cos branch is needed (via R2P and predicate selection). The FSELs with `P0` derived from the reduced-argument quadrant select between `sin` and `cos` polynomials.

[GAP] The exact per-iteration reconstruction of the Payne-Hanek accumulator from R11..R16 is not fully traced. The 6 partial products need to be combined via shifts and additions; the SHF.L.W.U32.HI at 0x05c0 and 0x05e0 implements this combination.

### 11h: sqrtf standard, fast path + local CALL slowpath

Fast path:

```
IADD R4, R0, -0xd000000                          ; bias-shifted input for range test
MUFU.RSQ R7, R0                                  ; y0 = 1/sqrt(x)
ISETP.GT.U32.AND P0, PT, R4, 0x727fffff, PT      ; test if fast path applicable
@!P0 BRA 0x140                                   ; if yes, fall through to NR
MOV R9, 0x130                                    ; else set return address
CALL.REL.NOINC 0x1d0                             ; and call slowpath

<fast path Newton-Raphson>
FMUL.FTZ R3, R0, R7                              ; q0 = x * y0 ≈ sqrt(x)
FMUL.FTZ R7, R7, 0.5                             ; y0/2
FFMA R0, -R3, R3, R0                             ; e = x - q0^2
FFMA R7, R0, R7, R3                              ; sqrt(x) = q0 + e * y0/2
```

One NR iteration starting from the hardware MUFU.RSQ estimate. The range test `R4 > 0x727fffff` after the `-0xd000000` bias catches denormals (very small), NaN, and infinity (very large).

Slowpath at 0x1d0:

```
LOP3.LUT P0, RZ, R0, 0x7fffffff, RZ, 0xc0, !PT   ; test if zero (mask sign, compare)
@!P0 MOV R2, R0                                   ; sqrt(0) = 0 or sqrt(-0) = -0
@!P0 BRA 0x300                                    ; return

FSETP.GEU.FTZ.AND P0, PT, R0, RZ, PT              ; test if negative
@!P0 MOV R2, 0x7fffffff                           ; return NaN (all 1s in significand)
@!P0 BRA 0x300

FSETP.GTU.FTZ.AND P0, PT, |R0|, +INF, PT          ; test if NaN (|x| > INF is only true for NaN)
@P0 FADD.FTZ R2, R0, 1                            ; propagate NaN
@P0 BRA 0x300

FSETP.NEU.FTZ.AND P0, PT, |R0|, +INF, PT          ; test if finite (not infinity)
@P0 FFMA R3, R0, 1.84467440737095516160e+19, RZ   ; scale by 2^64
@P0 MUFU.RSQ R2, R3                               ; refined RSQ
@P0 FMUL.FTZ R4, R3, R2                           ; NR iteration
@P0 FMUL.FTZ R8, R2, 0.5
@P0 FADD.FTZ R6, -R4, -RZ
@P0 FFMA R7, R4, R6, R3
@P0 FFMA R7, R7, R8, R4
@P0 FMUL.FTZ R2, R7, 2.3283064365386962891e-10    ; rescale by 2^-32

RET.REL.NODEC R2 0x0                              ; return, address in R9 (saved at CALL)
```

The slowpath handles edge cases explicitly and for denormals scales by `2^64` to bring the value into the normal range for MUFU.RSQ to work correctly, then applies the Newton-Raphson iteration and rescales by `2^-32` on the result (since sqrt of 2^64 is 2^32).

### 11i: rsqrtf, single MUFU.RSQ

```
FSETP.GEU.AND P0, PT, |R0|, 1.175494350822287508e-38, PT    ; subnormal test
@!P0 FMUL R0, R0, 16777216                                    ; scale by 2^24
MUFU.RSQ R9, R0                                               ; hardware rsqrt
@!P0 FMUL R9, R9, 4096                                        ; correct by 2^12 (= 2^(24/2))
```

The correction factor after subnormal scaling is `2^(scale/2) = 2^12 = 4096` because rsqrt halves the exponent. Four instructions total.

### 11j: __fdividef, MUFU.RCP + FMUL

```
FSETP.GEU.AND P0, PT, |R8|, 1.175494350822287508e-38, PT     ; test subnormal on divisor
@!P0 FMUL R8, R8, 16777216                                     ; scale divisor
@!P0 FMUL R0, R0, 16777216                                     ; scale numerator equally
MUFU.RCP R11, R8                                               ; 1/divisor
FMUL R11, R11, R0                                              ; dividend * (1/divisor)
```

Both numerator and divisor are scaled by the same 2^24 if the divisor is subnormal. This preserves the ratio `a/b` while moving both values into the normal range.

## Subnormal handling as a universal pattern

Four of the ten kernels (11d, 11e, 11i, 11j) use an identical subnormal handling skeleton:

```
FSETP.GEU.AND P0, PT, |R|, 1.175494350822287508e-38, PT    ; test if |x| < smallest normal
@!P0 FMUL R, R, 2^24                                         ; scale up if subnormal
<MUFU or polynomial operation>
@!P0 <correction>                                            ; subtract 24 for log, multiply by 4096 for rsqrt
```

The constant `1.175494350822287508e-38` is `FLT_MIN`, the smallest positive normal FP32. The scale factor `2^24 = 16777216` lifts any subnormal into the normal range (the largest subnormal is `(1 - 2^-23) * 2^-126`, so `2^24` times it is `(1 - 2^-23) * 2^-102`, safely normal).

The correction depends on the operation:
* For `log2`: subtract 24 from the result (since `log2(2^24 * x) = log2(x) + 24`)
* For `rsqrt`: multiply by `2^12` (since `rsqrt(2^24 * x) = rsqrt(x) / 2^12`)
* For operations like `fdividef` where both numerator and divisor are scaled by the same factor, no correction is needed because the scaling cancels in the ratio

This is a canonical pattern worth remembering when reading any SASS dump that uses MUFU on floats: if MUFU is preceded by a FSETP.GEU check and a scale-by-large-power-of-2, subnormal handling is active.

## Local CALL pattern (BSSY/CALL.REL.NOINC within kernel)

Both 11b (u64 div slowpath) and 11h (sqrtf NaN/Inf slowpath) use the same calling structure:

1. Main kernel executes up to a point where it may or may not need the slowpath.
2. `BSSY.RECONVERGENT B0, <sync_address>` establishes a reconvergence point before the divergent region.
3. A predicate or branch decides: fall through with inline fast path, or `CALL.REL.NOINC <slowpath_address>` to the local subroutine.
4. Before the CALL, `MOV Rn, <return_address>` is executed where `<return_address>` is the address of the instruction right after the CALL. The register `Rn` is chosen based on local register pressure (R4 in 11b, R9 in 11h).
5. After the CALL or the fast path, `BSYNC.RECONVERGENT B0` reconverges the warp.
6. The main kernel proceeds with the slowpath's result.
7. The slowpath code lives after the main kernel's EXIT, placed in the same `.text` section but unreachable from the normal control flow.
8. The slowpath ends with `RET.REL.NODEC Rn 0x0` where `Rn` is the register holding the return address.

This structure is a hybrid between inlining (no symbol lookup overhead, no ABI cost for caller-saved registers) and out-of-line code (code size economy, better I-cache behavior when the slowpath is rare). The choice of link register per-kernel based on local pressure is consistent with the helper calls seen in chapter 06.

## Opcode inventory (new in this chapter)

| Opcode | Usage | Semantics |
|---|---|---|
| `UI2F.U32.RP` | 11a, 11b | Uniform int32 to float, round toward +inf |
| `I2F.RP` | 11c | Signed int32 to float, round toward +inf (non-uniform) |
| `I2F.U64.RP` | 11b | Unsigned int64 to float, round toward +inf |
| `I2F.F64.S64` | 11g | Signed int64 to double |
| `I2FP.F32.S32` | 11d, 11g | Signed int32 to float (`FP` suffix disambiguates precision) |
| `F2I.FTZ.U32.TRUNC.NTZ` | 11a, 11b | Float to uint32 with FTZ, truncation, non-toward-zero rounding |
| `F2I.U64.TRUNC` | 11b | Float to uint64 with truncation |
| `F2F.F32.F64` | 11g | Double to float narrowing conversion |
| `IABS` | 11c | Integer absolute value |
| `FSEL` | 11d, 11g, 11h | Float select based on predicate |
| `SEL.64` | 11b | 64-bit select based on predicate |
| `FFMA.SAT` | 11f | FFMA with output saturated to `[0, 1]` |
| `FFMA.RM` | 11f | FFMA with round mode toward -inf |
| `MUFU.LG2` | 11e | Hardware log2 approximation |
| `MUFU.EX2` | 11f | Hardware 2^x approximation |
| `MUFU.RSQ` | 11h, 11i | Hardware reciprocal sqrt approximation |
| `R2P` | 11g | Register to predicates (unpack register bits to P0..P7) |
| `DMUL` | 11g | Double-precision multiply |
| `IMAD.SHL.U32` | 11g | Integer multiply-add with implicit left shift, unsigned |
| `IMAD.WIDE.U32` | 11b | 32×32→64 multiply accumulate, unsigned explicit |
| `SHF.L.W.U32.HI` | 11g | Funnel shift left, with wrap, high half, unsigned |
| `IADD3.X` | 11b | Three-input add with carry-in |
| `IADD.X` | 11b | Two-input add with carry-in |
| `IADD.64` | 11b | 64-bit integer add (no carry chain needed) |
| `ISETP.GE.U64.AND` | 11b | 64-bit unsigned greater-equal predicate |
| `ISETP.NE.S64.AND` | 11b | 64-bit signed not-equal predicate |

Twenty-six opcodes or opcode-modifier combinations new to the project.

## Modifier inventory (new)

| Modifier | Appears on | Semantics |
|---|---|---|
| `.FTZ` | FMUL, FADD, FSETP, F2I | Flush subnormals to zero (input and output) |
| `.SAT` | FFMA | Saturate result to `[0, 1]` |
| `.RM` | FFMA | Round mode: toward -inf |
| `.RP` | I2F, UI2F | Round mode: toward +inf |
| `.NEU` | FSETP | Not equal unordered (NaN comparisons return true) |
| `.GTU` | FSETP | Greater than unordered |
| `.NTZ` | F2I | Non-toward-zero rounding (resolves signed zero edge case) |
| `.W` | SHF.L | With wrap (funnel shift behavior) |
| `.CONSTANT` | LDG.E | Load from constant/read-only path |

The `.CONSTANT` modifier on LDG.E is the first observation of this variant in the project. It signals that the global memory region is read-only, allowing the GPU to use the constant cache path for the load. Used in 11g's Payne-Hanek table access.

## Patterns for later reference

Nine arithmetic patterns emerge from this chapter, each likely to appear in production kernels:

1. **Granlund-Montgomery integer division.** 5-stage reciprocal multiplication. Recognizable signature: UI2F.U32.RP → MUFU.RCP → IADD magic → F2I.FTZ → IMAD.HI.U32 correction chain.

2. **Multi-precision 64-bit arithmetic.** IMAD.WIDE.U32 + IADD3.X + IADD.X chains. Used anywhere 64-bit integer math appears at high precision.

3. **Signed-via-unsigned with LOP3 0x3c sign fix-up.** IABS + unsigned algorithm + sign XOR via LOP3 with LUT 0x3c.

4. **Subnormal pre-scale + post-correction.** FSETP.GEU + FMUL by 2^24 + MUFU + correction. Universal for FP math involving MUFU.

5. **Remez polynomial approximation via Horner's method.** Long chain of FFMAs with hardcoded coefficients. Recognizable by the monotonic pattern of FFMA with `.reuse` on the same multiplicand register.

6. **expf decomposition `exp(x) = 2^k * 2^f`.** FFMA.SAT + FFMA.RM + SHF.L integer part extraction + MUFU.EX2 + FMUL. The magic constants `0x437c0000 = 252.0` and `12582913 = 0x00C00001` are idiomatic.

7. **Newton-Raphson sqrt fast path.** MUFU.RSQ + FMUL (q0) + FMUL (y0/2) + FFMA (error) + FFMA (correction). Classic structure.

8. **Payne-Hanek range reduction.** LDG.E.CONSTANT table of 2/π + UIADD3 counter loop + IMAD.WIDE.U32 accumulation + F2F.F32.F64 final narrow. Only needed for trigonometric functions with arguments larger than ~2^17.

9. **Local CALL pattern.** BSSY.RECONVERGENT + conditional CALL.REL.NOINC + main body + BSYNC.RECONVERGENT + EXIT + slowpath body + RET.REL.NODEC. Used for rare slowpaths where inlining would bloat the common case.

## Observations worth flagging

**On R2P (register to predicates).** Opcode at 0x07c0 in 11g: `R2P PR, R3, 0x3`. Copies selected bits of R3 into predicate registers P0..P7. Rare but present. Useful when packed flags need to fan out into multiple predicated paths.

**On LDG.E.CONSTANT.** First observation of the `.CONSTANT` suffix on LDG.E in the project. Indicates read-only access to global memory via the constant cache path, benefiting from read-only caching behavior. Distinct from LDC (constant memory bank) and LDCU (uniform constant). Used when a read-only table is stored in regular global memory but accessed in a read-only pattern.

**On `.reuse` frequency.** Six occurrences in 11b, seven in 11g, zero in the simpler kernels (11a, 11c, 11e, 11f, 11i, 11j). The `.reuse` flag correlates with register pressure: dense kernels benefit more from operand cache reuse because they have more dependencies compressed into fewer register slots.

**On HFMA2 packed constant loading.** In 11d at 0x00c0: `HFMA2 R9, -RZ, RZ, 1.443359375, -0.2030029296875`. The two FP16 half-precision values are packed into a 32-bit register. The resulting R9 is then read as FP32 in the first polynomial FFMA at 0x0190. [GAP] Why ptxas chose HFMA2 packed loading over MOV immediate is not fully understood — the bit pattern of the two FP16s when interpreted as FP32 must correspond to a specific coefficient value, but the exact reasoning (whether this is a compiler optimization or an artifact of constant folding) needs further investigation.

**On the `0xffffffe` adjustment constant in integer division.** Used in 11a, 11b as a bias added to the reciprocal estimate before F2I conversion. Value is `(2^28 - 2)`. [HYP] This constant tunes the F2I rounding behavior to match the algorithm's correctness requirements, avoiding the need for a triple-correction pass.

**On `.CONSTANT` vs `.SYS` vs default LDG.** The descriptor-based LDG on SM120 has multiple cache path suffixes:
* Default LDG.E: standard L2 path with full caching
* LDG.E.CONSTANT: read-only path via constant cache hardware
* LDG.E.SYS: system memory coherent access (not observed in this chapter)

**On IADD3.X vs IADD3.** The `.X` suffix indicates carry-in from the condition code register (CC). IADD3.X reads CC, IADD3 doesn't. Essential for multi-precision arithmetic where the carry chains through multiple word operations.

**On SEL.64.** 64-bit version of SEL. Selects between two 64-bit register pairs based on a predicate. Used in 11b for the final u64 division result assembly.

## Gaps explicitly acknowledged

1. The exact bit encoding of the HFMA2 packed constants in 11d and 11e is not decoded to FP32 interpretation.
2. The Payne-Hanek reconstruction in 11g (combining R11..R16 partial products via SHF.L.W.U32.HI) is not traced to the end.
3. The magic constants `0x437c0000 = 252.0` and `12582913` in 11f are not formally verified as standard library idioms — [HYP] they match glibc's pattern but CUDA-specific verification is left for a future pass.
4. Control code bit fields (yield, stall, scoreboards) are read at the level of "stall count appears", not decoded bit-by-bit.
5. The `0xffffffe` adjustment in integer division is accepted as a tuning constant; its derivation from the algorithm's error analysis is not worked out.
6. The NCU-level cost comparison between inline u32 division (11a) and the hypothetical external `__cuda_sm20_div_u32` helper (not observed on SM120/CUDA 13) is not measured.
7. [HYP] The observation that CUDA 13.2 prefers inlining over external helpers needs validation against an older CUDA version to confirm this is a compiler change rather than an SM120-specific pattern.

## Open questions for future work

* Does compiling the same chapter on CUDA 11 or 12 produce `__cuda_sm20_div_u32` helpers? This would confirm the hypothesis that inlining aggressiveness increased between versions.
* Do chapter 06's u16 modulo helpers also inline under CUDA 13, or does u16 still use the external helper?
* Is there a threshold (in terms of source-level complexity) above which ptxas falls back to emitting named external helpers?
* For `sinf` with large arguments, is the Payne-Hanek table accessed via constant memory (c[0x4]) or global memory with LDG.E.CONSTANT? The two descriptors in 11g suggest global memory, but the `c[0x4][URZ]` lookup at 0x0190 suggests constant memory for the base pointer. Needs clarification.
* What is the exact precision of `__fdividef` vs `a/b` in SASS? Both compile to MUFU.RCP + FMUL + subnormal handling — there's no visible difference in 11j. Is the C-level difference only at the compiler front end?