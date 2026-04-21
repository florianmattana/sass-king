# Chapitre 11 — Patterns SASS communs

## Le but du chapitre

Reconnaître les **shapes structurelles récurrentes** dans les dumps SASS : GEMM mainloop, softmax, warp reduction, decode attention, quantization. Une fois que tu connais ces patterns, tu peux ouvrir n'importe quel kernel et identifier ses sections en quelques minutes.

## Pourquoi les patterns

La grande majorité des kernels production sur GPU sont des combinaisons de quelques patterns canoniques. Reconnaître ces patterns te permet de :
* Sauter les sections que tu connais déjà.
* Concentrer l'attention sur les parties novatrices.
* Comparer rapidement avec une implémentation de référence.

## Pattern 1 : GEMM mainloop (CUTLASS-style)

C'est le pattern le plus fondamental. Une multiplication de matrices avec tile blocking et software pipelining.

**Squelette** :
```
// === Prologue : prefetch first 2 K-tiles ===
LDGSTS [smem_A_buffer_0], desc[UR_A][R_offset_0.64] ;     × N (selon tile size)
LDGSTS [smem_B_buffer_0], desc[UR_B][R_offset_0.64] ;     × M
LDGDEPBAR ;                                                ← commit group 0

LDGSTS [smem_A_buffer_1], desc[UR_A][R_offset_1.64] ;     × N
LDGSTS [smem_B_buffer_1], desc[UR_B][R_offset_1.64] ;     × M
LDGDEPBAR ;                                                ← commit group 1

// === Main loop ===
loop_start:

DEPBAR.LE SB0, 0x1 ;                                       ← attendre (sauf 1 prefetch en flight)
BAR.SYNC.DEFER_BLOCKING 0x0 ;

// Load fragments from current K-tile
LDSM.16.M88.4 R12, [R_smem_A_current] ;
LDSM.16.M88.2 R20, [R_smem_B_current] ;

// MMA chain (unrolled K)
HMMA.16816.F32 R28, R12, R20, R28 ;
HMMA.16816.F32 R28, R13, R21, R28 ;
HMMA.16816.F32 R28, R14, R22, R28 ;
HMMA.16816.F32 R28, R15, R23, R28 ;

// Prefetch K-tile +2 in background
LDGSTS [smem_A_buffer_next], desc[UR_A][R_next_offset.64] ;
LDGSTS [smem_B_buffer_next], desc[UR_B][R_next_offset.64] ;
LDGDEPBAR ;

// Iterate
@P0 BRA loop_start ;

// === Epilogue : drain pipeline + write output ===
DEPBAR.LE SB0, 0x0 ;
... process last tiles ...
... STG.E to global output ...
EXIT ;
```

**Signaux à reconnaître** :
* Burst de LDGSTS (N+M par stage) suivi de LDGDEPBAR.
* DEPBAR.LE SB0 + BAR.SYNC pattern.
* LDSM B avant LDSM A (convention ptxas).
* Chain MMAs (D=C=même registre).
* Pattern "back-edge BRA" indiquant la K-loop.

**Ratio attendu** :
* MMAs / total instructions : 5-30%.
* LDGSTS / MMAs : 1-2 par MMA.

Si MMAs < 5% du total, le kernel est probablement memory-bound ou mal optimisé.

## Pattern 2 : Warp reduction (sum, max)

Quand tu réduis 32 valeurs (une par lane) en une seule, tu utilises SHFL.BFLY (butterfly shuffle).

**Squelette** (reduction sum) :
```
// Initial : R0 contient une valeur par lane
SHFL.BFLY PT, R1, R0, 0x10, 0x1f ;     échange lanes 0-15 vs 16-31
FADD R0, R0, R1 ;                       ajoute neighbor

SHFL.BFLY PT, R1, R0, 0x8, 0x1f ;       échange lanes 0-7 vs 8-15 (et symetrique)
FADD R0, R0, R1 ;

SHFL.BFLY PT, R1, R0, 0x4, 0x1f ;
FADD R0, R0, R1 ;

SHFL.BFLY PT, R1, R0, 0x2, 0x1f ;
FADD R0, R0, R1 ;

SHFL.BFLY PT, R1, R0, 0x1, 0x1f ;
FADD R0, R0, R1 ;

// Maintenant tous les lanes ont la même valeur = somme totale du warp
```

**Pour max** : remplace FADD par FMNMX (fmax).

**Signaux à reconnaître** :
* 5 SHFL.BFLY avec deltas 0x10, 0x8, 0x4, 0x2, 0x1 (puissances de 2 décroissantes).
* Suivi par opérateur de réduction (FADD, FMUL, FMNMX, IADD, etc.).
* Pattern compact, ~10 instructions.

**Latence** : ~50 cycles (5 × 10 cycles par SHFL+OP).

## Pattern 3 : Online softmax (FlashAttention-style)

L'astuce de l'online softmax permet de calculer attention en streaming, sans matérialiser la matrice S complète.

**Squelette par chunk** :
```
// Compute QK^T (MMA chain) → acc[]

// Local max reduction
FMNMX R_local_max, R_acc_0, R_acc_1 ;
FMNMX R_local_max, R_local_max, R_acc_2 ;
FMNMX R_local_max, R_local_max, R_acc_3 ;

// Warp reduction max
SHFL.BFLY PT, R_tmp, R_local_max, 0x10, 0x1f ; FMNMX R_local_max, R_local_max, R_tmp ;
SHFL.BFLY PT, R_tmp, R_local_max, 0x8, 0x1f ; FMNMX R_local_max, R_local_max, R_tmp ;
... (suite)

// Update running max
FMNMX R_new_m, R_m, R_local_max ;

// Compute alpha = exp(old_m - new_m) for rescaling
FADD R_diff, R_m, -R_new_m ;
MUFU.EX2 R_alpha, R_diff ;                    ← exp via MUFU.EX2

// Compute exp(s_ij - new_m) for new contributions
FADD R_e_0, R_acc_0, -R_new_m ;
MUFU.EX2 R_e_0, R_e_0 ;
... × 4 (one per acc)

// Local sum reduction (similar pattern)
FADD R_local_sum, R_e_0, R_e_1 ;
FADD R_local_sum, R_local_sum, R_e_2 ;
... + warp reduction

// Update running denominator
FFMA R_new_l, R_alpha, R_l, R_local_sum ;

// Rescale O
FMUL R_O_0, R_O_0, R_alpha ;
FMUL R_O_1, R_O_1, R_alpha ;
... etc

// Accumulate new contribution to O via FFMA with V
FFMA R_O_0, R_e_0, R_V_0, R_O_0 ;
... etc
```

**Signaux à reconnaître** :
* Mélange de FMNMX (max), FADD (différences), MUFU.EX2 (exp), FFMA (rescaling).
* Patterns warp-reduction (SHFL.BFLY) entre les sections.
* Beaucoup de FMUL pour les rescaling.

**Coût typique** : 50-100 instructions par chunk de softmax. Comparable au coût d'une chain MMA, donc ne devrait pas dominer le kernel si bien équilibré.

## Pattern 4 : Decode attention (batch=1)

Le cas extrême du LLM inference : un seul query token, un long sequence de keys/values. Memory-bound car peu de compute par byte loadé.

**Squelette** :
```
// Pour chaque K-tile dans le KV cache (long boucle)
loop_kv:
    LDG.E.128 R_K, [R_K_ptr.64] ;                  ← load K tile
    LDG.E.128 R_V, [R_V_ptr.64] ;                  ← load V tile
    
    // Petit MMA pour QK^T (juste 1-2 MMAs car batch=1)
    HMMA.16816.F32 R_acc, R_Q, R_K, R_acc ;
    
    // Online softmax (cf pattern 3)
    ... softmax pattern ...
    
    // Petit MMA pour PV
    HMMA.16816.F32 R_O, R_p, R_V, R_O ;
    
    @P0 BRA loop_kv ;

// Final normalization
FFMA R_O_norm, R_O, 1/R_l, RZ ;
STG.E ... R_O_norm ;
```

**Signaux à reconnaître** :
* Très peu de MMAs (1-4 par K-tile).
* Beaucoup de LDG.E (chaque K-tile loadé).
* Pattern softmax inline.
* Boucle longue (back-edge body fait 100-500 instructions).
* Ratio MMA/LDG très bas.

**Diagnostic** : si tu vois ce pattern, le kernel est memory-bound. Tu n'iras pas plus vite avec des MMAs plus rapides — il faut plus de bandwidth.

## Pattern 5 : Quantization (FP4, FP8)

Le kernel doit convertir entre dtypes (par exemple FP16 → FP4 + scaling).

**Squelette pour FP4 E2M1 encoding** :
```
// Pour chaque element :
FABS R_abs, R_value ;                 ← |val|

// Cascade d'if pour mapper magnitude → nibble
FSETP.GE.F32 P0, PT, R_abs, 5.0 ;
@P0 BRA case_7 ;
FSETP.GE.F32 P0, PT, R_abs, 3.5 ;
@P0 BRA case_6 ;
FSETP.GE.F32 P0, PT, R_abs, 2.5 ;
@P0 BRA case_5 ;
... (8 niveaux)
case_0:
MOV R_nibble, 0x0 ;
BRA done ;
case_1:
MOV R_nibble, 0x1 ;
BRA done ;
...
done:

// Combine sign + nibble + shift
LOP3 R_byte, R_sign, R_nibble, RZ, ... ;
```

**Signaux à reconnaître** :
* Cascades de FSETP + @P BRA (8 niveaux pour E2M1, plus pour autres formats).
* Beaucoup de IADD/IMAD pour packer les nibbles en bytes.
* Beaucoup de LOP3 (operations bitwise sur les bits packés).
* Beaucoup de LDS.U8 / STS.U8 si la quantization stocke en byte-packed.

**Pattern observé** : 8 niveaux × ~6 instructions = 48 instructions par quantization d'un élément. Multiplié par les éléments, c'est typiquement la phase la plus coûteuse d'un kernel quantization-heavy.

## Pattern 6 : Block scaling computation

Avant de faire un MMA `.SF`, il faut calculer les scale factors. Pour UE8M0, c'est :

```cpp
uint8_t scale = ceil(log2(max_abs(block))) + 127;
```

**Squelette SASS** :
```
// Find max in block
... reductions ...

// log2 via MUFU.LG2
MUFU.LG2 R_log, R_max ;

// Ceiling
F2I.CEIL.F32 R_ceil, R_log ;

// Add bias 127
IADD R_scale, R_ceil, 0x7f ;

// Convert to uint8 storage
... pack ...
```

**Signaux à reconnaître** :
* MUFU.LG2 ou MUFU.EX2 inversés pour scale computation.
* F2I.CEIL.F32 spécifiquement pour UE8M0 (ceiling).
* IADD avec 0x7f (127, le bias).

## Pattern 7 : Memory padding et bank conflict avoidance

Pour éviter les bank conflicts en shared memory, les kernels ajoutent du padding. Au niveau SASS, tu vois des adresses calculées avec stride non-power-of-2.

**Exemple** : tile 128×128 avec padding +4 par row :
```
// Compute address: row * (128 + 4) + col
IMAD R_addr, R_row, 0x84, R_col ;       ← stride 132 au lieu de 128
LDS R_data, [R_smem_base + R_addr] ;
```

Le `0x84` (132) au lieu de `0x80` (128) est le signal du padding.

## Comment matcher patterns dans la pratique

**Workflow** :

1. Identifie la région chaude (chap 10).
2. Compte les mnemonics présents dans cette région.
3. Compare avec les patterns ci-dessus :
   - Beaucoup de MMA + LDGSTS = GEMM mainloop.
   - Mélange MMA + MUFU.EX2 + SHFL = attention/softmax.
   - Beaucoup de FSETP + BRA = quantization.
   - Beaucoup de LDG sans MMA correspondant = decode/memory-bound.
4. Vérifie le ratio compute/memory pour confirmer.

**Tu n'as souvent pas besoin de comprendre toutes les instructions** — juste d'identifier le pattern global et de t'en assurer qu'il fonctionne comme attendu.

## Limites

Les patterns sont des **prototypes**. Un kernel réel mélange souvent plusieurs patterns (decode attention contient à la fois decode pattern + softmax + petit GEMM). Avec l'expérience, tu apprends à les **décomposer**.

## Ce qui suit

Le chapitre 12 attaque le **diagnostic** : comment identifier les bottlenecks dans un dump SASS. Tu sauras dire pourquoi tel kernel performe mal et où agir.
