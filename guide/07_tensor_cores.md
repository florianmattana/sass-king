# Chapitre 7 — Tensor cores : HMMA, QMMA, OMMA

## Le but du chapitre

Comprendre les **3 familles de MMA** sur SM120, savoir distinguer HMMA / QMMA / OMMA dans un dump, et reconnaître les patterns de chain et de block scaling. À la fin, tu peux dire d'un kernel "il utilise FP8 sans block scaling, throughput max ~250 TFLOPS" ou "il utilise OMMA mxf4nvf4 4X, throughput peak 900 TFLOPS atteint si pipelined correctement".

## Les 3 familles

Sur SM120 il existe **3 opcodes MMA distincts**, chacun avec son hardware pipeline et ses dtypes supportés :

| Famille | Low byte | Shape | Dtypes input | Path |
|---|---|---|---|---|
| **HMMA** | `0x3c` | m16n8k16 | FP16, BF16 | Tensor core standard |
| **QMMA** | `0x7a` | m16n8k32 | FP8, FP6, FP4 | `kind::f8f6f4` ou `kind::mxf8f6f4` (avec `.SF`) |
| **OMMA** | `0x7f` | m16n8k64 | FP4 only | `kind::mxf4nvf4` (toujours `.SF`) |

**Première chose à faire** quand tu lis un dump : compter les opcodes par famille. Tu sais immédiatement quelle voie le kernel utilise.

```bash
grep -oP '0x[0-9a-f]{16}' my_kernel.sass | grep -oP '..$' | sort | uniq -c
```

Si tu vois beaucoup de `7f`, c'est OMMA = path peak FP4.
Si tu vois beaucoup de `7a` avec `.SF` dans le mnemonic = QMMA scaled.
Si juste `3c` = HMMA classique.

## HMMA — Half precision MMA

```
HMMA.16816.F32 R12, R4, R8, R12 ;       /* 0x000000080c0c723c */
               D    A   B   C
```

**Décortique** :
* `.16816` = shape m=16, n=8, k=16.
* `.F32` = accumulateur FP32 (le plus courant).
* D, A, B, C sont des fragments distribués dans le warp.

**Variantes** :
* `.F16` = accumulateur FP16 (plus rapide, moins précis).
* `.BF16` = inputs BF16 au lieu de FP16.

**Throughput SM120** :
* HMMA F16 acc : ~60 TFLOPS.
* HMMA F32 acc : ~30 TFLOPS.

**Latence chain** : ~35 cycles par MMA en chain (chap 13e).

**Use case** : training, inference standard, partout où FP16/BF16 est suffisant et tu n'as pas besoin de FP8/FP4.

## QMMA — Quantized MMA

```
QMMA.16832.F32.E4M3.E4M3 R12, R4, R8, R12 ;       /* 0x000000080c0c727a */
                              standard FP8 E4M3 × E4M3, sans block scaling
```

**Décortique** :
* `.16832` = shape m=16, n=8, k=32. K **deux fois plus grand** que HMMA.
* `.F32` = accumulateur FP32.
* `.E4M3.E4M3` = dtypes A et B (un par opérande, ici les deux en E4M3).

**Avec block scaling** :
```
QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R4, R8, R12, R18, R31, URZ ;
                                                    ^^^  ^^^  ^^^
                                                    SFA  SFB  bid/tid
```

* `.SF` = block scaling activé.
* `.E8` = scale dtype UE8M0.
* 7 opérandes au lieu de 4 (les 3 derniers sont les scales).

**Dtypes possibles** :
* FP8 : E4M3, E5M2.
* FP6 : E3M2, E2M3.
* FP4 : E2M1.

Les combinaisons valides dépendent du kind. Sans `.SF` (kind::f8f6f4), beaucoup de combinaisons. Avec `.SF` (kind::mxf8f6f4), même chose mais avec scaling 1X.

**Throughput SM120** :
* QMMA E4M3 × E4M3 F32 : ~250-500 TFLOPS.
* QMMA E2M1 × E2M1 .SF (scale_vec::1X) : ~500 TFLOPS.

**Latence chain** : ~35 cycles par MMA.

**Use case** : inference moderne avec quantization FP8 ou FP4 standard. Si tu utilises CUTLASS GEMM FP8 ou FlashInfer FP8, tu vois du QMMA.

## OMMA — Octa-precision MMA (FP4 peak)

```
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X R12, R4, R8, R12, R18, R31, URZ ;
                                                    /* 0x7000000a0c0c747f */
```

**Décortique** :
* `.16864` = shape m=16, n=8, **k=64**. K **deux fois plus grand** que QMMA.
* `.SF` = block scaling obligatoire (pas d'OMMA sans SF).
* `.E2M1.E2M1` = inputs FP4.
* `.UE4M3` = scale dtype UE4M3 (plus flexible que UE8M0).
* `.4X` = scale_vec::4X (1 scale per 16 elements, le plus fin).

**Variantes** :
* `.2X` = scale_vec::2X (1 scale per 32 elements).
* `.4X` = scale_vec::4X (1 scale per 16 elements).

**Throughput SM120** :
* OMMA mxf4nvf4 4X : **900+ TFLOPS** (annoncé NVIDIA).
* En chain serial : ~254 TFLOPS observé (chap 16d).
* Le peak nécessite **pipelining parfait** entre les tiles.

**Latence chain** : ~29 cycles par MMA (chap 16d). Plus rapide que HMMA/QMMA.

**Use case** : production FP4 inference où tu veux le peak. FlashInfer FP4, TensorRT-LLM FP4, certains kernels Marlin compilés pour SM120+.

## Operand layout : comprendre les fragments

Les MMAs sont **warp-level** : les 32 threads contribuent collectivement aux fragments A, B, C, D. Tu vois 1 instruction MMA dans le SASS, mais elle exécute sur les 32 threads simultanément.

**Pour HMMA m16n8k16** (par thread) :
* A = 2 registres = 4 halfs (16-bit chacun).
* B = 1 registre = 2 halfs.
* C = 4 registres = 4 floats.
* D = 4 registres (souvent = C pour chain).

Si tu vois `HMMA.16816.F32 R12, R4, R8, R12` :
* D = R12, R13, R14, R15 (4 regs, accumulator out).
* A = R4, R5 (2 regs).
* B = R8 (1 reg).
* C = R12, R13, R14, R15 (4 regs, accumulator in = output, chain pattern).

**Pour QMMA m16n8k32** :
* A = 4 registres (k=32, donc 2× plus de data que HMMA).
* B = 2 registres.
* C = 4 registres.
* D = 4 registres.

**Pour OMMA m16n8k64** :
* A = 8 registres (k=64, donc 4× plus que HMMA).
* B = 4 registres.
* C = 4 registres.
* D = 4 registres.

**À reconnaître** : le nombre de registres dans A et B te dit immédiatement la shape (donc la famille).

## Le pattern chain

Une **chain MMA** est une série de MMAs successives qui accumulent dans le même registre. C'est le pattern canonique d'un GEMM le long de la dimension K (reduction axis).

```
HMMA.16816.F32 R12, R4, R8, RZ ;         1st MMA: D = A0 * B0 + 0
HMMA.16816.F32 R12, R5, R9, R12 ;        2nd: D = A1 * B1 + R12 (accumulate)
HMMA.16816.F32 R12, R6, R10, R12 ;       3rd: D = A2 * B2 + R12
HMMA.16816.F32 R12, R7, R11, R12 ;       4th: D = A3 * B3 + R12
```

**Caractéristiques** :
* **Premier MMA spécial** : C = RZ (pas d'accumulation). Souvent précédé de `CS2R.32 R12, SRZ` pour zero-init.
* **MMAs suivantes** : C = R_acc, D = R_acc (colocalisés).
* **Reuse flags** sur A et B si possible (pour économiser RF reads).

**Dans un dump, identifier une chain** : série d'MMAs avec destination identique (R12 dans l'exemple) et C = même registre = chain. Si tu vois 16 MMAs en chain, c'est probablement un K-loop unrolled (16 K-tiles).

**Si la chain a des wait masks à 0** : pipeline densément utilisé, optimal.
**Si wait masks non-zéro entre MMAs successifs** : ptxas a inséré une attente. Souvent à cause d'un load qui doit finir.

## Le pattern setup-MMA

Avant chaque MMA, il faut que les fragments soient en registres. Pattern typique :

```
LDSM.16.M88.4 R4, [R0+UR7] ;       load fragment A (4 LDSM tiles, 8 registres)
LDSM.16.M88.2 R8, [R0+UR8] ;       load fragment B (2 LDSM tiles, 4 registres)
HMMA.16816.F32 R12, R4, R8, R12 ;  compute MMA
```

**LDSM** est l'instruction qui charge directement depuis shared memory dans le layout fragment. Sans LDSM, il faudrait charger via plusieurs LDS et reformater — beaucoup moins efficient.

**Pourquoi B avant A** : observation empirique (chap 17). Ptxas émet systématiquement le LDSM B avant le LDSM A. Pas d'explication confirmée, hypothèse : ordre de scheduling pour le pipeline tensor core.

## Block scaling : .SF, scale_vec, dtypes

**Block scaling** = chaque groupe d'éléments partage un scale factor. Essentiel pour FP4 et FP6 où la précision intrinsèque est faible.

**Trois axes de configuration** :

### 1. scale_vec (granularité)
* `1X` : 1 scale per 32 elements. Implicite dans QMMA `.SF`.
* `2X` : 1 scale per 32 elements (mais pour k=64). OMMA `.2X`.
* `4X` : 1 scale per 16 elements. OMMA `.4X`. Le plus fin.

### 2. scale dtype
* `E8` (= UE8M0) : 8 bits exposant only. Power-of-2 scaling, gratuit côté hardware.
* `UE4M3` : 4 bits exp + 3 bits mantissa. Plus flexible, supporte 1.125, 1.5, etc.

### 3. kind
* `kind::f8f6f4` : sans scaling. QMMA (sans `.SF`).
* `kind::mxf8f6f4` : avec scaling 1X. QMMA `.SF`.
* `kind::mxf4nvf4` : FP4 peak avec scaling 2X/4X. OMMA `.SF`.

**Trade-offs** :
* 1X : moins de scales à charger, moins de regs, mais grosse granularité.
* 4X : 4 scales par fragment K, plus de regs pour les scales, mais précision fine.

**Pour identifier si un kernel utilise le peak path** :
1. Cherche les low bytes `0x7f` (= OMMA).
2. Vérifie qu'il y a `.SF.16864`.
3. Cherche `.4X` (peak path).
4. Cherche `.UE4M3` (scale dtype peak).

Si tu trouves les 4 conditions, c'est le path peak FP4. Sinon, autre voie.

## Reconnaître ce qui n'est PAS sur SM120

**Sur SM100a (B200, datacenter)** :
* `tcgen05.mma` = MMA qui écrit dans TMEM (pas registres). Hors-scope.
* `cta_group::2` = 2-CTA MMA via clusters. Hors-scope.

**Sur SM90a (Hopper)** :
* `wgmma.mma_async` = warp-group MMA (128 threads). Hors-scope.

Si tu vois ces instructions dans un dump, c'est pas SM120, et ton toolkit ne s'applique pas.

## Performance practique : combien faut-il de MMAs en chain ?

Pour amortir l'overhead du setup (LDSM, calcul d'addresses, scheduling), il faut une **chain MMA suffisamment longue**.

**Règle heuristique** :
* Chain de **4-8 MMAs** : minimum décent.
* Chain de **16-32 MMAs** : optimal pour peak.
* Chain de **<4 MMAs** : tu paies plus de setup que de compute. Souvent le cas dans des decode batch=1 — c'est pourquoi GEMV est si pénalisant pour les tensor cores.

Dans un audit, mesure la longueur des chains. Si toutes tes chains font 1-2 MMAs, tu n'utilises pas les tensor cores efficacement.

## Bilan

Tu sais maintenant :
* Distinguer HMMA / QMMA / OMMA via opcodes et shape.
* Identifier si block scaling est utilisé (`.SF`, `.E8`, `.UE4M3`).
* Reconnaître le path peak FP4 (OMMA mxf4nvf4 4X).
* Lire un fragment layout (combien de registres par opérande).
* Repérer un chain MMA et estimer son efficience.

C'est suffisant pour auditer 80% des kernels GEMM, attention, et inference.

## Ce qui suit

Le chapitre 8 attaque **cp.async et le bridge global → shared**. C'est ce qui permet aux kernels modernes d'overlap compute et memory transfer. Sans cp.async, tu fais du staging classique (LDG → STS → BAR.SYNC → LDS), et tu paies la latence à chaque étape.
