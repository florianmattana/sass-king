# Chapitre 6 — Memory hierarchy

## Le but du chapitre

Comprendre les **5 familles de loads/stores** que tu rencontres dans le SASS, leur latence approximative, leurs modifiers, et les patterns de coalescing / banking conflicts à reconnaître. À la fin, tu peux estimer si une région chaude est memory-bound et où le bottleneck se trouve dans la hiérarchie.

## La hiérarchie sur SM120

```
Registers           ← R0-R255, accès 1 cycle
    ↓
Constant memory     ← LDC, lecture broadcast, ~5-10 cycles
    ↓
Shared memory       ← LDS/STS, ~20-30 cycles si pas de bank conflict
    ↓
L1 cache            ← LDG hit L1, ~30 cycles
    ↓
L2 cache            ← LDG hit L2, ~150 cycles
    ↓
DRAM (HBM)          ← LDG miss, ~400-600 cycles
    ↓
Local memory        ← STL/LDL, register spill (en fait global), pire des deux mondes
```

Les latences sont approximatives (NVIDIA ne les publie pas). Ordres de grandeur stables sur Ampere/Ada/Blackwell.

## LDG — Load Global

Charge depuis la mémoire globale (HBM) vers un registre.

**Modifiers principaux** :

```
LDG.E R4, [R2.64] ;                    32-bit, addressing 64-bit
LDG.E.128 R4, [R2.64] ;                128-bit dans R4:R5:R6:R7
LDG.E.64 R4, [R2.64] ;                 64-bit dans R4:R5
LDG.E.CONSTANT.128 R4, [R2.64] ;       hint que c'est read-only (cache aggressif)
LDG.E.LTC128B.128 R4, [R2.64] ;        L2 cache hint, alignment 128B
LDG.E.SYS R4, [R2.64] ;                system-wide coherence (rare)
```

**Modifier `.E`** : Extended addressing 64-bit. Indispensable sur GPU moderne pour pouvoir adresser plus de 4GB.

**Modifier de taille** (`.32` implicite, `.64`, `.128`) : largeur du transfer **par thread**. Le `.128` est le plus efficient pour la bandwidth — un warp transfère 32 × 16 bytes = 512 bytes par instruction.

**Modifier de cache hint** :
* `.CONSTANT` : lecture read-only, agressivement cached. Utile pour des données ne changeant pas pendant le kernel.
* `.LTC*` : hints sur le L2 cache (line size, allocation policy).
* `.SYS` : system-wide (entre GPU et CPU), désactive certains caches. Rare.

## Coalescing : ce qui fait ou défait la bandwidth

**Définition simple** : si les 32 threads d'un warp font `LDG` sur 32 adresses consécutives alignées, le hardware combine les 32 transactions en **1 seule transaction de 128 bytes** (ou plus). Sinon, plusieurs transactions = perte de bandwidth.

**Exemples**.

Coalescé :
```cpp
float x = global_array[threadIdx.x];   // adresses 0, 4, 8, ..., 124 → 1 transaction 128B
```

Non coalescé :
```cpp
float x = global_array[threadIdx.x * 32];   // adresses 0, 128, 256, ... → 32 transactions
```

**Au niveau SASS**, tu ne vois **pas directement** le coalescing. C'est le pattern d'adresses qui détermine. Tu peux deviner :

* Si tu vois des `LDG.E.128` consécutifs avec adresses calculées simplement (`R7+0x10`, `R7+0x20`, etc.), probablement coalescé.
* Si tu vois des `LDG.E` (sans `.128`) avec adresses calculées via IMAD complexe, probablement non coalescé ou indirect.

**Diagnostic précis** : NCU. Métriques `l1tex__data_pipe_lsu_wavefronts*` et le warp stall reason `stall_lg_throttle`.

## LDS — Load Shared

Charge depuis shared memory.

**Variants size** :
```
LDS.U8 R4, [R2] ;          1 byte unsigned
LDS.S8 R4, [R2] ;          1 byte signed
LDS.U16 R4, [R2] ;         2 bytes unsigned
LDS.S16 R4, [R2] ;         2 bytes signed
LDS R4, [R2] ;             4 bytes (default)
LDS.64 R4, [R2] ;          8 bytes dans R4:R5
LDS.128 R4, [R2] ;         16 bytes dans R4:R5:R6:R7
```

**Latence** : ~20-30 cycles si pas de conflit. C'est rapide.

## Banking conflicts

Shared memory est divisée en **32 banks**, chaque bank fait 4 bytes wide. La bank d'une adresse est `(addr / 4) % 32`.

**Pas de conflit** : 32 threads accèdent 32 banks différents (typique : `addr = base + tid * 4`).

**Conflit N-way** : N threads accèdent au même bank avec des offsets différents. Le warp prend N cycles au lieu d'1.

**Broadcast (cas spécial)** : tous les threads accèdent **la même adresse**. Pas de conflit (broadcast natif, 1 cycle).

**Au niveau SASS** : tu ne vois pas le bank conflict directement. Tu peux suspecter :
* Si tu vois beaucoup de `LDS.U8` consécutifs avec les mêmes registres de base mais offsets différents (par exemple les 14 LDS.U8 du kernel FP4 attention chap audit).
* Si NCU rapporte `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum > 0`.

**Fix classique** : padding (ajouter 1 élément par row pour décaler les banks) ou layout transpose.

## LDC — Load Constant

Charge depuis constant memory. Très cacheable, latence courte.

```
LDC R1, c[0x0][0x37c] ;       32-bit depuis bank 0, offset 0x37c
LDC.64 R2, c[0x0][0x380] ;    64-bit dans R2:R3
LDCU UR5, c[0x0][0x3a0] ;     load uniform (résultat dans UR)
LDCU.64 UR8, c[0x0][0x358] ;  load uniform 64-bit
```

**Le bank `c[0x0]`** est le **kernel parameter buffer**. Les arguments de ton kernel (pointeurs, ints, etc.) sont accessibles via LDC depuis cette bank au début du kernel.

**LDCU vs LDC** : `LDCU` charge dans un uniform register (UR). Plus efficient si les 32 threads vont lire la même valeur.

**Pattern à reconnaître** : le **prologue** d'un kernel commence presque toujours par une rafale de LDC pour charger les paramètres :
```
LDC R1, c[0x0][0x37c] ;        load stack ptr
LDC R5, c[0x0][0x3a8] ;        load kernel arg
LDCU UR5, c[0x0][0x3a0] ;      load uniform arg
LDCU.64 UR10, c[0x0][0x358] ;  load 64-bit ptr arg
```

Compter les LDC te dit combien de paramètres le kernel utilise.

## LDGSTS — Load Global Store Shared (cp.async)

Instruction asynchrone : copie directement de global vers shared sans passer par les registres. C'est le SASS de `cp.async.ca.shared.global.*` en PTX.

```
LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;
                ^^^^^^^^^  ^^^  ^^^^^^^^^^^^^^^^^^
                modifiers  dest source
```

**Modifiers** :
* `.E` : Extended addressing 64-bit.
* `.LTC128B` : L2 cache hint, alignment 128B.
* `.128` : transfer size 128 bits par thread.

**Pourquoi c'est révolutionnaire** :
* **Overlap memory + compute** : l'instruction dispatche puis le warp continue à exécuter d'autres instructions pendant le transfer. Le warp ne stall que quand il fait `cp_async_wait_group<N>()`.
* **Pas de register staging** : économise le register file (LDG → STS classique consomme 4-16 registres temporaires).
* **Plus efficient pour le bus** : transfer direct global → shared sans aller-retour.

**Pattern production GEMM** :
```
// Prefetch tile K+1 pendant qu'on calcule sur tile K
LDGSTS [smem_buffer_K+1], desc[UR_global][R_offset_K+1.64] ;
LDGDEPBAR ;
DEPBAR.LE SB0, 0x1 ;             ← garde 1 prefetch en flight
BAR.SYNC.DEFER_BLOCKING 0x0 ;
LDS fragments tile K ;
HMMA chain ;
```

C'est le software pipelining canonique.

## LDGDEPBAR — Commit group async

```
LDGDEPBAR ;     /* 0x00000000000079af */
                /* 0x000e280000000000 */
```

Pas d'opérande. Sémantique : tous les LDGSTS émis depuis le dernier LDGDEPBAR sont **commit comme un groupe** dans SB0.

**Usage** : tu fais N LDGSTS pour prefetch, puis 1 LDGDEPBAR pour les regrouper. Plus tard, `DEPBAR.LE SB0, K` attend qu'au plus K groupes soient encore en flight.

## DEPBAR.LE SB0, N — Wait async memory

```
DEPBAR.LE SB0, 0x0 ;    wait all (no group pending)
DEPBAR.LE SB0, 0x1 ;    keep 1 group in flight (continue prefetching)
DEPBAR.LE SB0, 0x2 ;    keep 2 in flight
DEPBAR.LE SB0, 0x3 ;    keep 3 in flight
```

Le `N` est encodé sur 2 bits dans le control code (bits 38-39), donc max N=3.

**Pattern** :
* `N=0` : wait tail. Utilisé à la fin du kernel pour s'assurer que tout est landé.
* `N=1` ou plus : pipelining. Garde N prefetches en flight pendant qu'on consume les anciens.

## STG — Store Global

Écriture vers global memory.

```
STG.E desc[UR8][R2.64], R12 ;       store 32-bit
STG.E.128 desc[UR8][R2.64], R12 ;   store 128-bit (R12:R15)
```

**Coalescing s'applique aussi** aux stores. 32 threads écrivant 32 adresses consécutives = 1 transaction.

**Pattern à reconnaître** : l'**épilogue** d'un kernel finit typiquement par une rafale de STG, un par élément de résultat :
```
STG.E desc[UR_out][R_offset.64], R12 ;
STG.E desc[UR_out][R_offset.64+0x4], R13 ;
STG.E desc[UR_out][R_offset.64+0x8], R14 ;
STG.E desc[UR_out][R_offset.64+0xC], R15 ;
EXIT ;
```

## STS — Store Shared

Écriture vers shared memory. Latence et bank conflicts comme LDS.

```
STS [R4], R8 ;            32-bit
STS.U8 [R4], R8 ;         1 byte
STS.128 [R4], R8 ;        16 bytes (R8:R11)
```

**Pattern typique** : STS suivi de `BAR.SYNC.DEFER_BLOCKING 0x0` pour garantir que les writes sont visibles aux autres threads avant qu'ils ne lisent via LDS.

Sur SM120 moderne, `STS` est souvent **remplacé par LDGSTS** qui est plus efficient (pas de staging registers).

## STL / LDL — Local memory (spill)

```
STL.64 [R1], R2 ;              spill R2:R3 sur stack thread-local
... 50 instructions plus loin ...
LDL R8, [R11+UR7] ;             reload depuis stack
```

**Toujours mauvais signe**. Si tu vois STL/LDL, c'est ptxas qui a dû spill. Voir chapitre 4 (registres).

## Le descriptor pattern : `desc[UR][R]`

Sur SM120 moderne, l'addressing classique `[R.64]` est souvent remplacé par le descriptor pattern :

```
LDG.E R4, desc[UR8][R2.64] ;
STG.E desc[UR8][R2.64+0x4], R12 ;
LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;
```

**Comment ça se lit** :
* `UR8` contient le **descriptor** (pointeur de base + métadonnées de tensor).
* `R2.64` contient l'**offset** par-thread (pointeur 64-bit).
* L'adresse finale = base(UR8) + R2.

**Avantage** : le hardware peut optimiser l'accès via les métadonnées du descriptor (strides, dimensions). Plus expressif que pointer brut.

**Signal d'audit** : un kernel utilisant `desc[UR][R]` partout = addressing moderne. Si tu vois encore des `[R.64]` directs, c'est de l'addressing classique, peut-être code legacy.

## Bilan sur les latences

Quand tu lis une chain d'instructions et essayes d'estimer la latence totale :

| Instruction | Latence approx |
|---|---|
| LDC, LDCU | 5-10 cycles |
| LDS, STS (no conflict) | 20-30 cycles |
| LDS.128 | similaire |
| LDG hit L1 | 30 cycles |
| LDG hit L2 | 150 cycles |
| LDG miss → DRAM | 400-600 cycles |
| LDGSTS dispatch | ~5 cycles, completion 100-400 |
| STL/LDL | similaire à LDG |
| HMMA latency in chain | ~30-35 cycles |
| QMMA latency in chain | ~33-35 cycles |
| OMMA latency in chain | ~29 cycles |
| FFMA | 4 cycles |
| BRA | 4-8 cycles |

Note : ces chiffres sont indicatifs (NVIDIA ne les publie pas). Ils varient selon contention et microarchitecture.

## Ce qui suit

Le chapitre 7 attaque les **tensor cores** : HMMA, QMMA, OMMA. La famille d'instructions qui détermine ton throughput compute. Sans elles, tu fais 15 TFLOPS FP32 max. Avec elles bien utilisées, 900 TFLOPS FP4.
