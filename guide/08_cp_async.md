# Chapitre 8 — cp.async et le pipeline global → shared

## Le but du chapitre

Comprendre **comment les kernels modernes overlap memory transfer et compute** via cp.async (LDGSTS en SASS). Reconnaître le pattern de software pipelining dans le SASS, et savoir distinguer un kernel pipelined d'un kernel staging classique.

## Le problème que cp.async résout

**Avant cp.async** (Volta, Turing, début Ampere), pour amener une tile de global vers shared, tu devais :

```cpp
// Phase 1 : load from global to register
float4 data = global_array[idx];   // SASS : LDG.E.128

// Phase 2 : store register to shared
shared_array[idx] = data;          // SASS : STS.128

// Phase 3 : sync
__syncthreads();                   // SASS : BAR.SYNC.DEFER_BLOCKING 0x0
```

**Coût** : 4 staging registers × 32 threads × N tiles. Plus le scoreboard du LDG qui doit complete avant STS. Pour un kernel GEMM avec tile 128×128 FP16, c'est ~40 registres juste pour le staging. Drastiquement réduit l'occupancy.

**Avec cp.async** (Ampere+, et donc SM120), tu fais en une instruction :

```cpp
__pipeline_memcpy_async(shared_addr, global_addr, sizeof(float4));   
// SASS : LDGSTS.E.LTC128B.128
```

**Bénéfices** :
* **Pas de staging registers** : le transfer va direct global → shared.
* **Asynchrone** : l'instruction dispatche, le warp continue à exécuter d'autres choses.
* **Pipelined** : tu peux launcher N transfers et attendre seulement quand tu as besoin.

## L'instruction LDGSTS

```
LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;     /* 0x0000000002077fae */
                                                  /* 0x008fe6000b9a1a08 */
```

**Modifiers** :
* `.E` : Extended addressing 64-bit.
* `.LTC128B` : L2 cache hint, alignment 128 bytes.
* `.128` : transfer size 128 bits par thread.

**Opérandes** :
* `[R7]` : destination dans shared memory.
* `desc[UR8][R2.64]` : source dans global, via descriptor UR8 + offset 64-bit en R2:R3.

**Sémantique** : copie 16 bytes de `*desc[UR8][R2:R3]` vers shared `*R7`. Asynchrone.

**Low byte** : `0xae`. Famille distincte de LDG (0x81), LDS (0x84), STS (0x88).

## La synchronisation : LDGDEPBAR et DEPBAR.LE

cp.async est asynchrone. Pour que le warp puisse utiliser les données loadées, il faut **savoir quand elles sont arrivées**. C'est le rôle de LDGDEPBAR + DEPBAR.LE.

### LDGDEPBAR — Commit

```
LDGDEPBAR ;     /* 0x00000000000079af */
                /* 0x000e280000000000 */
```

**Pas d'opérande**. Sémantique : tous les LDGSTS émis depuis le dernier LDGDEPBAR sont **groupés ensemble** comme un commit group, tracké dans le scoreboard SB0.

### DEPBAR.LE SB0, N — Wait

```
DEPBAR.LE SB0, 0x0 ;    wait all (SB0 vide)
DEPBAR.LE SB0, 0x1 ;    keep 1 group in flight
DEPBAR.LE SB0, 0x2 ;    keep 2 groups
DEPBAR.LE SB0, 0x3 ;    keep 3 groups
```

**Sémantique** : attend que **au plus N groupes** soient encore pending dans SB0.

**Le N est encodé sur 2 bits** (bits 38-39 du control code), donc max N=3. Pour pipelined > 4 stages, restructurer.

## Le pattern software pipelining

Voici le pattern canonique de kernel GEMM moderne avec cp.async :

```cpp
// Pseudo-code

// Phase 1 : prefetch tile 0
async_copy(smem_buffer[0], global_tile[0]);
async_copy(smem_buffer[1], global_tile[1]);
cp_async_commit_group();       // group 0
cp_async_commit_group();       // group 1

// Phase 2 : main loop
for (int k = 0; k < num_tiles - 2; k++) {
    cp_async_wait_group<1>();   // attend que tout sauf 1 group soit landé
    __syncthreads();
    
    // Compute on tile k
    fragments_A = ldsm(smem_buffer[k % 2]);
    fragments_B = ldsm(smem_buffer[k % 2]);
    mma_chain(acc, A, B);
    
    // Prefetch tile k+2
    async_copy(smem_buffer[(k+2) % 2], global_tile[k+2]);
    cp_async_commit_group();
}

// Phase 3 : drain
cp_async_wait_group<0>();
// compute last tiles
```

**Au niveau SASS** :

```
// Prologue : prefetch 2 tiles
LDGSTS [smem_0_0], desc[UR_global][R_offset_0.64] ;
LDGSTS [smem_0_1], desc[UR_global][R_offset_0.64+0x10] ;
LDGSTS [smem_0_2], desc[UR_global][R_offset_0.64+0x20] ;
... (more LDGSTS for tile 0) ...
LDGDEPBAR ;                                                 ← commit group 0

LDGSTS [smem_1_0], desc[UR_global][R_offset_1.64] ;
LDGSTS [smem_1_1], desc[UR_global][R_offset_1.64+0x10] ;
... (tile 1) ...
LDGDEPBAR ;                                                 ← commit group 1

// Main loop body
loop_label:
DEPBAR.LE SB0, 0x1 ;                                        ← attend que tout sauf 1 group soit landé
BAR.SYNC.DEFER_BLOCKING 0x0 ;                              ← sync threads (visibility)

// Load fragments tile k from shared
LDSM.16.M88.4 R12, [R0+UR7] ;                              ← A fragments
LDSM.16.M88.2 R20, [R0+UR8] ;                              ← B fragments

// MMA chain
HMMA.16816.F32 R28, R12, R20, R28 ;
HMMA.16816.F32 R28, R13, R21, R28 ;
HMMA.16816.F32 R28, R14, R22, R28 ;
HMMA.16816.F32 R28, R15, R23, R28 ;

// Prefetch tile k+2 in background
LDGSTS [smem_next_0], desc[UR_global][R_next_offset.64] ;
LDGSTS [smem_next_1], desc[UR_global][R_next_offset.64+0x10] ;
... 
LDGDEPBAR ;

// Loop iteration
@P0 BRA loop_label ;
```

## Comment reconnaître ce pattern dans un dump

**3 signaux** :

1. **Présence de LDGSTS** (low byte `0xae`). Si zéro, le kernel n'utilise pas cp.async.

2. **Présence de LDGDEPBAR** (low byte `af`, exact opcode `0x79af`). Marque les commit points.

3. **Présence de DEPBAR.LE SB0** (low byte `1a` mais c'est aussi BAR.SYNC, à distinguer). Le mnemonic est explicite : `DEPBAR.LE SB0, 0xN`.

**Compter** :
```bash
grep -c "LDGSTS" my_kernel.sass         # nombre de cp.async
grep -c "LDGDEPBAR" my_kernel.sass      # nombre de commits
grep -c "DEPBAR.LE" my_kernel.sass      # nombre de waits
```

**Ratio attendu** : si N tiles prefetched par stage, et S stages, tu auras ~N×S LDGSTS, S+1 LDGDEPBAR (les commits), et 1-2 DEPBAR.LE (les waits).

## Diagnostic : combien de stages de pipeline ?

Dans le pattern ci-dessus, on a **2 stages** (deux tiles prefetched simultanément). Comment le voir dans le SASS ?

**Indicateur 1** : nombre de buffers shared. Si tu vois deux groupes d'addresses shared distinctes (`smem_0_*` et `smem_1_*`), 2 stages. Trois groupes, 3 stages.

**Indicateur 2** : valeur N dans `DEPBAR.LE SB0, N`. Si N=1, 2 stages. Si N=2, 3 stages. Si N=3, 4 stages.

**Indicateur 3** : nombre de LDGDEPBAR avant le main loop. S+1 LDGDEPBAR au prologue = S stages.

**Pourquoi on s'arrête à 4 stages** : limite hardware (N max = 3 dans DEPBAR.LE).

## Coût réel : amortir le setup

Le setup d'un pipeline (prologue : prefetch S tiles + commits) coûte. Si ta boucle K ne fait que 2-4 itérations, tu paies plus en setup qu'en bénéfice de l'overlap.

**Heuristique** :
* K iterations < S × 2 : pipeline pas avantageux.
* K iterations ≥ S × 4 : pipeline pleinement avantageux.

**Conséquence pratique** : pour un GEMM 256×256 avec tile 128×128 K_tile=32, tu as K = 256/32 = 8 tiles. Pipeline 2-stage avantageux. Pour un GEMV 1×4096 avec tile 1×128, tu as 1 K-iteration. Pipeline pas pertinent.

## Le mystère @!PT LDS RZ, [RZ]

Dans nos chapitres 18a-c, on a observé un pattern bizarre : **3 instructions `@!PT LDS RZ, [RZ]`** apparaissent juste avant chaque LDGSTS dans le main loop pipelined :

```
@!PT LDS RZ, [RZ] ;
@!PT LDS RZ, [RZ] ;
@!PT LDS RZ, [RZ] ;
LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;
```

`@!PT` est always-false donc l'instruction ne s'exécute jamais. C'est un **no-op explicite**. Pourquoi ptxas en met 3 par LDGSTS dans le pipeline ?

**Hypothèses non vérifiées** :
* Alignment du binaire sur frontière 64-byte (4 instructions × 16B = 64B).
* Sync forcé du scheduler.
* Workaround pour un erratum hardware.

C'est un **gap connu** (GAP-18-1 dans FINDINGS.md). Ne pas s'en inquiéter pour un audit, mais c'est intriguant.

## Performance attendue

**Sans cp.async** (staging classique LDG → STS → BAR.SYNC → LDS) :
* 30-40% de stalls memory dans la hot loop.
* Throughput tensor core ~50% du peak.

**Avec cp.async pipelined** :
* 5-15% de stalls memory.
* Throughput tensor core 75-90% du peak.

Le facteur 2× n'est pas exagéré pour les kernels memory-bound.

## Bilan

cp.async + le pattern software pipelining est **la primitive principale** des kernels modernes :
* CUTLASS l'utilise systématiquement depuis la 2.x.
* FlashAttention, FlashInfer en font la pierre angulaire de leur perf.
* Les inference engines (vLLM, TensorRT-LLM) en dépendent pour leur decode.

Reconnaître ce pattern dans un dump SASS = comprendre l'architecture moderne. Ne pas le voir = vieux code ou kernel non optimisé.

## Ce qui suit

Le chapitre 9 attaque **LDSM**, le pont entre shared memory et tensor cores. C'est ce qui permet aux MMA d'avoir leurs fragments efficacement, sans staging via plusieurs LDS.
