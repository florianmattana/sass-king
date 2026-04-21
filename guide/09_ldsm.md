# Chapitre 9 — LDSM : le pont shared memory → tensor core

## Le but du chapitre

Comprendre **pourquoi LDSM existe** et comment elle se combine avec MMA. Reconnaître les patterns de loading fragments dans un dump SASS, et savoir distinguer un setup MMA optimal d'un setup pathologique.

## Le problème

Tu as ta tile en shared memory. Tu veux la passer aux tensor cores. Mais les MMAs attendent les fragments **dans un layout très spécifique** distribué à travers les 32 threads du warp.

**Sans LDSM**, tu devrais :
1. Calculer pour chaque thread quelle portion de shared memory il doit lire.
2. Faire plusieurs LDS pour récupérer les bons éléments.
3. Reformater dans les registres pour matcher le layout fragment.

C'est laborieux et coûteux. Pour HMMA m16n8k16, chaque thread doit avoir :
* 4 halfs pour A (2 registres).
* 2 halfs pour B (1 registre).

Et la distribution n'est pas linéaire — elle dépend du lane et du quadrant. Pour un programmer C++ c'est invisible (tu utilises `wmma::fragment`), mais au niveau SASS c'est compliqué.

**LDSM résout ça** en une instruction : un warp lit une tile 8×8 de shared memory et chaque thread reçoit ses bytes dans le layout fragment, prêt pour MMA.

## L'instruction LDSM

```
LDSM.16.M88 R5, [R5+UR4] ;                    /* 0x... */
            ↑    ↑
            destination registers
            base address (shared memory)
```

**Modifiers** :
* `.16` : chaque element est 16-bit (half precision).
* `.M88` : matrix 8×8 (M = matrix, 88 = dimensions). Un tile 8×8 de halfs = 128 bytes total (64 elements × 2 bytes).
* `.MT88` : variant avec **transpose** (même matrix mais trans).

**Variants d'amplitude** (combien de tiles 8×8 chargés en une instruction) :
* `LDSM.16.M88` : 1 tile. Variant x1.
* `LDSM.16.M88.2` : 2 tiles. Variant x2.
* `LDSM.16.M88.4` : 4 tiles. Variant x4.

**Bits dans le control code** :
* Width (x1/x2/x4) : bits 8-9 (00=x1, 01=x2, 10=x4).
* Trans flag : bit 14.

## Distribution des tiles dans le warp

Pour `LDSM.16.M88` (variant x1) :
* 1 tile 8×8 = 64 elements.
* 32 threads chargent chacun 2 elements = 4 bytes par thread.
* Le résultat est dans 1 registre par thread.

Pour `LDSM.16.M88.2` (variant x2) :
* 2 tiles. Chaque thread reçoit 4 elements = 8 bytes = 2 registres.

Pour `LDSM.16.M88.4` (variant x4) :
* 4 tiles. Chaque thread reçoit 8 elements = 16 bytes = 4 registres.

**Pattern à reconnaître** : `.4` est utilisé pour fragment A en HMMA m16n8k16 (16×16 = 4 tiles 8×8). `.2` pour fragment B (16×8 = 2 tiles).

## Le pattern setup-MMA

Quasi tout MMA est précédé de LDSM :

```
LDSM.16.M88.4 R12, [R0+UR7] ;           load A fragment (4 tiles, 8 regs : R12-R19)
LDSM.16.M88.2 R20, [R0+UR8] ;           load B fragment (2 tiles, 4 regs : R20-R23)
HMMA.16816.F32 R28, R12, R20, R28 ;     compute MMA
```

**Observation surprenante** (chap 17) : ptxas émet **systématiquement le LDSM B avant le LDSM A**, malgré l'ordre logique inverse. C'est un choix de scheduling consistent, hypothèse non confirmée mais probablement lié au pipeline tensor core.

## Latency LDSM

Mesuré chap 17f :
* **~33 cycles** par LDSM en chain serial.
* **Chain overhead essentiellement zéro** : si tu enchaînes 4 LDSM, le total ≈ 4 × 33 = 132 cycles, pas plus.

**Comparaison avec HMMA** :
* HMMA en chain : ~35 cycles par MMA.
* LDSM dispatched + HMMA : si LDSM précèdent les HMMAs et sont déjà landés, HMMA dispatch immédiat. Bon scheduling = LDSM masqué par autres opérations.

## LDSM avec transpose

```
LDSM.16.MT88.4 R4, [R0+UR7] ;           load + transpose
```

Le `.MT88` charge une tile et la transpose. Utile pour les fragments B en row-major qui doivent devenir col-major pour MMA, ou inversement.

**Cost** : pas de surcoût significatif (mesure chap 17d). Le transpose est gratuit côté hardware.

**Pattern à reconnaître** : si A est row-major en shared mais le MMA attend col-major, ptxas émet `LDSM.MT88` au lieu de `LDSM.M88`.

## Coordination avec cp.async

Dans un kernel pipelined moderne :

```
// Prefetch tile via cp.async
LDGSTS [smem_buffer], desc[UR_global][R_offset.64] ;
LDGDEPBAR ;
DEPBAR.LE SB0, 0x0 ;            ← attend que cp.async soit landé
BAR.SYNC.DEFER_BLOCKING 0x0 ;   ← sync threads (visibility)

// Maintenant les données sont en shared, prêtes
LDSM.16.M88.4 R12, [R_smem_A] ;
LDSM.16.M88.2 R20, [R_smem_B] ;

// MMA
HMMA.16816.F32 R28, R12, R20, R28 ;
```

**Critique** : sans le `BAR.SYNC` après le DEPBAR.LE, certains threads pourraient lire shared memory que d'autres warps n'ont pas encore écrite. Le DEPBAR.LE est warp-local, le BAR.SYNC est block-wide.

## Anatomie du wait mask post-LDSM

Une HMMA qui suit immédiatement une LDSM doit attendre que la LDSM soit complete. Le wait mask de l'HMMA reflète cette dépendance.

**Pattern classique** (sources fraîches LDSM) :
```
LDSM.16.M88.4 R12, [R_smem_A] ;           /* assigned SB asgn = 4 (par exemple) */
LDSM.16.M88.2 R20, [R_smem_B] ;           /* assigned SB asgn = 5 */
HMMA.16816.F32 R28, R12, R20, R28 ;       /* wait mask = 0xff (attente tous SB) */
```

Wait mask `0xff` = attente très large. Ptxas a été pessimiste, ou il sait qu'il y a beaucoup de pendings.

**Pattern optimisé** (chain MMA déjà commencée) :
```
HMMA.16816.F32 R28, R12, R20, R28 ;       /* wait mask = 0x04 = juste SB2 attendu */
HMMA.16816.F32 R28, R13, R21, R28 ;       /* wait mask = 0x00 = chain, pas de wait */
HMMA.16816.F32 R28, R14, R22, R28 ;       /* wait mask = 0x00 */
HMMA.16816.F32 R28, R15, R23, R28 ;       /* wait mask = 0x00 */
```

Premier MMA attend, suivants enchaînent. Optimal.

## STMatrix : le store équivalent

Pour écrire un fragment de registres vers shared memory dans un layout matricielle, NVIDIA fournit `stmatrix` en PTX. En SASS, c'est `STSM` (probablement, à confirmer car peu utilisé sur SM120 dans nos chapitres).

Cas d'usage : un kernel qui calcule un MMA, accumule, puis veut écrire la tile vers shared pour un staging vers global. Pas observé dans nos chapitres récents, mais existe.

## Performance practique

**Pour un GEMM bien optimisé sur SM120** :
* 1 LDSM A (x4) + 1 LDSM B (x2) = ~66 cycles totaux pour charger les fragments.
* 4 HMMA en chain = ~4 × 35 = 140 cycles pour le compute.
* Ratio compute/setup : 140 / 66 ≈ 2:1.

C'est correct, mais on aimerait amortir mieux. C'est pourquoi les chains MMA longues sont préférables.

**Pour du FP4 (OMMA)** :
* 1 LDSM A (x4 sur k=64 nécessite plus de reads) + 1 LDSM B = ~80 cycles setup.
* 2 OMMA en chain = ~58 cycles compute.
* Ratio défavorable : 1:1.4.

C'est pourquoi le peak FP4 demande **beaucoup de pipelining** : il faut overlap les setups avec d'autres compute.

## Bilan

LDSM est l'instruction silencieuse qui rend les MMAs efficients. Sans elle, le coût de setup serait prohibitif. Avec elle, le ratio setup/compute reste raisonnable même pour FP4 peak.

**À retenir** :
* `.M88` = tile 8×8 halfs.
* `.MT88` = avec transpose.
* `.x1/.x2/.x4` = combien de tiles par instruction.
* Latency ~33 cycles, chain overhead nul.
* Toujours précéder MMA, B avant A par convention.

## Ce qui suit

Le chapitre 10 commence la **partie pratique** : comment lire un dump SASS de A à Z, identifier les sections (prologue, body, loops, epilogue), et naviguer rapidement. C'est là que les chapitres 1-9 deviennent applicables sur du vrai code.
