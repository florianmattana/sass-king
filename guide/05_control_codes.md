# Chapitre 5 — Control codes

## Le but du chapitre

Décoder la **ligne de droite** de chaque instruction SASS — celle qui contient le scheduling. Comprendre comment lire un wait mask, identifier un stall count élevé, et reconnaître les patterns de scheduling sain vs pathologique. À la fin, tu peux pointer **précisément** où un kernel ralentit et pourquoi.

## Pourquoi c'est essentiel

Le control code est la **seule source d'information** sur ce que ptxas a décidé concernant le scheduling. NCU te donne des stats agrégées ("ce kernel passe 30% du temps en stall_short_scoreboard"), mais le control code te dit **quelle instruction précise** stall et **sur quoi** elle attend.

Pour optimiser un kernel, tu dois lire les control codes de la région chaude.

## La ligne de droite

Reprenons l'exemple :

```
/*0160*/  LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;          /* 0x0000000002077fae */
                                                                  /* 0x008fe6000b9a1a08 */
                                                                  ^^^^^^^^^^^^^^^^^^^^^^
                                                                  control code (8 bytes = 64 bits)
```

Les 16 caractères hex `008fe6000b9a1a08` représentent 64 bits dans lesquels ptxas encode toute la politique d'exécution.

## Structure approximative (SM120)

La structure exacte n'est pas publiquement documentée. À partir d'observations empiriques (chapitres 1-18) et des conventions Turing/Ampere connues, voici notre meilleure compréhension :

```
Bit position:  63 ... 32 | 31 ... 16 | 15 ... 13 | 12 ... 8 | 7 ... 0
Field:         [other]   | [SB asgn] | [yield]   | [stall]  | [wait mask]
```

Lecture du control code de notre exemple `0x008fe6000b9a1a08` (low → high) :

* **Byte 0** (`0x08`) = wait mask. Bits set : bit 3 → attendre scoreboard 3.
* **Byte 1** (`0x1a`) = part of read wait / scheduling.
* **Bytes 2-3** (`0x9a 0x0b`) = scoreboard assignment.
* **Bytes 4-7** (`00 e6 8f 00`) = autres (yield, stall, reuse, etc.).

Pour un audit pratique, **tu n'as pas besoin de tout décoder**. Tu vas surtout regarder :
* **Byte 0** = wait mask.
* **Présence de stall count élevé** (impact visible dans le scheduling).

## Le wait mask (byte 0)

C'est le champ le plus directement actionnable. Chaque bit du byte 0 correspond à un scoreboard slot :

```
Bit 0 → wait sur SB0
Bit 1 → wait sur SB1
...
Bit 7 → wait sur SB7
```

**Valeurs typiques** :

| Wait mask | Signification |
|---|---|
| `0x00` | N'attend rien. Dispatch immédiat. |
| `0x01` | Attend SB0 (typiquement cp.async). |
| `0x04` | Attend SB2. |
| `0xff` | Attend tous les scoreboards (pessimiste, force ordre strict). |

**Cas concret 1 : MMA après load** :
```
LDG.E R4, [R2.64] ;    /* ... 0x000ea200... */    ← set scoreboard, ex SB1
... autres instructions ...
HMMA.16816.F32 R12, R4, R8, R12 ; /* ... 0x000fe800...02 */  ← wait mask 0x02 = wait SB1
```

L'HMMA attend que la LDG soit complete avant de pouvoir lire R4. Le wait mask `0x02` reflète cette dépendance.

**Cas concret 2 : MMA chain sans dépendance externe** :
```
HMMA.16816.F32 R12, R4, R8, RZ ;   /* wait mask 0x00 */
HMMA.16816.F32 R12, R5, R9, R12 ;  /* wait mask 0x00 */
HMMA.16816.F32 R12, R6, R10, R12 ; /* wait mask 0x00 */
HMMA.16816.F32 R12, R7, R11, R12 ; /* wait mask 0x00 */
```

Wait mask zéro partout = pas d'attente, pipeline densément utilisé. C'est ce qu'on veut voir dans une chain bien optimisée.

**Cas concret 3 : wait mask 0xff** :
```
HMMA.16816.F32 R12, R4, R8, R12 ; /* wait mask 0xff */
```

Wait mask `0xff` = attendre tout. Pessimiste. Soit ptxas n'a pas pu raisonner sur les dépendances, soit il s'est protégé d'un cas compliqué. Souvent vu après un point de synchronisation ou un appel de fonction.

## Le stall count

Champ dans le control code (approximativement bits 8-11 ou 12-15, 4 bits) qui force un nombre de cycles NOP **après dispatch** de l'instruction et avant de dispatcher la suivante.

Valeurs : 0 (pas de stall) à 15 (stall maximum).

**Pourquoi ptxas mettrait du stall** :
* La latence de pipeline du hardware demande un délai avant dispatch suivant.
* Conflit sur une ressource (RF bank, scoreboard slot, etc.).
* Pessimisme du compilateur sur une dépendance compliquée.

**Application** : compter le total de cycles d'une région chaude.

```
Total cycles ≈ Σ(1 + stall_count_i + wait_cycles_i)
             pour chaque instruction de la région
```

Si tu vois beaucoup de stall counts à 4-6 sur une chain MMA, perte de throughput. Une chain HMMA bien optimisée devrait avoir des stalls de 0-1.

## Le yield flag

Un bit (probablement bit 13) qui indique au scheduler que **le warp peut céder** le pipeline après cette instruction. Permet à un autre warp de dispatcher pendant que ce warp attend.

**Interprétation** :
* Yield set = ptxas pense que cette instruction sera suivie d'un wait long, autant laisser un autre warp s'exécuter en attendant.
* Yield clear = on continue de dispatcher en séquence.

Tu n'as quasiment jamais besoin de lire le yield directement. C'est un signal indirect.

## Scoreboard assignments

Quand une instruction long-latency dispatche, ptxas lui assigne un **scoreboard slot** (SB0-SB7) qui sera "pending" jusqu'à completion. L'instruction consommatrice plus tard utilise ce slot dans son wait mask.

**Bits dans le control code** : approximativement bits 17-19 pour le write SB (3 bits = 0-7), et autres bits pour le read SB.

**Convention SM120 observée** :
* **SB0 dédié à cp.async** (LDGSTS).
* SB1-SB7 utilisés pour le reste (loads global classiques, MMAs, etc.).

**Pratique** : dans 95% des cas tu n'as pas à lire ce champ. Il suffit de savoir que l'assignment cohérent (write SB d'un producer = bit du wait mask d'un consumer) est ce qui synchronise.

## Reuse bits

Champ dans le control code (bits 58-61 approximativement, 4 bits) qui activent le register reuse cache pour chaque opérande source de l'instruction.

Doit être en cohérence avec le suffix `.reuse` sur l'opérande visible dans le mnemonic. Si tu vois `R4.reuse` dans l'instruction, le bit reuse correspondant est set dans le control code.

## Comment auditer un control code

**Workflow pratique** :

1. **Identifier la région chaude** (via NCU profiling ou analyse statique).
2. **Lister les wait masks** non-zéro dans cette région.
3. **Tracer les producers** : pour chaque wait mask, identifier l'instruction qui produit le scoreboard attendu. Mesurer la distance (en instructions) entre producer et consumer.
4. **Estimer le wait time** : si distance > latence du producer, le wait est gratuit. Si distance < latence, le warp stall = différence.
5. **Vérifier les stall counts** : repérer les valeurs élevées (>2) et chercher leur cause.

**Exemple d'audit** :

```
LDG.E R4, [R2.64] ;       /* SB asgn = 1, latence ~200 cycles */
ADD R5, R6, R7 ;           /* indep, exécute */
ADD R6, R5, R8 ;           /* indep, exécute */
... 30 autres instructions independent ...
FFMA R10, R4, R11, R12 ;  /* wait mask 0x02 = wait SB1 */
```

Distance LDG → FFMA = 32 instructions ≈ 32-64 cycles.
Latence LDG ≈ 200 cycles (cache miss) ou ~30 cycles (L1 hit).

Si miss : FFMA va stall ~150 cycles en attendant.
Si hit : FFMA dispatche sans stall.

L'audit suggère que ptxas a fait du **bon work** en mettant 32 instructions entre producer et consumer pour cacher la latence — mais reste bottleneck sur cache miss.

## Pattern à reconnaître : DEPBAR.LE

```
LDGSTS [R7], desc[UR8][R2.64] ;
LDGSTS [R7+0x10], desc[UR8][R2.64+0x10] ;
LDGSTS [R7+0x20], desc[UR8][R2.64+0x20] ;
LDGDEPBAR ;
DEPBAR.LE SB0, 0x0 ;   /* attendre que SB0 soit vide = tous LDGSTS complets */
LDS R8, [R12] ;        /* peut maintenant lire les données loadées */
```

Le `DEPBAR.LE SB0, N` est l'équivalent SASS de `cp_async_wait_group<N>()`. Le `N` est encodé dans bits 38-39 du control code (donc 0-3 max).

Lecture de la valeur N depuis le control code :
* `0x000080000000791a` → bits 38-39 = `00` → N=0.
* `0x000080400000791a` → bits 38-39 = `01` → N=1.
* `0x000080800000791a` → bits 38-39 = `10` → N=2.

## Pattern à reconnaître : BAR.SYNC

```
BAR.SYNC.DEFER_BLOCKING 0x0 ;   /* 0x000fec0000010000 */
```

Synchronisation thread block. Toutes les instructions précédentes du block doivent être complete pour que les threads passent. Le `0x0` est le barrier ID (de 0 à 15, plusieurs barriers indépendants possibles).

## Pattern à reconnaître : pas de wait, pas de stall, dual-issue actif

```
HMMA.16816.F32 R12, R4.reuse, R8.reuse, R12 ;  /* wait 0x00, stall 0 */
LDS R20, [R30] ;                                /* wait 0x00, stall 0 */
HMMA.16816.F32 R12, R5,        R8,        R12 ;  /* wait 0x00, stall 0 */
LDS R21, [R30+0x10] ;                            /* wait 0x00, stall 0 */
```

Densité maximale. Le hardware peut probablement dual-issue les MMAs et les LDS (différents pipelines). Pattern de chain MMA optimale.

## Bilan

Le control code est dense en information mais accessible :
* **Wait mask** = quoi attendre.
* **Stall count** = combien de cycles forcés.
* **Scoreboard assignments** = synchronisation producer-consumer.
* **Yield flag** = warp switching hint.
* **Reuse bits** = register reuse cache.

Pour 95% de l'audit, tu lis juste le wait mask et le stall count visible dans le scheduling. Le reste, tu sais que c'est là, tu peux creuser si nécessaire.

## Limites connues

La structure exacte bit-par-bit du control code n'est **pas entièrement décodée**. On a observé empiriquement les conventions principales mais certains bits restent ambigus. Le projet maintient ces gaps comme un chantier ouvert (voir FINDINGS.md, GAP-1, GAP-2).

## Ce qui suit

Le chapitre 6 attaque la **memory hierarchy** : LDG, LDS, LDC, LDGSTS, et leurs variants. C'est la deuxième famille d'instructions la plus présente dans un dump après MMA, et la principale source de stalls.
