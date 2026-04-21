# Chapitre 3 — Anatomie d'une instruction SASS

## Le but du chapitre

À la fin de ce chapitre, tu dois pouvoir prendre n'importe quelle ligne de SASS et identifier ses 5 composants principaux : offset, mnemonic, opérandes, opcode bytes, control code. Tu dois savoir lire chaque champ et comprendre ce qu'il t'apprend.

C'est la **grammaire de base**. Sans ça, le reste du dump est du bruit.

## La ligne de référence

On va décortiquer cette ligne complète, qu'on rencontre dans presque tous les kernels Blackwell modernes :

```
        /*0160*/                   LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;          /* 0x0000000002077fae */
                                                                                          /* 0x008fe6000b9a1a08 */
```

Cette ligne contient **5 champs distincts** :

```
        /*0160*/                   LDGSTS.E.LTC128B.128         [R7], desc[UR8][R2.64] ;          /* 0x0000000002077fae */
        ^^^^^^^                    ^^^^^^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^^^^^^
        offset                     mnemonic                     opérandes                          opcode bytes
                                                                                                  /* 0x008fe6000b9a1a08 */
                                                                                                  ^^^^^^^^^^^^^^^^^^^^^^
                                                                                                  control code
```

Chacun de ces champs te dit quelque chose de différent. On les attaque un par un.

## Champ 1 — Offset

Le `/*0160*/` au début de la ligne est l'offset hexadécimal de l'instruction dans le binaire du kernel.

**Trois propriétés à connaître** :

1. **Chaque instruction fait 16 bytes.** Donc les offsets incrémentent de 0x10. Si tu vois 0x0000, l'instruction suivante est 0x0010, puis 0x0020, etc. Toujours.

2. **Les offsets recommencent à 0 pour chaque kernel** dans un binaire multi-kernel. Si tu as deux kernels A et B dans le même binaire, l'offset 0x100 dans A n'est pas la même adresse physique que 0x100 dans B.

3. **C'est la cible des branches.** Quand tu vois `BRA 0x890`, ça veut dire "saute à l'instruction à l'offset 0x890 dans ce kernel".

**Application pratique : identifier les boucles.**

Pour qu'il y ait une boucle, il faut une back-edge : un BRA dont la cible est inférieure à la source. Exemple :

```
/*0890*/  IADD R4, R4, 0x1 ;
/*08a0*/  ISETP.LT.U32 P0, PT, R4, R20, PT ;
/*08b0*/  @P0 BRA 0x0890 ;       <-- back-edge : cible 0x890 < source 0x8b0
```

Pour identifier toutes les boucles dans un dump, un script Python qui parse les `BRA 0x*` et compare source/cible te donne la liste en quelques secondes.

**Application pratique : mesurer la taille d'un body.**

Si une back-edge va de 0x9030 à 0x5270, le body de la boucle fait `0x9030 - 0x5270 = 0x3dc0` bytes = 985 instructions. Tu sais immédiatement que c'est une grosse boucle.

## Champ 2 — Mnemonic

Le `LDGSTS.E.LTC128B.128` est le mnemonic. C'est le nom de l'instruction.

**Structure** : `BASE.MODIFIER.MODIFIER...MODIFIER`. Chaque modifier est séparé par un `.` et apporte une spécialisation du comportement.

Pour notre exemple :
* **`LDGSTS`** = base. Famille "Load Global Store Shared" (cp.async).
* **`.E`** = Extended addressing (64-bit pointer).
* **`.LTC128B`** = L2 cache hint, alignment 128 bytes.
* **`.128`** = transfer size = 128 bits (16 bytes par thread).

**La base te dit la famille d'instruction.** `HMMA`, `QMMA`, `OMMA` = MMAs FP16/FP8/FP4. `LDG`, `LDS`, `LDC` = loads global/shared/constant. `STG`, `STS` = stores. `BRA`, `BSSY` = control flow. Etc.

**Les modifiers te disent le comportement précis.** Deux instructions avec même base mais modifiers différents ne font pas la même chose. Par exemple :
* `LDG.E R4, [R2.64]` = load 32-bit
* `LDG.E.128 R4, [R2.64]` = load 128-bit dans R4:R5:R6:R7

**Conventions à connaître** :

* **Ordre des modifiers est significatif.** Tu ne peux pas écrire `LDGSTS.128.E.LTC128B`. C'est toujours `BASE.E.cache.size`.
* **Les modifiers "par défaut" sont silencieux.** `QMMA.16832.F32.E4M3.E4M3` n'a pas `.SF` car le default est sans block scaling. Si tu vois `.SF`, c'est un choix explicite.
* **Pour les MMA spécifiquement**, l'ordre est : `BASE.SF?.shape.acc_dtype.A_dtype.B_dtype.scale_dtype?.scale_vec?`. Connaître cet ordre te permet de lire un mnemonic complexe en une passe.

## Champ 3 — Opérandes

Les opérandes sont entre le mnemonic et le `;`, séparés par des virgules. L'ordre dépend de l'instruction, mais la **première opérande est typiquement la destination**.

Pour `LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64]` :
* `[R7]` = destination (adresse shared memory pointée par R7)
* `desc[UR8][R2.64]` = source (descriptor en UR8 + offset 64-bit en R2:R3)

Pour une MMA :
```
HMMA.16816.F32 R12, R4, R8, R12 ;
               D    A   B   C
```

* D = destination accumulator (4 registres consécutifs : R12, R13, R14, R15).
* A = fragment matrix (2 registres : R4, R5).
* B = fragment matrix (1 registre : R8).
* C = accumulator d'entrée (4 registres : R12, R13, R14, R15).

Note : **D et C colocalisés (R12 = R12)** = pattern de chain. Le résultat de la MMA précédente est consommé puis écrasé par le résultat de celle-ci.

**Types d'opérandes que tu rencontreras** :

* **Registre simple** : `R0`, `R12`, `R255`, `RZ` (zéro fixé).
* **Registre uniforme** : `UR0`, `UR8`, `URZ`. Partagé entre tous les threads du warp.
* **Predicate** : `P0`, `P7`, `PT` (always-true). Booléen 1-bit.
* **Adresse mémoire** : `[R7]` (offset par-thread), `[R7+UR5]` (base uniforme + offset par-thread), `[R7+UR5+0x100]` (avec immediate offset).
* **Descriptor** : `desc[UR8][R2.64]`. Modern addressing avec descriptor en UR + pointer 64-bit. Très commun sur SM120.
* **Constant memory** : `c[0x0][0x37c]` = bank 0, offset 0x37c.
* **Immediate** : `0x10`, `0x100`, `0xffff`. Valeur littérale.
* **Special register** : `SR_TID.X`, `SR_CTAID.X`, `SR_LANEID`, `SRZ`.

**Astuce de lecture** : compte le nombre d'opérandes. Pour les MMA :
* 4 opérandes (D, A, B, C) = MMA standard.
* 7 opérandes (D, A, B, C, SFA, SFB, URZ) = MMA avec block scaling (`.SF`).

Tu peux distinguer immédiatement.

## Champ 4 — Opcode bytes

À droite de la première ligne, entre `/* */`, tu vois 16 caractères hex : `0x0000000002077fae`. Ce sont les **8 bytes** (64 bits) d'encodage machine de l'instruction. C'est ce que le hardware décode.

**Le low byte** (les deux derniers caractères, `ae`) **identifie la famille d'opcode**. Sur SM120 :
* `0x3b` = LDSM
* `0x3c` = HMMA
* `0x7a` = QMMA
* `0x7f` = OMMA
* `0x81` = LDG
* `0x82` = LDC
* `0x84` = LDS
* `0x86` = STG
* `0x88` = STS
* `0xae` = LDGSTS

Pour notre exemple, le low byte `ae` confirme que c'est bien un LDGSTS.

**Astuce d'audit** : pour compter combien d'instructions de chaque famille un kernel utilise :
```bash
grep -oP '0x[0-9a-f]+' my_kernel.sass | grep -oP '..$' | sort | uniq -c | sort -rn
```

Ça te donne la distribution. Beaucoup de `0x7a` = beaucoup de QMMAs, kernel intensif tensor core. Beaucoup de `0x81` = beaucoup de loads global, peut-être memory-bound.

**Les autres bytes** encodent les opérandes (registres source/dest, immédiats) et certains modifiers qui ne sont pas dans le control code. La topologie exacte bit-par-bit reste partiellement non décodée — voir gaps.

## Champ 5 — Control code

Sur la **deuxième ligne** de chaque instruction, tu vois 16 caractères hex : `0x008fe6000b9a1a08`. Ce sont **les 8 bytes du control code**. C'est distinct de l'opcode bytes.

Le control code contient **toute la politique d'exécution** que ptxas a choisie pour cette instruction :
* Sur quoi attendre (wait mask).
* Combien de cycles stall.
* Quel scoreboard set quand l'instruction termine.
* Si yield au scheduler après.
* Quels opérandes utilisent le reuse cache.

C'est le champ le plus dense en information de scheduling. Voir l'entrée "Control code" du glossaire pour la structure détaillée.

**Pour un audit rapide** : le **byte 0** (deux derniers caractères de la ligne, ici `08`) est le wait mask. `0x00` = pas d'attente. `0xff` = attendre tout. Les valeurs intermédiaires correspondent aux scoreboards spécifiques attendus.

## Mettre tout ensemble : lire une ligne en 30 secondes

Reprends notre ligne :
```
/*0160*/                   LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;          /* 0x0000000002077fae */
                                                                                  /* 0x008fe6000b9a1a08 */
```

En 30 secondes tu lis :

1. **Offset** `0x0160` → 22ème instruction du kernel.
2. **Mnemonic** `LDGSTS.E.LTC128B.128` → cp.async, 64-bit addressing, L2 hint avec align 128B, transfer 16 bytes.
3. **Opérandes** `[R7]` ← `desc[UR8][R2.64]` → destination shared memory à l'adresse en R7, source via descriptor UR8 avec offset 64-bit en R2:R3.
4. **Low byte** `ae` → confirme LDGSTS.
5. **Wait mask** `08` → attend scoreboard slot 3 (bit 3 set).

Tu sais maintenant ce que cette instruction fait, où elle prend ses données, et quelle synchronisation elle requiert. Sans rien lire de plus.

## Cas particuliers à connaître

**Prédicat avant le mnemonic** :
```
@P0 BRA 0x890 ;
@!P0 LDG.E R4, [R2.64] ;
@PT NOP ;
@!PT LDS RZ, [RZ] ;
```

Si tu vois `@P0`, l'instruction ne s'exécute que si P0 est vrai pour ce thread. `@!P0` = négation. `@PT` = toujours vrai (rare). `@!PT` = toujours faux, no-op effectif.

**Reuse flag sur opérande** :
```
HMMA.16816.F32 R12, R4.reuse, R8.reuse, R12 ;
                    ^^^^^^    ^^^^^^
                    register reuse cache activé
```

Le `.reuse` sur un opérande active le cache de reuse au niveau matériel. Voir entrée "Reuse flag" du glossaire.

**Cas où il n'y a pas d'opérande** :
```
LDGDEPBAR ;     /* 0x00000000000079af */
EXIT ;          /* 0x000000000000794d */
```

Certaines instructions n'ont pas d'opérande. Elles font une action atomique (commit, exit) et leur sémantique est dans le mnemonic seul.

## Exercice mental

Prends cette ligne et identifie ses 5 champs :

```
/*9220*/  QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R4, R12, RZ, R18, R31, URZ ;  /* 0x71f0120c040c747a */
                                                                            /* 0x000fe2000028feff */
```

Tu dois pouvoir dire en 30 secondes :
* Offset 0x9220, c'est probablement loin dans le kernel (>2000ème instruction).
* QMMA avec block scaling (.SF), shape m16n8k32, accumulateur FP32, inputs FP4 E2M1, scale UE8M0.
* 7 opérandes : D=R12, A=R4, B=R12, C=RZ (premier MMA de chain, pas d'accumulation), SFA=R18, SFB=R31, bid/tid=URZ.
* Low byte `7a` = QMMA confirmé.
* Wait mask `ff` = attend tous les scoreboards (sources fraîches du LDS qui précède).

Si ça te parle, tu maitrises l'anatomie. Tu peux passer aux chapitres suivants qui creusent chaque famille d'instruction.

## Ce qui suit

Le chapitre 4 attaque les **registres** : quels types existent (R, UR, P, UP, SR), comment ptxas les alloue, et pourquoi voir `R255` ou `STL/LDL` est un signal critique.
