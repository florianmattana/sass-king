# Chapitre 4 — Registres et register file

## Le but du chapitre

Comprendre les **5 types de registres** que tu rencontres dans le SASS, comment ptxas les alloue, et quels signaux te disent qu'un kernel a un problème de register pressure. À la fin, tu sauras lire un dump et estimer si le kernel est sain ou pathologique côté ressources.

## Pourquoi c'est critique

Les registres sont **la ressource la plus contraignante** sur GPU. Plus un thread utilise de registres, moins de threads peuvent être résidents simultanément sur un SM (occupancy). Ptxas doit constamment arbitrer entre :

* Garder une valeur en registre (rapide, mais consomme un slot du register file).
* La recharger depuis la mémoire (gratuit en registres, mais coûte une LDG/LDS).
* La spill vers la local memory (catastrophique en perf).

Quand tu lis un SASS, tu observes les conséquences de ces arbitrages.

## Les 5 types de registres

### 1. Registres généraux (R0 – R255)

Les registres **par-thread**, 32-bit chacun, jusqu'à 256. C'est où vivent les variables locales du thread (compteurs, accumulateurs, pointeurs, etc.).

```
LDG.E R4, [R2.64] ;       R4 reçoit la valeur loadée
FFMA R8, R4, R5, R6 ;     R8 = R4 * R5 + R6
```

Sur SM120, chaque thread utilise **au minimum 24 registres**, même un kernel trivial. C'est un floor hardware. Tu peux limiter le maximum via `-maxrregcount=N`, mais pas descendre sous 24.

**Variantes** :
* `R2.64` = paire R2:R3 traitée comme valeur 64-bit. Utilisé surtout pour les pointeurs.
* `R12, R13, R14, R15` (4 regs consécutifs) = format typique d'un fragment accumulator HMMA F32 m16n8k16.

### 2. RZ (Register Zero)

Pseudo-registre qui vaut **toujours 0**. Pas de slot dans le register file (gratuit).

```
MOV R4, RZ ;                          R4 = 0
HMMA.16816.F32 R12, R4, R8, RZ ;     premier MMA d'une chain : C = 0 (pas d'accumulation précédente)
IADD R0, RZ, -R3 ;                    R0 = -R3
@!PT LDS RZ, [RZ] ;                   load discardé (no-op effectif)
```

**Pattern à reconnaître** : `C = RZ` dans le premier MMA d'une chain. Les MMAs suivantes ont `C = R_acc`.

### 3. Registres uniformes (UR0 – UR63, URZ)

64 registres **partagés entre les 32 threads d'un warp**. Stockent des valeurs identiques pour tous les threads (typiquement adresses de base, constantes loadées depuis constant memory).

```
LDC.64 UR4, c[0x0][0x380] ;             load 64-bit constant uniforme (kernel param)
LDGSTS.E.LTC128B.128 [R7], desc[UR8][R2.64] ;     UR8 = base address (uniforme)
ULEA UR4, UR6, UR4, 0x18 ;              arithmétique uniforme dédiée
```

**Pourquoi ça existe** : si les 32 threads ont besoin de la même valeur, mettre 32 copies dans des Rn distincts gaspille 31 slots du register file. Un seul UR suffit.

**Pipeline distinct** : les instructions uniformes (`ULEA`, `UIADD3`, `UMOV`, etc.) tournent sur un pipeline matériel séparé du pipeline scalaire. Ça permet du dual-issue (instruction uniforme + instruction par-thread en parallèle).

### 4. Predicates (P0 – P7, PT) et uniform predicates (UP0 – UP7, UPT)

Booléens 1-bit. 8 par thread (P0-P7) + un constant true (PT).

```
ISETP.NE.U32.AND P0, PT, R5, RZ, PT ;     P0 = (R5 != 0)
@P0 BRA 0x890 ;                            si P0 vrai, branche
@!P0 LDG.E R4, [R2.64] ;                   si P0 faux, load
FSETP.GT.F32 P1, PT, R10, RZ, PT ;         P1 = (R10 > 0.0f)
```

**Variantes uniformes** : UP0-UP7, UPT. Évalués au niveau warp. Utilisés avec instructions uniformes pour éviter la divergence.

### 5. Special Registers (SR_*)

Registres matériels read-only. Tu accèdes via `S2R` (vers R) ou `S2UR` (vers UR).

```
S2R R20, SR_TID.X ;          R20 = threadIdx.x (par thread)
S2UR UR6, SR_CTAID.X ;       UR6 = blockIdx.x (uniforme par block)
S2R R1, SR_LANEID ;          R1 = lane (0-31)
S2R R2, SR_CLOCK_LO ;        R2 = compteur de cycles bas (clock64() etc.)
CS2R R4, SRZ ;               R4 = 0 (alternative compacte à MOV R4, RZ)
```

Liste des SR communs :
* `SR_TID.X/Y/Z` : threadIdx
* `SR_CTAID.X/Y/Z` : blockIdx
* `SR_LANEID` : lane index
* `SR_WARPID` : warp index dans le block
* `SR_CLOCK_LO/HI` : cycle counter
* `SRZ` : zero

**Le prologue de presque tout kernel** commence par 1-2 `S2R` + 1-2 `S2UR` pour récupérer threadIdx, blockIdx, etc. C'est une signature de "début de kernel".

## Le register file matériel

Le register file (RF) est physiquement sur chaque SM. Sur SM120, **256 KB par SM** (= 65 536 registres 32-bit).

**Calcul d'occupancy** :
```
warps_residents_max = (256 KB) / (registres_par_thread × 32 bytes_per_register × 32 threads_per_warp)
                    = 65536 / (regs × 32)
```

Exemples :
* 24 regs (floor) → 85 warps résidents max → ~2720 threads par SM (largement sub-limited).
* 64 regs → 32 warps → 1024 threads.
* 128 regs → 16 warps → 512 threads.
* 256 regs → 8 warps → 256 threads (très bas, occupancy critique).

L'occupancy max sur SM120 est typiquement **2048 threads par SM**, donc tu deviens occupancy-limited dès que tu utilises plus de ~32 registres.

## Banks du register file

Le RF est divisé en **banks** (typiquement 4 banks sur SM120). Une lecture d'instruction nécessite plusieurs registres simultanément. Si plusieurs registres demandés sont dans le même bank, il faut plusieurs cycles pour les lire = stall.

**Bank assignment grossier** : `bank(Rn) = n % 4`.

**Conséquence pratique** :
* `FFMA R12, R4, R8, R0` → R12, R4, R8, R0 sont tous bank 0 → conflit massif.
* `FFMA R12, R5, R9, R2` → différents banks → OK.

Ptxas tente d'allouer les opérandes des instructions critiques (FFMA, MMA) dans des banks différentes. Quand il n'y arrive pas, il utilise le **register reuse cache** pour contourner.

## Le register reuse cache

Petit buffer matériel (4 slots par warp) qui cache les opérandes lus récemment. Activé via le suffix `.reuse` sur un opérande **et** les bits correspondants dans le control code.

```
HMMA.16816.F32 R12, R4.reuse, R8.reuse, R12 ;       active reuse pour R4 et R8
HMMA.16816.F32 R12, R4,        R8,        R12 ;     R4 et R8 lus depuis cache, pas RF
```

**Bénéfices** :
* Évite les bank conflicts.
* Économise le throughput RF.
* Peut activer le dual-issue.

**Limites** :
* 4 slots seulement par warp.
* Activer reuse sans réutilisation pollue le cache.

**Pattern à reconnaître** : si tu vois beaucoup de `.reuse` dans une chain MMA, ptxas optimise correctement. Si tu vois aucun `.reuse`, vérifie pourquoi — soit le pattern ne le permet pas, soit ptxas a raté.

## Register pressure et spill

Quand un kernel demande **plus de registres** que ptxas n'en a alloué, ptxas doit **spill** : stocker temporairement les valeurs en local memory (qui est en fait global memory, par-thread).

**Signature dans le SASS** :
```
STL.64 [R1], R2 ;              spill R2:R3 sur stack thread-local
... 50 instructions plus loin ...
LDL R8, [R11+UR7] ;             reload depuis stack
```

**Pourquoi c'est critique** :
* STL/LDL passent par le L1 cache. Hit = ~30 cycles. Miss = 200-400 cycles.
* Le scoreboard de la LDL doit attendre, bloquant le warp.
* Plus de spills = plus de stalls memory-bound additionnels.

**Comment diagnostiquer** :
```bash
nvcc -arch=sm_120 -Xptxas -v my_kernel.cu
# Output : ptxas info : Used 256 registers, 16 stack frame, 16 bytes spill stores, 24 bytes spill loads
```

Si tu vois "spill stores" / "spill loads" non nuls, tu as du spill. À éliminer si possible.

**Comment réduire la pressure** :
* `__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)` : donne un target à ptxas. Il va essayer d'utiliser moins de registres pour permettre l'occupancy demandée.
* Simplifier les live ranges : ne garde pas des valeurs longtemps si tu ne les utilises pas immédiatement.
* Casser le kernel en plusieurs : si une fonction a 200 lignes, peut-être qu'elle peut être deux kernels.
* Utiliser `__restrict__` sur les pointeurs : permet à ptxas de faire des hypothèses plus agressives sur l'aliasing.

## Comment savoir si un kernel a un problème de pressure

**Méthode rapide depuis le dump** : regarde le **plus haut Rn** utilisé.

```bash
grep -oP 'R\d+' my_kernel.sass | sort -t'R' -k2 -n | tail -5
```

Si tu vois `R254`, le kernel utilise probablement ~255 registres. Combiné avec présence de STL/LDL = pression critique.

```bash
grep -c "STL\|LDL" my_kernel.sass
```

Si ça retourne > 0, tu as des spills. Si ça retourne > 50, c'est sévère.

**Méthode précise** : compile avec `-Xptxas -v`. Le comptage exact est dans la sortie.

## Pattern à reconnaître : initialisation par CS2R

Pour zero-initializer un accumulateur, ptxas a deux options :

```
MOV R12, RZ ;
MOV R13, RZ ;
MOV R14, RZ ;
MOV R15, RZ ;
```

vs

```
CS2R.32 R12, SRZ ;     zero R12-R15 d'un coup (ou via une seule instruction)
```

Le `CS2R` est plus compact et peut zero-init plusieurs registres en une instruction. Tu le verras typiquement avant un MMA chain pour initialiser l'accumulateur à zéro.

## Pattern à reconnaître : pointer 64-bit

Un pointer 64-bit utilise une **paire de registres** (R, R+1). En SASS, c'est noté `R.64` :

```
LDG.E R4, [R2.64] ;             R2:R3 contient pointer 64-bit
STG.E desc[UR8][R2.64], R12 ;   même chose
```

Le `.64` n'est pas un suffix d'instruction mais un **suffix d'opérande** indiquant l'usage 64-bit du registre. Le hardware lit R2 puis R3 implicitement.

## Pattern à reconnaître : indirect addressing

Quand le code fait `array[index]`, ptxas génère typiquement :

```
IMAD.WIDE R4, R_index, 0x4, R_array_base ;     R4:R5 = array_base + index*4 (64-bit)
LDG.E R6, [R4.64] ;                             load array[index]
```

`IMAD.WIDE` produit un résultat 64-bit. C'est très commun pour les calculs d'adresse.

## Bilan

À ce stade tu sais :
* Distinguer R, UR, P, UP, SR.
* Lire un kernel et estimer l'occupancy (via plus haut Rn).
* Détecter du spill (STL/LDL).
* Reconnaître les patterns courants (zero-init, pointer 64-bit, indirect addressing, reuse).

Tu peux poser les bonnes questions sur n'importe quel kernel : "combien de registres ?", "y a-t-il du spill ?", "est-ce que les MMAs utilisent reuse ?". C'est la première étape de tout audit.

## Ce qui suit

Le chapitre 5 attaque les **control codes** — la ligne de droite de chaque instruction qui contient toute la politique de scheduling. C'est là que se cachent les vraies causes de stalls.
