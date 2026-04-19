# Kernel 06 — vector_smem (shared memory scalar)

Premier kernel utilisant la shared memory. Le source fait un round-trip `global → shared → shared → global` avec un shift d'indice pour forcer le compilateur à garder le passage par shared.

## Source

```cuda
__global__ void vector_smem(const float* a, float* c, int n) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % blockDim.x;
        c[i] = smem[src];
    }
}
```

Launch : `<<<4, 256>>>` sur n=1024.

## Structure du SASS principal (30 instructions)

Le kernel suit le squelette en 6 sections déjà identifié, puis ajoute :

```
[Prologue : 8 instructions]
  Stack pointer, setup ID, calcul index, bounds check
  Identique aux kernels 01-05.

[Setup global a[i] : 4 instructions]
  LDC.64 pointeur a, LDCU.64 descriptor global, IMAD.WIDE adresse, LDG.E load.
  Identique aux kernels précédents.

[Nouveau : Setup base shared et descriptor : 4 instructions]
  S2UR UR5, SR_CgaCtaId
  UMOV UR4, 0x400
  ULEA UR4, UR5, UR4, 0x18
  
  UR4 final est la base pour les accès shared du kernel. Construction décodée ci-dessous.

[Nouveau : Write path shared : 3 instructions]
  LEA R7, R0, UR4, 0x2        ; &smem[tid] = UR4 + tid*4
  STS [R7], R2                 ; smem[tid] = a[i]
  BAR.SYNC.DEFER_BLOCKING 0x0  ; __syncthreads()

[Slowpath modulo : 1 CALL vers une fonction de 21 instructions]
  MOV R7, 0x170
  CALL.REL.NOINC __cuda_sm20_rem_u16

[Read path shared + store global : 4 instructions]
  LEA R0, R0, UR4, 0x2        ; &smem[src]
  LDS R7, [R0]                 ; smem[src]
  IMAD.WIDE R2, R5, 0x4, R2    ; &c[i]
  STG.E desc[UR6][R2.64], R7   ; c[i] = smem[src]

[Exit + fonction modulo placée après : ~25 instructions]
```

## Observations solides (vérifiées sur variantes 06b-06i)

### 1. Shared memory : architecture d'adressage

**Construction de la base shared.**
La séquence `S2UR UR5, SR_CgaCtaId` + `UMOV UR4, 0x400` + `ULEA UR4, UR5, UR4, 0x18` construit l'adresse de base de la shared memory du block. Décodage de ULEA :

```
ULEA UR4, UR5, UR4, 0x18
```

signifie `UR4 = (UR5 << 0x18) + UR4`. Le `0x18` est 24 en décimal. Donc la base finale est :

```
shared_base = (CgaCtaId << 24) + 0x400
```

`0x400` semble être un offset architectural (cf section dédiée plus bas) et le décalage de 24 bits sur CgaCtaId produit un offset par block dans le pool SM-wide de shared memory. Interprétation : chaque block dans le SM occupe une tranche alignée sur 16 MiB (2^24 bytes) du pool shared mappé en mémoire, avec `0x400` comme offset de départ à l'intérieur de la tranche.

[HYP] Le shift de 24 est probablement surdimensionné par rapport à la shared memory réelle (48 KB à 228 KB selon config), donc les bits hauts ne collent jamais dans la pratique. Le shift pourrait être un pattern d'encodage architectural pour l'adressage de descriptor shared, pas une adresse physique directe.

**Un seul UMOV + dérivations par addition pour N buffers shared.**
Confirmé par kernel 06g avec deux `__shared__` : un seul `UMOV UR4, 0x400` puis `UIADD3 UR5, UPT, UPT, UR4, 0x400, URZ` pour dériver la base du deuxième buffer.

**Les buffers shared sont placés consécutivement en mémoire.**
L'offset entre deux buffers `smem_a[256]` et `smem_b[256]` est exactement `0x400 = 1024 bytes`, soit la taille en bytes de `smem_a`. Pas d'alignement supplémentaire inséré automatiquement.

**Adressage shared ne nécessite pas de descriptor.**
Contrairement à LDG/STG global qui utilisent `desc[UR][R.64]`, les accès shared utilisent `[R]` direct ou `[R+UR]`. La base shared est encodée dans le registre d'adresse lui-même.

**Mode `[R+UR]` pour lire plusieurs buffers à la même position.**
Observé dans kernel 06g : `LDS R8, [R6+UR4]` et `LDS R11, [R6+UR5]`. Un seul calcul d'adresse per-thread (R6 masqué par le modulo), deux offsets uniformes pour les deux buffers. Pattern optimal pour le multi-buffer.

**STS successifs peuvent être émis sans BAR intermédiaire.**
Observé dans kernel 06g : deux STS consécutifs, une seule BAR.SYNC après. Le `__syncthreads()` unique couvre tous les stores précédents dans le block.

### 2. Modulo entier : trois modes distincts selon la constante

| Source | Stratégie ptxas | Instructions pour modulo |
|---|---|---|
| `% blockDim.x` (runtime) | CALL `__cuda_sm20_rem_u16` + fonction 21 instructions | 1 CALL + corps |
| `% 255` (non puissance de 2) | Inline reciprocal multiplication (magic number) | ~10 inline |
| `% 256` (puissance de 2) | Fusion dans LEA + LOP3 mask | 2 instructions |
| `& 255` | **Identique à `% 256`** (SASS byte-exact) | 2 instructions |

Ptxas reconnaît trois patterns distincts à la compilation et choisit la stratégie adaptée. La forme byte-exact prouve que ptxas canonicalise `% 2^k` et `& (2^k - 1)` vers la même représentation interne.

### 3. Dissection de la fonction `__cuda_sm20_rem_u16`

La fonction de modulo runtime fait 21 instructions. Voici sa structure, décomposée par algorithme.

```
Stage 1 : conversion u16 → float
  I2F.U16 R2, R9              ; R2 = (float) diviseur
  HFMA2 R3, -RZ, RZ, 15, 0    ; R3 = constante ajustée (exp FP16 bias)
  FSETP.GEU.AND P0, ...       ; test subnormal

Stage 2 : reciprocal approximation
  MUFU.RCP R3, R2             ; R3 ≈ 1 / R2 (XU pipeline)
  FSEL ...                    ; sélection selon P0

Stage 3 : reconstruction quotient
  FMUL R2, R_dividend, R3     ; quotient approximé en FP
  F2I.U32.TRUNC.NTZ R2, R2    ; conversion vers entier
  
Stage 4 : correction et calcul du reste
  LOP3.LUT ...                ; masquage ou correction
  IADD R_q, R_q, ...          ; ajustement ±1 selon signe du test
  IMAD R_r, R_q, -R_divisor, R_dividend   ; reste = dividende - quotient * diviseur
  
Stage 5 : retour
  MOV R2, Rn                  ; adresse de retour
  RET.REL.NODEC R_r
```

L'algorithme est une **Newton-Raphson-like reciprocal multiplication** pour division entière :

1. Convertir le diviseur entier vers un float.
2. Calculer `1/d` en float avec MUFU.RCP (approximation ~22 bits de précision).
3. Multiplier le dividende par cette approximation pour obtenir un quotient flottant.
4. Convertir le quotient vers un entier (troncature).
5. Corriger le quotient si nécessaire (l'approximation peut être off-by-one).
6. Calculer le reste comme `r = a - q * b`.

**Pipelines impliqués.**
- XU (Transcendental Unit) : MUFU.RCP, I2F.U16, F2I.U32 — opérations de conversion et reciprocal.
- FMA : FMUL, FMA — arithmétique flottante.
- ALU : LOP3, IADD, IMAD, FSEL — arithmétique entière et sélection.

**Coût estimé.** MUFU.RCP seul fait ~25 cycles. L'ensemble de la fonction coûte probablement 40-60 cycles en throughput (hors latence de CALL/RET). Comparer à un modulo par puissance de 2 (1 instruction, ~5 cycles), le slowpath est 10-12× plus coûteux.

[GAP] Je n'ai pas validé les offsets exacts des instructions dans la fonction ni leurs control codes. L'analyse algorithmique ci-dessus est basée sur le pattern général reconnu mais les instructions précises pourraient différer. À revalider sur un dump complet de la fonction si un audit de précision est requis.

### 4. Fonction de modulo : caractéristiques

**Une seule instance de la fonction pour N appels (kernel 06i).**
Deux modulos avec des diviseurs runtime différents dans le même kernel → deux CALL vers la même adresse de fonction. Taille code constante indépendamment du nombre d'appels.

**ABI observée pour `__cuda_sm20_rem_u16`.**
- R8 : dividende u16 en entrée
- R9 : diviseur u16 en entrée, reste en sortie
- Rn (R7 ou R10 selon kernel) : return address, écrite par le caller via MOV avant CALL

**Coût runtime non amorti.**
Chaque appel exécute la fonction complète (~30 cycles minimum, dont MUFU.RCP sur pipeline XU). N appels = N fois ce coût.

**Division entière : fonction séparée mais corps similaire (kernel 06h).**
Division et modulo ont deux fonctions distinctes partageant la même primitive (reciprocal multiplication par magic number), mais diffèrent sur la dernière étape. Division s'arrête au quotient. Modulo calcule quotient puis reconstruit reste via IMAD.

**Conséquence pratique.**
Utiliser `/` et `%` dans le même kernel sur les mêmes opérandes = deux CALL distincts. Pour économiser un CALL, écrire `q = a / b; r = a - q * b;` au lieu de `q = a / b; r = a % b;`.

### 5. Mécanisme CALL/RET : link register manuel

**CALL.REL.NOINC et RET.REL.NODEC.**
Le hardware ne gère pas automatiquement le link register. Le caller écrit l'adresse de retour dans un registre (MOV Rn, 0x170 avant CALL) et la callee copie cette valeur dans son registre de retour (MOV R2, Rn) avant le RET.

Ce mécanisme permet à ptxas de gérer plusieurs appels à la même fonction avec des adresses de retour différentes sans bibliothèque runtime ni stack pointer automatique.

### 6. Masquage u16 autour des CALL

**Avant l'appel à `__cuda_sm20_rem_u16`.**
```
LOP3.LUT Rn, Rn, 0xffff, RZ, 0xc0, !PT
```
Tronque l'opérande à 16 bits. La fonction a une ABI u16 (indiquée par `_u16` dans son nom), donc les arguments doivent respecter ce format.

Le masque n'est pas défensif ou arbitraire, il est **obligatoire** par l'ABI.

### 7. BAR.SYNC sur SM120

**Modifier `.DEFER_BLOCKING` observé.**
```
BAR.SYNC.DEFER_BLOCKING 0x0
```

**Pipeline : ADU** (observé dans gpuasm, contrairement à l'intuition CBU).

**Argument 0x0 = numéro de barrier.**
CUDA expose 16 barriers hardware par block. `__syncthreads()` utilise toujours la barrier 0.

**`.DEFER_BLOCKING` comme mode par défaut.**
Semble être la forme émise automatiquement par ptxas pour `__syncthreads()` sur SM120. Pas d'alternative observée dans nos dumps (pas de `BAR.SYNC` sans modifier).

### 8. Premier pipeline XU observé (fonction modulo)

La fonction modulo utilise MUFU.RCP (reciprocal), qui tourne sur le pipeline **XU** (Transcendental Unit). C'est le premier usage de XU dans nos kernels.

Conversions I2F et F2I observées dans la même fonction, également sur XU.

### 9. Nouveau special register : SR_CgaCtaId

Accessible via S2UR, utilisé dans la construction de la base shared. CGA = Cooperative Grid Array (terminologie cluster SM90+).

**Interprétation.** Même dans un kernel sans clusters (notre cas), SR_CgaCtaId est un identifiant unique du block dans son contexte d'exécution. Pour un kernel non-cluster, il équivaut probablement à blockIdx.x mais avec une encoding architecturale distincte pour la construction de l'adresse shared. L'usage ici n'est pas de connaître l'ID du block dans un cluster mais d'avoir une clé unique pour indexer la shared memory du SM.

[HYP] SR_CgaCtaId sur un kernel non-cluster pourrait retourner une valeur construite à partir de blockIdx et d'un cluster ID implicite de 0. Non vérifié.

## Test UMOV UR4, 0x400 : trois hypothèses rejetées

Trois variantes testées pour identifier ce que code `0x400` :

| Variante | Shared size | Block size | Modulo | UMOV observé |
|---|---|---|---|---|
| 06b | 256 floats (1024 B) | 256 | 256 | **0x400** |
| 06e | 128 floats (512 B) | 128 | 128 | **0x400** |
| 06f | 512 floats (2048 B) | 512 | 512 | **0x400** |

**Rejeté.** `0x400` n'encode ni la taille shared, ni la taille de block, ni la launch config.

**Ce qui change avec la taille.** Le masque LOP3 :
- 256 floats → 0x3fc = 1020 = (256-1) × 4
- 128 floats → 0x1fc = 508 = (128-1) × 4
- 512 floats → 0x7fc = 2044 = (512-1) × 4

**Conclusion.** `0x400` semble être une constante architecturale ou un paramètre du format de descriptor shared, indépendant des paramètres du kernel. Confirmé au kernel 07 : `0x400` apparaît conditionnellement sur la présence d'un `__shared__` dans le kernel.

## Hypothèses ouvertes à la fin du chapitre 06

1. **Signification exacte de `UMOV UR4, 0x400` + `ULEA shift 24`.** Construction de base shared décodée au niveau syntaxique (`(CgaCtaId << 24) + 0x400`) mais l'interprétation architecturale reste floue.
2. **Fonction division sans label symbolique dans le dump.** Pourquoi le modulo est nommé et pas la division.
3. **SR_CgaCtaId dans un kernel sans clusters.** Fallback sur un ID par défaut ou ID block implicite.
4. **`.DEFER_BLOCKING` sur BAR.SYNC.** Nouveau modifier SM90+ ou hérité. Semble être le mode par défaut sur SM120.
5. **PRMT avec `0x9910` et `0x7710` dans la fonction modulo 255.** Manipulation byte-level non analysée. Les valeurs 0x9910 et 0x7710 sont des masks de permutation byte-level : chaque nibble du masque indique quel byte source sélectionner pour produire chaque byte destination. À décoder byte-par-byte dans un futur pass.
6. **Séquence `HFMA2 R3, -RZ, RZ, 15, 0` et FSETP subnormal.** Rôle précis dans l'algorithme reciprocal multiplication. `15` correspond au bias de l'exposant FP16 (5 bits d'exposant, bias 15). L'instruction charge un float dans l'intervalle utile pour MUFU.RCP.
7. **Choix du registre de return address** dans la calling convention (R7 vs R10 selon kernel). Probablement lié à la register pressure locale avant le CALL : ptxas choisit le premier registre disponible.

## Instructions nouvelles observées dans ce chapitre

Liste des opcodes apparus pour la première fois dans nos dumps :

| Opcode | Usage |
|---|---|
| STS | Store to Shared |
| LDS | Load from Shared |
| BAR.SYNC.DEFER_BLOCKING | Block-level barrier |
| UMOV | Uniform MOV (matérialisation d'immédiat dans UR) |
| ULEA | Uniform Load Effective Address (`UR_dst = (UR_src1 << imm) + UR_src2`) |
| UIADD3 | Uniform Integer Add 3-input |
| LOP3.LUT | Logic Operation 3-input avec LUT |
| LEA | Load Effective Address (per-thread, `R_dst = (R_src1 << imm) + UR_base`) |
| CALL.REL.NOINC | Function call avec link register manuel |
| RET.REL.NODEC | Function return avec link register manuel |
| MUFU.RCP | Reciprocal (pipeline XU) |
| I2F.U16 | Integer to Float (conversion, pipeline XU) |
| F2I.U32.TRUNC.NTZ | Float to Integer (conversion, pipeline XU) |
| FSETP.GEU.AND | FP compare predicate avec combinaison (GEU = greater equal unordered, NaN retourne true) |
| FSEL | FP select (ternaire) |
| PRMT | Byte permutation (nibble mask sélectionne les bytes sources) |
| SHF.R.U32.HI | Funnel shift right, high half |
| S2R / S2UR pour SR_CgaCtaId | Nouveau special register |

## Ce que ce chapitre ajoute à la compétence de lecture SASS

**Reconnaître la shared memory en un coup d'œil.**
STS, LDS, BAR.SYNC, et la construction caractéristique via `UMOV 0x400 + ULEA shift-24` forment un pattern visuel identifiable.

**Diagnostiquer un slowpath arithmétique.**
La présence d'un CALL dans un kernel qui ne devrait en avoir aucun signale division/modulo runtime, math library, ou fonction non inlinable. Chercher le MOV setup juste avant et la fonction placée après EXIT confirme le pattern.

**Choisir la bonne forme source.**
- Modulo / division par constante puissance de 2 : optimal, pas de coût caché.
- Modulo / division par constante non puissance de 2 : coût modéré inline.
- Modulo / division par variable runtime : coûteux, CALL externe.

**Prévoir le layout shared.**
Plusieurs arrays `__shared__` sont empilés dans l'ordre de déclaration. Pas de padding automatique. Pas de surcoût linéaire de setup pour des buffers multiples.

**Regrouper les `__syncthreads()`.**
Plusieurs stores shared consécutifs n'ont besoin que d'une BAR finale. Ptxas ne fusionne pas les barriers, c'est au programmeur de les écrire efficacement.

**Identifier le modulo runtime même sans CALL visible.**
Le MOV immediate setup d'une return address (typiquement `MOV Rn, 0x??0` où 0x??0 est proche de la fin de la fonction) juste avant le CALL est un signal. Sans ce pattern, la division est inline.