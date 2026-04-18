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
  
  UR4 final est la base pour les accès shared du kernel. Construction non encore 
  entièrement comprise (cf hypothèses ouvertes).

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

ptxas reconnaît trois patterns distincts à la compilation et choisit la stratégie adaptée. La forme byte-exact prouve que ptxas canonicalise `% 2^k` et `& (2^k - 1)` vers la même représentation interne.

### 3. Fonction de modulo : caractéristiques

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

### 4. Mécanisme CALL/RET : link register manuel

**CALL.REL.NOINC et RET.REL.NODEC.**
Le hardware ne gère pas automatiquement le link register. Le caller écrit l'adresse de retour dans un registre (MOV Rn, 0x170 avant CALL) et la callee copie cette valeur dans son registre de retour (MOV R2, Rn) avant le RET.

Ce mécanisme permet à ptxas de gérer plusieurs appels à la même fonction avec des adresses de retour différentes sans bibliothèque runtime ni stack pointer automatique.

### 5. Masquage u16 autour des CALL

**Avant l'appel à `__cuda_sm20_rem_u16`.**
```
LOP3.LUT Rn, Rn, 0xffff, RZ, 0xc0, !PT
```
Tronque l'opérande à 16 bits. La fonction a une ABI u16 (indiquée par `_u16` dans son nom), donc les arguments doivent respecter ce format.

Le masque n'est pas défensif ou arbitraire, il est **obligatoire** par l'ABI.

### 6. BAR.SYNC sur SM120

**Modifier `.DEFER_BLOCKING` observé.**
```
BAR.SYNC.DEFER_BLOCKING 0x0
```

**Pipeline : ADU** (observé dans gpuasm, contrairement à l'intuition CBU).

**Argument 0x0 = numéro de barrier.**
CUDA expose 16 barriers hardware par block. `__syncthreads()` utilise toujours la barrier 0.

### 7. Premier pipeline XU observé (fonction modulo)

La fonction modulo utilise MUFU.RCP (reciprocal), qui tourne sur le pipeline **XU** (Transcendental Unit). C'est le premier usage de XU dans nos kernels.

Conversions I2F et F2I observées dans la même fonction, également sur XU.

### 8. Nouveau special register : SR_CgaCtaId

Accessible via S2UR, utilisé dans la construction de la base shared. CGA = Cooperative Grid Array (terminologie cluster SM90+).

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

**Conclusion.** `0x400` semble être une constante architecturale ou un paramètre du format de descriptor shared, indépendant des paramètres du kernel. À traiter au kernel 07.

## Hypothèses ouvertes à la fin du chapitre 06

1. **Signification de `UMOV UR4, 0x400`.** Constante architecturale probable, rôle exact inconnu.
2. **Fonction division sans label symbolique dans le dump.** Pourquoi le modulo est nommé et pas la division.
3. **SR_CgaCtaId dans un kernel sans clusters.** Fallback sur un ID par défaut ou ID block implicite.
4. **`.DEFER_BLOCKING` sur BAR.SYNC.** Nouveau modifier SM90+ ou hérité.
5. **PRMT avec `0x9910` et `0x7710` dans la fonction modulo 255.** Manipulation byte-level non analysée.
6. **Séquence `HFMA2 R3, -RZ, RZ, 15, 0` et FSETP subnormal.** Rôle précis dans l'algorithme reciprocal multiplication.
7. **Choix du registre de return address** dans la calling convention (R7 vs R10 selon kernel).

## Instructions nouvelles observées dans ce chapitre

Liste des opcodes apparus pour la première fois dans nos dumps :

| Opcode | Usage |
|---|---|
| STS | Store to Shared |
| LDS | Load from Shared |
| BAR.SYNC.DEFER_BLOCKING | Block-level barrier |
| UMOV | Uniform MOV (matérialisation d'immédiat dans UR) |
| ULEA | Uniform Load Effective Address |
| UIADD3 | Uniform Integer Add 3-input |
| LOP3.LUT | Logic Operation 3-input avec LUT |
| LEA | Load Effective Address (per-thread) |
| CALL.REL.NOINC | Function call avec link register manuel |
| RET.REL.NODEC | Function return avec link register manuel |
| MUFU.RCP | Reciprocal (pipeline XU) |
| I2F.U16 | Integer to Float (conversion, pipeline XU) |
| F2I.U32.TRUNC.NTZ | Float to Integer (conversion, pipeline XU) |
| FSETP.GEU.AND | FP compare predicate avec combinaison |
| FSEL | FP select (ternaire) |
| PRMT | Byte permutation |
| SHF.R.U32.HI | Funnel shift right, high half |
| S2R / S2UR pour SR_CgaCtaId | Nouveau special register |

## Ce que ce chapitre ajoute à la compétence de lecture SASS

**Reconnaître la shared memory en un coup d'œil.**
STS, LDS, BAR.SYNC, et la construction caractéristique via `UMOV + ULEA` forment un pattern visuel identifiable.

**Diagnostiquer un slowpath arithmétique.**
La présence d'un CALL dans un kernel qui ne devrait en avoir aucun signale division/modulo runtime, math library, ou fonction non inlinable. Chercher le MOV setup juste avant et la fonction placée après EXIT confirme le pattern.

**Choisir la bonne forme source.**
- Modulo / division par constante puissance de 2 : optimal, pas de coût caché.
- Modulo / division par constante non puissance de 2 : coût modéré inline.
- Modulo / division par variable runtime : coûteux, CALL externe.

**Prévoir le layout shared.**
Plusieurs arrays `__shared__` sont empilés dans l'ordre de déclaration. Pas de padding automatique. Pas de surcoût linéaire de setup pour des buffers multiples.

**Regrouper les `__syncthreads()`.**
Plusieurs stores shared consécutifs n'ont besoin que d'une BAR finale. ptxas ne fusionne pas les barriers, c'est au programmeur de les écrire efficacement.