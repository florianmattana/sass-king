# Chapitre 2 — CUDA C++ → PTX → SASS

## Trois langages, trois compilateurs, trois optimiseurs

Quand tu écris un kernel CUDA, le code passe par trois représentations distinctes avant d'arriver au GPU :

```
CUDA C++  →  cicc  →  PTX  →  ptxas  →  SASS  →  GPU
```

Chaque étape a son propre langage, son propre optimiseur, ses propres garanties. Pour influencer le SASS final, il faut savoir **où agir** dans cette chaîne.

## Étape 1 — CUDA C++

C'est ce que tu écris. Du C++ avec quelques extensions (`__global__`, `__device__`, `<<<>>>`, intrinsics comme `__shfl_sync`, `__half2float`, etc.).

À ce niveau, tu décris **la sémantique** : "fais une matmul", "lis ce tableau", "synchronise les threads". Tu ne contrôles pas comment c'est implémenté.

**Ce que tu contrôles à ce niveau** :
* Algorithme global (tiling, blocking, layouts).
* Choix des dtypes (FP16, FP8, FP4).
* Patterns de mémoire (coalescé, padding contre bank conflicts).
* Pragmas pour guider le compilateur (`#pragma unroll N`, `__launch_bounds__(threads, blocks_per_sm)`).
* Inlining via `__forceinline__` ou pas.

**Ce que tu ne contrôles pas** :
* Allocation des registres.
* Ordre exact d'exécution des instructions.
* Choix entre prédication et branche.
* Vectorisation.

**Le compilateur de cette étape** : `cicc` (CUDA Internal C++ Compiler). C'est essentiellement un frontend LLVM qui produit du PTX. Tu interagis rarement avec lui directement.

## Étape 2 — PTX

PTX est un langage **assembleur abstrait**. Il a des instructions, des registres virtuels (en nombre illimité), des conventions d'appel. Mais il n'est pas exécuté directement par le GPU — il est compilé en SASS.

**Caractéristique clé** : PTX est **stable à travers les architectures**. Le même PTX peut être compilé pour SM80, SM90, SM120 et produire du SASS valide à chaque fois (avec des perfs différentes selon les capacités matérielles).

**Exemple PTX** :
```ptx
// Un MMA FP4 block-scaled
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0
    {%0,%1,%2,%3},
    {%4,%5,%6,%7},
    {%8,%9},
    {%10,%11,%12,%13},
    {%14},{%15,%16},
    {%17},{%18,%19};
```

Cette instruction PTX décrit "fais un MMA block-scaled de cette forme avec ces dtypes". Comment c'est traduit en SASS dépend de l'architecture cible.

**Documentation** : NVIDIA documente PTX dans le **CUDA Parallel Thread Execution ISA Manual** (https://docs.nvidia.com/cuda/parallel-thread-execution/). Tu peux lire en détail toutes les instructions PTX, leurs sémantiques, leurs versions.

**Quand tu écris du `asm volatile`** dans ton code CUDA, tu écris du PTX. Pas du SASS.

**Ce que tu peux contrôler en passant par PTX inline** :
* Choisir une instruction PTX précise (par exemple `mma.sync.aligned.kind::mxf4nvf4` pour forcer l'usage du path FP4 peak).
* Bypass les abstractions C++ et atteindre directement les capacités matérielles.
* Garantir une sémantique précise (ex : memory ordering avec `fence.sc`).

**Ce que tu ne contrôles pas via PTX** :
* L'allocation de registres physiques (PTX a des registres virtuels).
* Le scheduling exact (ptxas reschedule librement).
* Le choix d'opcodes SASS spécifiques (plusieurs SASS peuvent réaliser le même PTX).

## Étape 3 — SASS

SASS est le **vrai langage machine** du GPU. Spécifique à chaque compute capability. Non documenté publiquement.

C'est le code que `cuobjdump --dump-sass` t'extrait. C'est le code que le hardware exécute.

**Exemple SASS** (correspondant approximativement au PTX précédent) :
```
QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R4, R8, R12, R18, R31, URZ ;   /* 0x71f0120c040c747a */
                                                                    /* 0x000fe2000028fe0c */
```

Cette instruction SASS est le résultat final, optimisé par ptxas, qui sera décodé par le GPU. Tu peux compter exactement combien d'opérandes, voir quels registres sont utilisés, et lire le control code (deuxième ligne) pour voir le scheduling choisi.

**Le compilateur de cette étape** : `ptxas`. C'est le compilateur PTX → SASS. C'est lui qui fait :
* Allocation des registres physiques (R0-R255).
* Scheduling des instructions (ordre, stall counts, dual-issue).
* Application des optimisations bas niveau (loop unrolling, dead code elimination, register reuse cache).
* Choix entre prédication et branche pour les conditions.
* Vectorisation des loads/stores (`LDG.E.128` au lieu de 4× `LDG.E`).

**Ptxas est une boîte noire**. Il a beaucoup de heuristiques, peu de documentation, et change de comportement entre versions de CUDA Toolkit. Le projet entier vise à **reverse-engineerer** ce qu'il fait.

## Ce qui se passe à chaque étape, schématiquement

Prends cette ligne C++ :
```cpp
for (int k = 0; k < 4; k++) {
    acc += A[k] * B[k];
}
```

**Après cicc (PTX)** : approximativement
```ptx
mov.u32 %r1, 0;
loop_start:
    setp.lt.s32 %p1, %r1, 4;
    @!%p1 bra loop_end;
    
    mul.lo.s32 %r2, %r1, 4;          // k * sizeof(float)
    add.s32 %r3, %r_A_base, %r2;
    ld.global.f32 %f1, [%r3];        // load A[k]
    
    add.s32 %r4, %r_B_base, %r2;
    ld.global.f32 %f2, [%r4];        // load B[k]
    
    fma.rn.f32 %f_acc, %f1, %f2, %f_acc;
    
    add.s32 %r1, %r1, 1;
    bra loop_start;
loop_end:
```

**Après ptxas (SASS)** : si ptxas full-unroll (boucle constante de 4 itérations) :
```
LDG.E R4, [R10]              // A[0]
LDG.E R5, [R20]              // B[0]
LDG.E R6, [R10+0x4]          // A[1]
LDG.E R7, [R20+0x4]          // B[1]
LDG.E R8, [R10+0x8]          // A[2]
LDG.E R9, [R20+0x8]          // B[2]
LDG.E R12, [R10+0xC]         // A[3]
LDG.E R13, [R20+0xC]         // B[3]
FFMA R14, R4, R5, R14        // acc = A[0]*B[0] + acc
FFMA R14, R6, R7, R14        // acc = A[1]*B[1] + acc
FFMA R14, R8, R9, R14        // acc = A[2]*B[2] + acc
FFMA R14, R12, R13, R14      // acc = A[3]*B[3] + acc
```

Note les transformations :
* La boucle a disparu (full unroll).
* Les loads sont batchés en avance.
* Les FFMAs sont chainées avec destination = source (acc dans R14 partout).
* Pas de `setp` ni de `bra` : tout est séquentiel.

C'est ptxas qui a décidé tout ça. Ton code C++ n'a pas demandé d'unroll, le `#pragma unroll` n'était pas obligatoire — ptxas a inféré.

## Quel niveau cibler quand tu optimises ?

**Pour un changement algorithmique** : C++. Restructure ton code, change le tile size, fusionne des kernels.

**Pour une instruction précise** : PTX inline. Si tu veux forcer l'usage de `mma.sync.aligned.kind::mxf4nvf4`, écris-le explicitement en `asm volatile`. Le compilateur C++ ne génère pas toujours les instructions PTX optimales.

**Pour comprendre ce qui se passe** : SASS. C'est la seule façon de savoir si tes intentions algorithmiques + tes hints PTX ont survécu à ptxas et produisent ce que tu attends.

**Pour modifier le SASS directement** : tu ne peux pas. Il n'y a pas d'assembleur SASS officiel public. Tu peux théoriquement utiliser MaxAS / Turing AS / OpenAS pour des archis anciennes, mais rien n'existe pour Blackwell. La seule façon d'influencer le SASS est de jouer avec C++ et PTX.

## Comment dumper le SASS d'un binaire

Si tu as un binaire (`a.out`, `.so`, `.cubin`) :
```bash
cuobjdump --dump-sass a.out > a.sass
```

Si tu as un `.cu` source :
```bash
nvcc -arch=sm_120 -cubin -o my_kernel.cubin my_kernel.cu
cuobjdump --dump-sass my_kernel.cubin > my_kernel.sass
```

Si tu veux voir le PTX intermédiaire :
```bash
nvcc -arch=sm_120 -ptx -o my_kernel.ptx my_kernel.cu
```

Si tu veux les deux PTX et SASS dans le même binaire :
```bash
nvcc -arch=sm_120 -O3 my_kernel.cu -o a.out
cuobjdump --dump-sass --dump-ptx a.out
```

Si tu veux les choix d'allocation de registres et autres warnings ptxas :
```bash
nvcc -arch=sm_120 -Xptxas -v my_kernel.cu
# Output : 
# ptxas info : Used N registers, M bytes smem, ...
# ptxas info : Compile time: ...
```

Si ton binaire est un fat binary (compilé pour plusieurs architectures), `cuobjdump` te montrera tous les SASS. Filtre avec `--gpu-architecture=sm_120` pour ne garder que ce qui t'intéresse.

## Influence des paramètres de compilation

Plusieurs flags `nvcc` / `ptxas` influencent le SASS généré :

* **`-O0` / `-O1` / `-O2` / `-O3`** : niveau d'optimisation. Tu utiliseras quasi toujours `-O3` en production. `-O0` est utile pour comprendre ce que serait le SASS "naif" sans optimisation, à comparer avec `-O3` pour voir ce que ptxas a fait.

* **`-Xptxas -O3`** : force `-O3` côté ptxas même si nvcc passe autre chose.

* **`-Xptxas -v`** : verbose, te donne register usage et autres infos.

* **`-maxrregcount=N`** : limite le nombre de registres par thread. Si N est faible, ptxas va spill vers local memory.

* **`-arch=sm_120`** vs **`-arch=sm_120a`** : la différence entre profil générique et arch-specific. Voir glossaire SM120 specifics.

* **`-Xptxas --warn-on-spills`** : warning explicite si ptxas spill.

* **`-G`** : debug info, désactive certaines optimisations. À éviter pour mesurer la perf, à utiliser pour debug.

## Versions de CUDA Toolkit

Le SASS généré pour le **même** code C++ peut différer entre **CUDA 12.3, 12.6, 13.0, etc.** Ptxas évolue, ses heuristiques changent.

**Implications pour ton workflow** :
* Quand tu compares un SASS "à un mois d'écart", vérifie que c'est la même version de toolkit. Sinon ce n'est pas la même comparaison.
* Quand tu rapportes une observation SASS publique (blog, repo), précise la version : `CUDA 12.6 + nvcc V12.6.85 + ptxas V12.6.x`.
* Quand tu mets à jour le toolkit, **rebuilde et compare** ton SASS avant/après. Si performance diffère, regarde le diff SASS pour identifier le changement.

## Limites de l'introspection

Même avec tout ce dump SASS, certaines choses restent **opaques** :

* **Latences exactes** : tu peux mesurer (microbenchmark via clock64()) mais NVIDIA ne les publie pas.
* **Bandwidth interne** : tu peux observer des patterns mais le découplage entre register file, L1, et tensor cores n'est pas documenté.
* **Decisions ptxas** : tu vois le résultat, pas le raisonnement. Pourquoi ptxas a unrollé 4× et pas 8× sur cette boucle particulière ? Tu peux deviner, pas savoir.

C'est précisément ce que le projet SASS King essaye de combler en accumulant des observations empiriques.

## Ce qui suit

Avec le pourquoi (ch 1) et la chaîne de compilation (ch 2) posés, le chapitre 3 attaque la **lecture pratique** du SASS : comment décortiquer une ligne de dump, identifier les sections d'un kernel, et reconnaître les patterns courants. Tu commences à devenir autonome.
