# Chapitre 12 — Diagnostiquer les problèmes

## Le but du chapitre

Une fois que tu sais lire un dump SASS et reconnaître les patterns, l'étape suivante est de **diagnostiquer pourquoi un kernel performe mal**. Ce chapitre couvre les **5 catégories principales de problèmes** que tu peux identifier directement dans le SASS, leurs signatures, et les fixes possibles.

## Catégorie 1 : Stalls scoreboard (memory waits)

**Symptôme** : NCU rapporte `stall_long_scoreboard` ou `stall_lg_throttle` élevé. Le warp attend que des loads memory complètent.

**Signature SASS** :
* Wait masks non-zéro fréquents dans la hot loop.
* Distance courte entre LDG/LDGSTS et leur consommateur (donc latence non cachée).

**Diagnostic précis** :
```
LDG.E R4, [R2.64] ;           /* SB asgn = 1 */    ← producer
ADD R5, R6, R7 ;              /* indep */
FFMA R8, R4, R9, R10 ;        /* wait mask 0x02 (wait SB1) */    ← consumer 2 instructions plus tard
```

Distance = 2 instructions = ~2-4 cycles. Latence LDG L1 hit = 30 cycles. Donc FFMA stall ~26 cycles à chaque iteration.

**Fixes** :
* **Réordonner** : déplacer le LDG plus tôt dans le code. Plus de distance = plus de latence cachée.
* **Prefetch** : utiliser cp.async (LDGSTS) pour overlap.
* **Plus d'instructions independent** entre producer et consumer.
* **Utiliser cache hints** (`.CONSTANT` pour read-only) pour augmenter hit rate L1.

## Catégorie 2 : Register spill (STL/LDL)

**Symptôme** : NCU rapporte `stall_lg_throttle` élevé même sans operations global. Performance pathologique.

**Signature SASS** :
```
STL.64 [R1+0x0], R2 ;          ← spill
STL.64 [R1+0x8], R8 ;
STL.64 [R1+0x10], R12 ;
... (plus tard) ...
LDL R20, [R11+UR7] ;           ← reload
LDL R21, [R11+UR7+0x4] ;
```

**Diagnostic** :
```bash
grep -c "STL\|LDL" mon_kernel.sass    # > 0 = spill
nvcc -arch=sm_120 -Xptxas -v ...      # confirme le compte exact
```

**Fixes** :
* **Augmenter register budget** via `__launch_bounds__(THREADS, MIN_BLOCKS_PER_SM)`.
* **Simplifier les live ranges** : ne pas garder des valeurs longtemps si pas utilisées.
* **Casser le kernel** en plusieurs si trop complexe.
* **Utiliser `__restrict__`** sur les pointeurs pour permettre plus d'optimisations.

## Catégorie 3 : Bank conflicts (shared memory)

**Symptôme** : NCU rapporte `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum > 0`.

**Signature SASS** : pas directement visible. Tu peux suspecter via :
* Beaucoup de `LDS.U8` consécutifs sur addresses adjacentes (chaque thread lit 1 byte à un offset différent du même bank).
* Layout shared memory non-padded pour des strides power-of-2.

**Diagnostic** : nécessite NCU + analyse du layout shared memory.

**Fixes** :
* **Padding** : ajouter +4 ou +1 element par row pour décaler les banks.
* **Layout transpose** : changer row-major en col-major si l'access pattern le permet.
* **Vectoriser** : `LDS.128` au lieu de 4× LDS séparés.

## Catégorie 4 : Faible utilisation tensor core

**Symptôme** : NCU rapporte `sm__pipe_tensor_op_hmma_cycles_active_realtime.pct_of_peak_sustained_active` faible (<50% pour un kernel GEMM-heavy).

**Signature SASS** :

* **Trop peu de MMAs** par rapport au reste du kernel. Si <5% des instructions sont des MMAs dans un kernel GEMM, problème.
* **Chains MMA courtes** (1-2 MMAs entre setups). Le coût de setup dilue le throughput compute.
* **Mauvais dtype choisi** : utilisation de HMMA F32 acc alors que F16 acc suffirait (2× perte throughput).
* **Pas de path peak** : QMMA `.SF` (~500 TFLOPS) au lieu de OMMA `.4X` (900 TFLOPS) pour FP4.

**Diagnostic** :
```bash
grep -c "HMMA\|QMMA\|OMMA" mon_kernel.sass    # nombre total de MMAs
# Comparer avec total instructions et taille de la hot loop
```

**Fixes** :
* **Augmenter le tile size** : plus de K iterations unrollées = chains plus longues.
* **Choisir le bon path** : compiler pour `sm_120a` et utiliser `kind::mxf4nvf4` pour FP4 peak.
* **Utiliser F16 acc** quand la précision le permet.
* **Restructurer** pour amortir les setups.

## Catégorie 5 : Divergence excessive

**Symptôme** : Performance en dessous des attentes sans cause évidente. NCU peut ne pas pointer directement.

**Signature SASS** :
* **Beaucoup de paires BSSY/BSYNC** (>5 par hot loop iteration).
* **Body de divergence non trivial** entre BSSY et BSYNC (>10 instructions).
* **Patterns prédiqués alternés** : `@P0 instr ; @!P0 instr ; @P0 instr ;` etc.

**Cas typique** : encoding cascade comme `encode_fp4_e2m1` qui a 8 niveaux de comparaison. Chaque thread peut potentiellement prendre une branche différente.

**Diagnostic** :
```bash
grep -c "BSSY\|BSYNC" mon_kernel.sass     # nombre de divergence regions
```

**Fixes** :
* **Group threads par chemin** : restructurer pour que des warps entiers prennent le même chemin (reduce divergence à warp granularity).
* **Predication forced** : pour des branches courtes, forcer la prédication via `__forceinline__` ou structure du code.
* **Lookup table** : remplacer une cascade if/else par une LDC depuis constant memory.

## Catégorie 6 : Pas assez d'occupancy

**Symptôme** : NCU rapporte `sm__warps_active.avg.pct_of_peak_sustained_active` faible (<50%).

**Signature SASS** :
* Plus haut Rn élevé (R200+).
* Beaucoup de shared memory utilisée (vérifier `cuobjdump --dump-sass --print-args` ou `-Xptxas -v`).

**Diagnostic** :
```bash
# Plus haut Rn
grep -oP 'R\d+' mon_kernel.sass | sort -t'R' -k2 -n -u | tail -3

# Resources usage
nvcc -arch=sm_120 -Xptxas -v ...
# Output: ptxas info : Used N registers, M bytes smem
```

**Fixes** :
* **`__launch_bounds__`** : dire à ptxas combien d'occupancy tu veux.
* **Réduire shared memory** : utiliser des tiles plus petits, ou recalculer au lieu de cacher.
* **Réduire register pressure** (cf catégorie 2).

## Catégorie 7 : Pipeline cp.async pas optimal

**Symptôme** : Memory stalls élevés malgré utilisation de cp.async.

**Signature SASS** :
* **Pas assez de stages** : `DEPBAR.LE SB0, 0x0` (waitall) à chaque iteration au lieu de `0x1` ou `0x2`.
* **Setup déséquilibré** : trop de LDGSTS par stage, pas assez de compute pour caché.
* **BAR.SYNC trop fréquente** : si tous les warps doivent re-sync à chaque iteration, perte.

**Diagnostic** :
```bash
grep -nE "DEPBAR.LE SB0" mon_kernel.sass
# Lire le N : si toujours 0, pas de pipelining réel
```

**Fixes** :
* **Augmenter stages** : 3 stages au lieu de 2.
* **Vérifier balance** : compute par stage doit être ≥ memory transfer time.
* **Réduire BAR.SYNC** : utiliser DEPBAR.LE warp-local quand possible.

## Méthodologie d'audit complet

Workflow de diagnostic d'un kernel slow :

### Étape 1 : NCU pour identifier la catégorie
```bash
ncu --set full -o ncu_report mon_binaire
ncu --import ncu_report.ncu-rep --print-summary
```

Regarde les top stall reasons. Ça te dit dans quelle catégorie chercher.

### Étape 2 : SASS pour confirmer la cause précise
```bash
cuobjdump --dump-sass mon_binaire > mon_kernel.sass

# Selon catégorie, extraire la signature correspondante
grep ...
```

### Étape 3 : Hypothèse et test
Formule une hypothèse : "le ralentissement est causé par X parce que Y".

Modifie le code (ou les paramètres compile) pour tester.

Recompile, re-dump SASS, vérifie que la signature a changé.

Re-mesure avec NCU pour confirmer l'amélioration.

### Étape 4 : Itérer
La plupart des optimisations gagnent 10-30%. Pour un facteur 2x ou plus, c'est souvent une combinaison de plusieurs fixes.

## Erreurs courantes en audit

**Erreur 1 : Optimiser sans NCU**
Tu lis le SASS, tu vois quelque chose qui te paraît sous-optimal, tu changes le code. Sans NCU, tu ne sais pas si c'est vraiment le bottleneck.

**Toujours mesurer d'abord, puis optimiser le vrai bottleneck.**

**Erreur 2 : Confondre symptôme et cause**
NCU dit "stall_long_scoreboard 50%". Tu changes les LDG. Mais en fait le stall était causé par un BAR.SYNC qui forçait tous les warps à attendre.

**Le SASS te dit la cause précise. NCU te dit le symptôme.**

**Erreur 3 : Ignorer les warnings ptxas**
`-Xptxas -v` te donne register usage et spills. Si tu ne lis pas ces warnings, tu peux passer à côté de spill catastrophique.

**Erreur 4 : Optimiser avant de comprendre**
Lire 5000 lignes de SASS pour identifier le bottleneck prend du temps. Mais c'est toujours plus rapide que de tester 10 modifications au hasard.

**Investis du temps dans le diagnostic.**

## Limites du diagnostic SASS

Tu ne peux pas tout diagnostiquer depuis le SASS seul :

* **Bandwidth contention global** : si plusieurs SMs accèdent au même cache line, le SASS ne le voit pas.
* **TMA / async details** : sur SM120 c'est limité, mais sur SM90+ il y a plein d'infos invisibles.
* **Power throttling** : pas dans le SASS.
* **Driver overhead** : pas dans le SASS.

Ces choses nécessitent NCU avec des métriques système, ou des outils externes.

## Bilan

Tu sais maintenant identifier 7 catégories de problèmes :
1. Stalls scoreboard (memory waits)
2. Register spill
3. Bank conflicts shared memory
4. Faible utilisation tensor core
5. Divergence excessive
6. Pas assez d'occupancy
7. Pipeline cp.async pas optimal

Pour chacune : signature SASS, diagnostic, fix possible. Combinéavec NCU, tu as un workflow complet.

## Ce qui suit

Le chapitre 13 fait un **cas d'étude complet** : on prend un kernel réel, on l'audite étape par étape de A à Z. C'est l'application pratique de tout ce qui précède.
