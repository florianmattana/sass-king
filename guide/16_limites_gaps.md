# Chapitre 16 — Limites actuelles et gaps identifiés

## Le but du chapitre

Faire l'inventaire **honnête** de ce que le projet SASS King ne couvre pas (encore). Identifier les chantiers ouverts, les incertitudes, les questions non résolues. Si tu lis ce guide pour apprendre le SASS SM120, tu dois savoir où sont les trous pour ne pas tomber dedans.

## Le principe

Ce chapitre ne fait pas d'excuse. Il constate. Un projet de reverse engineering d'un ISA propriétaire est **par nature incomplet**. La valeur du projet est dans ce qui est couvert **et** dans l'explicitation de ce qui ne l'est pas.

## Gaps par catégorie

### A. Control flow

#### GAP-audit-1 : Mapping C++ loop → SASS
Une boucle `for (...)` en C++ peut devenir, selon ptxas :
* Une back-edge BRA (cas canonique).
* Un full unroll (si trip count est constant).
* Un unroll partiel + boucle réduite.
* Une structure complètement réorganisée si le compilo fait du loop fusion.

Le **mapping exact** n'est pas étudié. Dans le cas FP4 attention (chap audit), une `for` avec trip count dynamique a produit 0 back-edge BRA, ce qui suggère soit unroll dynamique soit inlining agressif. Non résolu.

#### GAP-audit-2 : Détection de boucle sans back-edge
Si ptxas unrolle complètement, il n'y a plus de boucle au niveau SASS. Comment reconstruire l'intention source ? Besoin de patterns de détection (séquences d'instructions avec offsets croissants réguliers).

#### GAP-audit-3 : Divergence et prédication
Le projet n'a pas encore de chapitre dédié aux patterns de divergence. BSSY/BSYNC, @P instructions, prédication vs branche — tout ça est mentionné mais pas systématiquement étudié. Le chapitre 20 (planifié) est supposé couvrir ces aspects.

### B. Thread/warp scope

#### GAP-audit-4 : Thread vs warp scope
Certaines instructions sont **par-thread** (FFMA, LDG), d'autres **par-warp** (MMA, SHFL, BAR.SYNC), d'autres **par-block** (BAR.SYNC avec id). Le projet n'a pas de documentation systématique de chaque instruction avec son scope.

### C. Encodage SASS

#### GAP-17-1 : Interaction width × trans dans LDSM
Les deux bits width (8-9) et le bit trans (14) sont identifiés, mais leur combinaison dans certains cas n'est pas testée. Peut-il y avoir un conflit ou une restriction ?

#### GAP-control-code-1 : Structure bit-par-bit du control code
La structure générale est comprise (wait mask, stall, scoreboard, yield, reuse). Mais la répartition exacte de chaque bit n'est pas confirmée pour tous les cas. Certains bits restent ambigus.

#### GAP-18-1 : Fonction des `@!PT LDS RZ, [RZ]` avant LDGSTS
Pattern observé systématiquement dans les kernels pipelined (3 instances avant chaque LDGSTS). Hypothèses multiples non testées :
* Alignment binaire (64-byte frontière).
* Sync forcé du scheduler.
* Workaround hardware.

### D. Tensor cores

#### GAP-FP4-1 : OMMA .2X vs .4X mesurée
Le chapitre 16 documente OMMA `.4X` (scale_vec::4X). Les mesures comparatives avec `.2X` ne sont pas encore faites. Différence de throughput attendue : négligeable, mais non confirmée.

#### GAP-FP4-2 : Kind::mxf4nvf4 vs kind::mxf8f6f4 avec FP4
Quand tu compiles `mma.sync kind::mxf8f6f4` avec inputs FP4, est-ce que ptxas génère OMMA ou QMMA ? Non vérifié. Hypothèse : QMMA (le path non-peak), car kind::mxf8f6f4 est associé à QMMA.

#### GAP-audit-5 : Template specialization effects
Dans le kernel FP4 attention avec HEAD_DIM=64 vs HEAD_DIM=128, on voit différents nombres de QMMAs (2 vs 4). Le mécanisme n'est pas compris. Est-ce une optimisation ptxas ? Un effet du template ? Non résolu.

### E. Memory

#### GAP-memory-1 : Effets de LTC hints
Les hints `.LTC128B`, `.LTC256B`, etc. influencent le comportement L2. Mais leur effet mesuré précis n'est pas documenté. Gain de perf réel vs placebo ? Non testé.

#### GAP-memory-2 : Coalescing pathologique
Les patterns de load non-coalescé produisent combien de transactions exactement ? Non mesuré systématiquement. L'audit NCU nécessite des métriques spécifiques.

### F. SM120 specifics

#### GAP-sm120a-1 : Différences concrètes sm_120 vs sm_120a
On sait que sm_120a est le profil "architecture-specific" qui débloque certaines instructions. Mais **quelles instructions exactement** sont débloquées ? Liste non exhaustive.

#### GAP-sm120-tmem : TMEM sur consumer
Est-ce que SM120 a vraiment **pas** de TMEM ? Ou juste pas exposé ? Documenté comme "hors-scope" mais pas vérifié expérimentalement.

### G. Audit méthodologique

#### GAP-audit-6 : Framework d'audit
Le chapitre 13 (cas d'étude) présente un workflow, mais il n'est pas formalisé. Besoin d'un "audit checklist" complet pour standardiser.

#### GAP-audit-7 : Qualification de confiance
Quand tu dis "je pense que ce kernel est memory-bound", quel est ton niveau de confiance ? Pas de framework pour qualifier. Besoin d'échelle (e.g., [high-confidence], [medium-confidence], [speculative]).

## Limites par architecture

Le projet couvre **uniquement SM120**. N'offre **pas** :

* Couverture SM80 (Ampere) : pas de docs sur `wmma` legacy, pas de pattern async memory barrière.
* Couverture SM89 (Ada) : absence totale.
* Couverture SM90/SM90a (Hopper) : pas de `wgmma`, pas de TMA, pas de clusters.
* Couverture SM100/SM100a (Blackwell datacenter) : pas de `tcgen05`, pas de TMEM, pas de cta_group::2.

Contributeurs avec du hardware autre que RTX 50 series sont bienvenus pour commencer à couvrir ces architectures. Le framework est là, manque les hands-on.

## Limites par sujet

### Sujets couverts mais superficiels

* **Memory hierarchy** : les chapitres couvrent LDG/LDS/LDC/LDGSTS mais pas en profondeur les interactions avec les caches.
* **Control codes** : la structure générale est comprise mais certains bits restent non décodés.
* **Tensor cores** : HMMA/QMMA/OMMA couverts pour les cas canoniques. Les edge cases (strange shapes, dtypes inhabituels) pas testés.

### Sujets non couverts

* **Atomics** : RED, ATOM. Pas étudiés.
* **Texture loads** : TLD, TEX. Pas étudiés (rares en compute moderne).
* **Integer arithmetic detaillée** : IMAD.WIDE, IMUL.WIDE, leurs cas spéciaux.
* **Floating point spéciaux** : FSETP modes (FTZ, DAZ), denormals, subnormals.
* **Warp-level synchronization detaillée** : BAR.SYNC avec ID, SYNCS.
* **Indirect function calls** : pas vu, pas étudié.

## Pièges méthodologiques encore présents

### Benchmarking bruité

Les mesures `%clock` peuvent varier selon :
* Autres kernels sur le SM.
* State caches.
* Clock throttling.
* Variations de scheduling.

Sur-interpréter des différences de quelques cycles = erreur. Pas d'outil automatisé pour détecter les aberrations.

### Code non représentatif

Les microbenchmarks sont par nature artificiels. Un kernel production avec 10 000 instructions se comporte différemment d'un kernel avec 100. Les observations faites en microbenchmark peuvent ne pas se généraliser.

### Toolkit drift

CUDA Toolkit évolue. Ptxas 12.6 et 12.9 peuvent générer du SASS différent pour le même code. Nos observations sont datées. Il faut mettre à jour quand nouveau toolkit majeur.

## Ambiguités dans la documentation existante

Même au sein du projet, certains chapitres contiennent des formulations ambiguës ou des claims à revisiter :

* Chapitre 3 (early) : certains claims sur le prologue "architectural tax" pourraient être reformulés plus précisément.
* Chapitre 16a-d : FP4 peak throughput annoncé par NVIDIA à 900+ TFLOPS. Nos mesures en chain serial : ~254 TFLOPS. L'écart n'est pas complètement expliqué (pipelining et occupancy nécessaires).
* Plusieurs [HYP] n'ont pas été updates en [RES] même quand des observations supplémentaires auraient permis.

Un gros cleanup pass sur la cohérence des tags serait bénéfique.

## Outils manquants

* **Assembleur SM120** : impossible de modifier le SASS directement.
* **Diff visuel** : comparer deux SASS demande encore des efforts manuels.
* **Annotation automatique** : mapping C++ → SASS, aujourd'hui ad-hoc.
* **Ecosystème** : autres projets de rev-eng blackbox sur SM120 ? Pas vraiment.

## Roadmap (priorités court terme)

1. **Chapitre 20 (control flow)** : combler GAP-audit-1, 2, 3, 4.
2. **Chapitre 21 (OMMA comparisons)** : combler GAP-FP4-1, 2.
3. **Chapitre 22 (divergence patterns)** : approfondir GAP-audit-3.
4. **Review cohérence tags** : reviser les chapitres antérieurs pour cohérence [OBS]/[INF]/[HYP].
5. **Tooling** : scripts reusables publiés dans /tools/ du repo.

## Roadmap (priorités long terme)

* Ouvrir le projet à **contributeurs externes** avec autre hardware (H100, B200, RTX 4090).
* Publier des **blog posts** résumant les findings clés (pour la visibilité dans la communauté).
* Construire un **outil d'annotation SASS interactif** (HTML ou web).
* Éventuellement un **benchmark suite** reproductible avec résultats agrégés.

## Conclusion honnête

Ce guide et le projet SASS King sont **un début**, pas une fin. Il couvre les fondamentaux de SM120 suffisamment pour auditer 70-80% des kernels que tu rencontreras en pratique. Pour les 20-30% restants, tu vas rencontrer des patterns non documentés, des opcodes inattendus, des comportements mystérieux.

Deux attitudes possibles :
1. **Passif** : "Ah, pas documenté. Je ne sais pas." Tu restes dépendant des outils.
2. **Actif** : "Pas documenté. Je teste via controlled variation. Je documente. Je contribue."

Le projet existe pour permettre l'attitude 2. Tu peux ajouter un chapitre à n'importe quel moment. Tu peux formuler une hypothèse et la tester. Tu peux pointer un gap mal documenté et le combler.

L'ISA SASS SM120 est un territoire **partiellement cartographié**. La carte s'améliore avec chaque contributeur. Si ce guide t'aide à lire le SASS, considère rendre la faveur en ajoutant quelque chose.

## Fin du guide

Tu as maintenant :
* **Partie 1** (ch 1-2) : le pourquoi, la chaîne de compilation.
* **Partie 2** (ch 3-5) : l'anatomie d'une instruction, les registres, les control codes.
* **Partie 3** (ch 6-9) : mémoire, tensor cores, cp.async, LDSM.
* **Partie 4** (ch 10-12) : lire un dump, reconnaître les patterns, diagnostiquer.
* **Partie 5** (ch 13-16) : un cas d'étude complet, outillage, méthodologie, limites.

À ça s'ajoute le **glossaire** (~80 termes) comme référence et les **chapitres techniques numérotés** du repo (FINDINGS.md et les chapitres de reverse engineering).

Tu es équipé. À toi d'ouvrir les dumps.
