# Chapitre 1 — Pourquoi le SASS

## Le problème

Tu as un kernel CUDA. Tu l'as écrit en C++, tu utilises peut-être du PTX inline, tu profiles avec NCU. Tu vois que ton kernel atteint 250 TFLOPS sur un peak théorique de 500. Tu as 50% de throughput perdu. La question est : **pourquoi ?**

NCU te donne 200 métriques. La plupart sont corrélées. La plupart pointent vers des **symptômes** : "stall_lg_throttle", "stall_short_scoreboard", "stall_wait". Ces métriques te disent que le kernel attend, mais elles ne te disent **pas pourquoi**.

Pour répondre au "pourquoi", il faut ouvrir la boîte. C'est là que le SASS rentre.

## Ce que le SASS te donne

Le SASS, c'est **le code que ton GPU exécute réellement**. Pas le code que tu as écrit. Pas le PTX que NVIDIA documente. Le code machine, instruction par instruction, avec :

* La séquence exacte des instructions choisie par ptxas.
* Les registres assignés à chaque variable.
* Les contraintes de scheduling (wait masks, stall counts).
* Les choix d'unrolling, de vectorisation, de prédication.
* Les instructions exactes que les tensor cores reçoivent.

Quand tu lis un dump SASS, tu peux dire : "ici, ptxas a unrollé la boucle K par 2, mais pas la boucle N. La conséquence est que chaque MMA attend un load qui pourrait être prefetched. Voilà le bottleneck."

C'est un niveau de granularité que **ni NCU ni le compilateur ne te donnent**.

## Pourquoi personne ne le fait

Le SASS n'est **pas documenté publiquement** par NVIDIA. Il y a de la doc pour CUDA C++. Il y a de la doc pour PTX. Pour le SASS, il y a `cuobjdump --dump-sass` et c'est tout. Pas de manuel, pas de référence d'opcodes, pas d'explication du format des instructions.

Le dernier travail systématique sur le SASS date de 2018-2019 — Jia et al. (Citadel) sur Volta et Turing. Depuis : rien sur Ampere, Hopper, Blackwell. Pour SM120 (Blackwell consumer) spécifiquement : zéro.

Pourquoi ? Parce que c'est laborieux. Reverse-engineer un ISA non documenté, c'est des milliers d'heures de microbenchmarks, de comparaisons, de corrélations. Et ça change à chaque génération.

## Ce que ça débloque pour toi

Maîtriser le SASS te donne plusieurs capacités que peu de gens ont :

**Diagnostiquer un kernel à 100% précis.** Quand un kernel performe mal, tu peux pointer l'instruction exacte qui stall. Tu sais distinguer un stall scoreboard d'un bank conflict d'un register spill. Tu peux dire : "cette FFMA attend 30 cycles parce que ptxas a alloué R8 et R12 dans le même bank et le reuse cache n'est pas activé."

**Auditer un kernel optimisé écrit par d'autres.** FlashAttention, FlashInfer, Marlin, CUTLASS — tu peux ouvrir leur SASS et comprendre *pourquoi* ils sont rapides. Quels patterns ils utilisent. Quels trade-offs ils ont fait. Et reproduire ces techniques dans tes propres kernels.

**Détecter les régressions du compilateur.** Quand tu mets à jour CUDA Toolkit et que ton kernel devient 10% plus lent sans que tu aies touché au code, tu peux comparer le SASS avant/après et identifier exactement ce que ptxas a changé.

**Comprendre les annonces matérielles.** Quand NVIDIA dit "B200 fait 4.5 PFLOPS FP4 peak", tu peux vérifier dans le SASS si ton kernel utilise effectivement le pipeline matériel concerné (OMMA `kind::mxf4nvf4` sur SM120, ou `tcgen05.mma` sur SM100a), ou s'il utilise un fallback plus lent.

**Être crédible auprès des bonnes personnes.** Les kernel engineers de NVIDIA, les maintainers de CUTLASS, les équipes inference de Together/Fireworks/Meta — ces gens-là parlent SASS au quotidien. Si tu peux discuter avec eux au niveau instruction, tu existes dans leur monde. Sinon tu es un user de plus.

## L'audience

Ce projet n'est pas pour 50 000 followers GitHub. Il est pour **50 à 200 personnes** : les kernel engineers qui codent CUTLASS, qui maintiennent FlashInfer, qui optimisent les decode passes de vLLM, qui font de la performance perf engineering chez Anthropic, OpenAI, Mistral. Ces personnes vivent dans le SASS. Le projet vise à leur donner **la référence publique** qui n'existe pas encore.

## Le scope

Strictement **SM120 (Blackwell consumer, RTX 50 series)**.

Pas SM90a (Hopper). Pas SM100a (Blackwell datacenter). Pas SM89 (Ada). Pas SM80 (Ampere).

Un seul focus, exhaustif. Quand tu t'attaques à un ISA non documenté, la profondeur sur une seule architecture vaut bien plus que la surface sur cinq.

Tu travailles sur RTX 5070 Ti, c'est ce que tu as sous la main, c'est ce que tu peux benchmarker. Les autres archis viendront via contributeurs si elles viennent.

## Ce que ce projet n'est pas

* **Pas un manuel d'introduction à CUDA.** Si tu ne sais pas ce qu'est un warp ou comment lancer un kernel, ce n'est pas le bon endroit pour apprendre. Va lire le CUDA Programming Guide d'abord.

* **Pas un guide d'optimisation général.** Les techniques d'optimisation sont liées au contexte spécifique du kernel. Le projet documente des observations SASS, pas des recettes universelles.

* **Pas un substitut à la doc PTX.** Le PTX est documenté par NVIDIA, lis ça aussi. Le SASS King ajoute une couche au-dessous, il ne remplace pas l'au-dessus.

* **Pas exhaustif (encore).** Beaucoup de gaps connus. Les chapitres documentent ce qui est observé, pas tout ce qui existe. Le projet est itératif.

## Comment ce guide est structuré

Le guide complet contient :

* Un **glossaire** (~80 termes) qui définit chaque concept et instruction. À utiliser comme référence pendant que tu lis les chapitres ou audites du SASS.

* Des **chapitres narratifs** (cette partie) qui expliquent le pourquoi, le comment, et la méthodologie. Lecture séquentielle.

* Des **chapitres techniques numérotés** (`tensor_cores/13_*`, `tensor_cores/14_*`, etc.) qui contiennent le travail empirique : kernels minimaux, dumps SASS annotés, hypothèses validées par microbenchmark. Lecture par sujet.

* Un fichier **FINDINGS.md** qui aggrege toutes les observations, hypothèses, et résolutions cross-chapitre. Référence transversale.

L'ordre conseillé : **chapitre narratif → glossaire pour les termes inconnus → chapitre technique pour creuser**.

## Ce qui suit

Le chapitre 2 explique la chaîne de compilation CUDA C++ → PTX → SASS, et qui fait quels choix à chaque étape. Sans cette compréhension, tu ne peux pas savoir où agir pour influencer le SASS généré.
