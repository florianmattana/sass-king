# Chapitre 13 — Cas d'étude : audit guidé d'un kernel

## Le but du chapitre

Mettre en pratique tout ce qui précède sur un cas réel. On prend un kernel, on suit la méthodologie complète, on tire des conclusions honnêtes (y compris les limites). À la fin, tu as un template mental pour faire la même chose sur n'importe quel autre kernel.

Ce chapitre s'inspire d'une vraie tentative d'audit (chap audit du repo), avec ses succès et ses échecs.

## Le kernel cible

`fused_fp4_attention<HEAD_DIM>` du repo fp4-fused-attention-sm120.

**Description du kernel** (ce que tu sais avant d'auditer) :
* Attention fused : QK^T → softmax → ×V en une passe.
* Quantization à la volée : Q, K en FP4 E2M1 avec scale UE8M0.
* V reste en FP32, accumulation en FP32.
* Template HEAD_DIM=64 et 128.
* Tile Q : BQ=64. K_TILES = HEAD_DIM/32. N_TILES = BQ/8 = 8.

**Ton hypothèse de départ** : kernel attention production-quality, devrait avoir un pattern GEMM mainloop pour QK^T, suivi softmax online, suivi PV accumulation.

## Étape 0 : préparer

```bash
cd ~/fp4-fused-attention-sm120
make test_attention                                    # rebuild
cuobjdump --dump-sass build/test_attention > /tmp/test_attention.sass
wc -l /tmp/test_attention.sass
# 16355 lignes
```

## Étape 1 : identifier les kernels

```bash
grep -n "Function" /tmp/test_attention.sass
```

Sortie :
```
21:    Function : _Z19fused_fp4_attentionILi64EEvPKfS1_PfS1_iii
6618:  Function : _Z19fused_fp4_attentionILi128EEvPKfS1_PfS1_iii
14991: Function : _Z9debug_mmaPKfS0_Pf
```

3 kernels :
* `fused_fp4_attention<64>` : lignes 21-6617 (~6600 lignes).
* `fused_fp4_attention<128>` : lignes 6618-14990 (~8400 lignes).
* `debug_mma` : lignes 14991-fin (~1300 lignes).

On focus sur HEAD_DIM=64 d'abord (plus simple).

## Étape 2 : stats globales du kernel HD=64

```bash
sed -n '21,6617p' /tmp/test_attention.sass | awk '/^.*\/\*[0-9a-f]+\*\//{print $2}' | sort | uniq -c | sort -rn | head -20
```

Sortie (extrait):
```
   1245 IMAD
    893 LDS.U8
    586 IADD
    412 LDC
    380 LDG.E
    269 BRA
    156 STG.E
     71 BSSY.RECONVERGENT
     71 BSYNC.RECONVERGENT
     38 LDGSTS.E.LTC128B.128
     16 HMMA.16816.F32
      4 BAR.SYNC.DEFER_BLOCKING
      2 QMMA.SF.16832.F32.E2M1.E2M1.E8
      1 EXIT
```

**Lecture immédiate** :
* Beaucoup d'IMAD/IADD (1800+) : massive arithmétique d'addresses.
* 893 LDS.U8 : énormément de byte loads. Cohérent avec quantization FP4.
* 269 BRA : beaucoup de branches.
* 71 BSSY/BSYNC : beaucoup de divergence.
* **2 QMMAs seulement** : c'est très peu.
* 16 HMMA : encore moins courant.

**Observation surprenante** : on s'attendait à beaucoup de QMMAs (path FP4 production). On en trouve 2. C'est anormalement bas.

## Étape 3 : identifier les sections

**Total instructions** :
```bash
sed -n '21,6617p' /tmp/test_attention.sass | grep -cE '/\*[0-9a-f]+\*/'
# 3296 instructions
```

3296 instructions, ~52 KB de code. Kernel complexe.

**Identifier les boucles** :
```bash
sed -n '21,6617p' /tmp/test_attention.sass | python3 -c "
import sys, re
for line in sys.stdin:
    m_src = re.search(r'/\*([0-9a-f]+)\*/', line)
    m_tgt = re.search(r'BRA 0x([0-9a-f]+)', line)
    if m_src and m_tgt:
        src = int(m_src.group(1), 16)
        tgt = int(m_tgt.group(1), 16)
        if tgt < src:
            print(f'BACK-EDGE: 0x{src:04x} -> 0x{tgt:04x} ({(src-tgt)//16} insts in body)')
"
```

Sortie :
```
(rien)
```

**0 back-edge BRA**. Le kernel n'a pas de boucle SASS classique.

**C'est très inattendu**. Le code source contient `for (int seq_tile = 0; seq_tile < num_seq_tiles; seq_tile++)` avec `num_seq_tiles` dynamique (calculé en runtime depuis `seq_k`). Cette boucle **devrait** produire une back-edge.

## Étape 4 : enquête sur les MMAs

**Localiser les 2 QMMAs** :
```bash
sed -n '21,6617p' /tmp/test_attention.sass | grep -nE "QMMA|HMMA|OMMA"
```

Sortie :
```
4699:  /*9220*/  QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R4, R12, RZ, R18, R31, URZ ;
4725:  /*92f0*/  QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R8, R28, R12, R19, R34, URZ ;
```

2 QMMAs aux offsets 0x9220 et 0x92f0. Distance = 13 instructions.

**Pattern** : 
* 1ère QMMA : C = RZ → premier MMA d'une chain (pas d'accumulation précédente).
* 2ème QMMA : C = R12 → chain (accumulate).

Donc une chain de **2 MMAs** seulement.

**Si le code source dit N_TILES=8 × K_TILES=2 = 16 MMAs attendues en full unroll**, on devrait voir 16. On en voit 2.

**Hypothèses possibles** :
1. Binaire stale (compilé avec ancienne version du source qui avait N_TILES=1 ?).
2. ptxas a fait une optimisation non documentée.
3. Notre compréhension de l'unroll est incorrecte.

## Étape 5 : tenter de confirmer le pattern attendu

Cherchons d'autres signaux pour confirmer combien d'iterations :

**Compter les FMUL × 0.125** (softmax_scale = 1/sqrt(64) = 0.125) :
```bash
sed -n '21,6617p' /tmp/test_attention.sass | grep -c "FMUL.*0.125"
```

Sortie : `4`

Le code source fait `acc[i] *= softmax_scale` × 4 components × 8 n_tiles = 32 attendu.
On en voit **4**.

**Compter les MUFU.EX2** (expf) :
```bash
sed -n '21,6617p' /tmp/test_attention.sass | grep -c "MUFU.EX2"
```

Sortie : `8`

Le code fait expf 4 fois par n_tile × 8 n_tiles = 32 attendu.
On en voit **8**.

**Conclusion** : le ratio observé/attendu est ~1/4 partout. Soit le kernel exécute 2 n_tiles unrollés (au lieu de 8), soit il y a une boucle cachée qu'on n'a pas vue, soit le binaire est stale.

## Étape 6 : prendre du recul

À ce stade, on a deux choix :

**Option A** : Continuer à creuser jusqu'à comprendre.
**Option B** : Reconnaître que le toolkit n'est pas prêt et documenter les gaps.

C'est l'option B qui a été choisie dans le projet réel. Pourquoi ?

* On peut **plaquer une narrative propre** sur ces 2 QMMAs (par exemple "ptxas a fait une optimisation X qui dédupe les MMAs"). Mais sans confirmation expérimentale, c'est de la fiction.
* Les outils dont on dispose (chapitres 1-18) ne couvrent pas le cas "boucle dynamique sans back-edge BRA".
* Le binaire pourrait être stale (date du 13 avril, version source actuelle).

**Décision honnête** : on stoppe l'audit, on documente ce qu'on a observé, et on identifie les gaps.

## Étape 7 : documenter les gaps

7 gaps identifiés (cf chap audit gaps + plan chapitre 20) :

1. C++ loop → SASS mapping pas compris.
2. Loop detection sans back-edge BRA pas documenté.
3. Patterns de divergence et prédication pas couverts.
4. Thread vs warp scope pas formalisé.
5. Effets de template specialization pas étudiés.
6. Pas de méthodologie d'audit documentée.
7. Pas de framework de qualification de confiance.

Plan : chapitre 20 (control flow) doit combler les gaps 1, 2, 4. Chapitres 21+ pour les autres.

## Ce qu'on a quand même appris

Même un audit "raté" produit des observations utiles :

**Observation [OBS-fp4-1]** : Les MMAs utilisées sont **QMMA.SF.16832.F32.E2M1.E2M1.E8** (kind::mxf8f6f4 scale_vec::1X), pas le path peak `kind::mxf4nvf4` (OMMA `.4X`). 

Si l'objectif du kernel est de maximiser le throughput FP4, **migration vers OMMA serait un gain** (~2× theoretical).

**Observation [OBS-fp4-2]** : Quantization très lourde (893 LDS.U8, 1800+ IMAD). Cela suggère que la phase quantization domine le kernel. Pour un kernel attention production, on aimerait que le compute MMA domine.

**Observation [OBS-fp4-3]** : Beaucoup de divergence (71 BSSY/BSYNC pairs). Cohérent avec l'encoding FP4 cascade (8 niveaux). Optimisation possible : remplacer la cascade par une lookup table.

Ces 3 observations sont **actionnables** et peuvent guider une optimisation. Sans avoir compris l'intégralité du kernel.

## La leçon méta

**Un audit honnête vaut mieux qu'un audit prétendu complet.**

Quand tu rencontres des observations que tu ne peux pas expliquer, deux options :
1. Inventer une narrative qui semble cohérente.
2. Documenter le gap et stopper.

L'option 1 te fait paraître competent. L'option 2 te fait progresser réellement. Et si tu publies ton audit, l'option 2 est la seule honnête.

## Workflow récapitulatif

Pour tout kernel à auditer :

1. **Préparer** : dump SASS propre, identifier les kernels.
2. **Stats globales** : count mnemonics, identifier le caractère du kernel.
3. **Sections** : prologue / body / epilogue. Boucles via back-edges.
4. **MMAs** : localiser, compter, identifier la famille.
5. **Pipeline cp.async** : LDGSTS / LDGDEPBAR / DEPBAR.LE.
6. **Indicateurs problèmes** : spill, occupancy, divergence.
7. **Hypothèses** : formuler ce que tu penses, avec niveau de confiance.
8. **Validation** : NCU, recompile, comparison.
9. **Itérer** ou **documenter les gaps**.

## Ce qui suit

Le chapitre 14 attaque l'**outillage** : comment dumper, annoter, automatiser ton workflow d'audit. Quels scripts utiliser, quels outils existent, quelles limites.
