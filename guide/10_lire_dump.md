# Chapitre 10 — Lire un dump SASS de A à Z

## Le but du chapitre

Tu reçois un dump SASS d'un kernel inconnu. Comment t'y prendre ? Ce chapitre te donne **une méthodologie systématique** : identifier les sections, naviguer efficacement, et extraire les premières observations utiles. À la fin, tu peux ouvrir n'importe quel SASS et répondre en 5 minutes aux questions de base : "qu'est-ce qu'il fait, qu'est-ce qu'il utilise, est-ce qu'il a l'air sain ?".

## Étape 0 : préparer le dump

Avant d'analyser, génère le dump proprement.

```bash
# Si tu as un binaire (.cubin, .so, .out)
cuobjdump --dump-sass mon_binaire > mon_kernel.sass

# Si tu as un .cu source
nvcc -arch=sm_120 -cubin -o mon_kernel.cubin mon_kernel.cu
cuobjdump --dump-sass mon_kernel.cubin > mon_kernel.sass

# Pour aussi avoir le PTX (utile pour cross-référencer)
cuobjdump --dump-sass --dump-ptx mon_binaire > mon_kernel_full.sass
```

Le fichier .sass est typiquement 100 KB à 5 MB pour un kernel non trivial. Quelques milliers à dizaines de milliers de lignes. Pas lisible séquentiellement.

## Étape 1 : identifier les kernels

Un binaire peut contenir plusieurs kernels (templates instanciés, kernels distincts). Identifie d'abord la liste :

```bash
grep -n "Function" mon_kernel.sass
```

Sortie typique :
```
21:        Function : _Z19fused_fp4_attentionILi64EEvPKfS1_PfS1_iii
6618:      Function : _Z19fused_fp4_attentionILi128EEvPKfS1_PfS1_iii
14991:     Function : _Z9debug_mmaPKfS0_Pf
```

Tu vois 3 kernels. Le deuxième est `fused_fp4_attention<128>` (template avec HEAD_DIM=128). Tu peux **demangle** les noms pour les rendre lisibles :

```bash
c++filt _Z19fused_fp4_attentionILi64EEvPKfS1_PfS1_iii
# Output: fused_fp4_attention<64>(float const*, float const*, float*, float const*, int, int, int)
```

**Note les ranges de lignes** par kernel. Tu vas analyser un kernel à la fois.

## Étape 2 : extraire les stats globales

Pour le kernel cible (disons HD=64, lignes 21-6618), compte les principaux mnemonics :

```bash
sed -n '21,6618p' mon_kernel.sass | awk '/^.*\/\*[0-9a-f]+\*\//{print $2}' | sort | uniq -c | sort -rn | head -20
```

Sortie typique :
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
* Beaucoup d'IMAD et IADD = beaucoup d'arithmétique d'addresses (calculs d'indices).
* Beaucoup de LDS.U8 = quantization fine grain (FP4 ?), beaucoup de byte reads.
* 269 BRA = beaucoup de branches, probablement encoding paths multiples.
* 71 BSSY/BSYNC = beaucoup de divergence reconvergence.
* 38 LDGSTS = pipeline cp.async actif.
* 16 HMMA + 2 QMMA = compute mixte. Peu de QMMA = pas un kernel GEMM intensif.

En 1 minute, tu sais que c'est un **kernel quantization-heavy avec un peu de compute MMA**. Sans avoir lu le source.

## Étape 3 : taille et complexité

Combien d'instructions ?

```bash
sed -n '21,6618p' mon_kernel.sass | grep -cE '/\*[0-9a-f]+\*/'
```

Disons 3296 instructions. Multiplie par 16 bytes = ~52 KB de code.

**Heuristique** :
* < 500 instructions : kernel trivial.
* 500-2000 : kernel "normal" (GEMM tile, attention head simple).
* 2000-5000 : kernel complexe (fused attention, multi-stage pipeline).
* > 5000 : très complexe (peut-être indice de templates explosés ou code redondant).

3296 instructions = complexe mais pas absurde.

## Étape 4 : identifier les sections

Un kernel typique a 3 phases :

1. **Prologue** : setup (load params, calcul indices, init accumulateurs).
2. **Body** : main compute (boucles, MMAs, etc.).
3. **Epilogue** : write results, cleanup.

**Comment les délimiter rapidement** :

* **Début prologue** = ligne après `Function :`.
* **Fin prologue** = la première instruction "structurelle" (BAR.SYNC, début de boucle, premier load du compute).
* **Début epilogue** = avant la dernière rafale de STG.E.
* **Fin epilogue** = `EXIT`.

**Pattern prologue typique** :
```
LDC R1, c[0x0][0x37c] ;        // load stack pointer
LDC R5, c[0x0][0x3a8] ;        // load arg
LDCU UR5, c[0x0][0x3a0] ;      // load uniform arg
S2R R20, SR_TID.X ;            // threadIdx.x
S2UR UR4, SR_CTAID.X ;         // blockIdx.x
... arithmétique sur indices ...
... peut-être quelques BRA forward ...
```

Le prologue fait typiquement 50-300 instructions.

**Pattern epilogue typique** :
```
STG.E desc[UR_out][R_offset.64], R12 ;
STG.E desc[UR_out][R_offset.64+0x4], R13 ;
STG.E desc[UR_out][R_offset.64+0x8], R14 ;
STG.E desc[UR_out][R_offset.64+0xC], R15 ;
EXIT ;
```

Le epilogue fait typiquement 20-100 instructions.

**Le body** est le reste. C'est là que tu vas concentrer ton analyse.

## Étape 5 : identifier les boucles

```bash
# Extraire toutes les BRA, identifier les back-edges
sed -n '21,6618p' mon_kernel.sass | python3 -c "
import sys, re
for line in sys.stdin:
    m_src = re.search(r'/\*([0-9a-f]+)\*/', line)
    m_tgt = re.search(r'BRA 0x([0-9a-f]+)', line)
    if m_src and m_tgt:
        src = int(m_src.group(1), 16)
        tgt = int(m_tgt.group(1), 16)
        if tgt < src:
            print(f'BACK-EDGE: 0x{src:04x} -> 0x{tgt:04x} (body size ~{src-tgt:#x} bytes)')
"
```

Sortie typique :
```
BACK-EDGE: 0x9030 -> 0x5270 (body size ~0x3dc0 bytes)
BACK-EDGE: 0x4680 -> 0x0890 (body size ~0x3df0 bytes)
```

2 boucles principales dans le kernel. Body size 0x3dc0 ≈ 985 instructions. Ce sont les **régions chaudes** à analyser en priorité.

**Note** : absence de back-edge ne signifie pas absence de boucle — ptxas peut entièrement unroller les boucles constantes. Mais pour les boucles dynamiques (trip count en runtime), il y aura toujours une back-edge.

## Étape 6 : identifier les régions MMA

Les MMAs sont la partie compute. Localise-les :

```bash
sed -n '21,6618p' mon_kernel.sass | grep -nE "QMMA|HMMA|OMMA"
```

Sortie typique :
```
4699:  /*9220*/  QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R4, R12, RZ, R18, R31, URZ ;
4725:  /*92f0*/  QMMA.SF.16832.F32.E2M1.E2M1.E8 R12, R8, R28, R12, R19, R34, URZ ;
```

2 QMMAs autour des lignes 4699-4725 (offset 0x9220-0x92f0). C'est la zone de compute.

**Note** : c'est étrangement peu pour un kernel attention. Soit ptxas a été très efficace, soit il y a quelque chose d'inattendu (cas du kernel FP4 attention chap audit, où on a observé seulement 2 QMMAs au lieu des 16 attendues — gap non résolu).

## Étape 7 : identifier le pipeline cp.async

Si le kernel utilise cp.async (pipeline software), tu dois voir :

```bash
sed -n '21,6618p' mon_kernel.sass | grep -nE "LDGSTS|LDGDEPBAR|DEPBAR"
```

Compte les occurrences. Typiquement :
* N LDGSTS par stage × S stages.
* S+1 LDGDEPBAR.
* 1-2 DEPBAR.LE.

Pour notre kernel :
```
38 LDGSTS, 4 LDGDEPBAR, 2 DEPBAR.LE
```

Suggère 3-4 stages de pipeline, ~10 LDGSTS par stage.

## Étape 8 : indicateurs de problèmes

Cherche les signaux pathologiques :

**Spill** :
```bash
grep -c "STL\|LDL" mon_kernel.sass
```

Si > 0, register pressure. Si > 50, sévère.

**Plus haut Rn utilisé** :
```bash
sed -n '21,6618p' mon_kernel.sass | grep -oP 'R\d+' | sort -t'R' -k2 -n -u | tail -5
```

Si tu vois `R254`, kernel utilise ~255 regs, occupancy critique.

**CALL non-inlined** :
```bash
grep -n "CALL.REL" mon_kernel.sass
```

Présence de CALL = function non-inlinée. Si dans une hot loop, problème.

**BAR.SYNC dans une boucle** :
Identifie les BAR.SYNC et vérifie si ils sont dans le range d'une back-edge. Si oui, sync à chaque itération = potentiel bottleneck.

## Étape 9 : focus sur la région chaude

Une fois les boucles et MMAs identifiées, **zoom sur la région body principale** :

```bash
# Extraire la boucle principale (lignes ~5000-6000 dans notre exemple)
sed -n '5270,9030p' mon_kernel.sass > hot_loop.sass

# Analyse focalisée
wc -l hot_loop.sass
grep -c "MMA\|LDGSTS\|LDS\|LDG" hot_loop.sass
```

C'est là que tu passes le plus de temps. C'est là que les optimisations comptent.

## Étape 10 : poser des hypothèses

Après les 9 étapes précédentes, tu as une compréhension de surface. Maintenant, formule des **hypothèses** :

* "Ce kernel a 3 stages de pipeline, c'est cohérent avec un GEMM moderne."
* "Le ratio compute/load semble faible (16 MMAs pour 380 LDG), peut-être memory-bound."
* "Beaucoup de divergence (71 BSSY/BSYNC), cause probable : conditional branches dans encode_fp4."
* "Pas de spill détecté, mais R200 utilisé donc occupancy probablement limitée."

Ensuite, valide ces hypothèses :
* Mesure NCU pour confirmer memory-bound vs compute-bound.
* Vérifie register usage avec `-Xptxas -v`.
* Compare avec un kernel de référence sur le même use case.

## Cheat sheet : commandes Bash utiles

```bash
# Liste des kernels
grep -n "Function" mon_kernel.sass

# Demangle un nom mangled
c++filt _Z19fused_fp4_attention...

# Stats mnemonics
sed -n 'START,ENDp' mon_kernel.sass | awk '/^.*\/\*[0-9a-f]+\*\//{print $2}' | sort | uniq -c | sort -rn

# Compter instructions
sed -n 'START,ENDp' mon_kernel.sass | grep -cE '/\*[0-9a-f]+\*/'

# Distribution des opcodes (par low byte)
grep -oP '/\* 0x[0-9a-f]{16} \*/' mon_kernel.sass | grep -oP '..(\s|$)' | sort | uniq -c | sort -rn

# Identifier MMAs
grep -nE "HMMA|QMMA|OMMA" mon_kernel.sass

# Identifier cp.async
grep -nE "LDGSTS|LDGDEPBAR|DEPBAR" mon_kernel.sass

# Identifier spills
grep -nE "STL|LDL" mon_kernel.sass

# Identifier sync
grep -nE "BAR.SYNC|BSSY|BSYNC" mon_kernel.sass
```

## Bilan

Avec cette méthodologie, tu peux :
* Ouvrir un dump SASS inconnu et te repérer en 5 minutes.
* Identifier les sections, boucles, hot regions.
* Compter les types d'instructions et déduire la nature du kernel.
* Détecter les problèmes courants (spill, occupancy, divergence excessive).
* Formuler des hypothèses à valider via NCU ou autre.

Tu n'as **pas encore** identifié pourquoi tel kernel performe mal — pour ça, le chapitre 12 (diagnostic). Mais tu as la base.

## Ce qui suit

Le chapitre 11 attaque les **patterns communs** : qu'est-ce qu'un GEMM mainloop ressemble en SASS, comment reconnaître un softmax, une reduction, un decode attention. Les "shapes" structurelles que tu rencontres encore et encore.
