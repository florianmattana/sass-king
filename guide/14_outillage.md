# Chapitre 14 — Outillage : dumper, annoter, automatiser

## Le but du chapitre

Connaître les **outils existants** pour analyser du SASS, savoir lesquels utiliser quand, et reconnaître leurs limites. Construire un workflow efficace au lieu de lire 5000 lignes de SASS à la main.

## Outils officiels NVIDIA

### cuobjdump

Le couteau suisse pour dumper du SASS depuis un binaire compilé.

```bash
# Dump SASS d'un binaire (cubin, .so, .out)
cuobjdump --dump-sass mon_binaire

# Dump le PTX intermédiaire (utile pour cross-référencer)
cuobjdump --dump-ptx mon_binaire

# Les deux
cuobjdump --dump-sass --dump-ptx mon_binaire

# Filter par architecture (utile pour fat binaries)
cuobjdump --dump-sass --gpu-architecture=sm_120 mon_binaire

# Lister les sections (sans dump complet)
cuobjdump mon_binaire
```

**Output structuré** : chaque kernel commence par `Function : <mangled_name>`. Le SASS suit avec offsets, mnemonics, opérandes, et opcode bytes.

**Limites** :
* Le format du dump est conçu pour la lecture humaine, pas pour le parsing strict.
* Pas de mapping ligne C++ ↔ SASS direct (sauf si compilé avec `-G`, qui change le SASS).

### nvcc avec options ptxas

Pour obtenir des informations sur ce que ptxas a fait :

```bash
# Verbose : register usage, spills, smem usage
nvcc -arch=sm_120 -Xptxas -v mon_kernel.cu

# Output exemple :
# ptxas info : Compiling entry function '_Z19fused_fp4_attentionILi64EE...' for 'sm_120'
# ptxas info : Function properties for _Z19fused_fp4_attentionILi64EE...
#     N stack, M bytes spill stores, K bytes spill loads
# ptxas info : Used X registers, Y bytes smem, Z bytes cmem[0]

# Warnings sur les spills
nvcc -arch=sm_120 -Xptxas --warn-on-spills mon_kernel.cu

# Force optimization level
nvcc -arch=sm_120 -Xptxas -O3 mon_kernel.cu

# Disable register allocation optimization (pour debug)
nvcc -arch=sm_120 -Xptxas -O0 mon_kernel.cu
```

### Nsight Compute (NCU)

Le profileur officiel. Mesure des dizaines de métriques pendant l'exécution.

```bash
# Profiler simple
ncu mon_binaire

# Set complet de métriques
ncu --set full -o report mon_binaire

# Voir le report
ncu --import report.ncu-rep

# Print summary text
ncu --import report.ncu-rep --print-summary

# Source view (annote chaque ligne C++ avec les metrics)
ncu --set full --section SourceCounters mon_binaire
```

**Métriques essentielles à connaître** :
* `sm__throughput.avg.pct_of_peak_sustained_active` : utilisation SM globale.
* `gpu__time_active.avg` : temps total d'exécution.
* `smsp__warp_issue_stalled_*` : raisons des stalls (dispatch, scoreboard, etc.).
* `l1tex__data_pipe_lsu_wavefronts*` : load coalescing.
* `l1tex__data_bank_conflicts_pipe_lsu_mem_shared*` : bank conflicts shared.
* `sm__pipe_tensor_op_*_cycles_active*` : utilisation tensor cores.

NCU est **l'outil de référence** pour le diagnostic perf. Tu lui demandes un rapport, il te dit où le temps est passé.

### nvdisasm

Disassembler de plus bas niveau, plus brut que cuobjdump :

```bash
nvdisasm --section .text mon_kernel.cubin
```

Utile pour parser ou analyser à un niveau plus mécanique.

## Outils tiers (community)

### Maxas / TuringAS / OpenAS

**Assembleurs SASS open source**, par génération :
* MaxAs : Maxwell.
* TuringAS : Turing.
* OpenAS : effort plus récent.

**Pour Blackwell (SM120) : rien d'officiel ni de community** au moment de l'écriture. Tu ne peux pas écrire du SASS Blackwell directement, juste le lire.

**Conséquence** : si tu veux modifier le SASS, tu dois passer par C++/PTX inline.

### gpuasm.com

Site web qui annote les opcodes SASS avec leur sémantique (basé sur reverse engineering communautaire).

Limite : couverture variable selon les architectures. Pour Blackwell récent, peut être incomplet.

### Citadel papers (Jia et al.)

Référence académique pour Volta/Turing :
* "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" (2018).
* Utile comme méthodologie de référence, mais ne couvre pas SM80+.

### Marlin / Turing source

Le code source de Marlin (W4A16 inference kernels) contient beaucoup de PTX inline et est une mine d'or pour comprendre comment les pros écrivent. Lecture conseillée.

## Scripts maison utiles

Tu vas vite vouloir automatiser. Quelques scripts qui m'ont servi :

### Script 1 : extraire stats d'un kernel

```bash
#!/bin/bash
# stats_kernel.sh : extract stats for a specific kernel from a SASS dump
# Usage: ./stats_kernel.sh mon_kernel.sass START_LINE END_LINE

SASS=$1
START=$2
END=$3

echo "=== Top 20 mnemonics ==="
sed -n "${START},${END}p" "$SASS" | awk '/^.*\/\*[0-9a-f]+\*\//{print $2}' | \
    sort | uniq -c | sort -rn | head -20

echo "=== Stats ==="
echo "Total instructions: $(sed -n "${START},${END}p" "$SASS" | grep -cE '/\*[0-9a-f]+\*/')"
echo "MMAs: $(sed -n "${START},${END}p" "$SASS" | grep -cE 'HMMA|QMMA|OMMA')"
echo "LDGs: $(sed -n "${START},${END}p" "$SASS" | grep -c 'LDG.E')"
echo "LDGSTS: $(sed -n "${START},${END}p" "$SASS" | grep -c 'LDGSTS')"
echo "BSSY pairs: $(sed -n "${START},${END}p" "$SASS" | grep -c 'BSSY')"
echo "Spill (STL/LDL): $(sed -n "${START},${END}p" "$SASS" | grep -cE 'STL|LDL')"

echo "=== Highest registers ==="
sed -n "${START},${END}p" "$SASS" | grep -oP 'R\d+' | sort -t'R' -k2 -n -u | tail -3
```

### Script 2 : identifier les boucles

```python
#!/usr/bin/env python3
# loops.py : identify back-edges (loops) in a SASS dump
import sys, re

filename = sys.argv[1]
start_line = int(sys.argv[2]) if len(sys.argv) > 2 else 1
end_line = int(sys.argv[3]) if len(sys.argv) > 3 else 999999

with open(filename) as f:
    lines = f.readlines()

backs = []
for lineno, line in enumerate(lines[start_line-1:end_line], start=start_line):
    m_src = re.search(r'/\*([0-9a-f]+)\*/', line)
    m_tgt = re.search(r'BRA 0x([0-9a-f]+)', line)
    if m_src and m_tgt:
        src = int(m_src.group(1), 16)
        tgt = int(m_tgt.group(1), 16)
        if tgt < src:
            backs.append((lineno, src, tgt))

print(f"Found {len(backs)} back-edges:")
for lineno, src, tgt in backs:
    body_size = (src - tgt) // 16
    print(f"  Line {lineno}: 0x{src:04x} -> 0x{tgt:04x}  ({body_size} insts in body)")
```

### Script 3 : annoter les wait masks

```python
#!/usr/bin/env python3
# wait_masks.py : extract instructions with non-zero wait masks
import sys, re

filename = sys.argv[1]

with open(filename) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    if '/*' in line and '*/' in line:
        # Look for control code on next line
        if i+1 < len(lines):
            ctrl_line = lines[i+1]
            m_ctrl = re.search(r'0x([0-9a-f]{16})', ctrl_line)
            if m_ctrl:
                ctrl = int(m_ctrl.group(1), 16)
                wait_mask = ctrl & 0xff
                if wait_mask != 0 and wait_mask != 0xff:
                    print(f"Line {i+1}: wait_mask=0x{wait_mask:02x} | {line.strip()[:100]}")
    i += 1
```

### Script 4 : annoter le SASS avec le code source (mapping)

C'est plus complexe. Si tu compiles avec `-lineinfo` (sans changer le SASS), tu peux croiser SASS et source via :
```bash
nvcc -arch=sm_120 -lineinfo -o my_kernel my_kernel.cu
cuobjdump --dump-sass --dump-line-info my_kernel
```

Le dump inclura des annotations de ligne source.

## Workflow type d'un audit

```bash
# 1. Compile et dump
nvcc -arch=sm_120 -O3 -lineinfo -Xptxas -v -o my_kernel my_kernel.cu 2>&1 | tee compile.log
cuobjdump --dump-sass my_kernel > my_kernel.sass

# 2. Stats globales
./stats_kernel.sh my_kernel.sass 1 10000

# 3. Identifier sections
grep -n "Function" my_kernel.sass

# 4. Identifier loops dans le kernel cible
python3 loops.py my_kernel.sass START END

# 5. Profile avec NCU
ncu --set full -o report my_kernel
ncu --import report.ncu-rep --print-summary

# 6. Identifier hot lines (NCU source view)
ncu --set full --section SourceCounters --import report.ncu-rep --print-summary

# 7. Zoom sur la région chaude
sed -n 'HOT_START,HOT_ENDp' my_kernel.sass

# 8. Hypothèse + fix + re-mesure
```

## Outils que je n'ai pas mais qui seraient utiles

* **Annotateur SASS interactif** (HTML/web) : vue scrollable avec coloration par pipeline, dépendances tracées entre instructions, mapping vers source. C'est sur la roadmap du projet SASS King mais pas encore implémenté.

* **Diff SASS visuel** : comparer deux dumps (avant/après optim) en mettant en évidence les différences. Possible avec `diff` mais peu lisible.

* **Simulator de scoreboard** : prendre un dump, simuler les latences, prédire les stalls. Existe pour certaines architectures académiquement, pas pour Blackwell.

## Ce que tu peux construire toi-même

Si tu veux contribuer à l'écosystème :

* **Parser SASS robuste** : la plupart des scripts ad-hoc cassent sur des cas particuliers. Un parser proprement écrit serait précieux.
* **Base de données d'opcodes** : un dictionnaire SM120 opcode → mnemonic → sémantique. Aujourd'hui dispersé entre nos chapitres.
* **Visualisation** : l'idée du HTML annoté est excellente, juste pas faite.
* **Diff intelligent** : compare deux SASS en alignant par instruction, pas par ligne.

Tous ces outils manquent à la communauté.

## Limites des outils existants

* **Pas de standard** : chacun écrit ses scripts ad-hoc.
* **Pas de doc opcodes** : tu re-discovers à chaque fois.
* **Pas d'assembleur SM120** : tu ne peux pas modifier le SASS.
* **NCU est lourd** : besoin de root parfois, setup complexe.

C'est dans ces limites que le projet SASS King essaye de progresser.

## Ce qui suit

Le chapitre 15 attaque la **méthodologie controlled variation** : comment construire de la connaissance fiable sur le SASS via des microbenchmarks bien conçus. C'est la méthodologie utilisée pour tous les chapitres techniques du projet.
