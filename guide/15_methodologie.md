# Chapitre 15 — Méthodologie : controlled variation

## Le but du chapitre

Expliquer la **méthodologie de reverse engineering** utilisée dans tout le projet SASS King : controlled variation. Comment construire de la connaissance **fiable** sur un ISA non documenté, via des microbenchmarks bien conçus. Comment distinguer observation / inférence / hypothèse. À la fin, tu peux contribuer au projet ou lancer ton propre effort de rev-eng sur un aspect non couvert.

## Le problème du reverse engineering SASS

Tu veux comprendre quelque chose sur SM120. Par exemple :
* "Combien de cycles une HMMA prend en chain ?"
* "Quelle est la latency d'un LDG hit L1 ?"
* "Le modifier `.reuse` a-t-il un effet mesurable ?"
* "L'instruction X a-t-elle un side effect Y ?"

NVIDIA ne publie pas ces réponses. Tu dois les découvrir empiriquement.

**Tentation naïve** : écrire un kernel qui utilise X, mesurer, conclure.

**Problème** : un kernel réel contient des centaines d'instructions. La mesure est la somme de tout. Tu ne peux pas isoler X.

**Solution** : **controlled variation**.

## Le principe

1. Écris deux kernels, A et B, qui sont **identiques sauf sur une seule variable**.
2. Mesure les deux.
3. La différence de mesure est attribuable à la variable changée.

**Exemple** : pour mesurer la latency chain d'HMMA :
* Kernel A : 100 HMMAs en chain.
* Kernel B : 200 HMMAs en chain.
* Mesure cycles_A et cycles_B.
* `latency_per_mma = (cycles_B - cycles_A) / 100`.

En divisant par l'écart d'instructions, tu extrais la latency marginale, pas affectée par le prologue/épilogue commun.

## L'architecture d'un microbenchmark

Un bon microbenchmark SASS suit ce template :

```cuda
__global__ void bench(uint32_t* out, int N) {
    uint32_t start, end;
    
    // Warm-up (ensure data in cache, avoid cold start)
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start));
    
    // === SECTION À MESURER ===
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...}, {...}, {...}, {...};"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...}, {...}, {...}, {...};"
        // ... N fois ...
    );
    // ========================
    
    asm volatile("mov.u32 %0, %%clock;" : "=r"(end));
    
    if (threadIdx.x == 0) {
        out[blockIdx.x] = end - start;
    }
}
```

**Principes** :

* **Timing avec `%clock`** : lecture du compteur de cycles SM. Précis au cycle.
* **Volatile** : force le compilateur à respecter l'ordre.
* **Éviter les dépendances extérieures** : pas de loads global pendant la section mesurée, pas de branches conditionnelles (évite la divergence).
* **Écriture en fin** : que le résultat soit stocké après la section, pas pendant.

## Exemple concret : mesurer HMMA chain latency

**Objectif** : combien de cycles une HMMA F32 m16n8k16 prend en chain sur SM120 ?

**Kernel A** (100 HMMAs en chain) :
```cuda
__global__ void bench_hmma_100(uint32_t* out) {
    float C[4] = {0, 0, 0, 0};
    // ... setup A, B fragments ...
    
    uint32_t start;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start));
    
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
                     : "r"(A_lo), "r"(A_hi), "r"(B));
    }
    
    uint32_t end;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(end));
    
    out[blockIdx.x] = end - start;
}
```

**Kernel B** : même chose mais boucle sur 200.

**Mesure** :
```bash
nvcc -arch=sm_120 -O3 bench.cu -o bench
./bench       # run plusieurs fois, prendre la médiane
```

Disons : kernel A = 3 550 cycles, kernel B = 7 050 cycles.

**Conclusion** : `(7050 - 3550) / 100 = 35 cycles par HMMA en chain`.

**Validation** : cette mesure doit être stable à travers plusieurs runs. Si tu as 35 ± 2 cycles, c'est fiable. Si tu as 35 ± 20 cycles, il y a du bruit et il faut investiguer.

## Le tagging obligatoire : [OBS] / [INF] / [HYP] / [RES]

Le projet SASS King impose un **tagging strict** sur toutes les affirmations pour distinguer la solidité.

### [OBS] — Observation

Un fait mesuré ou observé directement dans le SASS.

```markdown
[OBS-chap16d-1] : En chain serial de 128 OMMAs, chaque OMMA dispatche tous les
~29 cycles mesurés via `%clock` (moyenne sur 10 runs, écart-type 0.8 cycles).
```

**Caractéristiques** :
* Reproductible.
* Exact (pas d'extrapolation).
* Le lecteur peut vérifier en relançant le benchmark.

### [INF] — Inférence

Une déduction basée sur plusieurs observations. Pas directement mesuré, mais logiquement nécessaire.

```markdown
[INF-chap16d-1] : L'opcode low byte 0x7f identifie la famille OMMA, pas une
instance spécifique. Basé sur [OBS-chap16d-1] (tous les OMMA observés ont 0x7f)
et [OBS-chap16d-2] (variantes de shape encodées dans d'autres bits).
```

**Caractéristiques** :
* Logique mais pas directement prouvée.
* Autres explications possibles.
* Le lecteur doit faire confiance à la chaîne logique.

### [HYP] — Hypothèse

Une supposition basée sur du raisonnement mais sans confirmation empirique.

```markdown
[HYP-chap18-1] : Les 3 instructions `@!PT LDS RZ, [RZ]` avant chaque LDGSTS
servent au hardware à aligner le dispatch sur une frontière 64-byte. Hypothèse
motivée par le pattern 4 instructions × 16B = 64B. Non confirmée
expérimentalement.
```

**Caractéristiques** :
* Non prouvée.
* Explicitement marquée comme incertaine.
* Invitation à future validation.

### [RES] — Résolu

Une hypothèse qui a été testée et confirmée ou réfutée.

```markdown
[RES-chap17] : [HYP-chap17-3] "le LDSM width est encodé bits 8-9" est confirmé
par microbenchmark isolé (kernel 3 variantes avec widths 1/2/4, opcode bytes
diffèrent uniquement aux positions 8-9).
```

**Caractéristiques** :
* Hypothèse marquée comme résolue.
* Soit confirmée (devient [OBS] ou [INF] solide), soit réfutée.

### [GAP] — Gap

Une question ouverte, identifiée mais non résolue.

```markdown
[GAP-18-1] : La fonction exacte des 3 `@!PT LDS RZ, [RZ]` avant LDGSTS reste
inconnue. Plusieurs hypothèses plausibles non testées (alignment, sync, workaround).
```

**Caractéristiques** :
* Admission honnête d'ignorance.
* Reste documentée pour travail futur.

## Les pièges courants

### Piège 1 : Mesurer une seule fois

Les GPUs ont de la variabilité (thermal throttling, autres process, etc.). **Toujours** faire plusieurs runs et regarder médiane + écart-type.

### Piège 2 : Pas de warm-up

Le premier kernel est souvent plus lent (JIT compile, cold caches). Lance 1 fois au warm-up, mesure les suivants.

### Piège 3 : Optimisations agressives

`-O3` peut supprimer ta boucle si elle n'a pas d'effet visible. Utilise `asm volatile` et écris le résultat pour forcer le compilo à tout garder.

### Piège 4 : Dépendances cachées

Si tu changes X et Y bouge aussi, tu ne peux pas isoler X. Vérifie le SASS des deux versions pour confirmer que seul ce qui t'intéresse a changé.

### Piège 5 : Sur-interpréter

Tu mesures 35 cycles au lieu de 36 attendu. Ne dis pas "c'est significatif, il y a X". Dis : "dans la marge d'erreur". Sois prudent.

### Piège 6 : Fabuler une narrative

Tu observes un pattern bizarre (ex : 2 QMMAs au lieu de 16 attendues). Tentation : inventer une explication plausible. **Ne le fais pas**. Documente comme gap, continue.

## Workflow type pour un chapitre du projet

Quand tu ajoutes un chapitre de reverse engineering :

1. **Hypothèse initiale** : "Je pense que X fait Y sur SM120 parce que Z."

2. **Design du microbenchmark** : quel kernel minimal permet de tester X ? Quelles variantes (A/B/C) ?

3. **Implémentation** : C++ avec PTX inline, compile pour sm_120, dump SASS pour vérifier que le SASS produit correspond à l'intention.

4. **Mesure** : compile, run, collect N runs, calcule moyenne/médiane/écart-type.

5. **Analyse** :
   - Si cohérent avec l'hypothèse → confirmer ([OBS]).
   - Si incohérent → investiguer ou documenter ([GAP]).
   - Si partiellement cohérent → formuler [INF] et [HYP].

6. **Rédaction** :
   - Présenter les kernels (quoi, pourquoi).
   - Présenter les mesures (chiffres bruts).
   - Tagger chaque affirmation.
   - Lister les limites.

7. **Review interne** : relecture à J+1 avec œil critique. Est-ce que tu pourrais défendre chaque [OBS] devant un reviewer hostile ? Si non, downgrade en [HYP] ou [GAP].

## L'importance de la reproductibilité

Tous les kernels de benchmark doivent être **dans le repo**, compilables, runnable. Le lecteur doit pouvoir :
```bash
git clone ...
cd sass-king/tensor_cores/17_ldsm/
make
./run
```

Et obtenir des mesures similaires. Si quelqu'un arrive à des chiffres différents, c'est un signal (hardware différent, toolkit différent, bug dans le benchmark).

## Limites de la méthodologie

Certaines questions restent **difficiles** à attaquer par controlled variation :

* **Topologie du register file** : les bank conflicts dépendent de l'état interne, pas directement mesurable.
* **Behavior side effects** : si l'instruction X modifie un état caché consommé par Y plus tard, difficile à isoler.
* **Interactions multi-warp** : le scheduler alloue différemment selon d'autres warps. Difficile de contrôler.

Pour ces cas, d'autres méthodologies sont nécessaires :
* **Fuzzing** : générer des milliers de patterns random, chercher les anomalies. Utilisé par kuterdinel.com pour SM90a.
* **Power analysis** : si tu mesures la consommation, tu peux détecter quels circuits sont activés.
* **Side channel** : technique plus avancée.

SASS King se concentre sur controlled variation car c'est **le plus accessible** et donne des résultats actionables rapidement.

## Quand passer à autre chose

Tu peux passer des jours sur une question sans arriver à la résoudre. Règle empirique :

* Si après 2-3 variants de benchmark tu n'as pas convergé vers une explication stable, **documente comme [GAP]** et passe à autre chose.
* Les gaps s'accumulent et peuvent être attaqués plus tard avec nouvelles idées.
* Pas de honte à avoir des gaps — c'est le projet, pas un échec personnel.

## Bilan

La méthodologie controlled variation permet de construire de la connaissance fiable sur un ISA non documenté :

* **Isole une variable** à la fois.
* **Mesure précisément** avec `%clock`.
* **Tag rigoureusement** : [OBS] / [INF] / [HYP] / [RES] / [GAP].
* **Reproductible** : le code des benchmarks fait partie du chapitre.
* **Honnête sur les limites** : les gaps sont documentés, pas cachés.

C'est lente. C'est rigoureuse. C'est la seule façon de produire quelque chose d'utile à long terme.

## Ce qui suit

Le chapitre 16 fait le bilan : **limites actuelles du projet, gaps identifiés, chantiers ouverts**. Honnêteté complète sur ce qui reste à faire.
