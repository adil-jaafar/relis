# Carnet de bord RELIS

*État au 12 septembre 2026. Ce document décrit l'objectif, le fonctionnement, l'originalité et
l'avancement du projet. Il se met à jour à chaque plan terminé.*

**RELIS est un modèle de langage qui lit pour répondre, décide combien lire, et montre ce qu'il a
retenu.**

| | |
|---|---|
| Pré-entraînement | **1,009 bit par octet** en validation, terminé |
| Entraînement ruban | en cours, 1 500 pas, ≈ 23 h sur 2×T4 |
| Modèle | 159,8 M paramètres, entraîné de zéro |
| Dépôt | 139 tests, 50 commits |
| Spécification | [`docs/superpowers/specs/2026-09-06-relis-design.md`](superpowers/specs/2026-09-06-relis-design.md) |

---

## 1. Ce qu'on veut faire

Concevoir un modèle de langage hors des sentiers du Transformer, avec une idée d'architecture qui lui
soit propre, et le construire pour de vrai avec les moyens disponibles : un T4 sur Colab, deux T4 sur
Kaggle, trente heures de GPU par semaine.

Le pari est que la contrainte oblige à l'originalité. À cette échelle, on ne gagne pas en empilant des
paramètres. On gagne en changeant ce que le modèle *fait*. RELIS ne cherche pas à mémoriser le monde :
il cherche à savoir où regarder.

La V1 doit tenir en quelques semaines et produire trois choses : une conversation cohérente, des
mécanismes visibles et mesurés, et une comparaison honnête à un Transformer de même taille. Ensuite
viennent une phase entreprise, puis des sujets de thèse.

---

## 2. Comment RELIS fonctionne

C'est une **machine à lire**. Une tête de lecture, deux mémoires de taille fixe, et des actions. Tout
ce que vit le modèle pendant un tour de conversation est une seule séquence d'octets, le **ruban**,
qu'il parcourt en trois modes.

Il encode d'abord la demande. Puis il balaie l'historique du plus récent au plus ancien, puis les
documents joints, *en sachant ce qu'il cherche*. À chaque en-tête et à chaque fin de segment, il
décide. Enfin il répond en vidant ce qu'il a accumulé.

### Un ruban réel

Produit par le générateur d'épisodes. Les codes marqués `←` sont les positions où le modèle décide :
elles pèsent cinq fois plus que le texte dans la fonction de coût. Les décisions ne sont pas une tête
séparée, ce sont des octets prédits comme les autres.

```
mode   ruban
────── ─────────────────────────────────────────────────────────────────────────────
ENC    ENC              Quelle est l'adresse du serveur ?
SCAN   SCAN CHAN        chan=history
SCAN      SEG           role=assistant;t=2026-09-20T14:03
SCAN      HDR READ   ←  Très bien, continuons.
SCAN      DEC NEXT   ←  rien d'utile ici, passer aux documents
SCAN   CHAN            chan=docs
SCAN      SEG           type=doc;name=notes_0.txt;bytes=395
SCAN      HDR SKIP   ←  sauté d'après son nom : le contenu n'entre jamais dans le ruban
SCAN      DEC CONT   ←
SCAN      SEG           type=doc;name=notes_1.txt;bytes=453
SCAN      HDR SKIP   ←  sauté
SCAN      DEC CONT   ←
SCAN      SEG           type=doc;name=serveur_27.txt;bytes=258
SCAN      HDR READ   ←  …L'adresse du serveur est 10.238.231.131…
SCAN      DEC STOP   ←  j'en sais assez, j'arrête de lire
GEN    GEN             L'adresse du serveur est 10.238.231.131.
GEN    END        ←     fin du tour
```

### La boucle de lecture

```
   ┌──────────┐     ┌───────────────────┐  STOP  ┌──────────┐     ┌─────┐
   │  Encode  │────▶│  Balaie et décide │───────▶│  Génère  │────▶│ END │
   │ la demande│     │ READ SKIP CONT NEXT│        │octet/octet│    └─────┘
   └──────────┘     └───────────────────┘        └──────────┘
         ▲                                              │
         └──────────────────────────────────────────────┘
            NOTE « il manque la ville » puis REFRESH
            Mémoire remise à zéro, Buffer et réponse partielle conservés,
            on relit plus profond
```

Le calcul dépensé n'est pas fixe : il vaut le nombre d'octets réellement lus plus le nombre d'octets
écrits. Une question simple coûte deux segments, une question difficile en coûte cent.

### Les deux mémoires

La **Mémoire** est l'état récurrent des couches, environ 1,5 Mo, et sa taille ne dépend pas de ce qui
a été lu. C'est ce qui donne le contexte illimité : lire dix kilo-octets ou dix méga-octets n'occupe
pas un octet de plus.

Le **Buffer** est un jeu de 32 vecteurs que le modèle écrit tous les 64 octets et relit à chaque
couche. C'est là que se forme l'intention : ce qu'il cherche, ce qu'il a trouvé, ce qu'il lui manque.
Une sonde entraînée séparément le traduira en français pour la démonstration.

Il n'y a **pas de tokenizer**. Le vocabulaire est exactement 256 octets, dont dix-sept sont réservés
comme codes de contrôle. Aucune langue n'est avantagée, aucun mot n'est hors vocabulaire, et les
décisions vivent dans le même alphabet que le texte.

### Architecture

| | |
|---|---|
| Mélangeur | Gated DeltaNet, 12 couches |
| Attention | locale, fenêtre 512, 4 couches |
| Dimension | 768, 8 têtes de 64 |
| Buffer | 32 slots, écriture tous les 64 octets, visible tous les 512 |
| Mémoire | ≈ 1,5 Mo, constante |
| Corpus de pré-entraînement | 1,3 Go : Wikipédia FR, FineWeb-2 fr, code Python et JavaScript |

---

## 3. Ce qu'il apporte d'original

Le niveau octet n'est pas l'originalité de RELIS. Plusieurs travaux récents ont montré qu'on peut se
passer de tokenizer sans rien perdre : MambaByte avec une récurrence linéaire, le Byte Latent
Transformer et H-Net avec une hiérarchie de patchs. RELIS s'inscrit dans cette lignée, et sa version
V1 y est même la plus simple, sans hiérarchie.

L'originalité est ailleurs, et elle tient en une phrase : **aucun de ces modèles ne décide combien
lire**.

| Approche | Sans tokenizer | Hiérarchie | Contexte | Décide quoi lire |
|---|---|---|---|---|
| Transformer + BPE | non | — | fenêtre | non |
| MambaByte | oui | non | état constant | non |
| Byte Latent Transformer | oui | par entropie | fenêtre | non |
| H-Net | oui | apprise | fenêtre | non |
| **RELIS V1** | **oui** | V2 | **état constant** | **oui** |

**1. Le calcul suit la difficulté.** Le modèle émet STOP quand il en sait assez et SKIP quand un
document ne le concerne pas. Ce ne sont pas des heuristiques externes : ce sont des octets qu'il
prédit, appris contre un oracle qui connaît la bonne réponse parce que les épisodes sont construits
pour cela.

**2. Le contexte est illimité par construction.** Pas de fenêtre à étendre, pas d'attention
quadratique à approximer. L'état est de taille fixe, donc le coût est linéaire en octets effectivement
lus. Un historique de deux cents kilo-octets n'est pas un problème d'architecture, c'est un problème
de temps.

**3. Récupérer sans index.** La lecture est conditionnée par la requête, encodée avant le balayage. Il
n'y a ni base vectorielle, ni découpage préalable, ni score de similarité. Le modèle retrouve
l'information en lisant, comme le ferait quelqu'un qui feuillette. C'est du RAG par conception plutôt
que par tuyauterie.

**4. Il écrit pourquoi il relit.** Avant de relancer une lecture, le modèle émet NOTE puis une phrase
en clair, « il manque la ville », qui lui est relue à la passe suivante. La raison survit à la remise
à zéro de la mémoire et reste lisible par un humain. Ce n'est pas une explication reconstruite après
coup, c'est le mécanisme lui-même.

*Réserve de méthode : cette comparaison repose sur la littérature connue jusqu'à mi-2026. Des travaux
plus récents peuvent exister.*

---

## 4. Où on en est

### ✅ Plan 1 — Le cœur du modèle et le pré-entraînement

Treize tâches, de la couche récurrente au lancement sur Kaggle. Gated DeltaNet chunké écrit en PyTorch
pur et prouvé équivalent à sa récurrence pas à pas, attention locale à cache borné, Buffer, État à
mémoire constante, boucle d'entraînement en demi-précision avec reprise automatique entre sessions.

- 2 500 pas, 1,3 Go d'octets vus, 39 heures sur deux T4
- **1,009 bit par octet** en validation, meilleur que les 1,3 à 1,6 attendus
- Le français est acquis : grammaire, accords, structure. Les erreurs sont factuelles, pas
  linguistiques

### ✅ Plan 2a — Rubans, oracle et entraînement à la lecture

Six tâches. Le format du ruban, le code NOTE, le REFRESH rendu entraînable exactement tel qu'il sert à
l'inférence par un masque de réinitialisation, sept familles d'épisodes synthétiques dont l'oracle
calcule les décisions exactes, et l'empaquetage en séquences de 16 384 octets.

- 12 919 séquences d'entraînement, 3,7 passages prévus
- Mélange du gradient : décisions 26 %, réponses 39 %, texte 35 %
- Reprise depuis les poids du pré-entraînement, en chargement strict

### 🟠 En cours — Entraînement ruban sur Kaggle

Mille cinq cents pas, environ vingt-trois heures, deux sessions. Le modèle apprend à décider. C'est
l'étape qui transforme un modèle de langage en machine à lire.

Ce qu'on surveille, à partir de la ligne de validation imprimée toutes les trente minutes :

```
[val] équilibrée 0.326 | acc 0.840 | n 2277 | STOP 0/133 SKIP 0/62 NOTE 0/30
      NEXT 0/27 END 4/103 REFRESH 17/30 READ 995/995 CONT 897/897
```

Au sixième pas, le modèle a trouvé la solution paresseuse : toujours répondre READ et CONT, les
classes majoritaires, ce qui donne déjà 84 % d'exactitude brute. Les cinq décisions qui comptent sont
à zéro. Le chiffre à suivre est l'**exactitude équilibrée**, et la sortie de zéro de **STOP**. Si rien
ne bouge au pas 300, un levier de rééquilibrage est prêt : `--override train.decision_balance=0.5`.

### ⬜ Plan 3 — Le contrôleur et la démonstration

La boucle d'inférence qui pilote réellement le modèle, et tout ce qui rend le mécanisme visible. C'est
le plan qui produit le « waw », et il ne consomme aucun quota GPU, donc il peut s'écrire pendant que
l'entraînement tourne.

- Contrôleur : encoder, balayer, décider, générer, relire ; décodage contraint en UTF-8
- Extraits et RECALL : copier exactement les noms, nombres et lignes lus dans les documents
- Sonde qui traduit le Buffer en français, évaluations aiguille sur des historiques de 1 ko à 200 ko
- Interface de démonstration : segments lus et sautés, position du STOP, raison des relectures

### ⬜ Plan 2b — Données réelles, échelle et témoin

Ce qui manque pour que les décisions apprises servent au-delà des épisodes synthétiques, et le témoin
scientifique qui rend la comparaison honnête.

- Dialogues français publics, étiquetés par un modèle professeur sur Colab
- Écart d'échelle : documents longs, historiques profonds, segments d'historique à sauter
- Baseline Transformer sur octets, même taille et mêmes données. Elle coûte autant de GPU que RELIS,
  donc elle passe en dernier

### ⬜ Plus tard — V2 et sujets de thèse

Hiérarchie octets vers concepts avec découpage appris, Lexique de cellules rappelables apprises à
l'entraînement, cache par segment pour ne pas relire l'inchangé, Buffer persistant entre les tours.

Du côté recherche : décider REFRESH d'après la saturation réelle de l'état plutôt que par heuristique,
apprendre les décisions par renforcement contre le coût de lecture, et formaliser ce que « lire pour
répondre » gagne sur l'attention globale.

---

## Repères pratiques

| Quoi | Où |
|---|---|
| Spécification de conception | [`docs/superpowers/specs/2026-09-06-relis-design.md`](superpowers/specs/2026-09-06-relis-design.md) |
| Plan 1 | [`docs/superpowers/plans/2026-09-06-relis-plan1-coeur.md`](superpowers/plans/2026-09-06-relis-plan1-coeur.md) |
| Plan 2a | [`docs/superpowers/plans/2026-09-07-relis-plan2a-rubans.md`](superpowers/plans/2026-09-07-relis-plan2a-rubans.md) |
| Procédure Kaggle et Colab | [`notebooks/KAGGLE.md`](../notebooks/KAGGLE.md) |
| Échantillonner un checkpoint | `python -m relis.infer.sample --repo <dépôt> --prompt "…"` |
| Construire des rubans | `python -m relis.data.pack --out shards/tapes --episodes 100000 …` |
