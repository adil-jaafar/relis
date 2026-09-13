# Carnet de bord RELIS

*État au 12 septembre 2026. Ce document décrit l'objectif, le fonctionnement, l'originalité et
l'avancement du projet. Il se met à jour à chaque plan terminé.*

**RELIS est un modèle de langage qui lit pour répondre, décide combien lire, et montre ce qu'il a
retenu.**

| | |
|---|---|
| Pré-entraînement | **1,009 bit par octet** en validation, terminé |
| Entraînement ruban | **terminé, 500 pas** en deux sessions (≈ 20 h sur 2×T4) ; équilibrée 0,927 en validation |
| Autonomie (pas 500) | **200/200 arrêts exacts**, réponses justes 91,5 %, 47 % du contexte économisé, lecture ×4,67 sur les cas durs ; aiguille 0,95 à 1 ko puis chute au-delà de la longueur vue à l'entraînement |
| Modèle | 159,8 M paramètres, entraîné de zéro |
| Dépôt | 198 tests, plan 3 fusionné dans main |
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

### ✅ Entraînement ruban sur Kaggle — 500 pas

Le modèle apprend à décider. C'est l'étape qui transforme un modèle de langage en machine à lire, et
elle a réussi bien plus vite que prévu.

**La thèse du projet est démontrée au pas 52.** Sur un épisode jamais vu, sondé en probabilité :

```
   1..9  CONT            CONT 0.99+   STOP 0.000
  10     STOP  <-- ici   CONT 0.088   STOP 0.903
```

Le modèle donne 90 % de probabilité à STOP au seul segment qui contient la réponse, et
essentiellement zéro aux neuf autres. Marge de 962 contre 1 au hasard.

| ≈ pas | équilibrée | STOP | SKIP | NEXT | NOTE |
|---|---|---|---|---|---|
| 13 | 0,321 | 0/133 | 0/62 | 0/27 | 0/30 |
| 26 | 0,673 | 66/133 | 62/62 | 0/27 | 0/30 |
| 39 | 0,826 | 124/133 | 61/62 | 16/27 | 8/30 |
| 52, épisodes **jamais vus** | **0,851** | 299/311 | 132/132 | 51/64 | 9/89 |
| 287, épisodes **jamais vus** (fin de session 1) | **0,915** | 303/311 | 132/132 | 64/64 | 35/89 |
| 500, validation (fin de session 2) | **0,927** | 125/133 | 62/62 | 27/27 | 16/30 |

Le score sur épisodes inédits dépasse celui de la validation : aucun surapprentissage, le mécanisme
généralise. SKIP et REFRESH sont à 100 %.

**Pas 287, fin de la première session (arrêt par budget de temps, 11,5 h).** STOP, SKIP, REFRESH et
NEXT sont acquis ; la sonde STOP donne 1,000 au bon segment et 0,000 aux neuf autres. Seule la
décision NOTE reste en retard, mais elle progresse encore (9 → 35 sur 89) : c'est la raison de la
seconde session. Perte 0,18, validation 0,51 bit par octet sur les rubans.

**NOTE reste à 10 %**, et la mesure est trop sévère pour lui : en forçage on exige l'émission à
l'octet exact où l'oracle l'a placée, alors qu'à l'inférence il suffit de déclencher au bon moment
approximatif. STOP choisit parmi trois à une position marquée ; NOTE se décide à chaque octet généré.
REFRESH à 100 % montre que le mécanisme fonctionne dès qu'il est enclenché.

**Calendrier revu.** Le masque de segment de l'attention locale coûte cher : environ 140 secondes par
pas, deux fois et demie le pré-entraînement. Les 1 500 pas prévus demanderaient 58 heures. Comme les
décisions saturent avant le pas 50, le run est ramené à `train.max_steps=500`, soit 1,2 passage sur
les données et deux sessions.

**Dérive de génération : légère.** Le français reste grammatical et s'est libéré du gabarit « commune
française » qui bloquait tout après le pré-entraînement. Le code dérive davantage. Le rappel à 10 %
du gradient tient la langue sans la figer.

### ✅ Plan 3 — Le contrôleur et la démonstration

La boucle d'inférence qui pilote réellement le modèle, et tout ce qui rend le mécanisme visible.
Six tâches, écrites sans consommer de quota GPU pendant que l'entraînement ruban tournait sur
Kaggle.

- Décodage contraint en UTF-8, cas limites câblés par restriction du masque plutôt qu'appris
- Contrôleur : encoder, balayer, décider, générer, relire (REFRESH) ; budgets pour rendre la main
  même sur un modèle qui n'arrête jamais
- Terminal de démonstration : documents joints, conversation persistante, segments lus et sautés
  affichés à l'écran, position du STOP, raison d'une relecture
- Harnais d'évaluation en autonomie contre l'oracle (`relis.eval.autonomy`) : le contrôleur décide
  seul, plus aucune décision n'est forcée — la seule mesure qui dit si le mécanisme survit à la
  composition de ses propres erreurs
- Évaluation aiguille (`relis.eval.needle`) : exactitude et octets lus en fonction de la longueur
  de l'historique (1 ko à 200 ko), pour vérifier que la Mémoire à taille constante ne décroche pas
  quand l'historique dépasse largement ce qui a été vu à l'entraînement
- Évaluation du calcul adaptatif (`relis.eval.adaptive`) : octets lus sur des cas faciles contre
  des cas difficiles, pour vérifier chiffres à l'appui que la lecture suit la difficulté plutôt
  que de tout lire systématiquement
- Mode d'emploi Colab (`notebooks/COLAB_EVAL.md`) pour exécuter les trois évaluations et une
  conversation d'essai sur un checkpoint réel

**Chiffres réels, checkpoint final du pas 500** (Colab, `notebooks/COLAB_EVAL.md`), tous sans aucun
arrêt par budget ; entre parenthèses, la valeur au pas 287 quand elle diffère :

| Évaluation | Résultat |
|---|---|
| Autonomie, 200 épisodes jamais vus, sept familles | **200 arrêts exacts sur 200** (0 trop tôt, 0 trop tard, 0 document utile sauté) ; réponses justes **91,5 %** (85,5 %) ; 453 octets lus en moyenne, 47 % du contexte économisé |
| Calcul adaptatif | cas faciles **98 octets** lus (justes 100 %, contre 95 %), cas difficiles **458 octets** (justes 100 %) : **×4,67** |
| Aiguille 1 ko | exactitude 0,95 |
| Aiguille 4 ko | 0,35, inchangé entre 287 et 500 pas |
| Aiguille 16 ko et 64 ko | 0,00 et 0,05, avec seulement 143 et 650 octets lus |

Deux démonstrations sur trois tiennent : « il réfléchit plus quand c'est dur » (le modèle lit près de
cinq fois plus quand la question l'exige) et la lecture des documents (aucun document utile sauté,
aucun arrêt prématuré sur les familles à documents). Les 8,5 % de réponses fausses ne viennent pas
de la lecture, qui s'arrête au bon endroit dans 100 % des cas, mais de la génération : la valeur
copiée est parfois altérée. Les 213 derniers pas ont fait passer ce chiffre de 14,5 % à 8,5 % sans
rien changer aux décisions, déjà saturées. C'est la piste Extraits/RECALL du plan 2b.

**« Il se souvient de très loin » ne tient pas encore, et la cause est connue.** Les épisodes
d'entraînement ont entre 3 et 14 paires de tours, soit au plus 1 ko d'historique environ ; c'est
exactement la taille où l'aiguille réussit à 95 %. À 4 ko l'historique est quatre fois plus long que
tout ce que le modèle a vu, à 64 ko cent fois. Il ne décroche pas parce que la Mémoire sature : il
s'arrête après quelques segments, parce qu'il n'a jamais appris à maintenir CONT sur des centaines de
segments. C'est un défaut de la distribution d'entraînement, pas de l'architecture, et il se corrige
avec des épisodes à historique long (voir la suite). Preuve par l'absence : 213 pas de plus n'ont pas
bougé l'aiguille d'un centième, ce sont bien les données qui manquent, pas les pas.

Extraits et RECALL (copier exactement les noms, nombres et lignes lus dans les documents) et la
sonde qui traduit le Buffer en français sont retirés du périmètre de ce plan : les deux exigent de
régénérer les rubans d'entraînement et de réentraîner (RECALL) ou dépendent de l'étiquetage par un
modèle professeur (la sonde). Ils sont renvoyés au plan 2b, avec l'application Gradio qui suivra
la CLI.

### ⬜ Plan 2b — Données réelles, échelle et témoin

Ce qui manque pour que les décisions apprises servent au-delà des épisodes synthétiques, et le témoin
scientifique qui rend la comparaison honnête.

- Dialogues français publics, étiquetés par un modèle professeur sur Colab
- **Historiques longs, en premier** (c'est ce qui manque à l'aiguille au-delà de 1 ko) : une
  famille d'épisodes à 50-150 paires de tours qui tient dans les 15 ko d'un ruban, puis, pour
  16 ko et au-delà, l'entraînement par fenêtres successives avec la Mémoire reportée d'une fenêtre à
  la suivante, ce pour quoi l'architecture est faite
- Écart d'échelle : documents longs, segments d'historique à sauter
- Baseline Transformer sur octets, même taille et mêmes données. Elle coûte autant de GPU que RELIS,
  donc elle passe en dernier
- Extraits et RECALL (copier exactement noms, nombres et lignes lus), reportés du plan 3 : exigent
  de régénérer les rubans d'entraînement
- Sonde qui traduit le Buffer en français, reportée du plan 3 : dépend de l'étiquetage par le
  modèle professeur ci-dessus

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
| Évaluer un checkpoint sur Colab | [`notebooks/COLAB_EVAL.md`](../notebooks/COLAB_EVAL.md) |
| Échantillonner un checkpoint | `python -m relis.infer.sample --repo <dépôt> --prompt "…"` |
| Construire des rubans | `python -m relis.data.pack --out shards/tapes --episodes 100000 …` |
