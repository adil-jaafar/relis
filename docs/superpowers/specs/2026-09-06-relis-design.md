# RELIS — un modèle de langage qui lit pour répondre, et relit quand il faut

*Spec de design, V1 (3-4 semaines). Date : 2026-09-06. Statut : validée en brainstorming, en attente de relecture avant plan d'implémentation.*

---

## 1. Vision

Les LLM actuels traitent la conversation comme une fenêtre : tout ce qui n'y tient pas est perdu, et tout ce qui y tient coûte de l'attention quadratique. RELIS renverse le principe. C'est **une machine à lire** : une tête de lecture, deux mémoires de taille fixe, et des actions.

- Le modèle **encode d'abord la demande**, puis **balaie** l'historique du plus récent au plus ancien, puis les documents, *en sachant ce qu'il cherche*. Il accumule une **Mémoire** (état récurrent, taille constante) et écrit incrémentalement son **intention** dans un **Buffer** de slots.
- Il **décide** par des **codes de contrôle** émis comme n'importe quel octet : s'arrêter de lire quand il en sait assez (STOP), sauter un document d'après son en-tête (SKIP), passer au canal suivant (NEXT), terminer (END), ou **relire** avec sa réponse partielle en tête quand Mémoire et Buffer sont consommés (REFRESH).
- Il **génère** en drainant Mémoire + Buffer, sans re-balayer par octet, et **rappelle** exactement (RECALL) les extraits relevés pendant la lecture : noms, nombres, lignes de code.

Conséquences : contexte illimité par construction (historique et documents), calcul proportionnel à la difficulté (Système 2 latent), aucune tokenisation (octets + codes de contrôle), aucune couche d'attention globale (récurrence linéaire), et une récupération d'information *par lecture orientée* plutôt que par index vectoriel (« RAG par conception »).

Pitch en une phrase : **un modèle qui lit pour répondre, décide combien lire, et montre ce qu'il a retenu.**

## 2. Cadre

| Élément | Décision |
|---|---|
| Livrable V1 | Système complet utilisable : conversation cohérente en CLI + démo Gradio, avec documents joints |
| Compute | Colab T4 (16 Go, fp16, pas de bf16) et Kaggle 2×T4 (30 h/semaine, sessions ≤ 12 h) |
| Budget réaliste | 80-120 GPU-heures sur 3-4 semaines, ~2 Go d'octets vus |
| Taille modèle | 120-150 M paramètres, entraîné de zéro |
| Aide externe | Hybride pragmatique : un professeur ouvert (Qwen2.5-1.5B-Instruct) **génère et étiquette des données** ; pas de distillation de logits (impossible sans tokenizer commun), pas de poids réutilisés |
| Données | Français, code (Python, JavaScript), dialogues, épisodes synthétiques de lecture |
| Succès V1 | Conversation cohérente simple ; mécanismes (lecture, arrêt, skip, refresh, Buffer) visibles et mesurés ; comparaison à une baseline Transformer sur octets |
| Suite | Phase entreprise, puis sujets de thèse (voir §13) |

Attente honnête : à 150 M paramètres et 2 Go de données, la fluidité sera celle d'un petit assistant (questions simples, français, bouts de code). Le « wow » vient de ce que le modèle **fait** et **montre**, pas de son éloquence.

## 3. Décisions de conception prises

| Question | Choix |
|---|---|
| Ordre de balayage de l'historique | Du plus récent au plus ancien ; octets d'un segment dans l'ordre naturel |
| Buffer lisible ? | Latent, plus une **sonde-traducteur** entraînée après coup (modèle figé) pour la démo |
| Contenu de la requête après REFRESH | Message + réponse partielle ; Mémoire réinitialisée, Buffer conservé |
| Canal documents | Fichiers joints à la conversation, **sans limite de taille ni de nombre** |
| Séparation des documents | Codification `[SEG] en-tête [HDR] contenu`, commune aux messages et aux documents |
| Décision après en-tête | READ ou SKIP (ajout V1, conséquence de l'illimité) |
| Découpage en patchs | Non en V1 : octets purs. Hiérarchie octets→concepts en V2 |
| Cellules rappelables (octets + embedding) | **Extraits** en V1 : cellules remplies pendant la lecture, rappelées par contenu (presse-papiers). **Lexique** en V2 : cellules apprises à l'entraînement. Un seul code d'action, RECALL |
| REFRESH à l'entraînement (plan 2) | Entraîné **exactement comme à l'inférence** par un masque de réinitialisation ; REFRESH redéfini : état GDN et fenêtre locale à zéro, le reste continue (§4.8) |
| Trace de la raison d'un REFRESH | En **octets**, code NOTE : le modèle écrit pourquoi il relit, la note est relue dans la requête de la passe suivante et visible dans la démo (§4.8) |
| Séquences d'entraînement ruban | Rubans rembourrés à un multiple de 512 puis **empaquetés** en séquences fixes de 16 384 octets ; Mémoire et slots remis à zéro à chaque début de ruban (§6.1) |
| Professeur | **Étiquetage seulement** (tours pertinents, résumés d'intention) sur des dialogues français publics ; aucune génération de dialogues (§5, §6.2) |

## 4. Architecture

### 4.1 Un réseau, trois modes

Un unique empilement de couches (mêmes poids partout) fonctionne dans trois modes signalés par un embedding de mode ajouté à l'entrée :

- **ENCODE** : lit la requête (dernier message, plus réponse partielle après REFRESH). Initialise la Mémoire, commence à écrire le Buffer.
- **SCAN** : balaie un canal segment par segment. Met à jour Mémoire et Buffer. Émet les décisions.
- **GENERATE** : émet la réponse octet par octet, met à jour la Mémoire avec ses propres sorties, lit le Buffer. Émet END ou REFRESH.

Aucun module distinct par phase. Le même lecteur lit et écrit ; seul le ruban change.

### 4.2 Vocabulaire : 256 octets, dont des codes de contrôle

Le vocabulaire est exactement 256 symboles. Les octets 0x00-0x1F, sauf `\t` (0x09), `\n` (0x0A) et `\r` (0x0D) qui restent du texte, sont réservés comme **codes de contrôle**. Tout octet de contrôle présent dans un texte d'entrée est remplacé par U+FFFD (EF BF BD).

| Octet | Nom | Type | Rôle |
|---|---|---|---|
| 0x01 | ENC | marqueur | Début d'encodage de la requête |
| 0x12 | PART | marqueur | Dans la requête : sépare le message de la réponse partielle (après REFRESH) |
| 0x0E | SCAN | marqueur | Début du balayage |
| 0x1C | CHAN | marqueur | Début d'un canal ; suivi d'un en-tête de canal (`chan=history` / `chan=docs`) |
| 0x02 | SEG | marqueur | Début d'un segment (message ou document) ; suivi de son en-tête |
| 0x1F | HDR | **décision** | Fin d'en-tête. Le modèle prédit READ ou SKIP |
| 0x11 | READ | action | Lire le contenu du segment |
| 0x08 | SKIP | action | Sauter le contenu ; le contrôleur passe au segment suivant |
| 0x10 | DEC | **décision** | Fin de segment. Le modèle prédit CONT, STOP ou NEXT |
| 0x07 | CONT | action | Continuer avec le segment suivant du canal |
| 0x04 | STOP | action | Assez d'information : passer à GENERATE |
| 0x06 | NEXT | action | Passer au canal suivant (historique → documents) |
| 0x0F | GEN | marqueur | Début de génération |
| 0x03 | END | action | Fin du tour |
| 0x05 | REFRESH | action | Réinitialiser la Mémoire, ré-encoder (message + PART + réponse partielle), re-balayer, puis reprendre la génération |
| 0x13 | RECALL | action | Rappeler une cellule (§4.9) : suivi de son adresse (2 octets en V1), puis le contrôleur développe les octets de la cellule |
| 0x14 | NOTE | action | Début d'une note en octets expliquant pourquoi le modèle va relire ; la note se termine par REFRESH et est relue dans la requête suivante (§4.8) |

Les **marqueurs** sont insérés par le contrôleur (entrées). Les **actions** sont *prédites* par le modèle aux positions de **décision** puis réinjectées dans le ruban comme octet suivant. Pendant GENERATE, chaque position est une décision parmi 256 octets où seuls les octets de texte, END, REFRESH, RECALL et NOTE sont autorisés ; après NOTE, seuls les octets de texte et REFRESH le sont.

**Adresse de cellule.** Les octets d'adresse sont pris dans 0x20-0xFF (224 valeurs) pour ne jamais entrer en collision avec les codes de contrôle. La plage du premier octet donne la classe et la longueur : 0x20-0x8F (112 valeurs) désigne un **Extrait** et il est suivi d'**un** seul octet, soit 112 × 224 = 25 088 adresses pour 4 096 cellules (12 bits utiles) ; 0x90-0xFF est réservé au **Lexique** de V2 et sera suivi de deux octets (112 × 224² ≈ 5,6 M adresses, assez pour 2^19 entrées). En V1 une adresse fait donc 2 octets. Pendant ces positions, l'automate UTF-8 est suspendu. Ainsi les décisions sont de simples prédictions du prochain octet : le modèle reste un modèle de langage autorégressif, sans tête d'action séparée ni RL en V1.

### 4.3 Le ruban

Toute la vie d'un tour est une seule séquence d'octets :

```
ENC  <octets du dernier message>
SCAN
CHAN chan=history HDR
  SEG role=assistant;t=...   HDR READ <msg n-1> DEC CONT
  SEG role=user;t=...        HDR READ <msg n-2> DEC CONT
  SEG role=assistant;t=...   HDR READ <msg n-3> DEC NEXT
CHAN chan=docs HDR
  SEG type=doc;name=notes.md;mime=text/markdown;bytes=48213;t=...  HDR SKIP
  SEG type=doc;name=api.py;mime=text/x-python;bytes=9120;t=...     HDR READ <contenu> DEC STOP
GEN  <octets de la réponse> END
```

Avec un rappel d'Extrait pendant la génération (le contenu développé est réinjecté dans la récurrence, en italique ici) :

```
GEN  La fonction s'appelle RECALL a1 a2 *parse_config_from_env* et prend ... END
```

Avec REFRESH :

```
GEN  <réponse partielle> NOTE il manque la date du contrat REFRESH
ENC  <dernier message> PART <réponse partielle> NOTE il manque la date du contrat
SCAN ... (nouveau balayage, Mémoire neuve, Buffer conservé) ... STOP
GEN  <suite de la réponse> END
```

**En-têtes.** Texte UTF-8 court (< 128 octets) de paires `clé=valeur` séparées par `;` ; dans les valeurs, `;` et `=` sont remplacés par une espace et les octets de contrôle assainis, de sorte que l'en-tête se parse sans ambiguïté. Messages : `role`, `t` (horodatage). Documents : `type=doc`, `name`, `mime`, `bytes`, `t` (date d'ajout). L'en-tête est lu avant la décision READ/SKIP : c'est ce qui permet d'ignorer un document d'après son nom, son type ou sa taille. Les en-têtes de canal (`chan=history`, `chan=docs`) n'ont pas de décision READ/SKIP : ils sont suivis directement du premier SEG.

**Canaux.** V1 : `history` puis `docs`. Le format admet d'autres canaux plus tard (outils, résultats d'exécution) sans changer le modèle.

**Ordre.** Historique : du message le plus récent au plus ancien. Documents : du plus récemment ajouté au plus ancien. Dans un segment, les octets sont dans l'ordre naturel.

**Illimité.** Ni le nombre ni la taille des segments ne sont bornés. Le contrôleur streame le contenu depuis le disque ; la Mémoire est de taille constante ; le temps est linéaire en octets effectivement lus (SKIP et STOP le réduisent).

### 4.4 Couche de base : Gated DeltaNet + attention locale

Le mélangeur séquentiel principal est un **Gated DeltaNet** (récurrence linéaire à règle delta et porte d'oubli), implémenté en **PyTorch pur** par l'algorithme par chunks (chunk = 64), sans kernel externe obligatoire. Choix motivé par le **rappel associatif** : retrouver un fait précis dans l'historique est le cœur du modèle, et la règle delta y est nettement supérieure à Mamba-2 à taille égale. L'état récurrent est maintenu en **fp32** même quand le reste est en fp16.

Une couche sur quatre est une **attention à fenêtre glissante** de 512 octets. Elle apporte la précision locale indispensable au niveau octet (orthographe, syntaxe du code) sans rompre la mémoire constante : la fenêtre est fixe, c'est un opérateur local comme une convolution.

Chaque bloc : RMSNorm → mélangeur (GDN ou SWA, précédé d'une convolution causale courte) → résidu → RMSNorm → MLP (SwiGLU) → résidu, plus la lecture du Buffer (§4.6).

**Taille V1** : d = 768, 16 blocs (12 GDN, 4 SWA en positions 4, 8, 12, 16), MLP ×4, GDN à 8 têtes de dimension 64 avec état 64. Environ 150 M paramètres (la lecture du Buffer utilise une dimension interne réduite de moitié pour tenir ce budget). Embedding d'octets 256×768 ; tête de sortie liée à l'embedding.

### 4.5 La Mémoire

La Mémoire est l'ensemble des états récurrents des 12 couches GDN (12 × 8 × 64 × 64 fp32 ≈ 1,5 Mo) plus les fenêtres des 4 couches SWA (4 × 512 positions). Sa taille ne dépend pas de la quantité lue : c'est ce qui donne le contexte illimité. Elle est **réinitialisée** à chaque REFRESH par ré-encodage de la requête, et à chaque nouveau tour.

### 4.6 Le Buffer d'intention

K = 32 slots de dimension d = 768, **partagés par toutes les couches**.

- **Initialisation** : vecteurs appris, à chaque nouveau tour utilisateur.
- **Lecture** (chaque bloc) : cross-attention des positions de la séquence vers les slots, avec projections propres à chaque couche. K est petit : coût négligeable. C'est par cette lecture que la requête, écrite dans le Buffer pendant ENCODE, **conditionne** tout le balayage.
- **Écriture** (une fois par chunk de 64 octets) : un module en sommet de pile met à jour les slots par attention slots → états cachés du chunk, avec porte apprise : `B ← B + g ⊙ Δ`, `g = σ(W [B ; Δ])`. La suite des écritures forme une récurrence à granularité chunk ; la rétropropagation la traverse sur toute la séquence d'entraînement. Les écritures sont calculées par chunk de 64 octets mais ne deviennent **visibles aux lectures qu'à chaque frontière de bloc de 512 octets** : cette latence permet de traiter 512 positions par passage dans la pile et rend les modes chunk et pas-à-pas numériquement équivalents.
- **Persistance** : conservé à travers les REFRESH d'un même tour ; remis à zéro au tour suivant (la persistance inter-tours est un sujet V2). À l'entraînement, la remise à zéro se fait par échantillon au début de chaque ruban, qui tombe sur une frontière de bloc (§6.1).

**Traducteur (sonde).** Un mini-décodeur (2 blocs GDN, d = 384) reçoit les 32 slots comme préfixe et génère en français « ce que j'ai retenu : demande, plan, faits ». Il est entraîné **après** le modèle principal, **modèle figé**, sur des résumés produits par le professeur pour (requête, tours pertinents). Il n'influence pas le modèle ; il le rend observable. Ce qu'il affiche est donc une *lecture* du Buffer, pas une garantie sur son contenu.

### 4.7 Décisions et calcul adaptatif

Trois points de décision, tous traités comme prédiction du prochain octet sous masque :

| Position | Choix autorisés | Effet |
|---|---|---|
| après HDR | READ, SKIP | Lire ou sauter le contenu du segment |
| après DEC | CONT, STOP, NEXT | Continuer, arrêter de lire, changer de canal |
| pendant GEN | octets texte valides, END, REFRESH, RECALL, NOTE | Générer, terminer, relire, rappeler un Extrait, noter pourquoi relire |
| après NOTE | octets texte valides, REFRESH | Écrire la raison, puis relire |

Cas limites fixés par le contrôleur : au dernier segment de l'historique, CONT est traité comme NEXT ; au dernier segment des documents (ou s'il n'y a aucun document), CONT et NEXT sont traités comme STOP. Le modèle n'a donc jamais à connaître la longueur des canaux.

Le calcul dépensé par tour = octets lus + octets générés. Question simple : quelques segments récents puis STOP. Question sur un vieux détail ou un document : plus de lecture. Réponse longue ou complexe : REFRESH. Tout cela est **affiché** dans la démo.

### 4.8 REFRESH

Avant de relire, le modèle **écrit pourquoi** : il émet NOTE puis une courte note en octets (« il manque la date du contrat », « réponse longue, suite »), puis REFRESH. Le contrôleur conserve la réponse partielle, la note et le Buffer, réinitialise la Mémoire, construit une nouvelle requête `ENC <message> PART <réponse partielle> NOTE <note>`, relance un balayage complet (avec ses propres décisions), puis reprend GENERATE en continuant la réponse. La réponse partielle et la note rendent chaque passe différente de la précédente : le modèle relit en sachant ce qu'il a déjà dit et ce qu'il cherche. La raison survit ainsi à la remise à zéro de la Mémoire par le ruban lui-même (et le Buffer, qui persiste, la porte aussi en latent) ; elle est affichée dans la démo. Le contrôleur plafonne à 8 REFRESH par tour pour éviter les boucles.

**Définition précise de la réinitialisation.** REFRESH remet à zéro l'**état récurrent des couches GDN** et vide la **fenêtre de l'attention locale** ; il ne touche ni à la convolution causale courte (filtre local de 4 octets), ni à la cadence d'écriture du Buffer (les frontières de 512 octets restent comptées depuis le début du ruban), ni aux slots. Cette définition est ce qui rend REFRESH **entraînable exactement comme il sert à l'inférence** : dans un ruban d'entraînement, la réinitialisation est un masque par position, réalisé en forçant la porte d'oubli du GDN à zéro (`log α = −10⁴`) et par un masque « même segment » dans l'attention locale, sans changer l'algorithme chunké ; l'équivalence « forward avec masque » = « forward, reset, forward » est testée.

En V1, la **position** des REFRESH est supervisée par une heuristique de longueur et par les épisodes synthétiques (§6.2), et la **note** par l'oracle : le mécanisme est appris et démontrable ; décider REFRESH d'après la saturation réelle de l'état est un sujet de thèse (§13).

### 4.9 Les Extraits (presse-papiers de lecture)

Une **cellule** est un couple (octets bruts, clé d'embedding). Le mécanisme est commun à deux magasins : les **Extraits**, remplis pendant la lecture (V1), et le **Lexique**, appris à l'entraînement (V2, §13). Le Buffer retient l'*intention* ; les Extraits retiennent le *texte exact*.

**Remplissage.** Pendant SCAN et READ, le contrôleur découpe le contenu lu en spans candidats : identifiants, nombres, chaînes entre guillemets, URL, lignes de code, et tout mot de 8 octets ou plus. Chaque span de longueur ≥ 8 devient une cellule dont la clé est la projection (d → 128) de l'état caché du sommet de pile à la fin du span. Plafond : 4 096 cellules, éviction des plus anciennes. Remise à zéro à chaque tour utilisateur, conservation à travers les REFRESH.

**Rappel.** À une position de génération, le modèle peut émettre RECALL. Une tête dédiée projette l'état caché en clé de requête (d → 128) ; la cellule la plus proche (produit scalaire sur ≤ 4 096 clés, négligeable) est sélectionnée si sa similarité dépasse un seuil de confiance, sinon le contrôleur annule le rappel et la génération reprend en octets. L'adresse de la cellule choisie est écrite dans le ruban (2 octets, §4.2) pour la trace et la reproductibilité, puis les octets de la cellule sont **réinjectés** dans le modèle en mode chunk (parallèle, rapide) pour que Mémoire et Buffer sachent ce qui a été dit.

**Ce que cela apporte.** Copie exacte des noms, nombres, identifiants et lignes lues dans les documents, là où les modèles sur octets de cette taille déforment les chaînes longues. Vitesse : un rappel émet 8 à 200 octets pour le prix d'une seule position autorégressive (l'adresse est écrite par le contrôleur, la réinjection est parallèle). Visibilité : la démo montre chaque rappel et sa source.

**Honnêteté sur la mémoire.** Les Extraits croissent avec ce qui est lu, côté hôte, hors de l'état récurrent. Plafonnés à 4 096 cellules, ils restent bornés ; on décrira RELIS comme « état récurrent constant plus un presse-papiers borné », pas comme une mémoire strictement constante.

**Ablation.** Le module est désactivable par configuration ; l'évaluation compare avec et sans (§9).

## 5. Données

Budget total ~2 Go d'octets vus pendant l'entraînement (pré-entraînement + ruban).

| Source | Usage | Volume cible |
|---|---|---|
| Wikipedia FR | pré-entraînement | ~600 Mo |
| FineWeb-2 (fr), sous-ensemble filtré qualité | pré-entraînement | ~400 Mo |
| The Stack (Python, JavaScript), filtré | pré-entraînement | ~300 Mo |
| Dialogues français publics : OpenAssistant oasst2 (fr, multi-tours), Aya (fr), French-Alpaca | ruban | ~150 Mo |
| Épisodes synthétiques de lecture (§6.2) | ruban | ~300 Mo (générés à la volée) |
| Rappel du corpus de pré-entraînement, en rubans-documents (20 % des séquences ruban) | ruban | pris dans les shards existants |
| Étiquettes du professeur : tours pertinents par question, résumé d'intention en une phrase | oracle, sonde | ~20 Mo |
| Jeux de test tenus à part (FR, code, épisodes) | évaluation | ~20 Mo |

Le professeur (Qwen2.5-1.5B-Instruct) tourne sur Colab en **étiquetage seulement** : sorties courtes, quelques heures de T4, pendant que Kaggle entraîne. Il ne génère pas de dialogues : 150 Mo de dialogues générés représenteraient ~40 M de tokens, hors de portée du budget Colab, pour un gain faible face aux jeux publics.

## 6. Entraînement

### 6.1 Étapes

1. **Pré-entraînement octets** (~60 % du compute). Prochain octet sur documents concaténés séparés par SEG + en-tête minimal, séquences de 4 096 à 8 192 octets, batch effectif ~0,5 M octets. Apprend le français, le code, la structure UTF-8, l'habitude des en-têtes.
2. **Entraînement ruban** (~35 %). Conversations et épisodes reformatés en rubans (§4.3) avec décisions fournies par l'oracle (§6.2). Chaque ruban est rembourré à un multiple de 512 octets (poids nul sur le rembourrage) et plusieurs rubans sont **empaquetés** en séquences fixes de 16 384 octets : formes constantes, graphe DDP statique, module d'écriture du Buffer toujours actif. Comme les débuts de ruban tombent sur des frontières de bloc, la Mémoire et les slots y sont remis à zéro par échantillon, exactement comme un nouveau tour à l'inférence. Chaque position porte quatre informations : octet, mode (ENCODE / SCAN / GENERATE), classe de poids de perte, drapeaux (réinitialisation Mémoire, réinitialisation slots). Les historiques trop longs sont tronqués côté ancien (la partie non lue n'existe pas dans le ruban puisque STOP tombe avant). L'entraînement reprend les poids du pré-entraînement avec un optimiseur neuf (taux 1e-4, warmup 5 %). Validation à chaque checkpoint : bits par octet, et **exactitude des décisions contre l'oracle** en forçage sur des épisodes tenus à part.
3. **Sonde Buffer** (~5 %). Modèle figé ; traducteur entraîné sur (slots, résumé du professeur).

### 6.2 L'oracle des décisions

Les codes de contrôle cibles sont **calculés**, jamais devinés :

- **Épisodes synthétiques** (source principale). Générateur procédural de conversations et de documents où l'information nécessaire est placée à des positions connues : faits injectés (« mon code postal est… » puis question 30 tours plus tard), suivi de variables, questions « qu'ai-je dit sur X », réponse présente dans un document joint parmi N (N de 1 à 50, tailles de 100 o à 200 ko), ou absente partout (le modèle doit dire qu'il ne sait pas après avoir tout lu). L'oracle en déduit exactement : SKIP pour les documents non pertinents (d'après l'en-tête : nom/type sans rapport), READ sinon ; STOP après le segment qui complète l'information ; NEXT si l'historique ne suffit pas ; CONT ailleurs.
- **Dialogues réels** (dialogues français publics). Le professeur indique, pour la dernière question, quels tours passés sont nécessaires (un appel par exemple). Défaut conservateur si l'étiquette est douteuse : tout lire (STOP au dernier segment).
- **REFRESH et NOTE**. Deux sources. (i) Épisodes synthétiques à deux informations : la réponse a besoin de deux faits ; le ruban lit jusqu'au premier, commence à répondre, puis émet `NOTE il manque <ce qui manque> REFRESH` et relit ; la note est templée par le générateur, qui sait ce qui manque. (ii) Heuristique de longueur pour les réponses longues : pour toute réponse dépassant L octets, L tiré dans [256, 1024], insérer `NOTE réponse longue, suite REFRESH` tous les L octets avec re-balayage complet. La variabilité évite l'apprentissage d'une longueur fixe et laisse le modèle émettre REFRESH à l'inférence.
- **Bruit**. 10 % des épisodes contiennent des en-têtes trompeurs (nom pertinent, contenu non) pour que SKIP ne repose pas uniquement sur le nom.
- **RECALL**. Plus longue correspondance : pendant la construction du ruban, le contrôleur remplit les Extraits exactement comme à l'inférence ; à chaque position de la réponse, si les k prochains octets (k ≥ 8) égalent une cellule présente, la cible devient RECALL + adresse de cette cellule, et les k octets sont réinjectés comme entrée à poids nul. La tête de rappel est entraînée par entropie croisée sur les cellules présentes dans l'épisode. Les épisodes synthétiques sont construits pour que la réponse cite des identifiants, nombres et lignes des documents lus.

### 6.3 Pertes et pondération

Entropie croisée sur tout le ruban, pondérée par type de position : octets de réponse et de note ×1 ; codes de contrôle ×5 (rares, critiques) ; octets balayés et octets de la requête ×0,1 (signal LM gratuit) ; en-têtes ×0,1 ; marqueurs insérés par le contrôleur, rembourrage, octets d'adresse et octets développés d'un rappel ×0 (jamais prédits par la tête d'octets). La tête de rappel a sa propre entropie croisée sur les cellules, pondérée ×1.

### 6.4 Stabilité en fp16 sur T4

fp16 avec loss scaling dynamique ; états récurrents, normalisations et softmax en fp32 ; RMSNorm partout ; clipping de gradient à 1,0 ; AdamW, warmup 2 000 pas, cosinus ; taux d'apprentissage 3e-4 (pré-entraînement) puis 1e-4 (ruban). Si les récurrences divergent malgré cela : bascule des opérations d'état en fp32 complet (coût ~1,5× accepté).

### 6.5 Sessions et reprise

Checkpoints (modèle, optimiseur, état du dataloader, RNG) toutes les 30 minutes vers le Hugging Face Hub ; reprise automatique au lancement d'une session Kaggle/Colab. Les notebooks lanceurs ne contiennent que la configuration ; le code vit dans le dépôt, installé par `pip install -e`.

### 6.6 Baseline

Un **Transformer sur octets** de même taille (~130 M), mêmes données, fenêtre 4 096, entraîné en pré-entraînement puis ruban. Si le budget manque, budget réduit à 50 % et signalé comme tel. Sans baseline, « hors du conventionnel » n'est qu'une affirmation.

## 7. Inférence

### 7.1 Contrôleur

Boucle explicite en Python autour du modèle :

```
encode(query)                        → Mémoire, Buffer
for channel in [history, docs]:
    for segment in channel (récent → ancien):
        feed(SEG + header + HDR); a = decide({READ, SKIP})
        if a == READ: stream(content, par chunks de 64)
        feed(DEC); a = decide({CONT, STOP, NEXT})
        if STOP: break → generate
        if NEXT: break → canal suivant
generate():
    loop: b = decide(texte valide ∪ {END, REFRESH, RECALL})
        END → fin ; REFRESH → reset Mémoire, encode(query + PART + partiel), rescan, continue
        RECALL → cellule = extraits.nearest(clé) ; si confiance < seuil : annuler, continuer en octets
                 sinon : écrire adresse (2 octets), réinjecter les octets de la cellule en mode chunk
```

Pendant READ, le contrôleur alimente aussi les Extraits (spans candidats, clés depuis l'état du sommet de pile). Chaque étape émet un **événement** (segment lu, sauté, STOP, NEXT, octet, RECALL avec sa source, REFRESH, END, snapshot du Buffer) consommé par le CLI et par l'interface.

### 7.2 Décodage contraint

Deux masques combinés à chaque position : le masque de phase (codes autorisés, §4.7) et un **automate UTF-8** qui interdit tout octet produisant une séquence invalide (par exemple un octet de tête après un octet de tête). Aucun caractère cassé n'est émis. RECALL n'est autorisé qu'à une frontière de caractère UTF-8 ; les octets d'adresse ne passent pas par la tête d'octets (ils sont écrits par le contrôleur d'après la cellule choisie) et l'automate reprend après la réinjection.

### 7.3 Coût et honnêteté

Le balayage traite les octets par chunks (débit élevé, ~10 k octets/s par T4 attendu). La génération est octet par octet à coût constant par octet : pas de cache qui grossit. L'historique est **re-balayé à chaque tour** puisque la lecture part du plus récent ; SKIP et STOP compensent, mais un historique de 100 ko lu intégralement coûte quelques secondes sur T4. Un cache d'états par segment est un travail V2 (§13). Les documents sont streamés depuis le disque et jamais chargés en entier en mémoire GPU.

## 8. Démo

- **CLI** : conversation, `/attach <fichier>` (n'importe quel nombre, n'importe quelle taille ; texte, markdown, code ; PDF converti en texte côté contrôleur), affichage des décisions.
- **Application Gradio** (lien partageable depuis Colab) montrant en direct : la liste des segments avec leur état (lu / sauté / non atteint), la position du STOP, la traduction du Buffer se formant pendant la lecture, les rappels d'Extraits surlignés dans la réponse avec un lien vers leur source, les REFRESH, un compteur « octets lus / octets écrits / % de l'historique économisé », et la réponse en streaming. Documents par glisser-déposer.

## 9. Évaluation

| Mesure | Ce qu'elle prouve |
|---|---|
| Bits par octet sur FR et code tenus à part, RELIS vs baseline | Qualité de modélisation à taille égale |
| Tâches aiguille, historique de 1 ko à 200 ko (au-delà des longueurs d'entraînement), exactitude vs longueur | Contexte illimité ; la baseline s'effondre au-delà de 4 ko |
| Décisions vs oracle : précision/rappel de STOP, SKIP, NEXT ; % d'octets économisés à exactitude égale | Calcul adaptatif réel |
| Documents : N de 1 à 200, taille jusqu'à 1 Mo, aiguille dans un seul | Illimité côté documents ; utilité de SKIP |
| Cent prompts FR jugés par le professeur, RELIS vs baseline (préférence aveugle) | Utilisabilité conversationnelle |
| REFRESH : cohérence des réponses longues avec et sans (ablation) | Le mécanisme sert |
| Extraits : exactitude des chaînes copiées (identifiants, nombres, URL) et octets/s de génération, avec et sans le module | Le presse-papiers corrige les copies et accélère |
| Sonde : accord humain sur 50 traductions du Buffer | Le Buffer porte bien l'intention |

## 10. Dépôt et modules

```
relis/
  tape/        codes de contrôle, en-têtes, construction et parsing des rubans, oracle
  model/
    gdn.py     Gated DeltaNet chunké PyTorch pur, état fp32
    swa.py     attention à fenêtre glissante 512
    buffer.py  slots : lecture par couche, module d'écriture par chunk
    extraits.py cellules (octets, clé), remplissage par spans, tête de rappel, seuil ; désactivable
    relis.py  assemblage, modes, API forward(bytes, mode, state, buffer)
    state.py   Mémoire : reset(), size_bytes(), to/from checkpoint
  data/        shards.py, dataset.py, prepare.py (pré-entraînement) ; dialogues.py (sources FR publiques),
               episodes.py (générateur synthétique + oracle), teacher.py (étiquetage Colab),
               pack.py (rubans rembourrés à 512, empaquetés en 16 384, quatre tableaux parallèles)
  train/       pretrain.py, tape_train.py, probe_train.py, reprise HF Hub
  infer/       controller.py (boucle, événements), constrain.py (UTF-8 + phases), cli.py, app.py
  eval/        bpb.py, needle.py, decisions.py, docs.py, judge.py
  baseline/    Transformer sur octets, même API que model/
  notebooks/   lanceurs Kaggle / Colab (configuration seulement)
  tests/
```

Chaque module se teste seul. Tests clés de la semaine 1 : équivalence numérique GDN chunké vs récurrent pas à pas ; mémoire GPU constante en lisant 1 M d'octets ; aller-retour ruban ↔ conversation ; oracle sur épisodes à réponse connue ; masque UTF-8 n'émettant jamais d'octet invalide.

Interfaces stables :

- `tape.build_turn(history, docs, query, oracle=None) -> (bytes, weights)` et `tape.parse(bytes)`.
- `RelisModel.forward(x, mode, state, buffer) -> (logits, state, buffer)` ; `State.reset()`.
- `Controller.turn(conversation, docs) -> Iterator[Event]`.
- La baseline expose la même API `forward`/`Controller` pour être évaluée par les mêmes scripts.

## 11. Plan sur quatre semaines

| Semaine | Livrables |
|---|---|
| S1 | GDN, SWA, Buffer, assemblage, tests d'équivalence et de mémoire constante ; pipeline de données ; **pré-entraînement lancé sur Kaggle dès le jour 4-5** |
| S2 | Rubans, en-têtes, oracle, générateur d'épisodes ; dialogues générés par le professeur sur Colab ; fin du pré-entraînement ; début de l'entraînement ruban ; lancement de la baseline |
| S3 | Contrôleur, décodage contraint, CLI ; évaluations aiguille, décisions, documents ; sonde Buffer ; **Extraits** (module, cibles RECALL dans les rubans, reprise courte de l'entraînement ruban avec le module actif) |
| S4 | Application Gradio, juge, ablation REFRESH, rapport ; marge pour les imprévus |

## 12. Risques

| Risque | Mitigation |
|---|---|
| Instabilité fp16 des récurrences | États et normalisations en fp32 ; fallback opérations d'état en fp32 complet |
| Débit du GDN PyTorch pur insuffisant | Algorithme chunké ; `torch.compile` ; fallback Triton (fonctionne sur T4, sm_75) |
| Sessions Kaggle interrompues | Checkpoints 30 min sur HF Hub, reprise automatique |
| Décisions mal apprises (STOP trop tôt, SKIP abusif) | Pondération ×5, bruit d'en-têtes, mesure vs oracle dès S3, curriculum (épisodes courts → longs) |
| Extrapolation au-delà des longueurs d'entraînement | Évaluée explicitement (§9) ; les récurrences extrapolent en général mieux qu'une attention positionnelle, sans garantie |
| Cohérence conversationnelle décevante | Attentes fixées (§2) ; la démo argumente par les mécanismes ; données d'instruction existantes plutôt que tout générer |
| Génération de données concurrençant l'entraînement | Professeur sur Colab, entraînement sur Kaggle |
| Rappel erroné produisant un bloc entier faux | Seuil de confiance avec repli en octets ; ablation ; module désactivable si les mesures §9 ne le justifient pas |
| Extraits en semaine 3 débordant sur la démo | Format RECALL réservé dès S1 ; le module est optionnel : s'il glisse, la V1 sort sans lui et il passe en V2 |

## 13. Hors périmètre V1 et feuille de route

**V2 (phase entreprise)** : **Lexique**, second magasin de cellules rappelables (jusqu'à 2^19 entrées) rempli à l'entraînement par sélection des spans qui compressent le mieux le corpus, rappelé par le même code RECALL et la même tête, éditable sans réentraînement (savoir séparé du calcul) ; hiérarchie octets → concepts (découpage dynamique appris, cœur sur concepts) ; cache d'états par segment pour ne pas re-balayer l'inchangé ; Buffer persistant entre les tours (mémoire de travail longue) ; canaux outils et exécution de code ; pré-ordonnancement des documents par pertinence probable pour rendre STOP plus tôt plus souvent ; quantification et serving.

**Sujets de thèse** : adressage sémantique des cellules (adresses prédites octet par octet via des codes issus d'une quantification résiduelle des clés, à la manière des identifiants sémantiques de la recherche générative), qui rendrait la tête de rappel superflue ; REFRESH décidé par saturation de l'état (mesure d'information résiduelle dans Mémoire + Buffer) ; apprentissage des décisions par renforcement contre le coût de lecture ; apprentissage continu du modèle par consolidation des Buffers ; sens de lecture choisi par le modèle ; comparaison formelle « lire pour répondre » vs attention globale en termes de complexité et de rappel.
