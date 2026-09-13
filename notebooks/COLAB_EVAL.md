# Évaluer un checkpoint RELIS sur Colab (T4 ou CPU)

Trois cellules Colab, à exécuter dans l'ordre. Elles ne réentraînent rien : elles chargent
`last.pt` depuis le Hub (`--repo jaafar2022/relis-v1-tape`, le dépôt du run ruban sur Kaggle) et
font tourner les trois évaluations en autonomie du plan 3 (spec §9), plus un essai de
conversation à la main.

## Cellule 1 — installation

```python
!pip install -q -e . --no-deps
```

`--no-deps` évite de réinstaller `torch` (déjà présent sur Colab, souvent une version différente
de celle du `pyproject.toml`) ; les autres dépendances de RELIS sont légères.

## Cellule 2 — jeton Hugging Face

Ajouter `HF_TOKEN` dans *Secrets* (icône clé à gauche), cocher l'accès au notebook, puis :

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Un jeton en lecture suffit : ces cellules ne font que télécharger `last.pt`, elles ne poussent
rien sur le Hub.

## Cellule 3 — les trois évaluations, puis une conversation

```python
!python -m relis.eval.autonomy --repo jaafar2022/relis-v1-tape --n 200
!python -m relis.eval.needle --repo jaafar2022/relis-v1-tape --sizes 1000,4000,16000,64000 --n 20
!python -m relis.eval.adaptive --repo jaafar2022/relis-v1-tape --n 40
!python -m relis.eval.distractor --repo jaafar2022/relis-v1-tape
!python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --conversation demo/conv.json --ask "Quelle est l'adresse du serveur ?" --max_gen 128
!git checkout demo/conv.json
!python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --conversation demo/conv_anodine.json --doc demo/notes_3.txt --doc demo/reunion_12.txt --ask "Dans quelle salle a lieu la réunion ?" --max_gen 128
!git checkout demo/conv_anodine.json
```

Les deux appels de la CLI partent de `demo/conv.json`, un historique de huit tours dans la forme vue à
l'entraînement (voir `demo/README.md`) ; `git checkout` le remet à l'état initial après chaque appel,
puisque la CLI y ajoute la question et la réponse.

Sur GPU, ajouter `--device cuda` accélère nettement les deux premières évaluations (des centaines
de tours de scan). Sans `--device`, chaque script détecte seul `cuda` si disponible, sinon `cpu`.

Les trois évaluations acceptent `--max_gen` (256 par défaut) : c'est le plafond
`Budget.max_gen_bytes`, une garantie que si le checkpoint chargé n'émet jamais le code END, le
script rend quand même la main au lieu de tourner indéfiniment octet par octet.

---

## Comment lire chaque sortie

### `relis.eval.autonomy` — décisions en autonomie contre l'oracle

```
[autonomie] checkpoint au pas <n>
autonomie n=200 | réponses justes 0.xxx | exact <k> trop_tot <k> trop_tard <k> doc_utile_saute <k> jamais_stop <k> | <b> octets lus en moyenne | <p> % économisés | arrêts par budget : <n>
```

Cinq classes s'excluent mutuellement et se répartissent tout l'échantillon :

- **`exact`** : le contrôleur s'est arrêté exactement là où l'oracle se serait arrêté (pour un cas
  « absent » avec documents, cela veut dire : tous les documents visités).
- **`trop_tot`** : arrêt avant d'avoir lu le segment (ou le document) qui contient la réponse — ou,
  pour un cas « absent » avec documents, avant d'avoir visité tous les documents — le fait n'a
  jamais été lu (ou vérifié), la réponse ne peut donc pas être juste, même si le modèle en invente
  une par hasard.
- **`trop_tard`** : le contrôleur a continué à lire (ou à ouvrir des documents) après avoir déjà
  tout ce qu'il fallait — la réponse peut être juste, mais du calcul a été gaspillé pour rien.
- **`doc_utile_saute`** : le canal documents a été ouvert jusqu'au document qui contenait la
  réponse, et celui-ci a été explicitement sauté (SKIP). Un arrêt avant même d'atteindre ce
  document — dans l'historique, ou dans les documents mais avant le bon — est classé `trop_tot`,
  pas ici : rien n'a été sauté puisque le document n'a jamais été en vue.
- **`jamais_stop`** : le balayage n'a jamais émis STOP (a heurté un plafond de `Budget`).

`réponses justes` est indépendant de ces classes (jugé sur le texte généré, pas sur l'arrêt) ;
`octets lus en moyenne` et `% économisés` disent combien le contrôleur a lu par rapport à
l'historique et aux documents complets qu'on lui a donnés (`% économisés` ne regarde que la
première passe, avant toute relecture, pour ne jamais devenir artificiellement négatif). La
dernière valeur, **`arrêts par budget`**, compte les cas où c'est un plafond de `Budget` — pas le
modèle — qui a forcé l'arrêt : le budget de chaque cas est désormais dimensionné sur sa taille
réelle (spec §9, F1), donc ce compte devrait rester à 0 en pratique ; s'il ne l'est pas, **toutes
les autres colonnes de cette ligne sont suspectes** et le rapport le signale explicitement.

### `relis.eval.needle` — exactitude contre la longueur de l'historique

```
   taille  exactitude  octets lus  secondes  arrêts par budget
     1000       0.xxx        xxxx      x.xx  arrêts par budget : 0
     4000       0.xxx        xxxx      x.xx  arrêts par budget : 0
    16000       0.xxx        xxxx      x.xx  arrêts par budget : 0
    64000       0.xxx        xxxx      x.xx  arrêts par budget : 0
```

Une ligne par taille d'historique demandée (1 ko, 4 ko, 16 ko, 64 ko, …), le fait à retrouver
placé à une position aléatoire dans chacun. Ce qu'on cherche : **l'exactitude doit rester à peu
près plate quand la taille croît**. RELIS n'a pas de fenêtre — sa Mémoire est de taille constante
quel que soit ce qu'il a lu — donc rien dans son architecture ne devrait le faire décrocher à 64
ko, ni à 200 ko. Un Transformer de même taille, lui, s'effondrerait dès que l'historique dépasse
sa fenêtre d'attention (spec §3) : c'est la comparaison implicite que ce tableau permet de faire.

**Avant de conclure quoi que ce soit sur une baisse d'exactitude, regarder d'abord `arrêts par
budget` sur la ligne concernée** (spec §9, F1) : le budget de chaque cas est dimensionné sur sa
taille réelle (`Budget.max_segments`/`max_read_bytes` calculés au cas, pas la constante
d'entraînement), donc ce compte devrait rester à 0 même à 200 ko ; s'il ne l'est pas, la baisse
d'exactitude sur cette ligne vient du plafond, pas de la Mémoire, et la ligne le dit explicitement
(« chiffre suspect »). Ce n'est qu'une fois ce compte à 0 partout qu'une baisse nette de
l'exactitude avec la taille indiquerait vraiment que le mécanisme de Mémoire ne tient pas aussi
bien qu'annoncé sur de longs historiques, malgré l'absence de fenêtre.

### `relis.eval.adaptive` — le calcul suit-il la difficulté ?

```
[adaptatif] checkpoint au pas <n>
faciles <b1> octets lus (justes 0.xxx) · difficiles <b2> octets lus (justes 0.xxx) · rapport ×<r> · arrêts par budget : <n>
```

« faciles » = le fait est dans les trois derniers échanges, le contrôleur devrait pouvoir
s'arrêter presque tout de suite. « difficiles » = le fait est enfoui dans le tiers le plus ancien
de l'historique, ou caché dans un document parmi plusieurs. Ce qu'on cherche : le **rapport
`difficiles / faciles` doit être nettement supérieur à 1** — c'est la preuve chiffrée que le
modèle ne lit pas tout systématiquement, qu'il lit *en fonction de* la difficulté réelle du cas.
Un rapport proche de 1 voudrait dire que le contrôleur lit toujours à peu près la même quantité,
qu'il ait ou non besoin de creuser — le calcul adaptatif ne serait alors qu'apparent. Cas
dégénéré à connaître : si « faciles » lit 0 octet en moyenne, `rapport` n'est plus un vrai ratio
mais le compte brut de `difficiles` (division protégée par un minimum de 1). Comme pour les deux
autres évaluations, vérifier `arrêts par budget` avant de faire confiance aux octets lus : non
nul, il signale que le budget — pas la difficulté du cas — a tranché l'arrêt.

### `relis.infer.cli` — un essai de conversation

Sans `--ask`, la commande ouvre une boucle interactive ; avec `--ask`, elle répond une fois puis
quitte. C'est la même primitive que les trois évaluations ci-dessus (`Controller.turn`), mais lue
à l'œil : elle affiche les segments lus et sautés, la position du STOP, et la raison d'un
éventuel REFRESH, plutôt que de les agréger en statistiques.

**Hors distribution à connaître (spec §9, F6)** : les épisodes d'entraînement comportent toujours
au moins un tour d'historique. La commande d'exemple ci-dessus (`--ask` en tout premier appel,
sans `--conversation` existant) fait donc voir au modèle un premier tour de conversation — avec ou
sans `--doc` joint — qu'il n'a jamais vu à l'entraînement dans cette position. Pour une
démonstration représentative de ce que le modèle a réellement appris, poser d'abord une question
anodine (en passant `--conversation conv.json` pour qu'elle soit enregistrée), puis poser la vraie
question dans un second appel qui relira ce fichier.

---

*Chiffres réels : à remplir une fois ces commandes exécutées sur le checkpoint Kaggle
(`jaafar2022/relis-v1-tape`) — voir `docs/CARNET.md`. Les essais manuels sur CPU avec un modèle
minuscule non entraîné (`tests/test_eval.py`, `runs/cli_smoke/last.pt`) ne vérifient que
l'absence d'exception, jamais la qualité des réponses.*
