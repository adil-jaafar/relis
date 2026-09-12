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
!python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --ask "Quelle est l'adresse du serveur ?"
```

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
autonomie n=200 | réponses justes 0.xxx | exact <k> trop_tot <k> trop_tard <k> doc_utile_saute <k> jamais_stop <k> | <b> octets lus en moyenne | <p> % économisés
```

Cinq classes s'excluent mutuellement et se répartissent tout l'échantillon :

- **`exact`** : le contrôleur s'est arrêté exactement là où l'oracle se serait arrêté.
- **`trop_tot`** : arrêt avant d'avoir lu le segment (ou le document) qui contient la réponse —
  le fait n'a jamais été lu, la réponse ne peut donc pas être juste, même si le modèle en invente
  une par hasard.
- **`trop_tard`** : le contrôleur a continué à lire (ou à ouvrir des documents) après avoir déjà
  tout ce qu'il fallait — la réponse peut être juste, mais du calcul a été gaspillé pour rien.
- **`doc_utile_saute`** : le document qui contenait la réponse a été sauté (ou jamais atteint).
- **`jamais_stop`** : le balayage n'a jamais émis STOP (a heurté un plafond de `Budget`).

`réponses justes` est indépendant de ces classes (jugé sur le texte généré, pas sur l'arrêt) ;
`octets lus en moyenne` et `% économisés` disent combien le contrôleur a lu par rapport à
l'historique et aux documents complets qu'on lui a donnés.

### `relis.eval.needle` — exactitude contre la longueur de l'historique

```
   taille  exactitude  octets lus  secondes
     1000       0.xxx        xxxx      x.xx
     4000       0.xxx        xxxx      x.xx
    16000       0.xxx        xxxx      x.xx
    64000       0.xxx        xxxx      x.xx
```

Une ligne par taille d'historique demandée (1 ko, 4 ko, 16 ko, 64 ko, …), le fait à retrouver
placé à une position aléatoire dans chacun. Ce qu'on cherche : **l'exactitude doit rester à peu
près plate quand la taille croît**. RELIS n'a pas de fenêtre — sa Mémoire est de taille constante
quel que soit ce qu'il a lu — donc rien dans son architecture ne devrait le faire décrocher à 64
ko. Un Transformer de même taille, lui, s'effondrerait dès que l'historique dépasse sa fenêtre
d'attention (spec §3) : c'est la comparaison implicite que ce tableau permet de faire. Une baisse
nette de l'exactitude avec la taille indiquerait que le mécanisme de Mémoire ne tient pas aussi
bien qu'annoncé sur de longs historiques, malgré l'absence de fenêtre.

### `relis.eval.adaptive` — le calcul suit-il la difficulté ?

```
[adaptatif] checkpoint au pas <n>
faciles <b1> octets lus (justes 0.xxx) · difficiles <b2> octets lus (justes 0.xxx) · rapport ×<r>
```

« faciles » = le fait est dans les trois derniers échanges, le contrôleur devrait pouvoir
s'arrêter presque tout de suite. « difficiles » = le fait est enfoui dans le tiers le plus ancien
de l'historique, ou caché dans un document parmi plusieurs. Ce qu'on cherche : le **rapport
`difficiles / faciles` doit être nettement supérieur à 1** — c'est la preuve chiffrée que le
modèle ne lit pas tout systématiquement, qu'il lit *en fonction de* la difficulté réelle du cas.
Un rapport proche de 1 voudrait dire que le contrôleur lit toujours à peu près la même quantité,
qu'il ait ou non besoin de creuser — le calcul adaptatif ne serait alors qu'apparent.

### `relis.infer.cli` — un essai de conversation

Sans `--ask`, la commande ouvre une boucle interactive ; avec `--ask`, elle répond une fois puis
quitte. C'est la même primitive que les trois évaluations ci-dessus (`Controller.turn`), mais lue
à l'œil : elle affiche les segments lus et sautés, la position du STOP, et la raison d'un
éventuel REFRESH, plutôt que de les agréger en statistiques.

---

*Chiffres réels : à remplir une fois ces commandes exécutées sur le checkpoint Kaggle
(`jaafar2022/relis-v1-tape`) — voir `docs/CARNET.md`. Les essais manuels sur CPU avec un modèle
minuscule non entraîné (`tests/test_eval.py`, `runs/cli_smoke/last.pt`) ne vérifient que
l'absence d'exception, jamais la qualité des réponses.*
