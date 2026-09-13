# Fichiers de démonstration

Un historique et deux documents écrits dans la forme que le modèle a vue à l'entraînement
(`relis/data/episodes.py`) : tours courts horodatés, documents d'une ligne dont le nom porte le sujet.

- `conv.json` : huit tours ; l'adresse du serveur est donnée au troisième.
- `reunion_12.txt` : document utile, il contient la salle de la réunion.
- `notes_3.txt` : document sans rapport, à sauter d'après son nom.

Depuis la racine du dépôt :

```
python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --conversation demo/conv.json     --ask "Quelle est l'adresse du serveur ?" --max_gen 128

python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --conversation demo/conv.json     --doc demo/notes_3.txt --doc demo/reunion_12.txt     --ask "Dans quelle salle a lieu la réunion ?" --max_gen 128
```

Résultat au pas 500 : la première commande fait ce qu'on attend, lecture des tours récents puis STOP
sur le tour qui contient l'adresse, documents jamais ouverts, réponse « L'adresse du serveur est
10.42.7.15. ».

La seconde **échoue de façon instructive** : le modèle s'arrête sur le même tour, celui de l'adresse du
serveur, et invente une salle. Aucun épisode d'entraînement ne contient de fait qui ne réponde pas à
la question ; le modèle a donc appris « s'arrêter sur une phrase en forme de fait » autant que
« s'arrêter sur le fait qui répond ». `python -m relis.eval.distractor --repo …` sépare les deux
hypothèses en quatre cas. Le remède est une famille d'épisodes à faits distracteurs (plan 2b).
Pour voir la lecture des documents fonctionner dès maintenant, enlever le tour de l'adresse du
serveur de `conv.json` : l'historique devient anodin, et le modèle passe aux documents.

Chaque appel ajoute la question et la réponse à `conv.json` ; `git checkout demo/conv.json` le remet
à l'état initial.

`conv_anodine.json` est cette conversation sans le tour de l'adresse **et sans « Parlons de la
réunion. »** : au pas 500, ce seul tour suffit à déclencher un STOP, le mot du sujet sans la réponse
étant lui aussi un appât jamais vu comme tel à l'entraînement ; le modèle répond alors « je ne trouve
pas cette information » sans ouvrir les documents. Avec un historique vraiment anodin, la sonde
`relis.eval.distractor` (cas C et D) montre que le passage aux documents, le saut sur le nom et la
lecture du bon document fonctionnent. La démo documents avec elle :

```
python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --conversation demo/conv_anodine.json \
    --doc demo/notes_3.txt --doc demo/reunion_12.txt \
    --ask "Dans quelle salle a lieu la réunion ?" --max_gen 128
```
