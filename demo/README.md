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

Attendu : première commande, lecture des tours récents puis STOP sur le tour qui contient l'adresse,
sans ouvrir les documents. Seconde commande, l'historique ne contient pas la salle, donc lecture
jusqu'au bout, NEXT vers les documents, `notes_3.txt` sauté sur son nom, `reunion_12.txt` lu, STOP.

Chaque appel ajoute la question et la réponse à `conv.json` ; `git checkout demo/conv.json` le remet
à l'état initial.
