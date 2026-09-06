# Lancer le pré-entraînement sur Kaggle (2×T4) ou Colab (T4)

## Une fois : préparer les shards (Colab ou machine locale, CPU suffit)

```bash
pip install -e .
huggingface-cli login          # ou : export HF_TOKEN=...
python -m relis.data.prepare --out shards/v1 --wiki_fr 600e6 --fineweb2_fr 400e6 --code 300e6
```
`bigcode/the-stack-smol-xl` (la part `--code`) est un jeu **contrôlé** : il faut accepter ses
conditions sur le Hub puis être authentifié — `huggingface-cli login` ou `HF_TOKEN` dans
l'environnement — **dès l'étape de préparation**, pas seulement pour les checkpoints.

Téléverser `shards/v1/train.bin` et `shards/v1/val.bin` comme **Kaggle Dataset** nommé `relis-shards`.

## Secrets Kaggle
Ajouter `HF_TOKEN` (jeton Hugging Face en écriture) dans *Add-ons → Secrets* et cocher le notebook.

## Cellule unique du notebook Kaggle (accélérateur : GPU T4 ×2)

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!git clone https://github.com/<utilisateur>/relis.git && cd relis && pip install -q -e .
%cd relis
!python -m relis.train.bench --config configs/pretrain_t4.yaml
!torchrun --nproc_per_node=2 -m relis.train.pretrain \
    --config configs/pretrain_t4.yaml --run_dir /kaggle/working/runs/v1 --static_graph \
    --override train.hub_repo=<utilisateur>/relis-v1-pretrain
```

Si `bench` a dû réduire le batch (message `[bench] batch réduit ...`), **recopier telle quelle** la
ligne `--override train.batch_size=… train.grad_accum=…` qu'il imprime et l'ajouter au `torchrun` :
elle conserve le batch effectif (`batch_size × grad_accum` constant).

`--static_graph` est recommandé pour le pré-entraînement (graphe DDP constant d'un pas à l'autre,
allreduce plus efficace) ; `--find_unused_parameters` ne sert qu'en cas d'erreur DDP sur un
paramètre sans gradient.

À chaque nouvelle session (Kaggle coupe à 12 h), relancer la même cellule : `train()` récupère
`last.pt` depuis le Hub et reprend au pas sauvegardé. `time_budget_hours: 11.5` fait sortir la
boucle proprement (sauvegarde locale + push) avant la coupure, et `hub_every_minutes: 120` évite de
saturer le quota LFS du dépôt — le checkpoint local, lui, reste écrit toutes les 30 min.

## Mémoire : `grad_checkpoint`
`configs/pretrain_t4.yaml` active `grad_checkpoint: true` (section `model`). Sans lui, un batch de
8 × 4096 demande plusieurs dizaines de Go d'activations, hors de portée des 15 Go d'un T4. Le
checkpointing recalcule chaque bloc à la rétropropagation : compter **~25-35 % de calcul en plus**
contre une mémoire d'activations réduite d'un ordre de grandeur.

## Calibrer `max_steps`
`train.max_steps` se fixe **à partir du débit mesuré par `bench`**, jamais d'un chiffre théorique :
`max_steps = octets_visés / (n_GPU × batch_size × grad_accum × seq_len)`.

Attente réaliste sur 2×T4 en mode eager (PyTorch pur, `grad_checkpoint` actif) : **6 000 à
10 000 octets/s par GPU**, soit environ **1 Go d'octets vus dans les 30 h de quota GPU hebdomadaire**
de Kaggle ; le reste du budget d'octets passe sur la semaine 2 — cohérent avec le plan §11 de la
spec, où le pré-entraînement s'étale sur S1-S2.

## Colab (1×T4)
Même cellule sans `torchrun` : `python -m relis.train.pretrain ...`, et `--override train.grad_accum=16` pour conserver le batch effectif.

## Si le débit est trop bas (< 5 000 octets/s/GPU)
1. `--override train.seq_len=2048 train.grad_accum=16`.
2. Vérifier que `amp: true` est bien actif (`torch.cuda.is_available()`).
3. Envelopper `gdn_chunked` dans `torch.compile` (option `compile: true` à ajouter dans la boucle si nécessaire).
