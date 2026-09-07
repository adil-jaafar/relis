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
Kaggle le monte sous `/kaggle/input/datasets/<utilisateur>/relis-shards/` : ce chemin dépend du
compte, `pretrain.py` le retrouve seul (recherche du nom de fichier sous `/kaggle/input`) tant qu'il
n'y a qu'un seul `train.bin` attaché. Pour vérifier : `!find /kaggle/input -name "*.bin"`.

## Secrets Kaggle
Ajouter `HF_TOKEN` (jeton Hugging Face en écriture) dans *Add-ons → Secrets* et cocher le notebook.

## Cellule unique du notebook Kaggle (accélérateur : GPU T4 ×2)

Cette cellule est **idempotente** : elle se relance telle quelle à chaque session, que le clone existe
déjà ou non.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
if not os.path.exists("/kaggle/working/relis"):
    !git clone https://github.com/<utilisateur>/relis.git /kaggle/working/relis
%cd /kaggle/working/relis
!git pull -q && pip install -q -e .
!python -m relis.train.bench --config configs/pretrain_t4.yaml
!torchrun --nproc_per_node=2 -m relis.train.pretrain \
    --config configs/pretrain_t4.yaml --run_dir /kaggle/working/runs/v1 --static_graph \
    --override train.hub_repo=<utilisateur>/relis-v1-pretrain
```

Le `bench` ne sert qu'à la première session (il coûte ~2 min) ; le retirer ensuite.

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
**Garder les mêmes `--override` à chaque relance** : le planning du taux d'apprentissage est
recalculé depuis la ligne de commande.

## Mesures réelles (première session, 6 septembre 2026, 2×T4, `grad_checkpoint: true`)

| Mesure | Valeur |
|---|---|
| Mémoire GPU max, batch 8 × 4096 | 7,0 Go (aucune réduction de batch nécessaire) |
| Débit `bench`, mode eager | 4 700 octets/s par GPU |
| Octets par pas (2 GPU × 8 × 8 × 4096) | 524 288 |
| Durée d'un pas | ≈ 56 s |
| Perte au pas 40 (warmup 10 %) | 2,98, soit 4,3 bits/octet |

Au débit mesuré, `max_steps: 2500` (≈ 1,3 Go vus) prend ≈ 39 h, soit **trois sessions Kaggle**
et un peu plus d'une semaine de quota (30 h GPU/semaine).

## Warmup
`warmup_frac: 0.1` fixe le warmup à 10 % du run, quel que soit `max_steps`. Ne pas repasser à un
`warmup_steps` fixe sans vérifier qu'il reste petit devant `max_steps` : avec 2 000 pas de warmup
sur un run de 2 500, le modèle passerait 80 % du temps à chauffer.

## Mémoire : `grad_checkpoint`
`configs/pretrain_t4.yaml` active `grad_checkpoint: true` (section `model`). Sans lui, un batch de
8 × 4096 demande plusieurs dizaines de Go d'activations, hors de portée des 15 Go d'un T4. Le
checkpointing recalcule chaque bloc à la rétropropagation : compter **~25-35 % de calcul en plus**
contre une mémoire d'activations réduite d'un ordre de grandeur. Mesuré : 7,0 Go à batch 8.

## Calibrer `max_steps`
`train.max_steps` se fixe **à partir du débit mesuré par `bench`**, jamais d'un chiffre théorique :
`max_steps = octets_visés / (n_GPU × batch_size × grad_accum × seq_len)`.

## Colab (1×T4)
Même cellule sans `torchrun` : `python -m relis.train.pretrain ...`, et `--override train.grad_accum=16` pour conserver le batch effectif.

## Gagner du débit : `torch.compile` sur le cœur GDN
Le goulot en mode eager est le nombre de lancements de kernels Python (boucle sur les chunks du
Gated DeltaNet). Le cœur `gdn_chunked` est une fonction pure de tenseurs et se compile bien.
Mesurer d'abord, sans toucher au run en cours :

```python
!python -m relis.train.bench --config configs/pretrain_t4.yaml --compile
```

La première itération inclut la compilation (plusieurs minutes possibles) ; le débit affiché est
donc prudent. Si le gain est net (> 20 %), ajouter `train.compile=true` aux `--override` du
`torchrun`. Si la compilation échoue (Triton indisponible), la boucle l'annonce et continue en eager :
aucun risque pour le run.

## Si le débit est trop bas (< 5 000 octets/s/GPU)
1. `bench --compile` puis `train.compile=true` (voir ci-dessus).
2. `--override train.seq_len=2048 train.grad_accum=16`.
3. Vérifier que `amp: true` est bien actif (`torch.cuda.is_available()`).

## Entraînement ruban (plan 2a)

### Une fois : construire les rubans (CPU suffit, ~10 min pour 20 000 épisodes)
```bash
python -m relis.data.pack --out shards/tapes --episodes 20000 --seed 0 --seq_len 16384 --block 512 \
    --replay shards/v1/train.bin --replay_frac 0.2 --val_frac 0.02
```
Téléverser `shards/tapes/` (dossiers `train/` et `val/`, quatre `.bin` + `meta.json` chacun) comme Kaggle
Dataset `relis-tapes`.

### Cellule Kaggle
```python
%cd /kaggle/working/relis
!git pull -q && pip install -q -e .
!torchrun --nproc_per_node=2 -m relis.train.tape_train \
    --config configs/tape_t4.yaml --run_dir /kaggle/working/runs/tape1 --static_graph \
    --override train.hub_repo=<utilisateur>/relis-v1-tape train.init_from=<utilisateur>/relis-v1-pretrain \
               data.train_dir=/kaggle/input/datasets/<utilisateur>/relis-tapes/train \
               data.val_dir=/kaggle/input/datasets/<utilisateur>/relis-tapes/val
```
`init_from` ne sert qu'au premier lancement (poids du pré-entraînement) ; les sessions suivantes reprennent
depuis le Hub `relis-v1-tape`. À chaque checkpoint, la ligne `[val] {...}` donne l'exactitude des décisions
contre l'oracle, globale et par code (READ/SKIP/CONT/STOP/NEXT/END/REFRESH/NOTE) : c'est la métrique qui dit
si RELIS apprend à *décider*, indépendamment de la perte.
