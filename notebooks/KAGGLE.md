# Lancer le pré-entraînement sur Kaggle (2×T4) ou Colab (T4)

## Une fois : préparer les shards (Colab ou machine locale, CPU suffit)

```bash
pip install -e .
python -m relis.data.prepare --out shards/v1 --wiki_fr 600e6 --fineweb2_fr 400e6 --code 300e6
```
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
    --config configs/pretrain_t4.yaml --run_dir /kaggle/working/runs/v1 \
    --override train.hub_repo=<utilisateur>/relis-v1-pretrain
```

À chaque nouvelle session (Kaggle coupe à 12 h), relancer la même cellule : `train()` récupère `last.pt` depuis le Hub et reprend au pas sauvegardé.

## Calibrer `max_steps`
`bench` imprime le débit mesuré. Fixer `train.max_steps` pour viser ~2 Go d'octets vus au total, soit
`max_steps = 2e9 / (2 GPU × batch_size × grad_accum × seq_len)`. Avec la config par défaut (2 × 8 × 8 × 4096 = 524 288 octets/pas) cela donne ≈ 3 800 pas.

## Colab (1×T4)
Même cellule sans `torchrun` : `python -m relis.train.pretrain ...`, et `--override train.grad_accum=16` pour conserver le batch effectif.

## Si le débit est trop bas (< 5 000 octets/s/GPU)
1. `--override train.seq_len=2048 train.grad_accum=16`.
2. Vérifier que `amp: true` est bien actif (`torch.cuda.is_available()`).
3. Envelopper `gdn_chunked` dans `torch.compile` (option `compile: true` à ajouter dans la boucle si nécessaire).
