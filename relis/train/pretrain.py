"""Pré-entraînement octets (spec §6.1 étape 1).

    python -m relis.train.pretrain --config configs/pretrain_t4.yaml --run_dir runs/v1
    torchrun --nproc_per_node=2 -m relis.train.pretrain --config configs/pretrain_t4.yaml --run_dir runs/v1
"""
import argparse
import glob
import os

import torch
import yaml

from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.data.dataset import ByteWindows
from .loop import TrainConfig, train

# Racines où Kaggle / Colab montent les jeux de données attachés.
DEFAULT_SEARCH_ROOTS = ("/kaggle/input", "/content")


def resolve_data_path(path: str, search_roots=DEFAULT_SEARCH_ROOTS, tail: int = 1) -> str:
    """Renvoie `path` s'il existe ; sinon cherche un fichier du même nom sous les racines.

    Kaggle monte un dataset sous /kaggle/input/datasets/<utilisateur>/<slug>/…, un
    chemin qui dépend du compte : la configuration ne peut pas le connaître. Un
    seul fichier trouvé → on l'utilise (et on le dit) ; zéro ou plusieurs →
    erreur explicite avec la liste, pour que l'utilisateur passe --override.

    `tail=2` : recherche par les deux derniers segments (dossier parent + fichier), pour
    distinguer par exemple `train/meta.json` de `val/meta.json` (même nom de fichier).
    """
    if os.path.exists(path):
        return path
    name = os.path.basename(path)
    pattern_name = os.path.join(os.path.basename(os.path.dirname(path)), name) if tail == 2 else name
    found = []
    for root in search_roots:
        if os.path.isdir(root):
            found.extend(glob.glob(os.path.join(root, "**", pattern_name), recursive=True))
    found = sorted(set(found))
    if len(found) == 1:
        print(f"[data] {path} absent ; utilisation de {found[0]}")
        return found[0]
    if not found:
        raise FileNotFoundError(
            f"{path} introuvable et aucun '{pattern_name}' sous {list(search_roots)} ; "
            f"passer --override data.train_bin=… data.val_bin=…")
    raise FileNotFoundError(
        f"{path} introuvable et plusieurs '{pattern_name}' candidats : {found} ; "
        f"choisir avec --override data.train_bin=… data.val_bin=…")


def _apply_overrides(raw: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return raw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--find_unused_parameters", action="store_true",
                    help="DDP : tolère des paramètres sans gradient (coûteux)")
    ap.add_argument("--static_graph", action="store_true",
                    help="DDP : graphe constant d'un pas à l'autre (recommandé pour le pré-entraînement)")
    ap.add_argument("--override", action="extend", nargs="+", default=None,
                    help="ex. --override train.max_steps=10 train.batch_size=4")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw = _apply_overrides(yaml.safe_load(f), args.override or [])
    mcfg = RelisConfig(**raw["model"])
    tcfg = TrainConfig(**raw["train"])
    data = raw["data"]

    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = args.device
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        # device_id évite l'avertissement « barrier(): using the device under current context »
        torch.distributed.init_process_group("nccl", device_id=torch.device(device))

    model = RelisModel(mcfg).to(device)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"paramètres : {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()],
            find_unused_parameters=args.find_unused_parameters,
            static_graph=args.static_graph)

    train_ds = ByteWindows(resolve_data_path(data["train_bin"]), tcfg.seq_len)
    val_ds = ByteWindows(resolve_data_path(data["val_bin"]), tcfg.seq_len)
    out = train(model, tcfg, train_ds, val_ds, args.run_dir, device=device, resume=not args.no_resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(out)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
