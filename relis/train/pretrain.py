"""Pré-entraînement octets (spec §6.1 étape 1).

    python -m relis.train.pretrain --config configs/pretrain_t4.yaml --run_dir runs/v1
    torchrun --nproc_per_node=2 -m relis.train.pretrain --config configs/pretrain_t4.yaml --run_dir runs/v1
"""
import argparse
import os

import torch
import yaml

from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.data.dataset import ByteWindows
from .loop import TrainConfig, train


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
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    model = RelisModel(mcfg).to(device)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"paramètres : {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    train_ds = ByteWindows(data["train_bin"], tcfg.seq_len)
    val_ds = ByteWindows(data["val_bin"], tcfg.seq_len)
    out = train(model, tcfg, train_ds, val_ds, args.run_dir, device=device, resume=not args.no_resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(out)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
