"""Entraînement ruban (spec §6.1 étape 2) : reprend les poids pré-entraînés.

    python -m relis.train.tape_train --config configs/tape_t4.yaml --run_dir runs/tape1
    torchrun --nproc_per_node=2 -m relis.train.tape_train --config configs/tape_t4.yaml --run_dir runs/tape1
"""
import argparse
import os
import sys

import torch
import yaml

from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.data.pack import TapeWindows, decision_counts
from .loop import TrainConfig, train, decision_multipliers
from .metrics import decision_accuracy, format_decision_report
from .pretrain import _apply_overrides, resolve_data_path


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--static_graph", action="store_true")
    ap.add_argument("--find_unused_parameters", action="store_true")
    ap.add_argument("--override", action="extend", nargs="+", default=None)
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
        torch.distributed.init_process_group("nccl", device_id=torch.device(device))

    model = RelisModel(mcfg).to(device)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"paramètres : {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()],
            find_unused_parameters=args.find_unused_parameters, static_graph=args.static_graph)

    train_dir = os.path.dirname(resolve_data_path(os.path.join(data["train_dir"], "meta.json"), tail=2))
    val_dir = os.path.dirname(resolve_data_path(os.path.join(data["val_dir"], "meta.json"), tail=2))
    train_ds, val_ds = TapeWindows(train_dir), TapeWindows(val_dir)
    for tag, ds in (("train", train_ds), ("val", val_ds)):
        assert ds.seq_len == tcfg.seq_len + 1, (
            f"shard {tag} : seq_len empaqueté {ds.seq_len} incompatible avec train.seq_len "
            f"{tcfg.seq_len} ; les rubans sont décalés d'un octet, il faut "
            f"seq_len empaqueté == train.seq_len + 1 (ici {tcfg.seq_len + 1})")
    byte_mult = None
    if tcfg.decision_balance > 0:
        counts = decision_counts(train_dir)
        byte_mult = decision_multipliers(counts, tcfg.decision_balance)

    extra = lambda m: format_decision_report(decision_accuracy(
        m, val_ds, n_batches=tcfg.val_batches, batch_size=max(1, tcfg.batch_size), device=device))
    out = train(model, tcfg, train_ds, val_ds, args.run_dir, device=device,
                resume=not args.no_resume, extra_val=extra, byte_mult=byte_mult)
    if int(os.environ.get("RANK", "0")) == 0:
        print(out)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
