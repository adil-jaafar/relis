"""Boucle d'entraînement (spec §6.4, §6.5) : AMP fp16 avec loss scaling, AdamW,
warmup + cosinus, clipping à 1.0, checkpoints périodiques, reprise."""
import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import yaml

from .checkpoint import save_checkpoint, load_checkpoint, push_to_hub, pull_from_hub


@dataclass
class TrainConfig:
    seq_len: int = 4096
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 3e-4
    warmup_steps: int = 2000
    max_steps: int = 4000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_every_minutes: float = 30.0
    log_every: int = 20
    hub_repo: str | None = None
    hub_every_minutes: float = 120.0        # le Hub coûte cher : bien plus rare que le local
    time_budget_hours: float | None = None  # arrêt propre avant la coupure Kaggle (12 h)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw.get("train", {}))


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _loss(model, x, y, device, amp):
    x, y = x.to(device), y.to(device)
    core = _unwrap(model)
    state = core.new_state(x.shape[0], device)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.float16, enabled=amp and device.startswith("cuda")):
        logits, _ = model(x, 1, state)          # Mode.SCAN pour le pré-entraînement
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


def _make_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "slots" not in n else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def bits_per_byte(model, ds, seq_len, n_batches, batch_size, device) -> float:
    model.eval()
    g = torch.Generator().manual_seed(1234)
    total, count = 0.0, 0
    for _ in range(n_batches):
        x, y = ds.sample(batch_size, g)
        loss = _loss(model, x, y, device, amp=False)
        total += loss.item() * y.numel(); count += y.numel()
    model.train()
    return total / count / math.log(2)


def train(model, tcfg: TrainConfig, train_ds, val_ds, run_dir, device="cuda",
          resume=True, on_step=None) -> dict:
    is_main = int(os.environ.get("RANK", "0")) == 0
    t_start = time.time()
    model.to(device).train()
    opt = _make_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.amp and device.startswith("cuda"))
    step = 0
    if resume:
        if tcfg.hub_repo and is_main and not os.path.exists(os.path.join(run_dir, "last.pt")):
            pull_from_hub(run_dir, tcfg.hub_repo)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        info = load_checkpoint(run_dir, _unwrap(model), opt, scaler)
        if info is not None:
            step = info["step"]
    g = torch.Generator().manual_seed(1000 + step + int(os.environ.get("RANK", "0")))
    last_ckpt = time.time()
    last_hub = time.time()
    last_loss = float("nan")
    stopped_by_budget = False
    cfg_dict = _unwrap(model).cfg.to_dict()

    def _save(final: bool = False):
        """Sauvegarde locale ; push Hub seulement à la cadence hub_every_minutes ou en fin de run."""
        nonlocal last_hub
        if not is_main:
            return
        save_checkpoint(run_dir, _unwrap(model), opt, scaler, step, cfg_dict, {"train": asdict(tcfg)})
        if not tcfg.hub_repo:
            return
        if not (final or time.time() - last_hub > tcfg.hub_every_minutes * 60):
            return
        try:
            push_to_hub(run_dir, tcfg.hub_repo)
        except Exception as e:  # le Hub ne doit jamais tuer l'entraînement
            print(f"[hub] push échoué : {e}")
        last_hub = time.time()

    while step < tcfg.max_steps:
        if tcfg.time_budget_hours is not None and time.time() - t_start > tcfg.time_budget_hours * 3600:
            stopped_by_budget = True
            if is_main:
                print(f"[budget] arrêt propre après {step} pas")
            break
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, tcfg)
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = train_ds.sample(tcfg.batch_size, g)
            loss = _loss(model, x, y, device, tcfg.amp) / tcfg.grad_accum
            scaler.scale(loss).backward()
            acc += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        last_loss = acc
        if on_step is not None:
            on_step(step, acc)
        if is_main and step % tcfg.log_every == 0:
            print(f"step {step} loss {acc:.4f} bpb {acc/math.log(2):.3f} lr {lr_at(step, tcfg):.2e}")
        if time.time() - last_ckpt > tcfg.ckpt_every_minutes * 60:
            _save(); last_ckpt = time.time()
    _save(final=True)
    val_bpb = bits_per_byte(_unwrap(model), val_ds, tcfg.seq_len, n_batches=4,
                            batch_size=max(1, tcfg.batch_size // 2), device=device) if is_main else float("nan")
    if is_main:
        print(f"val bpb {val_bpb:.3f}")
    return {"step": step, "last_loss": last_loss, "val_bpb": val_bpb,
            "stopped_by_budget": stopped_by_budget}
