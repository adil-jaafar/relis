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
    warmup_frac: float | None = None        # si défini, warmup = warmup_frac × max_steps (prime sur warmup_steps)
    compile: bool = False                   # torch.compile sur le cœur GDN (opt-in, mesurer avec bench --compile)
    init_from: str | None = None            # poids seuls, si run_dir n'a pas de checkpoint

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw.get("train", {}))


def effective_warmup(cfg: TrainConfig) -> int:
    """Nombre de pas de warmup : fraction du run si warmup_frac est défini, sinon warmup_steps."""
    if cfg.warmup_frac is not None:
        return max(1, int(round(cfg.warmup_frac * cfg.max_steps)))
    return cfg.warmup_steps


def lr_at(step: int, cfg: TrainConfig) -> float:
    warmup = effective_warmup(cfg)
    if step < warmup:
        return cfg.lr * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, cfg.max_steps - warmup))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def enable_compile() -> bool:
    """Remplace relis.model.gdn.gdn_chunked par sa version torch.compile.

    Le cœur GDN est une fonction pure de tenseurs : c'est la seule cible sûre et
    utile (la boucle par blocs et l'État mutable ne se compilent pas bien). Les
    formes varient peu (pièces de `block` positions, plus une queue), donc peu de
    recompilations. Renvoie False, sans rien changer, si la compilation est
    indisponible (pas de compilateur C++, GPU non supporté, etc.).
    """
    import relis.model.gdn as gdn_mod
    original = gdn_mod.gdn_chunked
    try:
        compiled = torch.compile(original, dynamic=None)
    except Exception as e:  # environnement sans compilateur : on continue en eager
        print(f"[compile] indisponible, exécution eager : {e}")
        return False
    gdn_mod.gdn_chunked = compiled
    print("[compile] gdn_chunked compilé (la première itération sera lente)")
    return True


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _as_batch(b) -> dict:
    if isinstance(b, dict):
        return b
    x, y = b
    return {"x": x, "y": y}


def _loss(model, batch, device, amp, norm: "torch.Tensor | float | None" = None):
    """`norm` : diviseur explicite (Σ w sur le lot effectif complet, sous accumulation de
    gradient) ; si None, normalise localement (moyenne, ou moyenne pondérée si `w` est présent)."""
    b = _as_batch(batch)
    x, y = b["x"].to(device), b["y"].to(device)
    core = _unwrap(model)
    state = core.new_state(x.shape[0], device)
    kw = {}
    if b.get("reset") is not None:
        kw["reset"] = b["reset"].to(device)
    if b.get("slots_reset") is not None:
        kw["slots_reset"] = b["slots_reset"].to(device)
    mode = b["mode"].to(device) if b.get("mode") is not None else 1      # Mode.SCAN par défaut
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.float16, enabled=amp and device.startswith("cuda")):
        logits, _ = model(x, mode, state, **kw)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    if b.get("w") is None:
        return ce.sum() / norm if norm is not None else ce.mean()
    w = b["w"].to(device).reshape(-1).float()
    return (ce * w).sum() / norm if norm is not None else (ce * w).sum() / w.sum().clamp_min(1e-6)


def load_weights(model, init_from: str) -> int:
    """Charge les poids (seulement) depuis un last.pt local ou un dépôt Hub ; renvoie le pas source."""
    path = init_from
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(init_from, "last.pt", token=os.environ.get("HF_TOKEN"), local_dir="hub_init")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap(model).load_state_dict(payload["model"], strict=True)
    return int(payload.get("step", 0))


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
        b = ds.sample(batch_size, g)
        loss = _loss(model, b, device, amp=False)
        y = _as_batch(b)["y"]
        total += loss.item() * y.numel(); count += y.numel()
    model.train()
    return total / count / math.log(2)


def train(model, tcfg: TrainConfig, train_ds, val_ds, run_dir, device="cuda",
          resume=True, on_step=None, extra_val=None) -> dict:
    assert tcfg.seq_len >= _unwrap(model).cfg.block, (
        "seq_len doit être ≥ block : sinon le module d'écriture du Buffer ne reçoit "
        "aucun gradient et DDP échoue")
    is_main = int(os.environ.get("RANK", "0")) == 0
    t_start = time.time()
    if tcfg.compile:
        enable_compile()
    model.to(device).train()
    opt = _make_optimizer(model, tcfg)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=tcfg.amp and device.startswith("cuda"))
    step = 0
    if resume:
        if tcfg.hub_repo and is_main and not os.path.exists(os.path.join(run_dir, "last.pt")):
            pull_from_hub(run_dir, tcfg.hub_repo)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        info = load_checkpoint(run_dir, _unwrap(model), opt, scaler)
        if info is not None:
            step = info["step"]
    if step == 0 and tcfg.init_from:
        src_step = load_weights(model, tcfg.init_from)
        if is_main:
            print(f"[init_from] poids chargés depuis {tcfg.init_from} (pas source {src_step})")
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
        if extra_val is not None and not final:
            print(f"[val] {extra_val(_unwrap(model))}")
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
        batches = [train_ds.sample(tcfg.batch_size, g) for _ in range(tcfg.grad_accum)]
        w_total = 0.0
        for bb in batches:
            bd = _as_batch(bb)
            w_total += float(bd["w"].sum().item()) if bd.get("w") is not None else float(bd["y"].numel())
        w_total = max(w_total, 1e-6)
        acc = 0.0
        for bb in batches:
            # norm=w_total : Σ w·ce sur le lot effectif complet (tous les micro-lots de
            # l'accumulation), pas une moyenne de moyennes locales.
            loss = _loss(model, bb, device, tcfg.amp, norm=w_total)
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
    if is_main and extra_val is not None:
        print(f"[val] {extra_val(_unwrap(model))}")
    return {"step": step, "last_loss": last_loss, "val_bpb": val_bpb,
            "stopped_by_budget": stopped_by_budget}
