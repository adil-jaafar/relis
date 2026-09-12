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
    val_batches: int = 4                     # lots de validation (bits_per_byte, exactitude des décisions)
    hub_repo: str | None = None
    hub_every_minutes: float = 120.0        # le Hub coûte cher : bien plus rare que le local
    time_budget_hours: float | None = None  # arrêt propre avant la coupure Kaggle (12 h)
    warmup_frac: float | None = None        # si défini, warmup = warmup_frac × max_steps (prime sur warmup_steps)
    compile: bool = False                   # torch.compile sur le cœur GDN (opt-in, mesurer avec bench --compile)
    init_from: str | None = None            # poids seuls, si run_dir n'a pas de checkpoint
    decision_balance: float = 0.0           # exposant alpha du rééquilibrage des décisions rares ; 0 = désactivé

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


def _metric_label(batch) -> str:
    """`wce` (entropie croisée pondérée par position) si le lot porte des poids, sinon `bpb`."""
    return "wce" if _as_batch(batch).get("w") is not None else "bpb"


def _as_batch(b) -> dict:
    if isinstance(b, dict):
        return b
    x, y = b
    return {"x": x, "y": y}


def decision_multipliers(counts: dict, alpha: float) -> torch.Tensor:
    """Multiplicateur par octet `(256,)` float32 pour rééquilibrer les décisions rares.

    `alpha == 0` renvoie exactement un tenseur de 1 (désactivé). Sinon, pour chaque
    octet compté : `m_c = (N_tot / N_c) ** alpha`, puis normalisation par la moyenne
    pondérée par la population (`m_c /= (Σ_c N_c·m_c) / N_tot`) pour que la masse
    totale des décisions (`Σ_c N_c·m_c`) reste égale à `N_tot`. Les octets non
    comptés valent 1.0.
    """
    m = torch.ones(256, dtype=torch.float32)
    if alpha == 0.0 or not counts:
        return m
    n_tot = float(sum(counts.values()))
    for byte, n_c in counts.items():
        m[byte] = (n_tot / n_c) ** alpha
    weighted_mean = sum(n_c * m[byte].item() for byte, n_c in counts.items()) / n_tot
    for byte in counts:
        m[byte] = m[byte] / weighted_mean
    return m


def effective_weights(batch: dict, byte_mult, device) -> "torch.Tensor | None":
    """Poids par position effectivement utilisés pour la perte : `w` (poids ruban) si
    `byte_mult` est `None`, sinon `w * byte_mult[y]`. `None` si le lot n'a pas de `w`
    (ex. lot tuple de pré-entraînement) : `byte_mult` seul ne pondère rien.

    Point d'unicité pour `_loss` et `train` (calcul de `w_total`) : s'ils divergent,
    le taux d'apprentissage effectif change silencieusement.
    """
    b = _as_batch(batch)
    w = b.get("w")
    if w is None:
        return None
    w = w.to(device).reshape(-1).float()
    if byte_mult is None:
        return w
    y = b["y"].to(device).reshape(-1)
    return w * byte_mult.to(device)[y]


def _loss(model, batch, device, amp, norm: "torch.Tensor | float | None" = None, byte_mult=None):
    """`norm` : diviseur explicite (Σ w sur le lot effectif complet, sous accumulation de
    gradient) ; si None, normalise localement (moyenne, ou moyenne pondérée si `w` est présent).
    `byte_mult` : voir `effective_weights` — `None` laisse le comportement inchangé."""
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
    w = effective_weights(b, byte_mult, device)
    if w is None:
        return ce.sum() / norm if norm is not None else ce.mean()
    return (ce * w).sum() / norm if norm is not None else (ce * w).sum() / w.sum().clamp_min(1e-6)


def fetch_weights_path(init_from: str, local_dir: str = "hub_init") -> str:
    """Chemin local du `last.pt` de `init_from` ; ne télécharge que si nécessaire.

    Un fichier local est renvoyé tel quel. Sinon, si `local_dir/last.pt` existe déjà
    (téléchargé par le rang 0), il est renvoyé sans toucher au réseau : c'est ce qui
    permet aux rangs non maîtres de charger les poids après la barrière, sans que
    plusieurs processus écrivent le même fichier en même temps.
    """
    if os.path.isfile(init_from):
        return init_from
    cached = os.path.join(local_dir, "last.pt")
    if os.path.isfile(cached):
        return cached
    from huggingface_hub import hf_hub_download
    return hf_hub_download(init_from, "last.pt", token=os.environ.get("HF_TOKEN"), local_dir=local_dir)


def load_weights(model, path: str) -> int:
    """Charge les poids (seulement) depuis un `last.pt` local ; renvoie le pas source."""
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
    """Perte moyenne par position divisée par ln 2.

    Sur un jeu non pondéré c'est bien des bits par octet. Sur un jeu pondéré
    (rubans : `w` présent), `_loss` renvoie une entropie croisée **pondérée** par
    position : la valeur est alors une CE pondérée / ln 2, à lire comme `wce`, pas
    comme un bpb comparable au pré-entraînement. Le nom reste pour la compatibilité.
    """
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
          resume=True, on_step=None, extra_val=None, byte_mult=None) -> dict:
    """`extra_val` : `(model) -> dict | str` optionnel, appelé au(x) point(s) de
    sauvegarde et en fin de run ; le résultat est simplement imprimé (`f"[val] {…}"`),
    donc une chaîne déjà formatée (ex. `format_decision_report`) convient aussi bien
    qu'un dict brut.

    `byte_mult` : multiplicateur par octet `(256,)` optionnel (voir
    `decision_multipliers`), transmis tel quel à `effective_weights` dans `_loss` et
    dans le calcul de `w_total` ci-dessous — c'est essentiel : si les deux
    divergent, le taux d'apprentissage effectif change silencieusement."""
    assert tcfg.seq_len >= _unwrap(model).cfg.block, (
        "seq_len doit être ≥ block : sinon le module d'écriture du Buffer ne reçoit "
        "aucun gradient et DDP échoue")
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        if byte_mult is None:
            print(f"[balance] alpha={tcfg.decision_balance:.1f} (désactivé)")
        else:
            from relis.tape import codes as C
            items = sorted(((C.name(c), byte_mult[c].item()) for c in sorted(C.DECISION_CODES)),
                           key=lambda kv: -kv[1])
            mults = " ".join(f"{name} ×{v:.2f}" for name, v in items)
            print(f"[balance] alpha={tcfg.decision_balance:.1f} ; multiplicateurs : {mults}")
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
        # Un seul rang télécharge (écritures concurrentes dans hub_init/ sinon) ; la
        # barrière garantit que le fichier est complet avant que les autres le lisent.
        if is_main:
            fetch_weights_path(tcfg.init_from)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        src_step = load_weights(model, fetch_weights_path(tcfg.init_from))
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
            w = effective_weights(bd, byte_mult, device)
            w_total += float(w.sum().item()) if w is not None else float(bd["y"].numel())
        w_total = max(w_total, 1e-6)
        acc = 0.0
        for bb in batches:
            # norm=w_total : Σ w·ce sur le lot effectif complet (tous les micro-lots de
            # l'accumulation), pas une moyenne de moyennes locales.
            loss = _loss(model, bb, device, tcfg.amp, norm=w_total, byte_mult=byte_mult)
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
            unit = _metric_label(batches[0])
            print(f"step {step} loss {acc:.4f} {unit} {acc/math.log(2):.3f} lr {lr_at(step, tcfg):.2e}")
        if time.time() - last_ckpt > tcfg.ckpt_every_minutes * 60:
            _save(); last_ckpt = time.time()
    _save(final=True)
    val_bpb = bits_per_byte(_unwrap(model), val_ds, tcfg.seq_len, n_batches=tcfg.val_batches,
                            batch_size=max(1, tcfg.batch_size // 2), device=device) if is_main else float("nan")
    if is_main:
        unit = _metric_label(val_ds.sample(1, torch.Generator().manual_seed(0)))
        print(f"val {unit} {val_bpb:.3f}")
    if is_main and extra_val is not None:
        print(f"[val] {extra_val(_unwrap(model))}")
    return {"step": step, "last_loss": last_loss, "val_bpb": val_bpb,
            "stopped_by_budget": stopped_by_budget}
