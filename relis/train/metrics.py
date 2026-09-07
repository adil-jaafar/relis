"""Exactitude des décisions contre l'oracle (spec §6.1, §9), en forçage."""
import torch

from relis.tape import codes as C
from relis.tape.tape import WEIGHTS, W_HIGH
from .loop import _as_batch, _unwrap

_DECISIONS = sorted(C.DECISION_CODES)


@torch.no_grad()
def decision_accuracy(model, ds, n_batches: int, batch_size: int, device: str) -> dict:
    model.eval()
    g = torch.Generator().manual_seed(4321)
    correct, total = 0, 0
    per_code = {C.name(c): [0, 0] for c in _DECISIONS}
    for _ in range(n_batches):
        b = _as_batch(ds.sample(batch_size, g))
        x, y, w = b["x"].to(device), b["y"].to(device), b["w"].to(device)
        core = _unwrap(model)
        kw = {k: b[k].to(device) for k in ("reset", "slots_reset") if b.get(k) is not None}
        mode = b["mode"].to(device) if b.get("mode") is not None else 1
        logits, _ = model(x, mode, core.new_state(x.shape[0], device), **kw)
        pred = logits.argmax(-1)
        is_dec = torch.zeros_like(y, dtype=torch.bool)
        for c in _DECISIONS:
            is_dec |= y == c
        # positions de décision de l'oracle : classe de poids W_HIGH exactement
        # (un octet valant un code de décision peut apparaître dans du texte parcouru)
        is_dec &= (w - WEIGHTS[W_HIGH]).abs() < 1e-6
        for c in _DECISIONS:
            sel = is_dec & (y == c)
            n = int(sel.sum())
            if n:
                k = int((pred[sel] == c).sum())
                per_code[C.name(c)][0] += k; per_code[C.name(c)][1] += n
                correct += k; total += n
    model.train()
    return {"acc": correct / total if total else float("nan"), "n": total, "per_code": per_code}
