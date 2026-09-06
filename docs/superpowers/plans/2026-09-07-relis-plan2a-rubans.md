# RELIS — Plan 2a : rubans, oracle, épisodes synthétiques et entraînement ruban

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apprendre à RELIS à *lire pour répondre* : construire des rubans complets (requête, balayage avec décisions, génération, REFRESH avec NOTE) à partir d'épisodes synthétiques dont l'oracle connaît les bonnes décisions, les empaqueter en séquences fixes, et poursuivre l'entraînement des poids pré-entraînés sur ces rubans avec une perte pondérée et une métrique d'exactitude des décisions.

**Architecture:** Le modèle reçoit deux masques par position : `reset` (REFRESH exact : porte d'oubli du GDN forcée à zéro, masque de segment dans l'attention locale) et `slots_reset` (nouveau ruban, sur une frontière de bloc). Le module `tape` transforme une spécification de ruban (`TapeSpec`) en quatre tableaux parallèles (octets, mode, classe de poids, réinitialisation). Le générateur d'épisodes produit des `TapeSpec` dont les décisions sont calculées par un oracle exact. `pack` rembourre chaque ruban à un multiple de 512 et en empaquette plusieurs dans des séquences de 16 384 octets. La boucle d'entraînement existante est généralisée (perte pondérée, masques, initialisation depuis un checkpoint) et `tape_train.py` l'utilise.

**Tech Stack:** Python ≥ 3.10, PyTorch ≥ 2.2, numpy, pytest ; code existant du plan 1 (`relis/model`, `relis/train/loop.py`, `relis/data/shards.py`).

**Spec:** `docs/superpowers/specs/2026-09-06-relis-design.md` — §3 (décisions plan 2), §4.2 (NOTE), §4.3 (ruban), §4.6 (slots), §4.7, §4.8 (REFRESH exact et NOTE), §6.1 étape 2, §6.2 (oracle, épisodes, REFRESH/NOTE), §6.3 (poids). Le plan 2b couvrira les dialogues publics, l'étiquetage par le professeur et la baseline Transformer (§5, §6.6).

## Global Constraints

- Branche `plan2-rubans` ; `main` reste intact pendant que le pré-entraînement tourne sur Kaggle. Le checkpoint de pré-entraînement doit rester chargeable : **aucun paramètre ajouté ni renommé** dans `RelisModel` (les masques sont des entrées, pas des poids).
- REFRESH = état GDN à zéro + fenêtre locale vidée ; la convolution courte, la cadence d'écriture du Buffer (`pending`), `seen` et les slots continuent (spec §4.8). Test d'arbitrage : « forward avec masque » = « forward, `reset_memory`, forward » à `atol=1e-4`.
- Porte d'oubli forcée : `RESET_LOG_ALPHA = -80.0` (`exp(-80) ≈ 1e-35`, nul en pratique, sans `-inf` ni perte de précision dans les sommes cumulées).
- Codes de contrôle (spec §4.2) : NOTE = 0x14 ; ensemble des décisions `DECISION_CODES = {READ, SKIP, CONT, STOP, NEXT, END, REFRESH, NOTE}`.
- Classes de poids (spec §6.3) : `WEIGHTS = (0.0, 0.1, 1.0, 5.0)` ; marqueurs et rembourrage 0 ; requête, en-têtes, contenu balayé 0,1 ; réponse et notes 1 ; décisions 5.
- Ruban : rembourrage à un multiple de `block` (512 en V1, 32 en tiny) avec l'octet 0x00, séquences empaquetées de longueur fixe (16 384 en V1) ; drapeaux : bit 0 = réinitialisation Mémoire, bit 1 = réinitialisation slots ; tout début de ruban porte les deux bits et tombe sur un multiple de `block`.
- Les cœurs numériques restent en fp32 sous autocast (règle du plan 1) ; les tests tournent sur CPU en moins de deux minutes, un test échoue avant le code qui le fait passer, un commit par tâche au minimum.
- Aucune dépendance nouvelle.

---

## Structure des fichiers

```
relis/tape/codes.py          (modifier) NOTE, DECISION_CODES
relis/tape/headers.py        (créer)   format_header / parse_header avec échappement
relis/tape/tape.py           (créer)   Segment, Pass, TapeSpec, Tape, build_tape, validate_spec
relis/model/gdn.py           (modifier) reset dans _project/forward/step, RESET_LOG_ALPHA
relis/model/swa.py           (modifier) segments dans le cache, seg_mask dans windowed_attention
relis/model/relis.py         (modifier) reset et slots_reset dans Block/_ckpt_block/_run_piece/forward/step
relis/model/state.py         (modifier) reset_memory redéfini, reset_all
relis/data/episodes.py       (créer)   générateur d'épisodes synthétiques + oracle
relis/data/pack.py           (créer)   pad_tape, pack_tapes, TapeShardWriter, TapeWindows, CLI
relis/train/loop.py          (modifier) batch dict, perte pondérée, masques, init_from, extra_val
relis/train/tape_train.py    (créer)   point d'entrée de l'entraînement ruban
relis/train/metrics.py       (créer)   decision_accuracy
configs/tape_t4.yaml         (créer)
notebooks/KAGGLE.md          (modifier) section « Entraînement ruban »
tests/test_headers.py, tests/test_tape.py, tests/test_reset.py, tests/test_episodes.py,
tests/test_pack.py, tests/test_tape_train.py  (créer) ; tests/test_relis.py (modifier un test)
```

---

### Task 1 : code NOTE, ensemble des décisions, en-têtes échappés

**Files:**
- Modify: `relis/tape/codes.py`
- Create: `relis/tape/headers.py`
- Test: `tests/test_headers.py`, `tests/test_codes.py` (ajout)

**Interfaces:**
- Produces: `codes.NOTE = 0x14`, `codes.DECISION_CODES: frozenset[int]`, `codes.name(b: int) -> str` (nom du code ou `"0x.."`) ; `headers.format_header(fields: dict) -> str` (valeurs : `;` et `=` remplacés par une espace, octets de contrôle assainis, résultat tronqué à 120 octets UTF-8 valides) ; `headers.parse_header(s: str) -> dict[str, str]`.

- [ ] **Step 1 : tests**

Ajouter à `tests/test_codes.py` :
```python
def test_note_and_decision_codes():
    assert C.NOTE == 0x14 and C.NOTE in C.CONTROL_CODES
    assert C.DECISION_CODES == frozenset({C.READ, C.SKIP, C.CONT, C.STOP, C.NEXT, C.END, C.REFRESH, C.NOTE})
    assert C.name(C.STOP) == "STOP" and C.name(0x41) == "0x41"
```

`tests/test_headers.py` :
```python
from relis.tape.headers import format_header, parse_header


def test_roundtrip_simple():
    h = format_header({"role": "user", "t": "2026-09-07T10:00"})
    assert h == "role=user;t=2026-09-07T10:00"
    assert parse_header(h) == {"role": "user", "t": "2026-09-07T10:00"}


def test_values_are_escaped_and_sanitized():
    h = format_header({"type": "doc", "name": "a=b;c\x01d.txt"})
    assert ";" not in h.split("name=")[1] and "=" not in h.split("name=")[1]
    assert "\x01" not in h
    assert parse_header(h)["type"] == "doc"


def test_header_is_truncated_to_valid_utf8():
    h = format_header({"name": "é" * 200})
    assert len(h.encode("utf-8")) <= 120
    h.encode("utf-8").decode("utf-8")   # ne lève pas


def test_parse_ignores_malformed_parts():
    assert parse_header("a=1;;b;c=2=3") == {"a": "1", "c": "2=3"}
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_codes.py tests/test_headers.py -q` → FAIL (`AttributeError`, `ModuleNotFoundError`).

- [ ] **Step 3 : implémenter**

Dans `relis/tape/codes.py`, sous `RECALL = 0x13` ajouter :
```python
NOTE = 0x14

DECISION_CODES = frozenset({READ, SKIP, CONT, STOP, NEXT, END, REFRESH, NOTE})

_NAMES = {ENC: "ENC", SEG: "SEG", END: "END", STOP: "STOP", REFRESH: "REFRESH", NEXT: "NEXT",
          CONT: "CONT", SKIP: "SKIP", SCAN: "SCAN", GEN: "GEN", DEC: "DEC", READ: "READ",
          PART: "PART", RECALL: "RECALL", NOTE: "NOTE", CHAN: "CHAN", HDR: "HDR"}


def name(b: int) -> str:
    return _NAMES.get(b, f"0x{b:02x}")
```
(placer `_NAMES` et `name` après la définition de `HDR`, c'est-à-dire en fin de section « Actions »).

`relis/tape/headers.py` :
```python
"""En-têtes de segment (spec §4.3) : `clé=valeur;clé=valeur`, valeurs échappées."""
from . import codes as C

MAX_BYTES = 120


def _clean(value) -> str:
    s = str(value).replace(";", " ").replace("=", " ")
    return C.sanitize(s.encode("utf-8")).decode("utf-8", errors="replace")


def format_header(fields: dict) -> str:
    s = ";".join(f"{k}={_clean(v)}" for k, v in fields.items())
    return s.encode("utf-8")[:MAX_BYTES].decode("utf-8", errors="ignore")


def parse_header(s: str) -> dict:
    out = {}
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k:
            out[k] = v
    return out
```

- [ ] **Step 4 : vérifier** — `python -m pytest tests/test_codes.py tests/test_headers.py -q` → PASS ; puis `python -m pytest -q` → tout PASS.

- [ ] **Step 5 : commit** — `git add relis/tape/codes.py relis/tape/headers.py tests/test_headers.py tests/test_codes.py && git commit -m "feat(tape): code NOTE, ensemble des décisions, en-têtes échappés"`

---

### Task 2 : masques de réinitialisation dans le modèle (REFRESH exact)

**Files:**
- Modify: `relis/model/gdn.py`, `relis/model/swa.py`, `relis/model/relis.py`, `relis/model/state.py`, `tests/test_relis.py` (un test)
- Test: `tests/test_reset.py`

**Interfaces:**
- Consumes: modules du plan 1.
- Produces:
  - `gdn.RESET_LOG_ALPHA = -80.0` ; `GatedDeltaNet.forward(x, S, conv_state, reset=None)` et `.step(x, S, conv_state, reset=None)` (`reset` : bool `(B,L)` / `(B,)`).
  - `SlidingWindowAttention.init_cache` renvoie aussi `"seg": LongTensor (B,)` et `"kseg": LongTensor (B,0)` ; `forward(x, cache, reset=None)`, `step(x, cache, reset=None)` ; `windowed_attention(..., seg_mask=None)` avec `seg_mask: bool (B,1,L,Lk)`.
  - `Block.forward(x, st_gdn, st_swa, slots, single_step=False, reset=None)`.
  - `RelisModel.forward(x, mode, state, reset=None, slots_reset=None)` ; `RelisModel.step(x, mode, state, reset=None)`. `slots_reset` = début d'un nouveau ruban : slots initiaux **et** convolution courte remise à zéro pour l'échantillon (combiné au bit `reset` du même endroit, l'échantillon repart exactement d'un état neuf) ; accepté seulement aux positions qui ouvrent un bloc (assert sinon).
  - `State.reset_memory()` : zéro sur les états GDN `S` **seulement** (la convolution est conservée), caches SWA vidés (`k`, `v`, `kseg` vides ; `seg += 1` ; `pos` conservé), `pending` et `seen` **conservés**. `State.reset_all()` : `reset_memory()` + slots initiaux + `pending` vidé + `seen = 0`.

- [ ] **Step 1 : tests**

`tests/test_reset.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape.codes import Mode


def _m():
    cfg = RelisConfig.tiny()
    return cfg, RelisModel(cfg).eval()


def test_reset_mask_equals_reset_memory_mid_block():
    cfg, m = _m()
    L, t = 2 * cfg.block + 20, cfg.block + 7
    x = torch.randint(0, 256, (1, L))
    mask = torch.zeros(1, L, dtype=torch.bool); mask[0, t] = True
    la, sa = m(x, int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask)
    st = m.new_state(1, "cpu")
    l1, st = m(x[:, :t], int(Mode.SCAN), st)
    st.reset_memory()
    l2, st = m(x[:, t:], int(Mode.SCAN), st)
    assert torch.allclose(la, torch.cat([l1, l2], 1), atol=1e-4), (la - torch.cat([l1, l2], 1)).abs().max()
    assert torch.allclose(sa.slots, st.slots, atol=1e-4)


def test_reset_per_sample_in_batch_equals_individual_runs():
    cfg, m = _m()
    L = 2 * cfg.block + 9
    x = torch.randint(0, 256, (2, L))
    mask = torch.zeros(2, L, dtype=torch.bool); mask[0, 13] = True; mask[1, cfg.block + 3] = True
    lb, _ = m(x, int(Mode.SCAN), m.new_state(2, "cpu"), reset=mask)
    for b in range(2):
        li, _ = m(x[b:b + 1], int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask[b:b + 1])
        assert torch.allclose(lb[b:b + 1], li, atol=1e-4)


def test_step_reset_equals_forward_reset():
    cfg, m = _m()
    L, t = cfg.block + 11, 6
    x = torch.randint(0, 256, (1, L))
    mask = torch.zeros(1, L, dtype=torch.bool); mask[0, t] = True
    lf, _ = m(x, int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask)
    st = m.new_state(1, "cpu"); outs = []
    for i in range(L):
        lg, st = m.step(x[:, i], int(Mode.SCAN), st, reset=mask[:, i])
        outs.append(lg)
    assert torch.allclose(lf, torch.stack(outs, 1), atol=1e-4)


def test_two_resets_in_one_chunk_and_no_nan():
    cfg, m = _m()
    L = cfg.block
    x = torch.randint(0, 256, (1, L))
    mask = torch.zeros(1, L, dtype=torch.bool); mask[0, 2] = True; mask[0, 4] = True
    lg, st = m(x, int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask)
    assert torch.isfinite(lg).all()
    st2 = m.new_state(1, "cpu")
    _, st2 = m(x[:, :4], int(Mode.SCAN), st2, reset=mask[:, :4]); st2.reset_memory()
    l2, st2 = m(x[:, 4:], int(Mode.SCAN), st2)
    assert torch.allclose(lg[:, 4:], l2, atol=1e-4)


def test_slots_reset_at_block_boundary_equals_fresh_state():
    cfg, m = _m()
    L = 2 * cfg.block
    x = torch.randint(0, 256, (2, L))
    reset = torch.zeros(2, L, dtype=torch.bool); reset[0, cfg.block] = True
    sreset = reset.clone()
    lb, sb = m(x, int(Mode.SCAN), m.new_state(2, "cpu"), reset=reset, slots_reset=sreset)
    # échantillon 0 : après la frontière, tout doit égaler un état neuf sur x[0, block:]
    lf, sf = m(x[0:1, cfg.block:], int(Mode.SCAN), m.new_state(1, "cpu"))
    assert torch.allclose(lb[0:1, cfg.block:], lf, atol=1e-4)
    assert torch.allclose(sb.slots[0:1], sf.slots, atol=1e-4)
    # échantillon 1 : inchangé par rapport à un run sans drapeaux
    ln, sn = m(x[1:2], int(Mode.SCAN), m.new_state(1, "cpu"))
    assert torch.allclose(lb[1:2], ln, atol=1e-4)
    assert torch.allclose(sb.slots[1:2], sn.slots, atol=1e-4)


def test_slots_reset_off_boundary_is_rejected():
    cfg, m = _m()
    x = torch.randint(0, 256, (1, cfg.block))
    sreset = torch.zeros(1, cfg.block, dtype=torch.bool); sreset[0, 5] = True
    try:
        m(x, int(Mode.SCAN), m.new_state(1, "cpu"), slots_reset=sreset)
        assert False, "attendu : AssertionError"
    except AssertionError:
        pass


def test_reset_memory_new_semantics():
    cfg, m = _m()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, cfg.block + 5)), int(Mode.SCAN), st)
    pend, seen = st.pending.clone(), st.seen
    conv_before = [g[1].clone() for g in st.gdn if g is not None]
    st.reset_memory()
    assert torch.equal(st.pending, pend) and st.seen == seen
    assert all(torch.count_nonzero(g[0]) == 0 for g in st.gdn if g is not None)
    assert all(torch.equal(g[1], c) for g, c in zip([g for g in st.gdn if g is not None], conv_before))
    assert all(c["k"].shape[2] == 0 and c["kseg"].shape[1] == 0 for c in st.swa if c is not None)
    st.reset_all()
    assert st.pending.shape[1] == 0 and st.seen == 0


def test_grad_flows_with_masks():
    cfg, m = _m(); m.train()
    L = 2 * cfg.block + 3
    x = torch.randint(0, 256, (2, L))
    reset = torch.zeros(2, L, dtype=torch.bool); reset[0, 10] = True; reset[1, cfg.block] = True
    sreset = torch.zeros(2, L, dtype=torch.bool); sreset[1, cfg.block] = True
    lg, _ = m(x, int(Mode.SCAN), m.new_state(2, "cpu"), reset=reset, slots_reset=sreset)
    lg.float().sum().backward()
    assert m.slots.grad is not None and torch.isfinite(m.slots.grad).all()
```

Dans `tests/test_relis.py`, remplacer le corps de `test_reset_memory_keeps_slots` par :
```python
def test_reset_memory_keeps_slots():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, 2 * cfg.block)), int(Mode.ENCODE), st)
    slots = st.slots.clone()
    st.reset_memory()
    assert torch.allclose(st.slots, slots)
    assert st.seen == 2 * cfg.block                      # la cadence d'écriture continue (spec §4.8)
    assert all(torch.count_nonzero(g[0]) == 0 for g in st.gdn if g is not None)
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_reset.py -q` → FAIL (`TypeError: unexpected keyword 'reset'`).

- [ ] **Step 3 : implémenter `gdn.py`**

Ajouter après les imports : `RESET_LOG_ALPHA = -80.0  # porte d'oubli « zéro » : exp(-80) ≈ 1e-35, sans -inf`.

Remplacer `_project`, `forward`, `step` de `GatedDeltaNet` par :
```python
    def _project(self, x, conv_state, reset=None):
        B, L, _ = x.shape
        H, dh = self.cfg.n_heads, self.cfg.head_dim
        qkv, conv_state = self.conv(self.qkv(x), conv_state)
        qkv = F.silu(qkv)
        q, k, v = qkv.split(self.cfg.inner, dim=-1)
        q = q.view(B, L, H, dh).transpose(1, 2)
        k = k.view(B, L, H, dh).transpose(1, 2)
        v = v.view(B, L, H, dh).transpose(1, 2)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        beta = torch.sigmoid(self.beta_proj(x).float()).transpose(1, 2)               # (B,H,L)
        log_alpha = -self.A_log.float().exp()[None, :, None] * F.softplus(self.dt_proj(x).float()).transpose(1, 2)
        if reset is not None:
            # REFRESH exact (spec §4.8) : l'état précédent est annulé à la position marquée
            log_alpha = log_alpha.masked_fill(reset.to(torch.bool)[:, None, :], RESET_LOG_ALPHA)
        return q, k, v.float(), log_alpha, beta, conv_state

    def forward(self, x, S, conv_state, reset=None):
        if S is None:
            S, conv_state = self.init_state(x.shape[0], x.device)
        q, k, v, log_alpha, beta, conv_state = self._project(x, conv_state, reset)
        o, S = gdn_chunked(q, k, v, log_alpha, beta, S, self.cfg.chunk)
        return self._output(o, x), S, conv_state

    def step(self, x, S, conv_state, reset=None):
        r = None if reset is None else reset.to(torch.bool).reshape(x.shape[0], 1)
        q, k, v, log_alpha, beta, conv_state = self._project(x, conv_state, r)
        o, S = gdn_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], log_alpha[:, :, 0], beta[:, :, 0], S)
        return self._output(o.unsqueeze(2), x), S, conv_state
```

- [ ] **Step 4 : implémenter `swa.py`**

Remplacer `windowed_attention` (signature + masque) et la classe par :
```python
def windowed_attention(q, k_all, v_all, qi, kj, slopes, window, dh, seg_mask=None):
    """Cœur d'attention local, fp32 garanti sous autocast.

    seg_mask : bool (B,1,L,Lk), True si requête et clé sont dans le même segment
    (aucun REFRESH entre elles). None = un seul segment.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        allowed = (kj <= qi) & (kj > qi - window)                    # (L, Lk)
        if seg_mask is not None:
            allowed = allowed[None, None] & seg_mask                   # (B,1,L,Lk)
        scores = (q @ k_all.transpose(-1, -2)) / dh ** 0.5           # (B,H,L,Lk)
        scores = scores - slopes[None, :, None, None] * (qi - kj).float()
        scores = scores.masked_fill(~allowed, float("-inf"))
        o = torch.softmax(scores, dim=-1) @ v_all
    return o


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.inner, bias=False)
        self.o_proj = nn.Linear(cfg.inner, cfg.d_model, bias=False)
        H = cfg.n_heads
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    def init_cache(self, B: int, device):
        H, dh = self.cfg.n_heads, self.cfg.head_dim
        return {"k": torch.zeros(B, H, 0, dh, device=device), "v": torch.zeros(B, H, 0, dh, device=device),
                "pos": 0, "seg": torch.zeros(B, dtype=torch.long, device=device),
                "kseg": torch.zeros(B, 0, dtype=torch.long, device=device)}

    def forward(self, x, cache, reset=None):
        B, L, _ = x.shape
        H, dh, W = self.cfg.n_heads, self.cfg.head_dim, self.cfg.window
        if cache is None:
            cache = self.init_cache(B, x.device)
        q, k, v = self.qkv(x).split(self.cfg.inner, dim=-1)
        q = q.view(B, L, H, dh).transpose(1, 2).float()
        k = k.view(B, L, H, dh).transpose(1, 2).float()
        v = v.view(B, L, H, dh).transpose(1, 2).float()
        k_all = torch.cat([cache["k"].float(), k], dim=2)
        v_all = torch.cat([cache["v"].float(), v], dim=2)
        Lc = cache["k"].shape[2]
        pos0 = cache["pos"]
        qi = torch.arange(pos0, pos0 + L, device=x.device)[:, None]
        kj = torch.arange(pos0 - Lc, pos0 + L, device=x.device)[None, :]
        # segments : un REFRESH à la position t ouvre un nouveau segment à partir de t
        inc = torch.zeros(B, L, dtype=torch.long, device=x.device) if reset is None \
            else reset.to(torch.long).reshape(B, L).cumsum(dim=1)
        qseg = cache["seg"][:, None] + inc                             # (B,L)
        kseg_all = torch.cat([cache["kseg"], qseg], dim=1)             # (B,Lc+L)
        seg_mask = kseg_all[:, None, None, :] == qseg[:, None, :, None]  # (B,1,L,Lc+L)
        o = windowed_attention(q, k_all, v_all, qi, kj, self.slopes, W, dh, seg_mask)
        y = self.o_proj(o.transpose(1, 2).reshape(B, L, H * dh).to(x.dtype))
        keep = W - 1
        new_cache = {"k": k_all[:, :, -keep:] if keep > 0 else k_all[:, :, :0],
                     "v": v_all[:, :, -keep:] if keep > 0 else v_all[:, :, :0],
                     "pos": pos0 + L, "seg": qseg[:, -1],
                     "kseg": kseg_all[:, -keep:] if keep > 0 else kseg_all[:, :0]}
        return y, new_cache

    def step(self, x, cache, reset=None):
        return self.forward(x, cache, reset)
```

- [ ] **Step 5 : implémenter `state.py`**

Remplacer `reset_memory` et `reset_all` par :
```python
    def reset_memory(self) -> None:
        """REFRESH (spec §4.8) : état GDN à zéro et fenêtre locale vidée ; la convolution
        courte, la cadence d'écriture du Buffer (`pending`, `seen`) et les slots continuent."""
        for i, g in enumerate(self.gdn):
            if g is not None:
                self.gdn[i] = (torch.zeros_like(g[0]), g[1])
        for i, c in enumerate(self.swa):
            if c is not None:
                self.swa[i] = {"k": c["k"][:, :, :0], "v": c["v"][:, :, :0], "pos": c["pos"],
                               "seg": c["seg"] + 1, "kseg": c["kseg"][:, :0]}

    def reset_all(self) -> None:
        """Nouveau tour utilisateur : Mémoire, cadence d'écriture et slots repartent de zéro."""
        self.reset_memory()
        for i, g in enumerate(self.gdn):
            if g is not None:
                self.gdn[i] = (g[0], torch.zeros_like(g[1]))
        self.pending = self.pending[:, :0]
        self.seen = 0
        self.slots = self._init_slots.clone()
```

Ajouter dans `size_bytes` la taille de `kseg` : après la ligne qui additionne `k` et `v`, ajouter `n += c["kseg"].numel() * c["kseg"].element_size()`.

- [ ] **Step 6 : implémenter `relis.py`**

Remplacer `Block.forward`, `_ckpt_block`, `_run_piece`, `forward`, `step` par :
```python
    def forward(self, x, st_gdn, st_swa, slots, single_step: bool = False, reset=None):
        h = self.norm1(x)
        if self.is_swa:
            y, st_swa = (self.mixer.step if single_step else self.mixer.forward)(h, st_swa, reset)
        else:
            S, cs = st_gdn
            y, S, cs = (self.mixer.step if single_step else self.mixer.forward)(h, S, cs, reset)
            st_gdn = (S, cs)
        x = x + y
        x = x + self.read(x, slots)
        x = x + self.mlp(self.norm2(x))
        return x, st_gdn, st_swa
```
```python
    @staticmethod
    def _ckpt_block(blk: Block, h, st_gdn, st_swa, slots, reset):
        """Exécute un bloc sous `torch.utils.checkpoint` (tenseurs seuls en entrée/sortie)."""
        if blk.is_swa:
            if st_swa is None:
                st_swa = blk.mixer.init_cache(h.shape[0], h.device)
            pos = st_swa["pos"]

            def fn(h_, k_, v_, seg_, kseg_, slots_, reset_):
                y_, _, c_ = blk(h_, None, {"k": k_, "v": v_, "pos": pos, "seg": seg_, "kseg": kseg_},
                                slots_, reset=reset_)
                return y_, c_["k"], c_["v"], c_["seg"], c_["kseg"]

            y, ck, cv, cseg, ckseg = ckpt.checkpoint(fn, h, st_swa["k"], st_swa["v"], st_swa["seg"],
                                                     st_swa["kseg"], slots, reset, use_reentrant=False)
            return y, None, {"k": ck, "v": cv, "pos": pos + h.shape[1], "seg": cseg, "kseg": ckseg}

        if st_gdn is None:
            st_gdn = blk.mixer.init_state(h.shape[0], h.device)

        def fn(h_, S_, cs_, slots_, reset_):
            y_, (S2, cs2), _ = blk(h_, (S_, cs_), None, slots_, reset=reset_)
            return y_, S2, cs2

        y, S, cs = ckpt.checkpoint(fn, h, st_gdn[0], st_gdn[1], slots, reset, use_reentrant=False)
        return y, (S, cs), None

    def _run_piece(self, x_ids, mode_ids, state: State, single_step: bool, reset=None, slots_reset=None):
        """Passe un morceau (≤ block positions, sans frontière interne) dans la pile."""
        B = x_ids.shape[0]
        if slots_reset is not None:
            assert not slots_reset[:, 1:].any(), "slots_reset n'est admis qu'en début de bloc"
            first = slots_reset[:, 0].to(torch.bool)
            if first.any():
                assert state.pending.shape[1] == 0, "slots_reset exige une frontière de bloc"
                fresh = expand_slots(self.slots, B).to(state.slots.dtype)
                state.slots = torch.where(first[:, None, None], fresh, state.slots)
                for i, g in enumerate(state.gdn):       # nouveau ruban : convolution courte à zéro
                    if g is not None:
                        state.gdn[i] = (g[0], g[1].masked_fill(first[:, None, None], 0.0))
        h = self.embed(x_ids) + self.mode_embed(mode_ids)
        use_ckpt = (self.cfg.grad_checkpoint and self.training
                    and torch.is_grad_enabled() and not single_step)
        for i, blk in enumerate(self.blocks):
            if use_ckpt:
                h, g, s = self._ckpt_block(blk, h, state.gdn[i], state.swa[i], state.slots, reset)
            else:
                h, g, s = blk(h, state.gdn[i], state.swa[i], state.slots, single_step=single_step, reset=reset)
            state.gdn[i], state.swa[i] = g, s
        top = self.final_norm(h)
        logits = top @ self.head_weight().t().to(top.dtype)
        state.pending = torch.cat([state.pending, top.float()], dim=1)
        state.seen += x_ids.shape[1]
        if state.pending.shape[1] == self.cfg.block:
            state.slots = self.writer.apply_blocks(state.pending.to(state.slots.dtype), state.slots)
            state.pending = state.pending[:, :0]
        return logits, state

    def forward(self, x, mode, state: State, reset=None, slots_reset=None):
        """`reset` et `slots_reset` : bool (B,L) ; voir spec §4.8 et §6.1."""
        B, L = x.shape
        if isinstance(mode, int):
            mode = torch.full((B, L), mode, dtype=torch.long, device=x.device)
        logits = []
        t = 0
        while t < L:
            room = self.cfg.block - state.pending.shape[1]
            n = min(room, L - t)
            lg, state = self._run_piece(
                x[:, t:t + n], mode[:, t:t + n], state, single_step=False,
                reset=None if reset is None else reset[:, t:t + n],
                slots_reset=None if slots_reset is None else slots_reset[:, t:t + n])
            logits.append(lg)
            t += n
        return torch.cat(logits, dim=1), state

    def step(self, x, mode, state: State, reset=None):
        B = x.shape[0]
        if isinstance(mode, int):
            mode = torch.full((B,), mode, dtype=torch.long, device=x.device)
        r = None if reset is None else reset.to(torch.bool).reshape(B, 1)
        lg, state = self._run_piece(x[:, None], mode[:, None], state, single_step=True, reset=r)
        return lg[:, 0], state
```
Ajouter `expand_slots` à l'import : `from .buffer import SlotRead, SlotWrite, init_slots, expand_slots`.

- [ ] **Step 7 : vérifier** — `python -m pytest tests/test_reset.py tests/test_relis.py tests/test_swa.py tests/test_gdn.py -q` → PASS ; `python -m pytest -q` → tout PASS. Si `test_reset_mask_equals_reset_memory_mid_block` échoue, comparer d'abord les slots (calendrier d'écriture) puis vérifier que `reset_memory` conserve bien la convolution : c'est la différence la plus probable.

- [ ] **Step 8 : commit** — `git add relis/model tests/test_reset.py tests/test_relis.py && git commit -m "feat(model): REFRESH exact par masque (GDN, attention locale), réinitialisation des slots par échantillon"`

---

### Task 3 : construction du ruban

**Files:**
- Create: `relis/tape/tape.py`
- Test: `tests/test_tape.py`

**Interfaces:**
- Consumes: `codes` (Task 1), `headers.format_header`.
- Produces: `Segment(header: str, content: bytes, read: bool = True, after: int = CONT)`, `Pass(history: list[Segment], docs: list[Segment] | None = None)`, `TapeSpec(query: bytes, passes: list[Pass], answer_parts: list[bytes], notes: list[bytes])`, `Tape` (champs `data: bytearray`, `mode: list[int]`, `wclass: list[int]`, `reset: list[bool]` ; méthodes `put`, `put_bytes`, `weights() -> list[float]`, `decisions() -> list[int]`, `__len__`), constantes `W_ZERO=0, W_LOW=1, W_ONE=2, W_HIGH=3`, `WEIGHTS=(0.0, 0.1, 1.0, 5.0)`, `validate_spec(spec)` (lève `ValueError`), `build_tape(spec) -> Tape`.

Disposition d'une passe `i` (spec §4.3) — chaque ligne : octets, mode, classe :
```
ENC                              ENCODE  0   (reset=True si i>0)
<query>                          ENCODE  1
[PART <réponses 0..i-1> NOTE <note i-1>]   ENCODE  0/1/0/1   (si i>0)
SCAN                             SCAN    0
CHAN chan=history HDR            SCAN    0 / 1 / 0
  par segment : SEG <hdr> HDR <READ|SKIP> [<contenu>] DEC <CONT|STOP|NEXT>
                                 SCAN    0 / 1 / 0 / 3 / 1 / 0 / 3
[CHAN chan=docs HDR + segments]  SCAN    (si p.docs is not None)
GEN                              GENERATE 0
<réponse i>                      GENERATE 2
NOTE <note i> REFRESH            GENERATE 3 / 2 / 3    (si i < dernière passe)
END                              GENERATE 3            (dernière passe)
```

- [ ] **Step 1 : tests**

`tests/test_tape.py` :
```python
import pytest
from relis.tape import codes as C
from relis.tape.tape import Segment, Pass, TapeSpec, build_tape, validate_spec, WEIGHTS, W_HIGH, W_ONE, W_LOW


def _seg(h, c, read=True, after=C.CONT):
    return Segment(header=h, content=c, read=read, after=after)


def _spec_two_passes():
    p1 = Pass(history=[_seg("role=assistant", b"ok"), _seg("role=user", b"code postal 75012", after=C.STOP)])
    p2 = Pass(history=[_seg("role=assistant", b"ok"), _seg("role=user", b"code postal 75012"),
                       _seg("role=user", b"ville Lyon", after=C.STOP)])
    return TapeSpec(query=b"code et ville ?", passes=[p1, p2],
                    answer_parts=[b"75012", b" et Lyon."], notes=[b"il manque la ville"])


def test_single_pass_layout_and_weights():
    spec = TapeSpec(query=b"q", passes=[Pass(history=[_seg("role=user", b"abc", after=C.STOP)])],
                    answer_parts=[b"rep"], notes=[])
    t = build_tape(spec)
    d = bytes(t.data)
    assert d[0] == C.ENC and d[1:2] == b"q" and d[2] == C.SCAN and d[3] == C.CHAN
    assert d.endswith(bytes([C.GEN]) + b"rep" + bytes([C.END]))
    assert t.decisions() == [C.READ, C.STOP, C.END]
    assert t.weights()[0] == 0.0 and t.weights()[1] == 0.1
    assert t.weights()[-1] == 5.0 and t.weights()[-2] == 1.0
    assert t.mode[0] == int(C.Mode.ENCODE) and t.mode[2] == int(C.Mode.SCAN) and t.mode[-1] == int(C.Mode.GENERATE)
    assert not any(t.reset)
    assert len(t) == len(t.mode) == len(t.wclass) == len(t.reset)


def test_two_passes_have_note_refresh_and_one_reset():
    t = build_tape(_spec_two_passes())
    d = bytes(t.data)
    assert d.count(bytes([C.REFRESH])) == 1 and d.count(bytes([C.NOTE])) == 2   # une émise, une relue
    assert sum(t.reset) == 1
    i = d.index(bytes([C.REFRESH]))
    assert d[i + 1] == C.ENC and t.reset[i + 1] is True
    # dans la requête de la passe 2 : PART + réponse partielle + NOTE + note, en mode ENCODE et poids 0,1
    j = d.index(bytes([C.PART]))
    assert d[j + 1:j + 6] == b"75012" and t.mode[j + 1] == int(C.Mode.ENCODE) and t.wclass[j + 1] == W_LOW
    # la NOTE émise en génération pèse 5, ses octets 1
    k = i - len(b"il manque la ville") - 1
    assert d[k] == C.NOTE and t.wclass[k] == W_HIGH and t.wclass[k + 1] == W_ONE
    assert t.decisions()[-1] == C.END and C.REFRESH in t.decisions() and C.NOTE in t.decisions()


def test_skip_and_docs_channel():
    docs = [_seg("type=doc;name=a.txt", b"rien", read=False), _seg("type=doc;name=b.txt", b"la date est le 3", after=C.STOP)]
    p = Pass(history=[_seg("role=user", b"salut", after=C.NEXT)], docs=docs)
    t = build_tape(TapeSpec(query=b"date ?", passes=[p], answer_parts=[b"le 3"], notes=[]))
    d = bytes(t.data)
    assert d.count(bytes([C.CHAN])) == 2 and b"chan=docs" in d
    assert t.decisions() == [C.READ, C.NEXT, C.SKIP, C.CONT, C.READ, C.STOP, C.END]
    assert b"rien" not in d                          # contenu sauté absent du ruban


def test_content_is_sanitized():
    p = Pass(history=[_seg("role=user", b"a\x05b", after=C.STOP)])
    t = build_tape(TapeSpec(query=b"q\x01", passes=[p], answer_parts=[b"r\x03"], notes=[]))
    d = bytes(t.data)
    assert d.count(bytes([C.REFRESH])) == 0 and d.count(bytes([C.END])) == 1 and d.count(bytes([C.ENC])) == 1


@pytest.mark.parametrize("bad", [
    dict(answer_parts=[b"a", b"b"]),                                      # trop de réponses
    dict(passes=[Pass(history=[_seg("role=user", b"x", after=C.CONT)])]),  # pas de STOP final
    dict(passes=[Pass(history=[_seg("role=user", b"x", after=C.NEXT)])]),  # NEXT sans canal docs
])
def test_validate_rejects(bad):
    base = dict(query=b"q", passes=[Pass(history=[_seg("role=user", b"x", after=C.STOP)])],
                answer_parts=[b"a"], notes=[])
    base.update(bad)
    with pytest.raises(ValueError):
        validate_spec(TapeSpec(**base))


def test_empty_history_is_allowed():
    t = build_tape(TapeSpec(query=b"bonjour", passes=[Pass(history=[])], answer_parts=[b"salut"], notes=[]))
    assert t.decisions() == [C.END]
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_tape.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3 : implémenter**

`relis/tape/tape.py` :
```python
"""Construction du ruban (spec §4.3, §6.3) : quatre tableaux parallèles par position."""
from dataclasses import dataclass, field

from . import codes as C

W_ZERO, W_LOW, W_ONE, W_HIGH = 0, 1, 2, 3
WEIGHTS = (0.0, 0.1, 1.0, 5.0)
ENCODE, SCAN, GENERATE = int(C.Mode.ENCODE), int(C.Mode.SCAN), int(C.Mode.GENERATE)


@dataclass
class Segment:
    header: str
    content: bytes
    read: bool = True
    after: int = C.CONT            # CONT, STOP ou NEXT


@dataclass
class Pass:
    history: list                  # Segment, du plus récent au plus ancien
    docs: list | None = None       # None : canal documents non ouvert


@dataclass
class TapeSpec:
    query: bytes
    passes: list                   # Pass ; len >= 1
    answer_parts: list             # bytes, une par passe
    notes: list                    # bytes, une par REFRESH (len(passes) - 1)


@dataclass
class Tape:
    data: bytearray = field(default_factory=bytearray)
    mode: list = field(default_factory=list)
    wclass: list = field(default_factory=list)
    reset: list = field(default_factory=list)

    def put(self, b: int, mode: int, w: int, reset: bool = False) -> None:
        self.data.append(b); self.mode.append(mode); self.wclass.append(w); self.reset.append(reset)

    def put_bytes(self, bs: bytes, mode: int, w: int) -> None:
        for b in bs:
            self.put(b, mode, w)

    def __len__(self) -> int:
        return len(self.data)

    def weights(self) -> list:
        return [WEIGHTS[w] for w in self.wclass]

    def decisions(self) -> list:
        return [b for b, w in zip(self.data, self.wclass) if w == W_HIGH]


def validate_spec(spec: TapeSpec) -> None:
    n = len(spec.passes)
    if n < 1:
        raise ValueError("au moins une passe")
    if len(spec.answer_parts) != n:
        raise ValueError("une réponse par passe")
    if len(spec.notes) != n - 1:
        raise ValueError("une note par REFRESH")
    for p in spec.passes:
        segs = list(p.history) + (list(p.docs) if p.docs is not None else [])
        for s in p.history[:-1]:
            if s.after not in (C.CONT, C.STOP):
                raise ValueError("NEXT n'est admis qu'au dernier segment de l'historique")
        if p.history and p.history[-1].after == C.NEXT and p.docs is None:
            raise ValueError("NEXT sans canal documents")
        if p.docs is not None and p.history and p.history[-1].after != C.NEXT:
            raise ValueError("le canal documents exige NEXT au dernier segment de l'historique")
        if p.docs is not None:
            for s in p.docs:
                if s.after not in (C.CONT, C.STOP):
                    raise ValueError("dans le canal documents : CONT ou STOP")
        for s in segs:
            if s.after not in (C.CONT, C.STOP, C.NEXT):
                raise ValueError("décision inconnue")
        if segs and segs[-1].after != C.STOP:
            raise ValueError("la dernière décision d'une passe doit être STOP")
        stops = [s for s in segs if s.after == C.STOP]
        if len(stops) > 1:
            raise ValueError("un seul STOP par passe")


def _segment(t: Tape, s: Segment, mode: int) -> None:
    t.put(C.SEG, mode, W_ZERO)
    t.put_bytes(s.header.encode("utf-8"), mode, W_LOW)
    t.put(C.HDR, mode, W_ZERO)
    t.put(C.READ if s.read else C.SKIP, mode, W_HIGH)
    if s.read:
        t.put_bytes(C.sanitize(s.content), mode, W_LOW)
    t.put(C.DEC, mode, W_ZERO)
    t.put(s.after, mode, W_HIGH)


def _channel(t: Tape, name: str, segs, mode: int) -> None:
    t.put(C.CHAN, mode, W_ZERO)
    t.put_bytes(f"chan={name}".encode("utf-8"), mode, W_LOW)
    t.put(C.HDR, mode, W_ZERO)
    for s in segs:
        _segment(t, s, mode)


def build_tape(spec: TapeSpec) -> Tape:
    validate_spec(spec)
    t = Tape()
    last = len(spec.passes) - 1
    for i, p in enumerate(spec.passes):
        t.put(C.ENC, ENCODE, W_ZERO, reset=(i > 0))
        t.put_bytes(C.sanitize(spec.query), ENCODE, W_LOW)
        if i > 0:
            t.put(C.PART, ENCODE, W_ZERO)
            t.put_bytes(C.sanitize(b"".join(spec.answer_parts[:i])), ENCODE, W_LOW)
            t.put(C.NOTE, ENCODE, W_ZERO)
            t.put_bytes(C.sanitize(spec.notes[i - 1]), ENCODE, W_LOW)
        t.put(C.SCAN, SCAN, W_ZERO)
        _channel(t, "history", p.history, SCAN)
        if p.docs is not None:
            _channel(t, "docs", p.docs, SCAN)
        t.put(C.GEN, GENERATE, W_ZERO)
        t.put_bytes(C.sanitize(spec.answer_parts[i]), GENERATE, W_ONE)
        if i < last:
            t.put(C.NOTE, GENERATE, W_HIGH)
            t.put_bytes(C.sanitize(spec.notes[i]), GENERATE, W_ONE)
            t.put(C.REFRESH, GENERATE, W_HIGH)
        else:
            t.put(C.END, GENERATE, W_HIGH)
    return t
```

- [ ] **Step 4 : vérifier** — `python -m pytest tests/test_tape.py -q` → PASS (le test `test_two_passes...` : si l'assertion sur `decisions()[-4:]` gêne, c'est que l'ordre attendu est `[..., STOP, NOTE, REFRESH, READ, ...]` en passe 1 puis `END` en fin ; la clause `or` la rend permissive volontairement). `python -m pytest -q` → tout PASS.

- [ ] **Step 5 : commit** — `git add relis/tape/tape.py tests/test_tape.py && git commit -m "feat(tape): construction du ruban avec modes, classes de poids et réinitialisations"`

---

### Task 4 : épisodes synthétiques et oracle

**Files:**
- Create: `relis/data/episodes.py`
- Test: `tests/test_episodes.py`

**Interfaces:**
- Consumes: `tape.Segment/Pass/TapeSpec`, `headers.format_header`, `codes`.
- Produces: `KINDS = ("fact_recall", "variable_tracking", "what_did_i_say", "doc_lookup", "absent", "two_facts", "long_answer")` ; `generate_episode(rng: random.Random, kind: str | None = None) -> TapeSpec` ; `iter_episodes(seed: int, n: int) -> Iterator[TapeSpec]` ; `oracle_pass(history_segments, needed_hist: set[int], docs, needed_doc: int | None, doc_relevant: list[bool]) -> Pass` (décisions exactes ; `history_segments` du plus récent au plus ancien ; les indices de `needed_hist` se réfèrent à cet ordre).

Règles de l'oracle (spec §6.2) :
- Historique, du plus récent au plus ancien : chaque segment est lu (`read=True`). Après le segment `k` : si tous les indices requis de l'historique sont ≤ `k` **et** aucun document n'est requis → `STOP` ; sinon si `k` est le dernier segment → `NEXT` s'il y a des documents, `STOP` sinon ; sinon, si rien n'est requis dans l'historique **et** qu'un document est requis → `NEXT` dès le premier segment (lu quand même) ; sinon `CONT`. Une information absente partout fait donc lire tout l'historique puis tous les en-têtes de documents.
- Documents, du plus récent au plus ancien : `read = doc_relevant[j]` (pertinence d'après l'en-tête, avec 10 % de bruit : un en-tête pertinent sur un contenu qui ne l'est pas, lu puis `CONT`) ; après le document `j` : `STOP` si le document requis a été lu (`j >= needed_doc`), sinon `CONT`, et `STOP` forcé au dernier.
- Historique vide et pas de document : `Pass(history=[])` (le contrôleur passe directement à GEN).

- [ ] **Step 1 : tests**

`tests/test_episodes.py` :
```python
import random
from collections import Counter
from relis.tape import codes as C
from relis.tape.tape import build_tape, Segment, validate_spec
from relis.data.episodes import KINDS, generate_episode, iter_episodes, oracle_pass


def _seg(i):
    return Segment(header=f"role=user;i={i}", content=f"tour {i}".encode())


def test_oracle_stops_at_oldest_needed_history_segment():
    hist = [_seg(i) for i in range(5)]           # 0 = le plus récent
    p = oracle_pass(hist, needed_hist={1, 3}, docs=[], needed_doc=None, doc_relevant=[])
    assert [s.after for s in p.history] == [C.CONT, C.CONT, C.CONT, C.STOP]
    assert p.docs is None and all(s.read for s in p.history)


def test_oracle_goes_to_docs_and_skips_by_header():
    hist = [_seg(0), _seg(1)]
    docs = [Segment("type=doc;name=a", b"a"), Segment("type=doc;name=b", b"b"), Segment("type=doc;name=c", b"c")]
    p = oracle_pass(hist, needed_hist=set(), docs=docs, needed_doc=1, doc_relevant=[False, True, False])
    assert [s.after for s in p.history] == [C.NEXT]           # rien de requis dans l'historique
    assert [(s.read, s.after) for s in p.docs] == [(False, C.CONT), (True, C.STOP)]


def test_oracle_absent_reads_everything_then_stops():
    hist = [_seg(0), _seg(1)]
    docs = [Segment("type=doc;name=a", b"a")]
    p = oracle_pass(hist, needed_hist=set(), docs=docs, needed_doc=None, doc_relevant=[False])
    assert [s.after for s in p.history] == [C.CONT, C.NEXT]
    assert [(s.read, s.after) for s in p.docs] == [(False, C.STOP)]


def test_every_kind_builds_a_valid_tape():
    rng = random.Random(0)
    for kind in KINDS:
        for _ in range(5):
            spec = generate_episode(rng, kind)
            validate_spec(spec)
            t = build_tape(spec)
            assert t.decisions()[-1] == C.END
            assert 100 < len(t) < 16_000, (kind, len(t))   # doit tenir dans une séquence de 16 384


def test_two_facts_has_exactly_one_refresh_with_note():
    rng = random.Random(1)
    spec = generate_episode(rng, "two_facts")
    assert len(spec.passes) == 2 and len(spec.notes) == 1 and spec.notes[0].startswith(b"il manque")
    t = build_tape(spec)
    assert sum(t.reset) == 1


def test_long_answer_refreshes_every_L_bytes():
    rng = random.Random(2)
    spec = generate_episode(rng, "long_answer")
    assert len(spec.passes) >= 2
    assert all(n == b"r\xc3\xa9ponse longue, suite" for n in spec.notes)
    assert all(256 <= len(part) <= 1024 for part in spec.answer_parts[:-1])


def test_doc_lookup_reads_needed_doc_and_answer_mentions_fact():
    rng = random.Random(3)
    spec = generate_episode(rng, "doc_lookup")
    p = spec.passes[0]
    assert p.docs is not None and any(s.read and s.after == C.STOP for s in p.docs)
    assert p.history[-1].after == C.NEXT


def test_iter_episodes_is_deterministic_and_covers_kinds():
    a = [build_tape(s).data for s in iter_episodes(seed=7, n=30)]
    b = [build_tape(s).data for s in iter_episodes(seed=7, n=30)]
    assert a == b
    counts = Counter()
    rng = random.Random(11)
    for _ in range(300):
        spec = generate_episode(rng)
        counts[len(spec.passes) > 1] += 1
    assert counts[True] > 20 and counts[False] > 100
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_episodes.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3 : implémenter**

`relis/data/episodes.py` :
```python
"""Épisodes synthétiques de lecture et oracle des décisions (spec §6.2).

Chaque épisode est une conversation construite pour que l'on sache exactement
où se trouve l'information nécessaire ; l'oracle en déduit READ/SKIP,
CONT/STOP/NEXT, et la position des REFRESH avec leur NOTE.
"""
import random

from relis.tape import codes as C
from relis.tape.headers import format_header
from relis.tape.tape import Segment, Pass, TapeSpec

KINDS = ("fact_recall", "variable_tracking", "what_did_i_say", "doc_lookup", "absent", "two_facts", "long_answer")
KIND_WEIGHTS = (22, 12, 14, 22, 8, 12, 10)

PRENOMS = ["Adil", "Camille", "Nadia", "Julien", "Inès", "Marc", "Sofia", "Karim", "Léa", "Youssef"]
VILLES = ["Lyon", "Paris", "Rabat", "Lille", "Casablanca", "Bordeaux", "Nantes", "Tunis", "Genève", "Montréal"]
OBJETS = ["le rapport", "la facture", "le contrat", "le devis", "la présentation", "le planning"]
SUJETS = ["la réunion", "le budget", "les vacances", "le projet RELIS", "la voiture", "le déménagement",
          "le stage", "la formation", "le serveur", "la thèse"]
FILLERS_USER = ["Merci pour ton aide.", "D'accord, je note.", "Peux-tu me rappeler l'heure de {sujet} ?",
                "Parlons de {sujet}.", "Je reviens vers toi demain.", "Qu'en penses-tu ?",
                "J'ai avancé sur {sujet} ce matin.", "On en reparle plus tard."]
FILLERS_ASSISTANT = ["Bien sûr.", "Entendu, je m'en occupe.", "Voici ce que je propose pour {sujet}.",
                     "Je reste disponible.", "C'est noté.", "Très bien, continuons."]

# faits : (phrase de l'utilisateur, question, réponse, libellé pour la NOTE, générateur de valeur)
FACTS = [
    ("Mon code postal est {v}.", "Quel est mon code postal ?", "Votre code postal est {v}.", "votre code postal",
     lambda r: f"{r.randint(10000, 95999)}"),
    ("Je m'appelle {v}.", "Comment je m'appelle ?", "Vous vous appelez {v}.", "votre prénom",
     lambda r: r.choice(PRENOMS)),
    ("J'habite à {v}.", "Où est-ce que j'habite ?", "Vous habitez à {v}.", "votre ville",
     lambda r: r.choice(VILLES)),
    ("Mon rendez-vous est le {v}.", "Quand est mon rendez-vous ?", "Votre rendez-vous est le {v}.",
     "la date de votre rendez-vous", lambda r: f"{r.randint(1, 28)} {r.choice(['mars', 'avril', 'mai', 'juin', 'octobre'])}"),
    ("Le mot de passe du wifi est {v}.", "Quel est le mot de passe du wifi ?", "Le mot de passe du wifi est {v}.",
     "le mot de passe du wifi", lambda r: "".join(r.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(8))),
    ("Mon numéro de dossier est {v}.", "Quel est mon numéro de dossier ?", "Votre numéro de dossier est {v}.",
     "votre numéro de dossier", lambda r: f"D-{r.randint(1000, 9999)}"),
]

DOC_TOPICS = [
    ("contrat", "Le contrat expire le {v}.", "Quand expire le contrat ?", "Le contrat expire le {v}.",
     lambda r: f"{r.randint(1, 28)}/{r.randint(1, 12):02d}/{r.randint(2026, 2029)}"),
    ("facture", "Le montant total de la facture est de {v} euros.", "Quel est le montant de la facture ?",
     "Le montant de la facture est de {v} euros.", lambda r: f"{r.randint(50, 9000)}"),
    ("reunion", "La réunion aura lieu en salle {v}.", "Dans quelle salle a lieu la réunion ?",
     "La réunion a lieu en salle {v}.", lambda r: f"{r.choice('ABCDE')}{r.randint(1, 40)}"),
    ("serveur", "L'adresse du serveur est {v}.", "Quelle est l'adresse du serveur ?", "L'adresse du serveur est {v}.",
     lambda r: f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"),
    ("recette", "Il faut {v} grammes de farine.", "Combien de farine faut-il ?", "Il faut {v} grammes de farine.",
     lambda r: f"{r.randint(100, 900)}"),
]
FILLER_DOC_NAMES = ["notes", "brouillon", "todo", "lecture", "journal", "annexe", "divers", "memo"]


def _stamp(rng, i):
    return f"2026-09-{rng.randint(1, 28):02d}T{rng.randint(8, 19):02d}:{i % 60:02d}"


def _filler_turn(rng, role, i):
    tpl = rng.choice(FILLERS_USER if role == "user" else FILLERS_ASSISTANT)
    text = tpl.format(sujet=rng.choice(SUJETS))
    return Segment(header=format_header({"role": role, "t": _stamp(rng, i)}), content=text.encode("utf-8"))


def _history(rng, n_pairs, injections):
    """Construit n_pairs paires (user, assistant), du plus ANCIEN au plus récent, en injectant
    `injections` = {index_de_tour_user: texte}. Renvoie la liste du plus RÉCENT au plus ancien."""
    turns = []
    for i in range(n_pairs):
        u = _filler_turn(rng, "user", 2 * i)
        if i in injections:
            u = Segment(header=u.header, content=injections[i].encode("utf-8"))
        turns.append(u)
        turns.append(_filler_turn(rng, "assistant", 2 * i + 1))
    return turns[::-1]


def _filler_doc(rng, i):
    name = f"{rng.choice(FILLER_DOC_NAMES)}_{i}.txt"
    body = " ".join(rng.choice(FILLERS_USER).format(sujet=rng.choice(SUJETS)) for _ in range(rng.randint(2, 25)))
    return Segment(header=format_header({"type": "doc", "name": name, "mime": "text/plain",
                                         "bytes": len(body.encode()), "t": _stamp(rng, i)}),
                   content=body.encode("utf-8"))


def oracle_pass(history_segments, needed_hist, docs, needed_doc, doc_relevant) -> Pass:
    hist = []
    collected = set()
    n = len(history_segments)
    for k, s in enumerate(history_segments):
        collected.add(k)
        if needed_hist and needed_hist <= collected and needed_doc is None:
            after = C.STOP                      # tout ce qu'il faut est lu
        elif k == n - 1:
            after = C.NEXT if docs else C.STOP  # fin de l'historique
        elif not needed_hist and needed_doc is not None:
            after = C.NEXT                      # rien dans l'historique, un document est requis
        else:
            after = C.CONT
        hist.append(Segment(header=s.header, content=s.content, read=True, after=after))
        if after in (C.STOP, C.NEXT):
            break
    if not docs:
        return Pass(history=hist, docs=None)
    if hist and hist[-1].after != C.NEXT:
        return Pass(history=hist, docs=None)
    out_docs = []
    got = False
    m = len(docs)
    for j, d in enumerate(docs):
        read = bool(doc_relevant[j])
        if read and needed_doc is not None and j == needed_doc:
            got = True
        after = C.STOP if (got or j == m - 1) else C.CONT
        out_docs.append(Segment(header=d.header, content=d.content, read=read, after=after))
        if after == C.STOP:
            break
    return Pass(history=hist, docs=out_docs)


def _fact_recall(rng):
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(3, 14); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f[0].format(v=v)})
    idx = 2 * (n - 1 - at) + 1                        # position dans l'ordre récent -> ancien
    p = oracle_pass(hist, {idx}, [], None, [])
    return TapeSpec(query=f[1].encode(), passes=[p], answer_parts=[f[2].format(v=v).encode()], notes=[])


def _variable_tracking(rng):
    n = rng.randint(3, 10); x = rng.randint(1, 9); ops = {}
    at = sorted(rng.sample(range(n), k=min(n, rng.randint(1, 3))))
    val = x
    ops[at[0]] = f"Note que x vaut {x}."
    for a in at[1:]:
        d = rng.randint(1, 5); val += d
        ops[a] = f"x augmente de {d}."
    hist = _history(rng, n, ops)
    needed = {2 * (n - 1 - a) + 1 for a in at}
    p = oracle_pass(hist, needed, [], None, [])
    return TapeSpec(query=b"Combien vaut x maintenant ?", passes=[p],
                    answer_parts=[f"x vaut {val}.".encode()], notes=[])


def _what_did_i_say(rng):
    sujet = rng.choice(SUJETS); v = rng.choice(["c'est urgent", "c'est reporté", "c'est terminé", "il faut un budget"])
    n = rng.randint(3, 12); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f"À propos de {sujet} : {v}."})
    p = oracle_pass(hist, {2 * (n - 1 - at) + 1}, [], None, [])
    return TapeSpec(query=f"Qu'ai-je dit sur {sujet} ?".encode(), passes=[p],
                    answer_parts=[f"Vous avez dit : « {v} ».".encode()], notes=[])


def _docs_for(rng, topic, sentence, n_docs, noise):
    docs, relevant = [], []
    target = rng.randint(0, n_docs - 1)
    for j in range(n_docs):
        if j == target:
            body = " ".join([rng.choice(FILLERS_ASSISTANT).format(sujet=rng.choice(SUJETS))
                             for _ in range(rng.randint(2, 40))] + [sentence] +
                            [rng.choice(FILLERS_USER).format(sujet=rng.choice(SUJETS)) for _ in range(rng.randint(0, 40))])
            name = f"{topic}_{rng.randint(1, 99)}.txt"
            docs.append(Segment(header=format_header({"type": "doc", "name": name, "mime": "text/plain",
                                                      "bytes": len(body.encode()), "t": _stamp(rng, j)}),
                                content=body.encode("utf-8")))
            relevant.append(True)
        else:
            d = _filler_doc(rng, j)
            if noise and rng.random() < 0.10:      # en-tête trompeur : nom pertinent, contenu non
                d = Segment(header=format_header({"type": "doc", "name": f"{topic}_ancien.txt", "mime": "text/plain",
                                                  "bytes": len(d.content), "t": _stamp(rng, j)}), content=d.content)
                relevant.append(True)
            else:
                relevant.append(False)
            docs.append(d)
    return docs, relevant, target


def _doc_lookup(rng):
    topic, tpl, q, a, gen = rng.choice(DOC_TOPICS); v = gen(rng)
    hist = _history(rng, rng.randint(1, 4), {})
    docs, relevant, target = _docs_for(rng, topic, tpl.format(v=v), rng.randint(1, 8), noise=True)
    p = oracle_pass(hist, set(), docs, target, relevant)
    return TapeSpec(query=q.encode(), passes=[p], answer_parts=[a.format(v=v).encode()], notes=[])


def _absent(rng):
    topic, tpl, q, a, gen = rng.choice(DOC_TOPICS)
    hist = _history(rng, rng.randint(1, 5), {})
    n_docs = rng.randint(0, 6)
    docs = [_filler_doc(rng, j) for j in range(n_docs)]
    p = oracle_pass(hist, set(), docs, None, [False] * n_docs)
    return TapeSpec(query=q.encode(), passes=[p],
                    answer_parts=[b"Je ne trouve pas cette information dans notre \xc3\xa9change ni dans les documents."],
                    notes=[])


def _two_facts(rng):
    fa, fb = rng.sample(FACTS, 2); va, vb = fa[4](rng), fb[4](rng)
    n = rng.randint(4, 14)
    at_a, at_b = sorted(rng.sample(range(n), 2), reverse=True)     # A plus récent que B
    hist = _history(rng, n, {at_a: fa[0].format(v=va), at_b: fb[0].format(v=vb)})
    ia, ib = 2 * (n - 1 - at_a) + 1, 2 * (n - 1 - at_b) + 1
    p1 = oracle_pass(hist, {ia}, [], None, [])
    p2 = oracle_pass(hist, {ia, ib}, [], None, [])
    q = f"{fa[1][:-2]} et {fb[3]} ?".encode()
    part1 = fa[2].format(v=va).encode()
    part2 = (" " + fb[2].format(v=vb)).encode()
    return TapeSpec(query=q, passes=[p1, p2], answer_parts=[part1, part2],
                    notes=[f"il manque {fb[3]}".encode()])


def _long_answer(rng):
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(2, 8); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f[0].format(v=v)})
    p = oracle_pass(hist, {2 * (n - 1 - at) + 1}, [], None, [])
    items = [f"{i + 1}. Point {i + 1} concernant {rng.choice(SUJETS)} : {rng.choice(FILLERS_ASSISTANT).format(sujet=rng.choice(SUJETS))}"
             for i in range(rng.randint(16, 40))]
    answer = (f[2].format(v=v) + " Voici le détail :\n" + "\n".join(items)).encode("utf-8")
    L = rng.randint(256, 1024)
    parts = [answer[i:i + L] for i in range(0, len(answer), L)]
    if len(parts) < 2:
        parts = [answer[:len(answer) // 2], answer[len(answer) // 2:]]
    passes = [p] * len(parts)
    notes = [b"r\xc3\xa9ponse longue, suite"] * (len(parts) - 1)
    return TapeSpec(query=(f[1] + " Donne-moi tous les détails.").encode("utf-8"), passes=passes,
                    answer_parts=parts, notes=notes)


_GEN = {"fact_recall": _fact_recall, "variable_tracking": _variable_tracking, "what_did_i_say": _what_did_i_say,
        "doc_lookup": _doc_lookup, "absent": _absent, "two_facts": _two_facts, "long_answer": _long_answer}


def generate_episode(rng: random.Random, kind: str | None = None) -> TapeSpec:
    if kind is None:
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS, k=1)[0]
    return _GEN[kind](rng)


def iter_episodes(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        yield generate_episode(rng)
```

- [ ] **Step 4 : vérifier** — `python -m pytest tests/test_episodes.py -q` → PASS. Si `test_every_kind_builds_a_valid_tape` échoue sur `validate_spec`, l'oracle produit une passe invalide : imprimer `[s.after for s in p.history]` et vérifier la règle « STOP unique et final ».

- [ ] **Step 5 : commit** — `git add relis/data/episodes.py tests/test_episodes.py && git commit -m "feat(data): épisodes synthétiques de lecture et oracle des décisions"`

---

### Task 5 : rembourrage, empaquetage et jeu de données ruban

**Files:**
- Create: `relis/data/pack.py`
- Test: `tests/test_pack.py`

**Interfaces:**
- Consumes: `Tape`, `build_tape`, `iter_episodes`, `WEIGHTS`, `codes.Mode`.
- Produces:
  - `PAD = 0x00`, `FLAG_RESET = 1`, `FLAG_SLOTS = 2`.
  - `pad_tape(tape: Tape, block: int) -> Tape` (rembourrage 0x00, mode SCAN, classe 0, reset False).
  - `pack_tapes(tapes: Iterable[Tape], seq_len: int, block: int) -> Iterator[dict]` : chaque élément `{"data": np.uint8[seq_len], "mode": np.uint8, "wclass": np.uint8, "flags": np.uint8, "n_tapes": int}` ; un ruban plus long que `seq_len` est ignoré et compté (`pack_tapes.skipped`, attribut posé sur la fonction après appel n'est pas fiable : renvoyer plutôt le compte via un paramètre `stats: dict` optionnel mis à jour en place).
  - `replay_tape(bin_path, seq_len, rng) -> Tape` : fenêtre brute du shard de pré-entraînement, mode SCAN, classe `W_ONE`, longueur exactement `seq_len`.
  - `class TapeShardWriter(out_dir)` : `add(packed: dict)`, `close()` ; écrit `data.bin`, `mode.bin`, `wclass.bin`, `flags.bin`, `meta.json` (`{"seq_len": L, "n": N}`).
  - `class TapeWindows(dir)` : `__len__ = n` ; `sample(batch_size, generator) -> dict` avec `x, y: Long (B, L-1)`, `w: Float (B, L-1)` (poids des cibles `y`), `mode, reset, slots_reset: (B, L-1)` (Long, Bool, Bool, alignés sur `x`). Attribut `has_decisions = True`.
  - CLI : `python -m relis.data.pack --out shards/tapes --episodes 20000 --seed 0 --seq_len 16384 --block 512 --replay shards/v1/train.bin --replay_frac 0.2 --val_frac 0.02`.

- [ ] **Step 1 : tests**

`tests/test_pack.py` :
```python
import json, os
import numpy as np
import torch
from relis.data.episodes import iter_episodes
from relis.data.pack import (pad_tape, pack_tapes, replay_tape, TapeShardWriter, TapeWindows,
                             PAD, FLAG_RESET, FLAG_SLOTS, build_shards)
from relis.data.shards import ShardWriter
from relis.tape.tape import build_tape, WEIGHTS


def test_pad_tape_aligns_to_block():
    t = build_tape(next(iter_episodes(0, 1)))
    p = pad_tape(t, 32)
    assert len(p) % 32 == 0 and len(p) - len(t) < 32
    assert all(b == PAD for b in p.data[len(t):]) and all(w == 0 for w in p.wclass[len(t):])


def test_pack_sets_flags_at_tape_starts_on_block_multiples():
    tapes = [build_tape(s) for s in iter_episodes(1, 40)]
    stats = {}
    seqs = list(pack_tapes(tapes, seq_len=4096, block=32, stats=stats))
    assert seqs and stats["skipped"] >= 0
    for s in seqs:
        assert s["data"].shape == (4096,) and s["data"].dtype == np.uint8
        starts = np.flatnonzero(s["flags"] & FLAG_SLOTS)
        assert len(starts) == s["n_tapes"] and starts[0] == 0
        assert all(st % 32 == 0 for st in starts)
        assert all(s["flags"][st] & FLAG_RESET for st in starts)
        # les resets internes (REFRESH) ne portent pas le bit slots
        internal = np.flatnonzero((s["flags"] & FLAG_RESET) & ~(s["flags"] & FLAG_SLOTS))
        assert all(st not in starts for st in internal)


def test_replay_tape_is_full_length_weight_one(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p); w.add(bytes(range(256)) * 40, "src=t"); w.close()
    import random
    t = replay_tape(p, 512, random.Random(0))
    assert len(t) == 512 and set(t.wclass) == {2} and not any(t.reset)


def test_writer_and_dataset_roundtrip(tmp_path):
    tapes = [build_tape(s) for s in iter_episodes(2, 30)]
    out = str(tmp_path / "tapes")
    wr = TapeShardWriter(out)
    n = 0
    for s in pack_tapes(tapes, seq_len=2048, block=32):
        wr.add(s); n += 1
    wr.close()
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["n"] == n and meta["seq_len"] == 2048
    ds = TapeWindows(out)
    assert len(ds) == n and ds.has_decisions
    b = ds.sample(3, torch.Generator().manual_seed(0))
    assert b["x"].shape == (3, 2047) and b["y"].shape == (3, 2047)
    assert torch.equal(b["x"][:, 1:], b["y"][:, :-1])
    assert b["w"].dtype == torch.float32 and set(b["w"].unique().tolist()) <= set(WEIGHTS)
    assert b["reset"].dtype == torch.bool and b["slots_reset"].dtype == torch.bool
    pos = torch.nonzero(b["slots_reset"])[:, 1]
    assert (pos % 32 == 0).all()


def test_build_shards_cli_function(tmp_path):
    p = str(tmp_path / "pre.bin")
    w = ShardWriter(p); w.add(("le chat dort. " * 300).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    stats = build_shards(out, episodes=60, seed=0, seq_len=2048, block=32, replay=p, replay_frac=0.2, val_frac=0.1)
    assert stats["train_seqs"] > 0 and stats["val_seqs"] > 0
    assert os.path.exists(os.path.join(out, "train", "data.bin")) and os.path.exists(os.path.join(out, "val", "meta.json"))
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_pack.py -q` → FAIL.

- [ ] **Step 3 : implémenter**

`relis/data/pack.py` :
```python
"""Rembourrage et empaquetage des rubans (spec §6.1 étape 2).

Quatre tableaux parallèles par position : octet, mode, classe de poids, drapeaux
(bit 0 : réinitialisation Mémoire, bit 1 : réinitialisation slots). Tout début de
ruban tombe sur un multiple de `block` et porte les deux bits.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from relis.tape import codes as C
from relis.tape.tape import Tape, WEIGHTS, W_ONE, build_tape
from .episodes import iter_episodes

PAD = 0x00
FLAG_RESET = 1
FLAG_SLOTS = 2
_WEIGHTS = torch.tensor(WEIGHTS, dtype=torch.float32)


def pad_tape(tape: Tape, block: int) -> Tape:
    n = (-len(tape)) % block
    out = Tape(bytearray(tape.data), list(tape.mode), list(tape.wclass), list(tape.reset))
    for _ in range(n):
        out.put(PAD, int(C.Mode.SCAN), 0)
    return out


def _to_arrays(tape: Tape):
    data = np.frombuffer(bytes(tape.data), dtype=np.uint8).copy()
    mode = np.asarray(tape.mode, dtype=np.uint8)
    wclass = np.asarray(tape.wclass, dtype=np.uint8)
    flags = np.asarray(tape.reset, dtype=np.uint8) * FLAG_RESET
    flags[0] |= FLAG_RESET | FLAG_SLOTS
    return data, mode, wclass, flags


def pack_tapes(tapes, seq_len: int, block: int, stats: dict | None = None):
    """Empaquetage glouton : on ajoute les rubans tant qu'ils tiennent ; le reste est rembourré."""
    assert seq_len % block == 0
    if stats is not None:
        stats.setdefault("skipped", 0); stats.setdefault("packed", 0)
    cur = []; used = 0; n_tapes = 0

    def flush():
        data = np.full(seq_len, PAD, dtype=np.uint8)
        mode = np.full(seq_len, int(C.Mode.SCAN), dtype=np.uint8)
        wclass = np.zeros(seq_len, dtype=np.uint8)
        flags = np.zeros(seq_len, dtype=np.uint8)
        off = 0
        for d, m, w, f in cur:
            data[off:off + len(d)] = d; mode[off:off + len(d)] = m
            wclass[off:off + len(d)] = w; flags[off:off + len(d)] = f
            off += len(d)
        return {"data": data, "mode": mode, "wclass": wclass, "flags": flags, "n_tapes": len(cur)}

    for t in tapes:
        p = pad_tape(t, block)
        if len(p) > seq_len:
            if stats is not None:
                stats["skipped"] += 1
            continue
        if used + len(p) > seq_len:
            yield flush()
            cur, used = [], 0
        cur.append(_to_arrays(p)); used += len(p)
        if stats is not None:
            stats["packed"] += 1
    if cur:
        yield flush()


def replay_tape(bin_path: str, seq_len: int, rng: random.Random) -> Tape:
    """Fenêtre brute du corpus de pré-entraînement, en ruban-document (mode SCAN, poids 1)."""
    data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    start = rng.randint(0, len(data) - seq_len - 1)
    t = Tape()
    for b in data[start:start + seq_len].tobytes():
        t.put(b, int(C.Mode.SCAN), W_ONE)
    return t


class TapeShardWriter:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.files = {k: open(os.path.join(out_dir, f"{k}.bin"), "wb") for k in ("data", "mode", "wclass", "flags")}
        self.n = 0
        self.seq_len = None

    def add(self, packed: dict) -> None:
        if self.seq_len is None:
            self.seq_len = int(packed["data"].shape[0])
        assert packed["data"].shape[0] == self.seq_len
        for k, f in self.files.items():
            f.write(np.ascontiguousarray(packed[k], dtype=np.uint8).tobytes())
        self.n += 1

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"seq_len": self.seq_len or 0, "n": self.n}, f)


class TapeWindows:
    """Séquences ruban empaquetées ; `sample` renvoie le lot décalé d'un octet avec ses masques."""
    has_decisions = True

    def __init__(self, dir: str):
        meta = json.load(open(os.path.join(dir, "meta.json"), encoding="utf-8"))
        self.seq_len, self.n = meta["seq_len"], meta["n"]
        shape = (self.n, self.seq_len)
        self.data = np.memmap(os.path.join(dir, "data.bin"), dtype=np.uint8, mode="r", shape=shape)
        self.mode = np.memmap(os.path.join(dir, "mode.bin"), dtype=np.uint8, mode="r", shape=shape)
        self.wclass = np.memmap(os.path.join(dir, "wclass.bin"), dtype=np.uint8, mode="r", shape=shape)
        self.flags = np.memmap(os.path.join(dir, "flags.bin"), dtype=np.uint8, mode="r", shape=shape)

    def __len__(self) -> int:
        return self.n

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict:
        idx = torch.randint(0, self.n, (batch_size,), generator=generator).tolist()
        data = torch.from_numpy(np.stack([self.data[i] for i in idx]).astype(np.int64))
        mode = torch.from_numpy(np.stack([self.mode[i] for i in idx]).astype(np.int64))
        wclass = torch.from_numpy(np.stack([self.wclass[i] for i in idx]).astype(np.int64))
        flags = torch.from_numpy(np.stack([self.flags[i] for i in idx]).astype(np.int64))
        return {"x": data[:, :-1], "y": data[:, 1:], "w": _WEIGHTS[wclass[:, 1:]],
                "mode": mode[:, :-1], "reset": (flags[:, :-1] & FLAG_RESET).bool(),
                "slots_reset": (flags[:, :-1] & FLAG_SLOTS).bool()}


def build_shards(out: str, episodes: int, seed: int, seq_len: int, block: int,
                 replay: str | None, replay_frac: float, val_frac: float, extra_tapes=None) -> dict:
    """Épisodes synthétiques (+ rubans externes, + rappel) → train/ et val/."""
    rng = random.Random(seed)
    tapes = [build_tape(s) for s in iter_episodes(seed, episodes)]
    if extra_tapes:
        tapes.extend(extra_tapes)
    rng.shuffle(tapes)
    n_val = max(1, int(len(tapes) * val_frac))
    splits = {"val": tapes[:n_val], "train": tapes[n_val:]}
    stats = {}
    for name, ts in splits.items():
        wr = TapeShardWriter(os.path.join(out, name))
        st = {}
        n = 0
        for s in pack_tapes(ts, seq_len, block, st):
            wr.add(s); n += 1
        if replay and replay_frac > 0 and n > 0:
            n_replay = int(round(n * replay_frac / max(1e-9, 1 - replay_frac)))
            for _ in range(n_replay):
                t = replay_tape(replay, seq_len, rng)
                d, m, w, f = _to_arrays(t)
                wr.add({"data": d, "mode": m, "wclass": w, "flags": f, "n_tapes": 1}); n += 1
        wr.close()
        stats[f"{name}_seqs"] = n
        stats[f"{name}_skipped"] = st.get("skipped", 0)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq_len", type=int, default=16384)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--replay", default=None, help="shard de pré-entraînement pour le rappel")
    ap.add_argument("--replay_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.02)
    args = ap.parse_args()
    stats = build_shards(args.out, args.episodes, args.seed, args.seq_len, args.block,
                         args.replay, args.replay_frac, args.val_frac)
    print(stats)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : vérifier** — `python -m pytest tests/test_pack.py -q` → PASS ; `python -m pytest -q` → tout PASS.

- [ ] **Step 5 : commit** — `git add relis/data/pack.py tests/test_pack.py && git commit -m "feat(data): rembourrage à 512, empaquetage en séquences fixes, jeu de données ruban"`

---

### Task 6 : boucle généralisée, métrique de décisions, entraînement ruban

**Files:**
- Modify: `relis/train/loop.py`
- Create: `relis/train/metrics.py`, `relis/train/tape_train.py`, `configs/tape_t4.yaml`
- Modify: `notebooks/KAGGLE.md`
- Test: `tests/test_tape_train.py`

**Interfaces:**
- Consumes: `TapeWindows`, `build_shards`, `RelisModel.forward(x, mode, state, reset, slots_reset)`, `checkpoint.load_checkpoint`, `pull_from_hub`.
- Produces:
  - `loop._as_batch(b) -> dict` (tuple `(x, y)` → `{"x","y"}`).
  - `loop._loss(model, batch: dict, device, amp)` : entropie croisée **pondérée** `Σ w·ce / Σ w` si `w` présent, moyenne sinon ; passe `mode` (défaut `1`), `reset`, `slots_reset` au modèle quand présents.
  - `TrainConfig.init_from: str | None = None` (chemin local vers `last.pt` ou dépôt Hub ; poids seuls, utilisé seulement si `run_dir` n'a pas de checkpoint à reprendre).
  - `loop.load_weights(model, init_from) -> int` (renvoie le pas du checkpoint source ; télécharge depuis le Hub si `init_from` n'est pas un fichier).
  - `train(..., extra_val=None)` : callable `model -> dict`, appelé (rang principal) à chaque sauvegarde et en fin de run, résultat imprimé.
  - `metrics.decision_accuracy(model, ds, n_batches, batch_size, device) -> dict` : `{"acc": float, "n": int, "per_code": {nom: [correct, total]}}` sur les positions où `y ∈ DECISION_CODES` et `w > 0`, prédiction = argmax sur les 256 octets.
  - `tape_train.main()` : `python -m relis.train.tape_train --config configs/tape_t4.yaml --run_dir runs/tape1 [--override ...]` (réutilise `_apply_overrides` et `resolve_data_path` de `pretrain.py` ; `data.train_dir`, `data.val_dir` sont des répertoires).

- [ ] **Step 1 : tests**

`tests/test_tape_train.py` :
```python
import os
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape import codes as C
from relis.data.pack import build_shards, TapeWindows
from relis.data.shards import ShardWriter
from relis.train.loop import TrainConfig, train, _loss, load_weights
from relis.train.metrics import decision_accuracy
from relis.train.checkpoint import save_checkpoint


def _tapes(tmp_path, seq_len=2048, block=32, episodes=60):
    pre = str(tmp_path / "pre.bin")
    w = ShardWriter(pre); w.add(("le chat dort. " * 400).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    build_shards(out, episodes=episodes, seed=0, seq_len=seq_len, block=block, replay=pre, replay_frac=0.2, val_frac=0.1)
    return TapeWindows(os.path.join(out, "train")), TapeWindows(os.path.join(out, "val"))


def _tcfg(**kw):
    base = dict(seq_len=2047, batch_size=2, grad_accum=1, lr=2e-3, warmup_steps=3, max_steps=15,
                weight_decay=0.1, grad_clip=1.0, amp=False, ckpt_every_minutes=1e9, log_every=5)
    base.update(kw)
    return TrainConfig(**base)


def test_weighted_loss_ignores_zero_weight_positions(tmp_path):
    tr, _ = _tapes(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    b = tr.sample(2, torch.Generator().manual_seed(0))
    l1 = _loss(m, b, "cpu", amp=False)
    b2 = {k: v.clone() for k, v in b.items()}
    zero = b2["w"] == 0
    b2["y"][zero] = (b2["y"][zero] + 7) % 256           # change les cibles là où le poids est nul
    l2 = _loss(m, b2, "cpu", amp=False)
    assert zero.any() and torch.allclose(l1, l2, atol=1e-6)


def test_loss_accepts_plain_tuple_batches():
    m = RelisModel(RelisConfig.tiny())
    x = torch.randint(0, 256, (1, 40)); y = torch.randint(0, 256, (1, 40))
    l = _loss(m, (x, y), "cpu", amp=False)
    assert torch.isfinite(l)


class _Oracle:
    """Faux modèle : prédit exactement la cible (logits one-hot)."""
    def __init__(self): self.cfg = RelisConfig.tiny(); self._y = None
    def eval(self): return self
    def train(self): return self
    def new_state(self, B, device): return None
    def __call__(self, x, mode, state, reset=None, slots_reset=None):
        return torch.nn.functional.one_hot(self._y, 256).float() * 10, state


def test_decision_accuracy_on_oracle_and_random(tmp_path):
    tr, _ = _tapes(tmp_path)
    o = _Oracle()
    import relis.train.metrics as M
    orig = tr.sample
    def spy(bs, g):
        b = orig(bs, g); o._y = b["y"]; return b
    tr.sample = spy
    r = M.decision_accuracy(o, tr, n_batches=2, batch_size=2, device="cpu")
    assert r["n"] > 0 and abs(r["acc"] - 1.0) < 1e-9
    assert set(r["per_code"]) <= {C.name(c) for c in C.DECISION_CODES}
    tr.sample = orig
    r2 = M.decision_accuracy(RelisModel(RelisConfig.tiny()), tr, n_batches=2, batch_size=2, device="cpu")
    assert 0.0 <= r2["acc"] <= 1.0 and r2["n"] > 0


def test_tape_training_decreases_loss_and_reports_decisions(tmp_path):
    torch.manual_seed(0)
    tr, va = _tapes(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    losses, seen = [], []
    out = train(m, _tcfg(), tr, va, str(tmp_path / "run"), device="cpu", resume=False,
                on_step=lambda s, l: losses.append(l),
                extra_val=lambda mm: seen.append(decision_accuracy(mm, va, 1, 2, "cpu")) or seen[-1])
    assert out["step"] == 15 and losses[-1] < losses[0]
    assert seen and "acc" in seen[-1]


def test_init_from_loads_weights_only(tmp_path):
    cfg = RelisConfig.tiny()
    src = RelisModel(cfg)
    save_checkpoint(str(tmp_path / "src"), src, None, None, step=42, cfg_dict=cfg.to_dict(), extra={})
    dst = RelisModel(cfg)
    step = load_weights(dst, str(tmp_path / "src" / "last.pt"))
    assert step == 42
    for a, b in zip(src.parameters(), dst.parameters()):
        assert torch.equal(a, b)
    tr, va = _tapes(tmp_path)
    out = train(dst, _tcfg(max_steps=2, init_from=str(tmp_path / "src" / "last.pt")), tr, va,
                str(tmp_path / "run2"), device="cpu", resume=True)
    assert out["step"] == 2      # init_from n'impose pas le pas source : le run repart de 0
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_tape_train.py -q` → FAIL.

- [ ] **Step 3 : implémenter `loop.py`**

Ajouter à `TrainConfig` : `init_from: str | None = None   # poids seuls, si run_dir n'a pas de checkpoint`.

Remplacer `_loss` par :
```python
def _as_batch(b) -> dict:
    if isinstance(b, dict):
        return b
    x, y = b
    return {"x": x, "y": y}


def _loss(model, batch, device, amp):
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
        return ce.mean()
    w = b["w"].to(device).reshape(-1).float()
    return (ce * w).sum() / w.sum().clamp_min(1e-6)
```

Ajouter après `_make_optimizer` :
```python
def load_weights(model, init_from: str) -> int:
    """Charge les poids (seulement) depuis un last.pt local ou un dépôt Hub ; renvoie le pas source."""
    path = init_from
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(init_from, "last.pt", token=os.environ.get("HF_TOKEN"), local_dir="hub_init")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = _unwrap(model).load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        print(f"[init_from] clés manquantes : {missing} ; inattendues : {unexpected}")
    return int(payload.get("step", 0))
```

Dans `train(...)` : signature `train(model, tcfg, train_ds, val_ds, run_dir, device="cuda", resume=True, on_step=None, extra_val=None)`. Après le bloc `if resume:` (et son `info`), ajouter :
```python
    if step == 0 and tcfg.init_from:
        src_step = load_weights(model, tcfg.init_from)
        if is_main:
            print(f"[init_from] poids chargés depuis {tcfg.init_from} (pas source {src_step})")
```
Dans `_save`, juste après `save_checkpoint(...)`, ajouter :
```python
        if extra_val is not None and not final:
            print(f"[val] {extra_val(_unwrap(model))}")
```
et en fin de `train`, après le calcul de `val_bpb` :
```python
    if is_main and extra_val is not None:
        print(f"[val] {extra_val(_unwrap(model))}")
```

Le `bits_per_byte` existant appelle `_loss(model, x, y, ...)` : le remplacer par `_loss(model, (x, y) if not isinstance(x, dict) else x, ...)`, en adaptant la boucle : `b = ds.sample(batch_size, g)` puis `loss = _loss(model, b, device, amp=False)` et `count += (_as_batch(b)["y"]).numel()`.

- [ ] **Step 4 : implémenter `metrics.py`**

```python
"""Exactitude des décisions contre l'oracle (spec §6.1, §9), en forçage."""
import torch

from relis.tape import codes as C
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
        is_dec &= w > 0
        for c in _DECISIONS:
            sel = is_dec & (y == c)
            n = int(sel.sum())
            if n:
                k = int((pred[sel] == c).sum())
                per_code[C.name(c)][0] += k; per_code[C.name(c)][1] += n
                correct += k; total += n
    model.train()
    return {"acc": correct / total if total else float("nan"), "n": total, "per_code": per_code}
```

- [ ] **Step 5 : implémenter `tape_train.py` et la configuration**

`relis/train/tape_train.py` :
```python
"""Entraînement ruban (spec §6.1 étape 2) : reprend les poids pré-entraînés.

    python -m relis.train.tape_train --config configs/tape_t4.yaml --run_dir runs/tape1
    torchrun --nproc_per_node=2 -m relis.train.tape_train --config configs/tape_t4.yaml --run_dir runs/tape1
"""
import argparse
import os

import torch
import yaml

from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.data.pack import TapeWindows
from .loop import TrainConfig, train
from .metrics import decision_accuracy
from .pretrain import _apply_overrides, resolve_data_path


def main(argv=None):
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

    train_dir = os.path.dirname(resolve_data_path(os.path.join(data["train_dir"], "meta.json")))
    val_dir = os.path.dirname(resolve_data_path(os.path.join(data["val_dir"], "meta.json")))
    train_ds, val_ds = TapeWindows(train_dir), TapeWindows(val_dir)
    extra = lambda m: decision_accuracy(m, val_ds, n_batches=4, batch_size=max(1, tcfg.batch_size), device=device)
    out = train(model, tcfg, train_ds, val_ds, args.run_dir, device=device,
                resume=not args.no_resume, extra_val=extra)
    if int(os.environ.get("RANK", "0")) == 0:
        print(out)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
```
Note : `resolve_data_path` cherche un fichier par son nom ; comme plusieurs `meta.json` peuvent exister sous `/kaggle/input`, préférer des chemins explicites dans la config Kaggle (`--override data.train_dir=…`).

`configs/tape_t4.yaml` :
```yaml
model:
  d_model: 768
  n_layers: 16
  swa_every: 4
  n_heads: 8
  head_dim: 64
  window: 512
  chunk: 64
  block: 512
  n_slots: 32
  mlp_mult: 4
  conv_kernel: 4
  grad_checkpoint: true
train:
  seq_len: 16383         # séquences empaquetées de 16 384 octets, décalées d'un
  batch_size: 2          # par GPU : 2 × 16 383 ≈ 32 k octets, même mémoire que le pré-entraînement
  grad_accum: 8          # 2 GPU × 2 × 8 × 16 383 ≈ 0,5 M octets par pas
  lr: 1.0e-4
  warmup_frac: 0.05
  max_steps: 1500        # ≈ 0,8 Go de rubans ; ajuster au débit mesuré
  weight_decay: 0.1
  grad_clip: 1.0
  amp: true
  ckpt_every_minutes: 30
  log_every: 20
  hub_repo: null         # ex. jaafar2022/relis-v1-tape
  hub_every_minutes: 120
  time_budget_hours: 11.5
  init_from: null        # ex. jaafar2022/relis-v1-pretrain (poids du pré-entraînement)
  compile: false
data:
  train_dir: /kaggle/input/relis-tapes/train
  val_dir: /kaggle/input/relis-tapes/val
```

Ajouter à `notebooks/KAGGLE.md` une section « Entraînement ruban (plan 2a) » :
````markdown
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
````

- [ ] **Step 6 : vérifier** — `python -m pytest tests/test_tape_train.py -q` → PASS ; `python -m pytest -q` → tout PASS (les tests du plan 1 sur `train`/`bits_per_byte` doivent rester verts : ils passent des tuples).

- [ ] **Step 7 : essai de bout en bout sur CPU (tiny)**

```bash
python -m relis.data.pack --out shards/tapes_tiny --episodes 200 --seq_len 512 --block 32 --val_frac 0.1
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/tiny.yaml", encoding="utf-8"))
c["train"].update(seq_len=511, batch_size=2, max_steps=30, warmup_steps=3, lr=2e-3)
c["data"] = {"train_dir": "shards/tapes_tiny/train", "val_dir": "shards/tapes_tiny/val"}
yaml.safe_dump(c, open("configs/tape_tiny.yaml", "w", encoding="utf-8"))
PY
python -m relis.train.tape_train --config configs/tape_tiny.yaml --run_dir runs/tape_tiny --device cpu
```
Expected : la perte pondérée descend nettement en 30 pas ; une ligne `[val] {'acc': ..., 'n': ..., 'per_code': {...}}` est imprimée en fin de run. Committer `configs/tape_tiny.yaml` (utile pour les tests futurs) ; `shards/` et `runs/` sont ignorés.

- [ ] **Step 8 : commit** — `git add relis/train/loop.py relis/train/metrics.py relis/train/tape_train.py configs/tape_t4.yaml configs/tape_tiny.yaml notebooks/KAGGLE.md tests/test_tape_train.py && git commit -m "feat(train): perte pondérée avec masques, exactitude des décisions, entraînement ruban depuis le pré-entraînement"`

---

## Auto-revue du plan

**Couverture de la spec (périmètre 2a).** §4.2 NOTE et `DECISION_CODES` : Task 1. §4.3 ruban complet, en-têtes échappés, canaux, décisions, NOTE/REFRESH/PART : Tasks 1, 3. §4.6 réinitialisation des slots par échantillon aux frontières de bloc : Task 2 (`slots_reset`) et Task 5 (drapeau bit 1 sur multiples de `block`). §4.7 décisions et positions : Task 3 (classes de poids ×5). §4.8 REFRESH exact et NOTE : Task 2 (masque, `reset_memory`), Task 3 (NOTE émise puis relue), Task 4 (`two_facts`, `long_answer`). §6.1 étape 2 : rembourrage 512, empaquetage 16 384, quatre tableaux, reprise des poids avec optimiseur neuf (1e-4, warmup 5 %), validation bpb + exactitude des décisions : Tasks 5, 6. §6.2 oracle exact, sept familles d'épisodes, bruit d'en-têtes 10 %, absent, REFRESH/NOTE : Task 4. §6.3 pondération : Tasks 3, 6. Hors périmètre volontaire (plan 2b) : §5 dialogues publics et étiquetage du professeur, §6.6 baseline.

**Placeholders.** Aucun ; chaque étape de code contient le code ; l'essai GPU réel reste une action utilisateur documentée dans `KAGGLE.md`.

**Cohérence des types.** `Tape.reset` (liste de bool) → `pack._to_arrays` (uint8 × FLAG_RESET) → `TapeWindows.sample` (`reset`, `slots_reset` bool alignés sur `x`) → `loop._loss` (`reset=`, `slots_reset=`) → `RelisModel.forward(x, mode, state, reset, slots_reset)` (Task 2) ; `mode` uint8 → Long `(B,L-1)` accepté par `forward` (tenseur) ; `w` = `WEIGHTS[wclass[:, 1:]]` est le poids de la cible `y`, cohérent avec `_loss` qui multiplie `ce` (calculée sur `y`) ; `decision_accuracy` lit les mêmes clés ; `Segment.after` ∈ {CONT, STOP, NEXT} vérifié par `validate_spec` ; `oracle_pass` renvoie des `Pass` valides (STOP unique et final) que `test_every_kind_builds_a_valid_tape` vérifie sur les sept familles ; `TrainConfig.init_from` est lu par `train` et testé par `test_init_from_loads_weights_only`.
