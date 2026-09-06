# RELIS — Plan 1 : le cœur du modèle et le pré-entraînement (semaine 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un paquet Python `relis` contenant le réseau complet de la spec (Gated DeltaNet chunké + attention locale + Buffer à slots + Mémoire d'état fixe), prouvé équivalent en mode chunk et en mode pas-à-pas, à mémoire constante, avec un script de pré-entraînement sur octets qui tourne sur Kaggle 2×T4 avec reprise depuis Hugging Face Hub.

**Architecture:** Un seul type de bloc résiduel (RMSNorm → mélangeur → lecture du Buffer → MLP), 16 blocs dont un sur quatre est une attention à fenêtre glissante et les autres du Gated DeltaNet en PyTorch pur. Le modèle traite une séquence par blocs de 512 octets : dans un bloc, toutes les positions lisent les slots du Buffer tels qu'ils étaient au début du bloc ; à la fin du bloc, le module d'écriture met à jour les slots chunk de 64 par chunk de 64. Le mode pas-à-pas (génération) suit exactement le même calendrier, ce qui rend les deux modes numériquement équivalents et testables l'un contre l'autre.

**Tech Stack:** Python ≥ 3.10, PyTorch ≥ 2.2 (CPU pour les tests, CUDA fp16 sur T4 pour l'entraînement), numpy, pytest, `datasets` (flux Hugging Face), `huggingface_hub` (checkpoints), PyYAML.

**Spec:** `docs/superpowers/specs/2026-09-06-relis-design.md` (§4.1, §4.2, §4.4, §4.5, §4.6, §5, §6.1 étape 1, §6.4, §6.5, §10, §11 S1). Les plans 2 à 4 couvriront les rubans et l'oracle (§4.3, §4.7, §4.8, §6.2, §6.3, §6.6), le contrôleur, les Extraits et les évaluations (§4.9, §7, §9), puis la démo (§8).

## Global Constraints

- Vocabulaire de sortie : exactement 256 symboles ; codes de contrôle dans 0x00-0x1F sauf 0x09, 0x0A, 0x0D (spec §4.2). Table des codes copiée verbatim dans `relis/tape/codes.py` (Task 2).
- Aucun kernel externe (pas de `mamba_ssm`, pas de `flash-linear-attention`, pas de Triton obligatoire) : PyTorch pur.
- L'état récurrent, les normalisations et les softmax sont calculés en **fp32** même sous autocast fp16 (spec §6.4). T4 n'a pas de bf16.
- Taille V1 : d = 768, 16 blocs, SWA aux positions 4, 8, 12, 16, 8 têtes de dimension 64, état 64, fenêtre 512, chunk 64, K = 32 slots, MLP ×4 (spec §4.4, §4.6).
- Mémoire constante : `State.size_bytes()` ne dépend pas du nombre d'octets lus (spec §4.5).
- Checkpoints toutes les 30 minutes, reprise automatique (spec §6.5).
- Tous les tests tournent sur CPU en moins de deux minutes au total, sur une configuration `tiny`.
- Un test échoue avant d'écrire le code qui le fait passer ; un commit par tâche au minimum.
- Précision apportée à la spec §4.6 par ce plan : les écritures du Buffer sont calculées par chunk de 64 octets mais **deviennent visibles aux lectures tous les 512 octets** (frontière de bloc). Cette latence de lecture rend les modes chunk et pas-à-pas équivalents et permet de traiter 512 positions par passage dans la pile.

---

## Structure des fichiers

```
pyproject.toml
README.md
.gitignore
configs/
  tiny.yaml                 configuration de test (CPU)
  pretrain_t4.yaml          configuration V1 pour Kaggle 2×T4
relis/
  __init__.py
  tape/
    __init__.py
    codes.py                codes de contrôle, sanitize(), adresses d'Extraits, Mode
  model/
    __init__.py
    config.py               RelisConfig (+ tiny())
    layers.py               RMSNorm fp32, SwiGLU, conv causale courte
    gdn.py                  Gated DeltaNet : gdn_chunked(), gdn_step(), module GatedDeltaNet
    swa.py                  attention à fenêtre glissante avec ALiBi, cache borné
    buffer.py               SlotRead (par bloc), SlotWrite (par chunk)
    state.py                State : états GDN, caches SWA, positions en attente ; reset(), size_bytes()
    relis.py               Block, RelisModel.forward() par blocs de 512, RelisModel.step()
  data/
    __init__.py
    shards.py               écriture/lecture de shards d'octets avec en-tête SEG
    prepare.py              flux HF → shards (wiki_fr, fineweb2_fr, code)
    dataset.py              fenêtres aléatoires depuis un memmap
  train/
    __init__.py
    checkpoint.py           save/load local + push/pull HF Hub
    loop.py                 boucle AMP fp16, AdamW, cosinus, clipping, journal
    pretrain.py             point d'entrée : config YAML → entraînement
notebooks/
  KAGGLE.md                 procédure de lancement Kaggle/Colab (une cellule)
tests/
  conftest.py
  test_codes.py
  test_layers.py
  test_gdn.py
  test_swa.py
  test_buffer.py
  test_relis.py
  test_shards.py
  test_dataset.py
  test_checkpoint.py
  test_loop.py
```

---

### Task 1 : squelette du paquet et dépôt git

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`, `relis/__init__.py`, `relis/tape/__init__.py`, `relis/model/__init__.py`, `relis/data/__init__.py`, `relis/train/__init__.py`, `tests/conftest.py`, `tests/test_import.py`

**Interfaces:**
- Produces: paquet importable `relis`, `relis.__version__ == "0.1.0"`.

- [ ] **Step 1 : initialiser le dépôt**

```bash
cd c:/Adil/PhD-2/hmi/ia102
git init -b main
```

- [ ] **Step 2 : écrire le test d'import**

`tests/test_import.py` :
```python
def test_import():
    import relis
    assert relis.__version__ == "0.1.0"
```

- [ ] **Step 3 : lancer le test, vérifier l'échec**

Run: `python -m pytest tests/test_import.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'relis'`

- [ ] **Step 4 : créer le paquet**

`pyproject.toml` :
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "relis"
version = "0.1.0"
description = "RELIS : un modèle de langage qui lit au lieu de se souvenir de tout"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2",
  "numpy>=1.24",
  "pyyaml>=6",
  "huggingface_hub>=0.23",
  "datasets>=2.19",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
include = ["relis*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`relis/__init__.py` :
```python
__version__ = "0.1.0"
```

Les quatre autres `__init__.py` sont vides.

`.gitignore` :
```
__pycache__/
*.pyc
*.egg-info/
build/
dist/
.pytest_cache/
shards/
runs/
*.bin
*.pt
```

`README.md` :
```markdown
# RELIS

Un modèle de langage qui lit pour répondre, décide combien lire, et montre ce qu'il a retenu.

Spec : `docs/superpowers/specs/2026-09-06-relis-design.md`.

    pip install -e ".[dev]"
    python -m pytest
```

`tests/conftest.py` :
```python
import torch
import pytest


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    yield
```

- [ ] **Step 5 : installer et vérifier**

Run: `pip install -e ".[dev]" && python -m pytest tests/test_import.py -v`
Expected: PASS

- [ ] **Step 6 : commit**

```bash
git add -A
git commit -m "chore: squelette du paquet relis et dépôt git"
```

---

### Task 2 : codes de contrôle et adresses

**Files:**
- Create: `relis/tape/codes.py`
- Test: `tests/test_codes.py`

**Interfaces:**
- Produces:
  - constantes `int` : `ENC=0x01, SEG=0x02, END=0x03, STOP=0x04, REFRESH=0x05, NEXT=0x06, CONT=0x07, SKIP=0x08, SCAN=0x0E, GEN=0x0F, DEC=0x10, READ=0x11, PART=0x12, RECALL=0x13, CHAN=0x1C, HDR=0x1F`
  - `TEXT_CONTROL = {0x09, 0x0A, 0x0D}` ; `CONTROL_CODES: frozenset[int]` (tous les octets 0x00-0x1F sauf `TEXT_CONTROL`)
  - `class Mode(IntEnum): ENCODE=0, SCAN=1, GENERATE=2`
  - `sanitize(data: bytes) -> bytes` : remplace chaque octet de `CONTROL_CODES` par `b"\xef\xbf\xbd"` (U+FFFD)
  - `encode_extrait_address(index: int) -> bytes` (2 octets) et `decode_address(data: bytes) -> tuple[str, int]` renvoyant `("extrait", index)` ; `ADDR_BASE=0x20`, `ADDR_RANGE=224`, `EXTRAIT_FIRST_MAX=0x8F`, `MAX_EXTRAITS=4096`
  - `address_length(first_byte: int) -> int` : 2 pour 0x20-0x8F, 3 pour 0x90-0xFF

- [ ] **Step 1 : écrire les tests**

`tests/test_codes.py` :
```python
import pytest
from relis.tape import codes as C


def test_control_table_matches_spec():
    assert (C.ENC, C.SEG, C.END, C.STOP, C.REFRESH, C.NEXT, C.CONT, C.SKIP) == (
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)
    assert (C.SCAN, C.GEN, C.DEC, C.READ, C.PART, C.RECALL, C.CHAN, C.HDR) == (
        0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x1C, 0x1F)


def test_control_codes_exclude_text_controls():
    assert 0x09 not in C.CONTROL_CODES
    assert 0x0A not in C.CONTROL_CODES
    assert 0x0D not in C.CONTROL_CODES
    assert C.ENC in C.CONTROL_CODES
    assert len(C.CONTROL_CODES) == 32 - 3


def test_sanitize_replaces_control_bytes_keeps_text_controls():
    raw = b"a\x01b\tc\nd\re\x1f"
    out = C.sanitize(raw)
    assert out == b"a\xef\xbf\xbdb\tc\nd\re\xef\xbf\xbd"
    assert all(b not in C.CONTROL_CODES for b in out)


def test_mode_values():
    assert int(C.Mode.ENCODE) == 0
    assert int(C.Mode.SCAN) == 1
    assert int(C.Mode.GENERATE) == 2


@pytest.mark.parametrize("index", [0, 1, 223, 224, 4095])
def test_extrait_address_roundtrip(index):
    addr = C.encode_extrait_address(index)
    assert len(addr) == 2
    assert C.ADDR_BASE <= addr[0] <= C.EXTRAIT_FIRST_MAX
    assert C.ADDR_BASE <= addr[1] <= 0xFF
    assert C.address_length(addr[0]) == 2
    assert C.decode_address(addr) == ("extrait", index)


def test_extrait_address_out_of_range():
    with pytest.raises(ValueError):
        C.encode_extrait_address(C.MAX_EXTRAITS)
    with pytest.raises(ValueError):
        C.encode_extrait_address(-1)


def test_address_length_lexique_reserved():
    assert C.address_length(0x90) == 3
    assert C.address_length(0xFF) == 3
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_codes.py -v`
Expected: FAIL avec `ImportError` / `AttributeError`

- [ ] **Step 3 : implémenter**

`relis/tape/codes.py` :
```python
"""Codes de contrôle du ruban (spec §4.2) et adresses de cellules.

Le vocabulaire est exactement 256 octets. Les octets 0x00-0x1F, sauf
\\t, \\n, \\r, sont réservés comme codes de contrôle.
"""
from enum import IntEnum

# Marqueurs (insérés par le contrôleur)
ENC = 0x01
SEG = 0x02
SCAN = 0x0E
GEN = 0x0F
DEC = 0x10
PART = 0x12
CHAN = 0x1C
HDR = 0x1F
# Actions (prédites par le modèle)
END = 0x03
STOP = 0x04
REFRESH = 0x05
NEXT = 0x06
CONT = 0x07
SKIP = 0x08
READ = 0x11
RECALL = 0x13

TEXT_CONTROL = frozenset({0x09, 0x0A, 0x0D})
CONTROL_CODES = frozenset(b for b in range(0x20) if b not in TEXT_CONTROL)

REPLACEMENT = b"\xef\xbf\xbd"  # U+FFFD


class Mode(IntEnum):
    ENCODE = 0
    SCAN = 1
    GENERATE = 2


_SANITIZE_TABLE = {b: REPLACEMENT for b in CONTROL_CODES}


def sanitize(data: bytes) -> bytes:
    """Remplace chaque octet de contrôle d'un texte d'entrée par U+FFFD."""
    if not any(b in CONTROL_CODES for b in data):
        return bytes(data)
    return b"".join(_SANITIZE_TABLE.get(b, bytes((b,))) for b in data)


# Adresses de cellules (spec §4.2) : octets dans 0x20-0xFF.
ADDR_BASE = 0x20
ADDR_RANGE = 0x100 - ADDR_BASE  # 224
EXTRAIT_FIRST_MAX = 0x8F        # premier octet 0x20-0x8F => Extrait, 2 octets
MAX_EXTRAITS = 4096


def address_length(first_byte: int) -> int:
    if first_byte < ADDR_BASE:
        raise ValueError(f"octet d'adresse invalide : {first_byte:#x}")
    return 2 if first_byte <= EXTRAIT_FIRST_MAX else 3


def encode_extrait_address(index: int) -> bytes:
    if not 0 <= index < MAX_EXTRAITS:
        raise ValueError(f"index d'Extrait hors plage : {index}")
    hi, lo = divmod(index, ADDR_RANGE)
    return bytes((ADDR_BASE + hi, ADDR_BASE + lo))


def decode_address(data: bytes) -> tuple[str, int]:
    if len(data) < 2:
        raise ValueError("adresse trop courte")
    first = data[0]
    if address_length(first) == 2:
        index = (first - ADDR_BASE) * ADDR_RANGE + (data[1] - ADDR_BASE)
        return ("extrait", index)
    if len(data) < 3:
        raise ValueError("adresse de Lexique : 3 octets attendus")
    index = ((first - 0x90) * ADDR_RANGE + (data[1] - ADDR_BASE)) * ADDR_RANGE + (data[2] - ADDR_BASE)
    return ("lexique", index)
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_codes.py -v`
Expected: 11 PASS

- [ ] **Step 5 : commit**

```bash
git add relis/tape/codes.py tests/test_codes.py
git commit -m "feat(tape): codes de contrôle et adresses d'Extraits"
```

---

### Task 3 : configuration et couches communes

**Files:**
- Create: `relis/model/config.py`, `relis/model/layers.py`, `configs/tiny.yaml`, `configs/pretrain_t4.yaml`
- Test: `tests/test_layers.py`

**Interfaces:**
- Produces:
  - `@dataclass RelisConfig(d_model=768, n_layers=16, swa_every=4, n_heads=8, head_dim=64, window=512, chunk=64, block=512, n_slots=32, mlp_mult=4, conv_kernel=4, vocab=256, n_modes=3)` avec `is_swa(i: int) -> bool` (vrai si `(i+1) % swa_every == 0`), `inner` (= `n_heads*head_dim`), `classmethod tiny()` (d_model=64, n_layers=4, swa_every=4, n_heads=2, head_dim=16, window=16, chunk=8, block=32, n_slots=4, mlp_mult=2, conv_kernel=4), `from_yaml(path)`.
  - `RMSNorm(d, eps=1e-6)` : calcul en fp32, sortie au dtype d'entrée.
  - `SwiGLU(d, mult)` : `w1, w3: Linear(d, mult*d, bias=False)`, `w2: Linear(mult*d, d, bias=False)`.
  - `CausalConv1d(channels, kernel)` : convolution dépthwise causale ; `forward(x: (B,L,C), conv_state: (B,C,kernel-1)|None) -> (y, new_conv_state)` ; `new_conv_state` = les `kernel-1` dernières entrées.

- [ ] **Step 1 : écrire les tests**

`tests/test_layers.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.layers import RMSNorm, SwiGLU, CausalConv1d


def test_config_defaults_match_spec():
    c = RelisConfig()
    assert (c.d_model, c.n_layers, c.n_heads, c.head_dim) == (768, 16, 8, 64)
    assert (c.window, c.chunk, c.block, c.n_slots) == (512, 64, 512, 32)
    assert [i for i in range(c.n_layers) if c.is_swa(i)] == [3, 7, 11, 15]
    assert c.inner == 512


def test_tiny_config_is_small_and_consistent():
    c = RelisConfig.tiny()
    assert c.d_model == 64 and c.n_layers == 4
    assert c.block % c.chunk == 0
    assert sum(c.is_swa(i) for i in range(c.n_layers)) == 1


def test_rmsnorm_fp32_and_shape():
    n = RMSNorm(8)
    x = torch.randn(2, 5, 8, dtype=torch.float16)
    y = n(x)
    assert y.dtype == torch.float16 and y.shape == x.shape
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y.float(), ref, atol=1e-2)


def test_swiglu_shape():
    m = SwiGLU(8, 2)
    assert m(torch.randn(3, 8)).shape == (3, 8)


def test_causal_conv_matches_streaming():
    conv = CausalConv1d(6, 4)
    x = torch.randn(2, 11, 6)
    y_full, st_full = conv(x, None)
    ys, st = [], None
    for t in range(11):
        y_t, st = conv(x[:, t:t + 1], st)
        ys.append(y_t)
    y_stream = torch.cat(ys, dim=1)
    assert torch.allclose(y_full, y_stream, atol=1e-6)
    assert torch.allclose(st_full, st, atol=1e-6)
    assert st.shape == (2, 6, 3)


def test_causal_conv_is_causal():
    conv = CausalConv1d(4, 4)
    x = torch.randn(1, 9, 4)
    y1, _ = conv(x, None)
    x2 = x.clone(); x2[:, 6] += 10.0
    y2, _ = conv(x2, None)
    assert torch.allclose(y1[:, :6], y2[:, :6])
    assert not torch.allclose(y1[:, 6], y2[:, 6])
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_layers.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/model/config.py` :
```python
from dataclasses import dataclass, asdict
import yaml


@dataclass
class RelisConfig:
    d_model: int = 768
    n_layers: int = 16
    swa_every: int = 4
    n_heads: int = 8
    head_dim: int = 64
    window: int = 512
    chunk: int = 64
    block: int = 512
    n_slots: int = 32
    mlp_mult: int = 4
    conv_kernel: int = 4
    vocab: int = 256
    n_modes: int = 3

    def __post_init__(self):
        assert self.block % self.chunk == 0, "block doit être un multiple de chunk"

    @property
    def inner(self) -> int:
        return self.n_heads * self.head_dim

    def is_swa(self, i: int) -> bool:
        return (i + 1) % self.swa_every == 0

    @classmethod
    def tiny(cls) -> "RelisConfig":
        return cls(d_model=64, n_layers=4, swa_every=4, n_heads=2, head_dim=16,
                   window=16, chunk=8, block=32, n_slots=4, mlp_mult=2, conv_kernel=4)

    @classmethod
    def from_yaml(cls, path: str) -> "RelisConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw.get("model", {}))

    def to_dict(self) -> dict:
        return asdict(self)
```

`relis/model/layers.py` :
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm calculée en fp32 (spec §6.4), sortie au dtype d'entrée."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int):
        super().__init__()
        h = mult * d
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalConv1d(nn.Module):
    """Convolution depthwise causale courte, avec état pour le mode pas-à-pas.

    x : (B, L, C). conv_state : (B, C, kernel-1) = dernières entrées vues.
    """

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.channels = channels
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=True)

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor | None):
        B, L, C = x.shape
        xt = x.transpose(1, 2)  # (B, C, L)
        if conv_state is None:
            conv_state = xt.new_zeros(B, C, self.kernel - 1)
        xcat = torch.cat([conv_state.to(xt.dtype), xt], dim=2)  # (B, C, k-1+L)
        y = self.conv(xcat)  # (B, C, L)
        new_state = xcat[:, :, -(self.kernel - 1):]
        return y.transpose(1, 2), new_state
```

`configs/tiny.yaml` :
```yaml
model:
  d_model: 64
  n_layers: 4
  swa_every: 4
  n_heads: 2
  head_dim: 16
  window: 16
  chunk: 8
  block: 32
  n_slots: 4
  mlp_mult: 2
  conv_kernel: 4
train:
  seq_len: 128
  batch_size: 4
  grad_accum: 1
  lr: 1.0e-3
  warmup_steps: 10
  max_steps: 60
  weight_decay: 0.1
  grad_clip: 1.0
  amp: false
  ckpt_every_minutes: 30
  log_every: 10
data:
  train_bin: shards/tiny/train.bin
  val_bin: shards/tiny/val.bin
```

`configs/pretrain_t4.yaml` :
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
train:
  seq_len: 4096
  batch_size: 8          # par GPU
  grad_accum: 8          # batch effectif 2 GPU × 8 × 8 × 4096 ≈ 0,5 M octets (spec §6.1)
  lr: 3.0e-4
  warmup_steps: 2000
  max_steps: 4000        # ≈ 2 Go d'octets ; ajuster selon le débit mesuré
  weight_decay: 0.1
  grad_clip: 1.0
  amp: true
  ckpt_every_minutes: 30
  log_every: 20
  hub_repo: null         # ex. "utilisateur/relis-v1-pretrain" ; token dans HF_TOKEN
data:
  train_bin: /kaggle/input/relis-shards/train.bin
  val_bin: /kaggle/input/relis-shards/val.bin
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_layers.py -v`
Expected: 6 PASS

- [ ] **Step 5 : commit**

```bash
git add relis/model/config.py relis/model/layers.py configs/ tests/test_layers.py
git commit -m "feat(model): configuration, RMSNorm fp32, SwiGLU, convolution causale"
```

---

### Task 4 : Gated DeltaNet chunké et pas-à-pas

**Files:**
- Create: `relis/model/gdn.py`
- Test: `tests/test_gdn.py`

**Interfaces:**
- Consumes: `RMSNorm`, `CausalConv1d`, `RelisConfig` (Task 3).
- Produces:
  - `gdn_chunked(q, k, v, log_alpha, beta, S, chunk) -> (o, S_new)` : `q,k: (B,H,L,dk)` fp32 avec `k` L2-normalisé, `v: (B,H,L,dv)`, `log_alpha, beta: (B,H,L)`, `S: (B,H,dk,dv)` fp32 ; renvoie `o: (B,H,L,dv)` fp32 et le nouvel état.
  - `gdn_step(q, k, v, log_alpha, beta, S) -> (o, S_new)` : mêmes formes sans dimension L.
  - `class GatedDeltaNet(cfg)` : `forward(x: (B,L,d), S: (B,H,dk,dv)|None, conv_state|None) -> (y: (B,L,d), S, conv_state)` ; `step(x: (B,1,d), S, conv_state)` même signature via `gdn_step` ; `init_state(B, device) -> (S, conv_state)`.

Récurrence de référence (convention vecteurs-lignes, `S ∈ R^{dk×dv}`, `o_t = q_t S_t`) :

```
S_t = α_t S_{t-1} + β_t k_tᵀ (v_t − α_t k_t S_{t-1})
```

Algorithme chunké (dérivé dans le plan, vérifié par le test d'équivalence) : dans un chunk de C positions, avec `γ_t = Π_{j≤t} α_j`, `Γ = diag(γ)`, `M_{t,i} = (γ_t/γ_i) β_i (k_t·k_i)` pour i<t (strictement triangulaire inférieure), `P_{t,i} = (γ_t/γ_i) β_i (q_t·k_i)` pour i≤t, `T = (I+M)^{-1}`, `U = T V`, `W = T Γ K` :

```
U_eff = U − W S_0
O     = Γ Q S_0 + P U_eff
S_C   = γ_C S_0 + Kᵀ diag(γ_C/γ_i · β_i) U_eff
```

`T`, `U`, `W`, `P` se calculent en parallèle sur tous les chunks ; seule la boucle sur `S` est séquentielle (L/64 itérations).

- [ ] **Step 1 : écrire les tests**

`tests/test_gdn.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.gdn import gdn_chunked, gdn_step, GatedDeltaNet


def _inputs(B=2, H=2, L=37, dk=8, dv=8):
    q = torch.randn(B, H, L, dk)
    k = torch.nn.functional.normalize(torch.randn(B, H, L, dk), dim=-1)
    v = torch.randn(B, H, L, dv)
    log_alpha = -torch.rand(B, H, L) * 0.5          # α ∈ (0.6, 1)
    beta = torch.sigmoid(torch.randn(B, H, L))
    S = torch.randn(B, H, dk, dv) * 0.1
    return q, k, v, log_alpha, beta, S


def _reference(q, k, v, log_alpha, beta, S):
    outs = []
    for t in range(q.shape[2]):
        o, S = gdn_step(q[:, :, t], k[:, :, t], v[:, :, t], log_alpha[:, :, t], beta[:, :, t], S)
        outs.append(o)
    return torch.stack(outs, dim=2), S


def test_step_matches_definition():
    q, k, v, la, be, S = _inputs(L=1)
    a = la[:, :, 0].exp()[..., None, None]; b = be[:, :, 0][..., None, None]
    kt = k[:, :, 0]; vt = v[:, :, 0]; qt = q[:, :, 0]
    kS = kt.unsqueeze(-2) @ S
    S_ref = a * S + b * (kt.unsqueeze(-1) @ (vt.unsqueeze(-2) - a * kS))
    o_ref = (qt.unsqueeze(-2) @ S_ref).squeeze(-2)
    o, S_new = gdn_step(qt, kt, vt, la[:, :, 0], be[:, :, 0], S)
    assert torch.allclose(S_new, S_ref, atol=1e-6)
    assert torch.allclose(o, o_ref, atol=1e-6)


def test_chunked_equals_recurrent_with_padding():
    q, k, v, la, be, S = _inputs(L=37)           # 37 n'est pas un multiple de 8
    o_ref, S_ref = _reference(q, k, v, la, be, S)
    o, S_new = gdn_chunked(q, k, v, la, be, S, chunk=8)
    assert torch.allclose(o, o_ref, atol=1e-4), (o - o_ref).abs().max()
    assert torch.allclose(S_new, S_ref, atol=1e-4)


def test_chunked_state_carry_across_calls():
    q, k, v, la, be, S = _inputs(L=40)
    o_ref, S_ref = _reference(q, k, v, la, be, S)
    o1, S1 = gdn_chunked(q[:, :, :13], k[:, :, :13], v[:, :, :13], la[:, :, :13], be[:, :, :13], S, chunk=8)
    o2, S2 = gdn_chunked(q[:, :, 13:], k[:, :, 13:], v[:, :, 13:], la[:, :, 13:], be[:, :, 13:], S1, chunk=8)
    assert torch.allclose(torch.cat([o1, o2], dim=2), o_ref, atol=1e-4)
    assert torch.allclose(S2, S_ref, atol=1e-4)


def test_module_forward_equals_steps():
    cfg = RelisConfig.tiny()
    m = GatedDeltaNet(cfg)
    x = torch.randn(2, 21, cfg.d_model)
    y_full, S_full, cs_full = m(x, None, None)
    S, cs = m.init_state(2, x.device)
    ys = []
    for t in range(21):
        y_t, S, cs = m.step(x[:, t:t + 1], S, cs)
        ys.append(y_t)
    y_steps = torch.cat(ys, dim=1)
    assert torch.allclose(y_full, y_steps, atol=1e-4), (y_full - y_steps).abs().max()
    assert torch.allclose(S_full, S, atol=1e-4)
    assert torch.allclose(cs_full, cs, atol=1e-5)


def test_state_is_fp32_under_fp16_inputs():
    cfg = RelisConfig.tiny()
    m = GatedDeltaNet(cfg)
    x = torch.randn(1, 9, cfg.d_model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, S, _ = m(x, None, None)
    assert S.dtype == torch.float32
    assert y.shape == x.shape
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_gdn.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/model/gdn.py` :
```python
"""Gated DeltaNet en PyTorch pur (spec §4.4).

Convention vecteurs-lignes : S ∈ R^{dk×dv}, o_t = q_t S_t,
S_t = α_t S_{t-1} + β_t k_tᵀ (v_t − α_t k_t S_{t-1}).
Tout le cœur numérique est en fp32 ; l'état est toujours fp32.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RelisConfig
from .layers import RMSNorm, CausalConv1d


def gdn_step(q, k, v, log_alpha, beta, S):
    """Un pas de récurrence. q,k:(B,H,dk) v:(B,H,dv) log_alpha,beta:(B,H) S:(B,H,dk,dv)."""
    q, k, v, S = q.float(), k.float(), v.float(), S.float()
    a = log_alpha.float().exp()[..., None, None]
    b = beta.float()[..., None, None]
    kS = k.unsqueeze(-2) @ S                                  # (B,H,1,dv)
    S_new = a * S + b * (k.unsqueeze(-1) @ (v.unsqueeze(-2) - a * kS))
    o = (q.unsqueeze(-2) @ S_new).squeeze(-2)                 # (B,H,dv)
    return o, S_new


def gdn_chunked(q, k, v, log_alpha, beta, S, chunk: int):
    """Forme chunkée, équivalente à gdn_step appliqué L fois.

    q,k:(B,H,L,dk) (k L2-normalisé) v:(B,H,L,dv) log_alpha,beta:(B,H,L) S:(B,H,dk,dv)
    """
    q, k, v = q.float(), k.float(), v.float()
    log_alpha, beta, S = log_alpha.float(), beta.float(), S.float()
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    C = chunk
    pad = (-L) % C
    if pad:
        # Positions de remplissage neutres : α=1 (log 0), β=0, k=q=v=0.
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        log_alpha = F.pad(log_alpha, (0, pad))
        beta = F.pad(beta, (0, pad))
    Lp = L + pad
    N = Lp // C
    q = q.view(B, H, N, C, dk)
    k = k.view(B, H, N, C, dk)
    v = v.view(B, H, N, C, dv)
    log_alpha = log_alpha.view(B, H, N, C)
    beta = beta.view(B, H, N, C)

    lg = torch.cumsum(log_alpha, dim=-1)                       # log γ_t dans le chunk
    gamma = lg.exp()                                           # (B,H,N,C)
    diff = lg.unsqueeze(-1) - lg.unsqueeze(-2)                 # (t,i) -> log(γ_t/γ_i)
    tri_strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    tri_incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(0)
    ratio_strict = diff.masked_fill(~tri_strict, float("-inf")).exp()
    ratio_incl = diff.masked_fill(~tri_incl, float("-inf")).exp()

    KK = k @ k.transpose(-1, -2)                               # (B,H,N,C,C)
    M = ratio_strict * beta.unsqueeze(-2) * KK                 # β_i sur la colonne i
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    T = torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False, unitriangular=True)
    U = T @ v                                                  # (B,H,N,C,dv)
    W = T @ (gamma.unsqueeze(-1) * k)                          # (B,H,N,C,dk)
    P = ratio_incl * beta.unsqueeze(-2) * (q @ k.transpose(-1, -2))
    gamma_C = gamma[..., -1]                                   # (B,H,N)
    decay_to_end = (lg[..., -1:] - lg).exp() * beta            # γ_C/γ_i · β_i

    outs = []
    for n in range(N):
        U_eff = U[:, :, n] - W[:, :, n] @ S                    # (B,H,C,dv)
        O = gamma[:, :, n, :, None] * (q[:, :, n] @ S) + P[:, :, n] @ U_eff
        S = gamma_C[:, :, n, None, None] * S + k[:, :, n].transpose(-1, -2) @ (decay_to_end[:, :, n, :, None] * U_eff)
        outs.append(O)
    o = torch.stack(outs, dim=2).reshape(B, H, Lp, dv)[:, :, :L]
    return o, S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        inner = cfg.inner
        self.qkv = nn.Linear(d, 3 * inner, bias=False)
        self.conv = CausalConv1d(3 * inner, cfg.conv_kernel)
        self.beta_proj = nn.Linear(d, H, bias=True)
        self.dt_proj = nn.Linear(d, H, bias=True)
        # A_log initialisé comme dans Mamba-2 : log(1..H)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        with torch.no_grad():
            # dt_bias : softplus^{-1} d'un dt dans [1e-3, 1e-1]
            dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.gate_proj = nn.Linear(d, inner, bias=False)
        self.o_norm = RMSNorm(dh)
        self.o_proj = nn.Linear(inner, d, bias=False)

    def init_state(self, B: int, device):
        S = torch.zeros(B, self.cfg.n_heads, self.cfg.head_dim, self.cfg.head_dim, device=device, dtype=torch.float32)
        cs = torch.zeros(B, 3 * self.cfg.inner, self.cfg.conv_kernel - 1, device=device, dtype=torch.float32)
        return S, cs

    def _project(self, x, conv_state):
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
        return q, k, v.float(), log_alpha, beta, conv_state

    def _output(self, o, x):
        B, H, L, dh = o.shape
        o = self.o_norm(o).transpose(1, 2).reshape(B, L, H * dh)
        gate = F.silu(self.gate_proj(x).float())
        return self.o_proj((o * gate).to(x.dtype))

    def forward(self, x, S, conv_state):
        if S is None:
            S, conv_state = self.init_state(x.shape[0], x.device)
        q, k, v, log_alpha, beta, conv_state = self._project(x, conv_state)
        o, S = gdn_chunked(q, k, v, log_alpha, beta, S, self.cfg.chunk)
        return self._output(o, x), S, conv_state

    def step(self, x, S, conv_state):
        q, k, v, log_alpha, beta, conv_state = self._project(x, conv_state)
        o, S = gdn_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], log_alpha[:, :, 0], beta[:, :, 0], S)
        return self._output(o.unsqueeze(2), x), S, conv_state
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_gdn.py -v`
Expected: 5 PASS. Si `test_chunked_equals_recurrent_with_padding` échoue avec une erreur > 1e-4, l'implémentation chunkée est fautive et la récurrence `gdn_step` est la référence : corriger `gdn_chunked`, jamais le test.

- [ ] **Step 5 : commit**

```bash
git add relis/model/gdn.py tests/test_gdn.py
git commit -m "feat(model): Gated DeltaNet chunké PyTorch pur, équivalent au pas-à-pas"
```

---

### Task 5 : attention à fenêtre glissante avec cache borné

**Files:**
- Create: `relis/model/swa.py`
- Test: `tests/test_swa.py`

**Interfaces:**
- Consumes: `RelisConfig` (Task 3).
- Produces: `class SlidingWindowAttention(cfg)` :
  - `forward(x: (B,L,d), cache: dict|None) -> (y: (B,L,d), cache)` où `cache = {"k": (B,H,Lc,dh), "v": (B,H,Lc,dh), "pos": int}` avec `Lc ≤ window-1` ; `pos` = nombre total de positions vues (pour le biais ALiBi relatif).
  - `step(x: (B,1,d), cache) -> (y, cache)` = `forward` avec L=1.
  - `init_cache(B, device) -> dict`.
  - Un token à la position absolue `i` attend aux positions `j` telles que `i-window < j ≤ i`. Biais ALiBi : `-slope_h * (i-j)`, `slope_h = 2^(-8(h+1)/H)`.

- [ ] **Step 1 : écrire les tests**

`tests/test_swa.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.swa import SlidingWindowAttention


def _naive(m, x):
    """Référence : attention complète masquée par fenêtre, ALiBi, en une passe."""
    B, L, _ = x.shape
    H, dh, W = m.cfg.n_heads, m.cfg.head_dim, m.cfg.window
    q, k, v = m.qkv(x).split(m.cfg.inner, dim=-1)
    q = q.view(B, L, H, dh).transpose(1, 2).float()
    k = k.view(B, L, H, dh).transpose(1, 2).float()
    v = v.view(B, L, H, dh).transpose(1, 2).float()
    i = torch.arange(L)[:, None]; j = torch.arange(L)[None, :]
    allowed = (j <= i) & (j > i - W)
    scores = (q @ k.transpose(-1, -2)) / dh ** 0.5
    scores = scores - m.slopes[None, :, None, None] * (i - j).float()
    scores = scores.masked_fill(~allowed, float("-inf"))
    o = torch.softmax(scores, dim=-1) @ v
    return m.o_proj(o.transpose(1, 2).reshape(B, L, H * dh))


def test_forward_matches_naive():
    cfg = RelisConfig.tiny()          # window 16
    m = SlidingWindowAttention(cfg)
    x = torch.randn(2, 45, cfg.d_model)
    y, cache = m(x, None)
    assert torch.allclose(y, _naive(m, x), atol=1e-5)
    assert cache["k"].shape[2] == cfg.window - 1
    assert cache["pos"] == 45


def test_forward_split_equals_whole():
    cfg = RelisConfig.tiny()
    m = SlidingWindowAttention(cfg)
    x = torch.randn(1, 50, cfg.d_model)
    y_all, c_all = m(x, None)
    y1, c = m(x[:, :7], None)
    y2, c = m(x[:, 7:30], c)
    y3, c = m(x[:, 30:], c)
    assert torch.allclose(torch.cat([y1, y2, y3], dim=1), y_all, atol=1e-5)
    assert torch.allclose(c["k"], c_all["k"], atol=1e-6)


def test_steps_equal_forward():
    cfg = RelisConfig.tiny()
    m = SlidingWindowAttention(cfg)
    x = torch.randn(2, 33, cfg.d_model)
    y_all, _ = m(x, None)
    c = m.init_cache(2, x.device)
    ys = []
    for t in range(33):
        y_t, c = m.step(x[:, t:t + 1], c)
        ys.append(y_t)
    assert torch.allclose(torch.cat(ys, dim=1), y_all, atol=1e-5)


def test_cache_is_bounded():
    cfg = RelisConfig.tiny()
    m = SlidingWindowAttention(cfg)
    _, c = m(torch.randn(1, 500, cfg.d_model), None)
    assert c["k"].shape[2] == cfg.window - 1 and c["v"].shape[2] == cfg.window - 1
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_swa.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/model/swa.py` :
```python
"""Attention à fenêtre glissante (spec §4.4) : opérateur local, cache borné à window-1.

Biais ALiBi (relatif) : la position absolue ne sert qu'à calculer i-j.
"""
import torch
import torch.nn as nn

from .config import RelisConfig


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
        return {"k": torch.zeros(B, H, 0, dh, device=device), "v": torch.zeros(B, H, 0, dh, device=device), "pos": 0}

    def forward(self, x, cache):
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
        # positions absolues des requêtes et des clés
        qi = torch.arange(pos0, pos0 + L, device=x.device)[:, None]
        kj = torch.arange(pos0 - Lc, pos0 + L, device=x.device)[None, :]
        allowed = (kj <= qi) & (kj > qi - W)
        scores = (q @ k_all.transpose(-1, -2)) / dh ** 0.5           # (B,H,L,Lc+L)
        scores = scores - self.slopes[None, :, None, None] * (qi - kj).float()
        scores = scores.masked_fill(~allowed, float("-inf"))
        o = torch.softmax(scores, dim=-1) @ v_all                    # fp32
        y = self.o_proj(o.transpose(1, 2).reshape(B, L, H * dh).to(x.dtype))
        keep = W - 1
        new_cache = {"k": k_all[:, :, -keep:] if keep > 0 else k_all[:, :, :0],
                     "v": v_all[:, :, -keep:] if keep > 0 else v_all[:, :, :0],
                     "pos": pos0 + L}
        return y, new_cache

    def step(self, x, cache):
        return self.forward(x, cache)
```

Note : `L` est toujours ≤ `block` (512) parce que `RelisModel` découpe la séquence en blocs (Task 7). La matrice de scores fait donc au plus `(B, H, 512, 1023)`.

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_swa.py -v`
Expected: 4 PASS

- [ ] **Step 5 : commit**

```bash
git add relis/model/swa.py tests/test_swa.py
git commit -m "feat(model): attention à fenêtre glissante ALiBi avec cache borné"
```

---

### Task 6 : Buffer d'intention (lecture par bloc, écriture par chunk)

**Files:**
- Create: `relis/model/buffer.py`
- Test: `tests/test_buffer.py`

**Interfaces:**
- Consumes: `RelisConfig`, `RMSNorm` (Task 3).
- Produces:
  - `class SlotRead(cfg)` : `forward(x: (B,L,d), slots: (B,K,d)) -> (B,L,d)` ; cross-attention multi-têtes des positions vers les slots dans une dimension interne réduite `r = cfg.inner // 2` (256 en V1, 16 en tiny ; `q: Linear(d, r)`, `kv: Linear(d, 2r)`, `o: Linear(r, d)`), projections propres à la couche ; retourne l'incrément résiduel (à ajouter par l'appelant). La dimension réduite maintient le total V1 autour de 150 M paramètres (16 lectures à pleine dimension coûteraient 38 M à elles seules).
  - `class SlotWrite(cfg)` : `forward(h_chunk: (B,C,d), slots: (B,K,d)) -> slots_new: (B,K,d)` avec `Δ = Attn(norm(slots) → norm(h_chunk))`, `g = σ(W_g [slots ; Δ])`, `slots + g ⊙ Δ` ; `apply_blocks(h: (B,L,d), slots)` applique `forward` séquentiellement sur chaque chunk de `cfg.chunk` positions de `h` (L doit être un multiple de `cfg.chunk`).
  - `init_slots(cfg) -> nn.Parameter (K,d)` : vecteurs appris ; `expand_slots(param, B) -> (B,K,d)`.

- [ ] **Step 1 : écrire les tests**

`tests/test_buffer.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.buffer import SlotRead, SlotWrite, init_slots, expand_slots


def test_slot_read_shape_and_dependence():
    cfg = RelisConfig.tiny()
    r = SlotRead(cfg)
    x = torch.randn(2, 9, cfg.d_model)
    slots = expand_slots(init_slots(cfg), 2)
    y = r(x, slots)
    assert y.shape == x.shape
    slots2 = slots + 1.0
    assert not torch.allclose(y, r(x, slots2))


def test_slot_write_gate_and_shape():
    cfg = RelisConfig.tiny()
    w = SlotWrite(cfg)
    slots = expand_slots(init_slots(cfg), 3)
    h = torch.randn(3, cfg.chunk, cfg.d_model)
    new = w(h, slots)
    assert new.shape == slots.shape
    assert not torch.allclose(new, slots)


def test_apply_blocks_is_sequential_over_chunks():
    cfg = RelisConfig.tiny()   # chunk 8
    w = SlotWrite(cfg)
    slots = expand_slots(init_slots(cfg), 1)
    h = torch.randn(1, 24, cfg.d_model)
    seq = slots
    for c in range(3):
        seq = w(h[:, c * 8:(c + 1) * 8], seq)
    assert torch.allclose(w.apply_blocks(h, slots), seq, atol=1e-6)


def test_init_slots_is_parameter():
    cfg = RelisConfig.tiny()
    p = init_slots(cfg)
    assert isinstance(p, torch.nn.Parameter) and p.shape == (cfg.n_slots, cfg.d_model)
    assert expand_slots(p, 5).shape == (5, cfg.n_slots, cfg.d_model)
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_buffer.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/model/buffer.py` :
```python
"""Buffer d'intention (spec §4.6) : K slots partagés par toutes les couches.

Lecture : chaque bloc fait une cross-attention positions -> slots.
Écriture : au sommet de la pile, une fois par chunk de 64 octets, slots <- slots + g ⊙ Δ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RelisConfig
from .layers import RMSNorm


def init_slots(cfg: RelisConfig) -> nn.Parameter:
    return nn.Parameter(torch.randn(cfg.n_slots, cfg.d_model) * 0.02)


def expand_slots(param: torch.Tensor, B: int) -> torch.Tensor:
    return param.unsqueeze(0).expand(B, -1, -1).contiguous()


def _mha(q, k, v, n_heads):
    """q:(B,Lq,d) k,v:(B,Lk,d) -> (B,Lq,d), softmax en fp32."""
    B, Lq, d = q.shape
    Lk = k.shape[1]
    dh = d // n_heads
    q = q.view(B, Lq, n_heads, dh).transpose(1, 2).float()
    k = k.view(B, Lk, n_heads, dh).transpose(1, 2).float()
    v = v.view(B, Lk, n_heads, dh).transpose(1, 2).float()
    att = torch.softmax((q @ k.transpose(-1, -2)) / dh ** 0.5, dim=-1)
    return (att @ v).transpose(1, 2).reshape(B, Lq, d)


class SlotRead(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        r = cfg.inner // 2            # dimension interne réduite (256 en V1)
        assert r % cfg.n_heads == 0
        self.norm_x = RMSNorm(d)
        self.norm_s = RMSNorm(d)
        self.q = nn.Linear(d, r, bias=False)
        self.kv = nn.Linear(d, 2 * r, bias=False)
        self.o = nn.Linear(r, d, bias=False)

    def forward(self, x, slots):
        k, v = self.kv(self.norm_s(slots)).chunk(2, dim=-1)
        y = _mha(self.q(self.norm_x(x)), k, v, self.cfg.n_heads)
        return self.o(y.to(x.dtype))


class SlotWrite(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.norm_s = RMSNorm(d)
        self.norm_h = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, d, bias=True)
        with torch.no_grad():
            self.gate.bias.fill_(-2.0)   # écritures prudentes au départ

    def forward(self, h_chunk, slots):
        k, v = self.kv(self.norm_h(h_chunk)).chunk(2, dim=-1)
        delta = self.o(_mha(self.q(self.norm_s(slots)), k, v, self.cfg.n_heads).to(slots.dtype))
        g = torch.sigmoid(self.gate(torch.cat([slots, delta], dim=-1)).float())
        return (slots.float() + g * delta.float()).to(slots.dtype)

    def apply_blocks(self, h, slots):
        L = h.shape[1]
        C = self.cfg.chunk
        assert L % C == 0, "apply_blocks attend un multiple de chunk"
        for c in range(L // C):
            slots = self.forward(h[:, c * C:(c + 1) * C], slots)
        return slots
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_buffer.py -v`
Expected: 4 PASS

- [ ] **Step 5 : commit**

```bash
git add relis/model/buffer.py tests/test_buffer.py
git commit -m "feat(model): Buffer d'intention, lecture par couche et écriture par chunk"
```

---

### Task 7 : État, bloc et modèle complet (mode bloc et mode pas-à-pas)

**Files:**
- Create: `relis/model/state.py`, `relis/model/relis.py`
- Test: `tests/test_relis.py`

**Interfaces:**
- Consumes: Tasks 3 à 6 ; `Mode` (Task 2).
- Produces:
  - `class State` (dataclass) : `gdn: list[tuple[S, conv_state] | None]` (une entrée par couche, `None` pour les couches SWA), `swa: list[dict | None]`, `slots: (B,K,d)`, `pending: (B,P,d)` (états du sommet de pile en attente d'écriture, `P < block`), `seen: int` (octets traités depuis la dernière réinitialisation). Méthodes : `State.init(model, B, device)`, `reset_memory()` (remet `gdn`, `swa`, `pending`, `seen` à zéro, **garde** `slots` : c'est le REFRESH de la spec §4.8), `reset_all()` (tout, slots compris, depuis les vecteurs appris), `size_bytes() -> int` (somme des octets des tenseurs de `gdn`, `swa`, `pending` ; plafonné par construction).
  - `class Block(cfg, is_swa: bool)` : `forward(x, state_gdn, state_swa, slots) -> (y, state_gdn, state_swa)` ; `step(...)` idem pour L=1.
  - `class RelisModel(cfg)` : `forward(x: LongTensor (B,L), mode: LongTensor (B,L) | int, state: State) -> (logits: (B,L,256), state)` ; `step(x: (B,), mode, state) -> (logits: (B,256), state)` ; `new_state(B, device) -> State`. La tête de sortie est liée à l'embedding d'octets.
  - Calendrier d'écriture : `forward` découpe l'entrée en morceaux dont les frontières tombent sur les multiples absolus de `cfg.block` (compte tenu de `state.seen`) ; à chaque frontière, `SlotWrite.apply_blocks` est appliqué sur les `block` derniers états du sommet de pile (`pending` + morceau courant) et `slots` est mis à jour. `step` suit le même calendrier.

- [ ] **Step 1 : écrire les tests**

`tests/test_relis.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.model.state import State
from relis.tape.codes import Mode


def _model():
    cfg = RelisConfig.tiny()   # block 32, chunk 8, window 16
    return cfg, RelisModel(cfg)


def test_forward_shapes_and_tied_head():
    cfg, m = _model()
    x = torch.randint(0, 256, (2, 70))
    st = m.new_state(2, x.device)
    logits, st = m(x, int(Mode.SCAN), st)
    assert logits.shape == (2, 70, 256)
    assert m.head_weight().data_ptr() == m.embed.weight.data_ptr()
    assert st.seen == 70 and st.pending.shape[1] == 70 % cfg.block


def test_causality():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 50))
    l1, _ = m(x, int(Mode.SCAN), m.new_state(1, x.device))
    x2 = x.clone(); x2[0, 40] = (x2[0, 40] + 1) % 256
    l2, _ = m(x2, int(Mode.SCAN), m.new_state(1, x.device))
    assert torch.allclose(l1[:, :40], l2[:, :40], atol=1e-5)
    assert not torch.allclose(l1[:, 40:], l2[:, 40:])


def test_mode_changes_output():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 12))
    a, _ = m(x, int(Mode.SCAN), m.new_state(1, x.device))
    b, _ = m(x, int(Mode.GENERATE), m.new_state(1, x.device))
    assert not torch.allclose(a, b)


def test_forward_equals_steps_across_block_boundaries():
    cfg, m = _model()
    L = 2 * cfg.block + 13            # traverse deux frontières d'écriture du Buffer
    x = torch.randint(0, 256, (2, L))
    mode = torch.full((2, L), int(Mode.SCAN))
    mode[:, 40:] = int(Mode.GENERATE)
    l_full, st_full = m(x, mode, m.new_state(2, x.device))
    st = m.new_state(2, x.device)
    outs = []
    for t in range(L):
        l_t, st = m.step(x[:, t], mode[:, t], st)
        outs.append(l_t)
    l_steps = torch.stack(outs, dim=1)
    assert torch.allclose(l_full, l_steps, atol=1e-4), (l_full - l_steps).abs().max()
    assert torch.allclose(st_full.slots, st.slots, atol=1e-4)


def test_forward_split_equals_whole():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 100))
    l_all, st_all = m(x, int(Mode.SCAN), m.new_state(1, x.device))
    st = m.new_state(1, x.device)
    l1, st = m(x[:, :5], int(Mode.SCAN), st)
    l2, st = m(x[:, 5:61], int(Mode.SCAN), st)
    l3, st = m(x[:, 61:], int(Mode.SCAN), st)
    assert torch.allclose(torch.cat([l1, l2, l3], dim=1), l_all, atol=1e-4)
    assert torch.allclose(st.slots, st_all.slots, atol=1e-4)


def test_slots_change_only_at_block_boundaries():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    s0 = st.slots.clone()
    _, st = m(torch.randint(0, 256, (1, cfg.block - 1)), int(Mode.SCAN), st)
    assert torch.allclose(st.slots, s0)
    _, st = m(torch.randint(0, 256, (1, 1)), int(Mode.SCAN), st)
    assert not torch.allclose(st.slots, s0)
    assert st.pending.shape[1] == 0


def test_memory_is_constant():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, 3 * cfg.block)), int(Mode.SCAN), st)
    size_a = st.size_bytes()
    _, st = m(torch.randint(0, 256, (1, 40 * cfg.block)), int(Mode.SCAN), st)
    assert st.size_bytes() == size_a


def test_reset_memory_keeps_slots():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, 2 * cfg.block)), int(Mode.ENCODE), st)
    slots = st.slots.clone()
    st.reset_memory()
    assert torch.allclose(st.slots, slots)
    assert st.seen == 0 and st.pending.shape[1] == 0
    assert all(torch.count_nonzero(g[0]) == 0 for g in st.gdn if g is not None)


def test_parameter_count_v1_in_range():
    m = RelisModel(RelisConfig())
    n = sum(p.numel() for p in m.parameters())
    assert 100e6 < n < 170e6, n
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_relis.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter l'état**

`relis/model/state.py` :
```python
"""Mémoire de RELIS (spec §4.5) : état récurrent de taille constante + slots du Buffer."""
from dataclasses import dataclass, field
import torch


@dataclass
class State:
    gdn: list                      # par couche : (S, conv_state) ou None
    swa: list                      # par couche : dict cache ou None
    slots: torch.Tensor            # (B, K, d)
    pending: torch.Tensor          # (B, P, d) états du sommet en attente d'écriture
    seen: int = 0
    _init_slots: torch.Tensor = field(default=None, repr=False)

    @classmethod
    def init(cls, model, B: int, device) -> "State":
        gdn, swa = [], []
        for i, blk in enumerate(model.blocks):
            if blk.is_swa:
                gdn.append(None); swa.append(blk.mixer.init_cache(B, device))
            else:
                gdn.append(blk.mixer.init_state(B, device)); swa.append(None)
        slots = model.slots.detach().unsqueeze(0).expand(B, -1, -1).contiguous().to(device)
        pending = torch.zeros(B, 0, model.cfg.d_model, device=device)
        return cls(gdn=gdn, swa=swa, slots=slots, pending=pending, seen=0, _init_slots=slots.clone())

    def reset_memory(self) -> None:
        """REFRESH (spec §4.8) : Mémoire remise à zéro, Buffer conservé."""
        for i, g in enumerate(self.gdn):
            if g is not None:
                self.gdn[i] = (torch.zeros_like(g[0]), torch.zeros_like(g[1]))
        for i, c in enumerate(self.swa):
            if c is not None:
                self.swa[i] = {"k": c["k"][:, :, :0], "v": c["v"][:, :, :0], "pos": 0}
        self.pending = self.pending[:, :0]
        self.seen = 0

    def reset_all(self) -> None:
        """Nouveau tour utilisateur : tout est remis à zéro, slots compris."""
        self.reset_memory()
        self.slots = self._init_slots.clone()

    def size_bytes(self) -> int:
        n = 0
        for g in self.gdn:
            if g is not None:
                n += g[0].numel() * g[0].element_size() + g[1].numel() * g[1].element_size()
        for c in self.swa:
            if c is not None:
                n += c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size()
        n += self.pending.numel() * self.pending.element_size()
        return n
```

- [ ] **Step 4 : implémenter le modèle**

`relis/model/relis.py` :
```python
"""Assemblage de RELIS (spec §4.1, §4.4, §4.6) : un réseau, trois modes.

forward() traite la séquence par blocs de cfg.block positions ; dans un bloc,
toutes les positions lisent les slots tels qu'ils étaient au début du bloc ;
à la frontière, SlotWrite met à jour les slots chunk par chunk. step() suit
exactement le même calendrier, d'où l'équivalence numérique des deux modes.
"""
import torch
import torch.nn as nn

from .config import RelisConfig
from .layers import RMSNorm, SwiGLU
from .gdn import GatedDeltaNet
from .swa import SlidingWindowAttention
from .buffer import SlotRead, SlotWrite, init_slots
from .state import State


class Block(nn.Module):
    def __init__(self, cfg: RelisConfig, is_swa: bool):
        super().__init__()
        self.is_swa = is_swa
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = SlidingWindowAttention(cfg) if is_swa else GatedDeltaNet(cfg)
        self.read = SlotRead(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_mult)

    def forward(self, x, st_gdn, st_swa, slots, single_step: bool = False):
        h = self.norm1(x)
        if self.is_swa:
            y, st_swa = (self.mixer.step if single_step else self.mixer.forward)(h, st_swa)
        else:
            S, cs = st_gdn
            y, S, cs = (self.mixer.step if single_step else self.mixer.forward)(h, S, cs)
            st_gdn = (S, cs)
        x = x + y
        x = x + self.read(x, slots)
        x = x + self.mlp(self.norm2(x))
        return x, st_gdn, st_swa


class RelisModel(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.mode_embed = nn.Embedding(cfg.n_modes, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, cfg.is_swa(i)) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.slots = init_slots(cfg)
        self.writer = SlotWrite(cfg)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.mode_embed.weight, std=0.02)

    def head_weight(self) -> torch.Tensor:
        return self.embed.weight   # tête liée (spec §4.4)

    def new_state(self, B: int, device) -> State:
        return State.init(self, B, device)

    def _run_piece(self, x_ids, mode_ids, state: State, single_step: bool):
        """Passe un morceau (≤ block positions, sans frontière interne) dans la pile."""
        h = self.embed(x_ids) + self.mode_embed(mode_ids)
        for i, blk in enumerate(self.blocks):
            h, g, s = blk(h, state.gdn[i], state.swa[i], state.slots, single_step=single_step)
            state.gdn[i], state.swa[i] = g, s
        top = self.final_norm(h)
        logits = top @ self.head_weight().t().to(top.dtype)
        # écriture différée : accumuler, écrire à la frontière de bloc
        state.pending = torch.cat([state.pending, top.float()], dim=1)
        state.seen += x_ids.shape[1]
        if state.pending.shape[1] == self.cfg.block:
            state.slots = self.writer.apply_blocks(state.pending.to(state.slots.dtype), state.slots)
            state.pending = state.pending[:, :0]
        return logits, state

    def forward(self, x, mode, state: State):
        B, L = x.shape
        if isinstance(mode, int):
            mode = torch.full((B, L), mode, dtype=torch.long, device=x.device)
        logits = []
        t = 0
        while t < L:
            room = self.cfg.block - state.pending.shape[1]
            n = min(room, L - t)
            lg, state = self._run_piece(x[:, t:t + n], mode[:, t:t + n], state, single_step=False)
            logits.append(lg)
            t += n
        return torch.cat(logits, dim=1), state

    def step(self, x, mode, state: State):
        B = x.shape[0]
        if isinstance(mode, int):
            mode = torch.full((B,), mode, dtype=torch.long, device=x.device)
        lg, state = self._run_piece(x[:, None], mode[:, None], state, single_step=True)
        return lg[:, 0], state
```

- [ ] **Step 5 : vérifier**

Run: `python -m pytest tests/test_relis.py -v`
Expected: 9 PASS. `test_forward_equals_steps_across_block_boundaries` est le test central de la semaine : s'il échoue, comparer d'abord `st_full.slots` et `st.slots` pour savoir si l'écart vient des mélangeurs ou du calendrier d'écriture.

- [ ] **Step 6 : lancer toute la suite**

Run: `python -m pytest -q`
Expected: tous PASS, durée < 60 s.

- [ ] **Step 7 : commit**

```bash
git add relis/model/state.py relis/model/relis.py tests/test_relis.py
git commit -m "feat(model): État à mémoire constante et modèle RELIS complet, modes bloc et pas-à-pas équivalents"
```

---

### Task 8 : shards d'octets et jeu de données

**Files:**
- Create: `relis/data/shards.py`, `relis/data/dataset.py`
- Test: `tests/test_shards.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `codes.SEG, codes.HDR, codes.sanitize` (Task 2).
- Produces:
  - `encode_document(content: bytes, header: str) -> bytes` = `SEG + header.encode() + HDR + sanitize(content)`.
  - `class ShardWriter(path: str)` : `add(content: bytes, header: str) -> int` (octets écrits), `close()`, attribut `total_bytes` ; écrit un unique fichier binaire brut.
  - `class ByteWindows(bin_path: str, seq_len: int)` (torch `Dataset` infini par échantillonnage) : `sample(batch_size, generator) -> (x: LongTensor (B, seq_len), y: LongTensor (B, seq_len))` où `y` est `x` décalé d'un octet ; lit via `numpy.memmap`.

- [ ] **Step 1 : écrire les tests**

`tests/test_shards.py` :
```python
import os
from relis.data.shards import encode_document, ShardWriter
from relis.tape import codes as C


def test_encode_document_layout():
    doc = encode_document(b"bonjour\x01monde", "src=test")
    assert doc[0] == C.SEG
    assert doc[1:9] == b"src=test"
    assert doc[9] == C.HDR
    assert doc[10:] == b"bonjour\xef\xbf\xbdmonde"


def test_shard_writer_concatenates(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p)
    n1 = w.add(b"aaa", "src=a")
    n2 = w.add(b"bb", "src=b")
    w.close()
    data = open(p, "rb").read()
    assert len(data) == n1 + n2 == w.total_bytes
    assert data.count(bytes([C.SEG])) == 2
```

`tests/test_dataset.py` :
```python
import torch
from relis.data.shards import ShardWriter
from relis.data.dataset import ByteWindows


def test_byte_windows_shift_and_range(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p)
    w.add(bytes(range(256)) * 20, "src=t")
    w.close()
    ds = ByteWindows(p, seq_len=16)
    g = torch.Generator().manual_seed(0)
    x, y = ds.sample(4, g)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.min() >= 0 and x.max() <= 255
    assert len(ds) == w.total_bytes
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_shards.py tests/test_dataset.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/data/shards.py` :
```python
"""Shards d'octets pour le pré-entraînement (spec §6.1) : documents concaténés,
chaque document précédé de SEG + en-tête minimal + HDR."""
from relis.tape import codes as C


def encode_document(content: bytes, header: str) -> bytes:
    return bytes([C.SEG]) + header.encode("utf-8") + bytes([C.HDR]) + C.sanitize(content)


class ShardWriter:
    def __init__(self, path: str):
        self._f = open(path, "wb")
        self.total_bytes = 0

    def add(self, content: bytes, header: str) -> int:
        data = encode_document(content, header)
        self._f.write(data)
        self.total_bytes += len(data)
        return len(data)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

`relis/data/dataset.py` :
```python
import numpy as np
import torch


class ByteWindows:
    """Fenêtres aléatoires de seq_len+1 octets dans un fichier binaire (memmap)."""

    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError("fichier trop court pour seq_len")

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator)
        rows = np.stack([self.data[s:s + self.seq_len + 1] for s in starts.tolist()])
        t = torch.from_numpy(rows.astype(np.int64))
        return t[:, :-1], t[:, 1:]
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_shards.py tests/test_dataset.py -v`
Expected: 3 PASS

- [ ] **Step 5 : commit**

```bash
git add relis/data/shards.py relis/data/dataset.py tests/test_shards.py tests/test_dataset.py
git commit -m "feat(data): shards d'octets avec en-tête SEG et fenêtres memmap"
```

---

### Task 9 : préparation des corpus (flux Hugging Face → shards)

**Files:**
- Create: `relis/data/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `ShardWriter` (Task 8).
- Produces:
  - `SOURCES: dict[str, callable]` : `"wiki_fr"`, `"fineweb2_fr"`, `"code"` ; chaque callable renvoie un itérateur de `(content: bytes, header: str)`.
  - `build_shards(sources: list[str], out_dir: str, max_bytes: dict[str, int], val_fraction: float = 0.005, iterator_factory=None)` : écrit `out_dir/train.bin` et `out_dir/val.bin` ; `iterator_factory(name)` permet d'injecter des itérateurs de test (sans réseau).
  - CLI : `python -m relis.data.prepare --out shards/v1 --wiki_fr 600e6 --fineweb2_fr 400e6 --code 300e6`.

- [ ] **Step 1 : écrire le test (sans réseau)**

`tests/test_prepare.py` :
```python
import os
from relis.data.prepare import build_shards


def _fake(name):
    for i in range(50):
        yield (f"document {name} numéro {i} ".encode() * 20, f"src={name};i={i}")


def test_build_shards_respects_budget_and_split(tmp_path):
    out = str(tmp_path)
    stats = build_shards(["a", "b"], out, {"a": 20_000, "b": 5_000}, val_fraction=0.1, iterator_factory=_fake)
    assert os.path.exists(os.path.join(out, "train.bin"))
    assert os.path.exists(os.path.join(out, "val.bin"))
    assert stats["a"] >= 20_000 and stats["a"] < 20_000 + 2_000
    assert stats["b"] >= 5_000 and stats["b"] < 5_000 + 2_000
    assert stats["val_bytes"] > 0 and stats["train_bytes"] > stats["val_bytes"]
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_prepare.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/data/prepare.py` :
```python
"""Flux Hugging Face -> shards d'octets (spec §5).

Sources V1 : Wikipedia FR, FineWeb-2 (fra_Latn), code Python/JavaScript.
Chaque source est un itérateur (content: bytes, header: str) ; le budget
en octets par source est respecté à un document près.
"""
import argparse
import os
import random

from .shards import ShardWriter


def _wiki_fr():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    for ex in ds:
        text = ex.get("text") or ""
        if len(text) < 200:
            continue
        yield text.encode("utf-8"), f"src=wiki_fr;title={ex.get('title', '')[:60]}"


def _fineweb2_fr():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True)
    for ex in ds:
        text = ex.get("text") or ""
        if len(text) < 500:
            continue
        yield text.encode("utf-8"), "src=fineweb2_fr"


def _code():
    from datasets import load_dataset
    for lang in ("python", "javascript"):
        ds = load_dataset("bigcode/the-stack-smol-xl", data_dir=f"data/{lang}", split="train", streaming=True)
        for ex in ds:
            code = ex.get("content") or ""
            if not 200 <= len(code) <= 100_000:
                continue
            path = (ex.get("path") or "").split("/")[-1][:60]
            yield code.encode("utf-8"), f"src=code;lang={lang};name={path}"


SOURCES = {"wiki_fr": _wiki_fr, "fineweb2_fr": _fineweb2_fr, "code": _code}


def build_shards(sources, out_dir, max_bytes, val_fraction=0.005, iterator_factory=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    factory = iterator_factory or (lambda name: SOURCES[name]())
    stats = {}
    with ShardWriter(os.path.join(out_dir, "train.bin")) as train, \
         ShardWriter(os.path.join(out_dir, "val.bin")) as val:
        for name in sources:
            budget = int(max_bytes[name])
            written = 0
            for content, header in factory(name):
                target = val if rng.random() < val_fraction else train
                written += target.add(content, header)
                if written >= budget:
                    break
            stats[name] = written
        stats["train_bytes"] = train.total_bytes
        stats["val_bytes"] = val.total_bytes
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for name in SOURCES:
        ap.add_argument(f"--{name}", type=float, default=0.0, help="budget en octets")
    ap.add_argument("--val_fraction", type=float, default=0.005)
    args = ap.parse_args()
    budgets = {name: getattr(args, name) for name in SOURCES if getattr(args, name) > 0}
    stats = build_shards(list(budgets), args.out, budgets, args.val_fraction)
    for k, v in stats.items():
        print(f"{k}: {v/1e6:.1f} Mo")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_prepare.py -v`
Expected: PASS

- [ ] **Step 5 : essai réseau court (manuel, non testé automatiquement)**

Run: `python -m relis.data.prepare --out shards/smoke --wiki_fr 2e6 --code 1e6`
Expected: affiche `wiki_fr: 2.0 Mo`, `code: 1.0 Mo`, fichiers `shards/smoke/train.bin` et `val.bin` créés. Si un jeu de données a changé de nom ou exige une authentification, corriger la fonction source correspondante ; ne pas modifier `build_shards`.

- [ ] **Step 6 : commit**

```bash
git add relis/data/prepare.py tests/test_prepare.py
git commit -m "feat(data): préparation des corpus FR et code en shards"
```

---

### Task 10 : checkpoints locaux et Hugging Face Hub

**Files:**
- Create: `relis/train/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `RelisModel` (Task 7).
- Produces:
  - `save_checkpoint(dir: str, model, optimizer, scaler, step: int, cfg_dict: dict, extra: dict) -> str` : écrit `dir/last.pt` de façon atomique (écriture dans `last.pt.tmp` puis renommage) ; contenu : `{"model", "optimizer", "scaler", "step", "cfg", "extra", "rng"}`.
  - `load_checkpoint(dir: str, model, optimizer=None, scaler=None) -> dict | None` : renvoie `None` si absent ; sinon restaure et renvoie `{"step", "cfg", "extra"}`.
  - `push_to_hub(dir: str, repo_id: str) -> None` et `pull_from_hub(dir: str, repo_id: str) -> bool` : `upload_file`/`hf_hub_download` de `last.pt`, jeton via `HF_TOKEN` ; `pull_from_hub` renvoie `False` si le dépôt ou le fichier n'existe pas. Aucun test réseau ; ces deux fonctions sont couvertes par l'essai manuel de la Task 12.

- [ ] **Step 1 : écrire le test**

`tests/test_checkpoint.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint, load_checkpoint


def test_roundtrip(tmp_path):
    cfg = RelisConfig.tiny()
    m = RelisModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (1, 8))
    logits, _ = m(x, 1, m.new_state(1, "cpu"))
    logits.sum().backward(); opt.step()
    path = save_checkpoint(str(tmp_path), m, opt, None, step=7, cfg_dict=cfg.to_dict(), extra={"note": "ok"})
    assert path.endswith("last.pt")

    m2 = RelisModel(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    info = load_checkpoint(str(tmp_path), m2, opt2)
    assert info["step"] == 7 and info["extra"]["note"] == "ok" and info["cfg"]["d_model"] == 64
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_load_missing_returns_none(tmp_path):
    m = RelisModel(RelisConfig.tiny())
    assert load_checkpoint(str(tmp_path), m) is None
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/train/checkpoint.py` :
```python
"""Checkpoints (spec §6.5) : écriture atomique locale, aller-retour Hugging Face Hub."""
import os
import torch


def save_checkpoint(dir, model, optimizer, scaler, step, cfg_dict, extra):
    os.makedirs(dir, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "cfg": cfg_dict,
        "extra": extra,
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    final = os.path.join(dir, "last.pt")
    tmp = final + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


def load_checkpoint(dir, model, optimizer=None, scaler=None):
    path = os.path.join(dir, "last.pt")
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    rng = payload.get("rng") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {"step": payload["step"], "cfg": payload["cfg"], "extra": payload.get("extra", {})}


def push_to_hub(dir, repo_id):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=os.path.join(dir, "last.pt"), path_in_repo="last.pt", repo_id=repo_id)


def pull_from_hub(dir, repo_id):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    os.makedirs(dir, exist_ok=True)
    try:
        p = hf_hub_download(repo_id, "last.pt", token=os.environ.get("HF_TOKEN"), local_dir=dir)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    if os.path.abspath(p) != os.path.abspath(os.path.join(dir, "last.pt")):
        os.replace(p, os.path.join(dir, "last.pt"))
    return True
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: 2 PASS

- [ ] **Step 5 : commit**

```bash
git add relis/train/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(train): checkpoints atomiques et aller-retour HF Hub"
```

---

### Task 11 : boucle d'entraînement AMP fp16 et point d'entrée

**Files:**
- Create: `relis/train/loop.py`, `relis/train/pretrain.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `RelisModel`, `RelisConfig`, `ByteWindows`, `save_checkpoint`, `load_checkpoint`, `push_to_hub`, `pull_from_hub`.
- Produces:
  - `@dataclass TrainConfig(seq_len, batch_size, grad_accum, lr, warmup_steps, max_steps, weight_decay, grad_clip, amp, ckpt_every_minutes, log_every, hub_repo=None)` avec `from_yaml(path)`.
  - `lr_at(step, cfg: TrainConfig) -> float` : warmup linéaire puis cosinus jusqu'à `0.1*lr`.
  - `train(model, tcfg, train_ds, val_ds, run_dir, device, resume=True, on_step=None) -> dict` : renvoie `{"step", "last_loss", "val_bpb"}` ; `on_step(step, loss)` est appelé à chaque pas (pour les tests) ; sauvegarde locale toutes les `ckpt_every_minutes` et à la fin ; pousse sur le Hub si `hub_repo` est défini ; à `resume=True`, tente `pull_from_hub` puis `load_checkpoint`.
  - `bits_per_byte(model, ds, seq_len, n_batches, batch_size, device) -> float`.
  - `pretrain.main()` : `python -m relis.train.pretrain --config configs/pretrain_t4.yaml --run_dir runs/v1 [--device cuda] [--override train.max_steps=10]`.
  - Multi-GPU : `train` accepte un modèle déjà enveloppé dans `torch.nn.parallel.DistributedDataParallel` ; `pretrain.main()` initialise `torch.distributed` si `WORLD_SIZE > 1` (lancement par `torchrun --nproc_per_node=2`).

- [ ] **Step 1 : écrire les tests**

`tests/test_loop.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.data.shards import ShardWriter
from relis.data.dataset import ByteWindows
from relis.train.loop import TrainConfig, lr_at, train, bits_per_byte


def _tiny_data(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p)
    w.add(b"le chat dort. le chien court. " * 400, "src=t")
    w.close()
    return ByteWindows(p, seq_len=32), ByteWindows(p, seq_len=32)


def _tcfg(**kw):
    base = dict(seq_len=32, batch_size=4, grad_accum=1, lr=3e-3, warmup_steps=5, max_steps=40,
                weight_decay=0.1, grad_clip=1.0, amp=False, ckpt_every_minutes=1e9, log_every=10)
    base.update(kw)
    return TrainConfig(**base)


def test_lr_schedule():
    c = _tcfg(lr=1.0, warmup_steps=10, max_steps=110)
    assert abs(lr_at(0, c)) < 1e-9
    assert abs(lr_at(10, c) - 1.0) < 1e-9
    assert 0.1 <= lr_at(110, c) <= 0.1 + 1e-9
    assert lr_at(60, c) < lr_at(20, c)


def test_loss_decreases_on_repetitive_data(tmp_path):
    torch.manual_seed(0)
    tr, va = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    losses = []
    out = train(m, _tcfg(), tr, va, str(tmp_path / "run"), device="cpu",
                resume=False, on_step=lambda s, l: losses.append(l))
    assert out["step"] == 40
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.8
    assert out["val_bpb"] < 8.0


def test_resume_continues_from_checkpoint(tmp_path):
    tr, va = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    run = str(tmp_path / "run")
    train(m, _tcfg(max_steps=6), tr, va, run, device="cpu", resume=False)
    m2 = RelisModel(RelisConfig.tiny())
    seen = []
    out = train(m2, _tcfg(max_steps=9), tr, va, run, device="cpu", resume=True,
                on_step=lambda s, l: seen.append(s))
    assert seen == [7, 8, 9] and out["step"] == 9


def test_bits_per_byte_range(tmp_path):
    tr, _ = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    bpb = bits_per_byte(m, tr, seq_len=32, n_batches=2, batch_size=2, device="cpu")
    assert 6.0 < bpb < 10.0     # ~8 bits/octet pour un modèle non entraîné
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_loop.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter la boucle**

`relis/train/loop.py` :
```python
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
    model.to(device).train()
    opt = _make_optimizer(model, tcfg)
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg.amp and device.startswith("cuda"))
    step = 0
    if resume:
        if tcfg.hub_repo and is_main and not os.path.exists(os.path.join(run_dir, "last.pt")):
            pull_from_hub(run_dir, tcfg.hub_repo)
        info = load_checkpoint(run_dir, _unwrap(model), opt, scaler)
        if info is not None:
            step = info["step"]
    g = torch.Generator().manual_seed(1000 + step + int(os.environ.get("RANK", "0")))
    last_ckpt = time.time()
    last_loss = float("nan")
    cfg_dict = _unwrap(model).cfg.to_dict()

    def _save():
        if not is_main:
            return
        save_checkpoint(run_dir, _unwrap(model), opt, scaler, step, cfg_dict, {"train": asdict(tcfg)})
        if tcfg.hub_repo:
            try:
                push_to_hub(run_dir, tcfg.hub_repo)
            except Exception as e:  # le Hub ne doit jamais tuer l'entraînement
                print(f"[hub] push échoué : {e}")

    while step < tcfg.max_steps:
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
    _save()
    val_bpb = bits_per_byte(_unwrap(model), val_ds, tcfg.seq_len, n_batches=4,
                            batch_size=max(1, tcfg.batch_size // 2), device=device) if is_main else float("nan")
    if is_main:
        print(f"val bpb {val_bpb:.3f}")
    return {"step": step, "last_loss": last_loss, "val_bpb": val_bpb}
```

- [ ] **Step 4 : implémenter le point d'entrée**

`relis/train/pretrain.py` :
```python
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
    ap.add_argument("--override", action="append", default=[], help="ex. train.max_steps=10")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw = _apply_overrides(yaml.safe_load(f), args.override)
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
```

- [ ] **Step 5 : vérifier**

Run: `python -m pytest tests/test_loop.py -v`
Expected: 4 PASS (le test de décroissance prend ~20-40 s sur CPU).

- [ ] **Step 6 : essai de bout en bout sur CPU avec la config tiny**

```bash
python - <<'PY'
from relis.data.shards import ShardWriter
import os
os.makedirs("shards/tiny", exist_ok=True)
for name in ("train", "val"):
    w = ShardWriter(f"shards/tiny/{name}.bin")
    w.add(("relis lit pour répondre. " * 2000).encode(), "src=tiny")
    w.close()
PY
python -m relis.train.pretrain --config configs/tiny.yaml --run_dir runs/tiny --device cpu
```
Expected: la perte descend nettement (sous 3,0 en 60 pas ; observé 5,1 → 2,4), `runs/tiny/last.pt` existe, `val bpb` affiché.

- [ ] **Step 7 : commit**

```bash
git add relis/train/loop.py relis/train/pretrain.py tests/test_loop.py
git commit -m "feat(train): boucle AMP fp16 avec reprise et point d'entrée de pré-entraînement"
```

---

### Task 12 : lancement Kaggle / Colab et mesure du débit

**Files:**
- Create: `notebooks/KAGGLE.md`, `relis/train/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Produces:
  - `bench.throughput(model, seq_len, batch_size, device, amp, steps=5) -> float` : octets par seconde en avant+arrière, après un pas de chauffe.
  - CLI `python -m relis.train.bench --config configs/pretrain_t4.yaml` : imprime le débit et l'estimation `max_steps` pour 2 Go.

- [ ] **Step 1 : écrire le test**

`tests/test_bench.py` :
```python
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.bench import throughput


def test_throughput_positive_cpu():
    m = RelisModel(RelisConfig.tiny())
    bps = throughput(m, seq_len=64, batch_size=2, device="cpu", amp=False, steps=2)
    assert bps > 0
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_bench.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3 : implémenter**

`relis/train/bench.py` :
```python
"""Mesure du débit (octets/s) pour calibrer max_steps sur le matériel réel."""
import argparse
import time

import torch
import torch.nn.functional as F
import yaml

from relis.model.config import RelisConfig
from relis.model.relis import RelisModel


def throughput(model, seq_len, batch_size, device, amp, steps=5) -> float:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.startswith("cuda"))

    def one():
        x = torch.randint(0, 256, (batch_size, seq_len + 1), device=device)
        state = model.new_state(batch_size, device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            logits, _ = model(x[:, :-1], 1, state)
        loss = F.cross_entropy(logits.float().reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    one()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return steps * batch_size * seq_len / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_gb", type=float, default=2.0)
    args = ap.parse_args()
    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = RelisConfig(**raw["model"]); t = raw["train"]
    model = RelisModel(mcfg)
    bps = throughput(model, t["seq_len"], t["batch_size"], args.device, t["amp"])
    per_step = t["batch_size"] * t["grad_accum"] * t["seq_len"]
    steps_needed = args.target_gb * 1e9 / per_step
    print(f"débit : {bps:,.0f} octets/s par GPU")
    print(f"octets par pas (1 GPU) : {per_step:,} ; pas pour {args.target_gb} Go : {steps_needed:,.0f}")
    print(f"heures pour {args.target_gb} Go sur 1 GPU : {args.target_gb*1e9/bps/3600:.1f}")
    if args.device.startswith("cuda"):
        print(f"mémoire GPU max : {torch.cuda.max_memory_allocated()/1e9:.2f} Go")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : vérifier**

Run: `python -m pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5 : écrire la procédure de lancement**

`notebooks/KAGGLE.md` :
````markdown
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
````

- [ ] **Step 6 : essai manuel sur GPU (à faire dès la première session Kaggle/Colab)**

Suivre `notebooks/KAGGLE.md` avec `--override train.max_steps=20 train.ckpt_every_minutes=0.1` et un `hub_repo` de test.
Expected : `bench` imprime un débit > 5 000 octets/s par T4 et une mémoire GPU max < 14 Go ; l'entraînement affiche la perte à chaque `log_every` ; `last.pt` apparaît dans le dépôt Hub ; une seconde exécution reprend à `step 20`. Noter le débit mesuré dans `notebooks/KAGGLE.md` et ajuster `train.max_steps` dans `configs/pretrain_t4.yaml`. Si la perte devient `nan` en fp16, appliquer le repli de la spec §6.4 : passer `amp: false` (fp32 complet) et le consigner.

- [ ] **Step 7 : commit**

```bash
git add notebooks/KAGGLE.md relis/train/bench.py tests/test_bench.py
git commit -m "feat(train): mesure du débit et procédure de lancement Kaggle/Colab"
```

---

### Task 13 : lancement du pré-entraînement V1

**Files:**
- Modify: `configs/pretrain_t4.yaml` (valeur de `train.max_steps` calibrée), `notebooks/KAGGLE.md` (débit mesuré)

- [ ] **Step 1 : préparer les shards complets** selon `notebooks/KAGGLE.md` (Wikipedia FR 600 Mo, FineWeb-2 fr 400 Mo, code 300 Mo) et les publier en Kaggle Dataset `relis-shards`.

- [ ] **Step 2 : fixer `max_steps`** dans `configs/pretrain_t4.yaml` d'après le débit de la Task 12 (viser ~1,3 Go pour le pré-entraînement, soit ~60 % du budget total de 2 Go de la spec §6.1 : `max_steps ≈ 1.3e9 / 524288 ≈ 2500`).

- [ ] **Step 3 : lancer** la cellule Kaggle avec le vrai `hub_repo`. Vérifier après 100 pas que la perte descend sous 2,5 (≈ 3,6 bits/octet) et qu'aucun `nan` n'apparaît.

- [ ] **Step 4 : commit**

```bash
git add configs/pretrain_t4.yaml notebooks/KAGGLE.md
git commit -m "chore(train): calibration du pré-entraînement V1 et lancement"
```

Le pré-entraînement tourne alors en fond pendant que le plan 2 (rubans, oracle, épisodes) démarre.

---

## Auto-revue du plan

**Couverture de la spec (périmètre semaine 1).** §4.1 modes : embedding de mode, Task 7. §4.2 codes : Task 2. §4.4 GDN + SWA + bloc + taille : Tasks 4, 5, 7 (test de comptage de paramètres). §4.5 Mémoire constante et réinitialisation : Task 7 (`size_bytes`, `reset_memory`, `reset_all`). §4.6 Buffer : Task 6 et calendrier d'écriture Task 7 ; la précision « visible tous les 512 octets » est notée dans les contraintes globales et devra être reportée dans la spec §4.6 (une phrase) lors du premier commit du plan. §5 corpus : Task 9. §6.1 étape 1 : Tasks 8, 11. §6.4 stabilité : fp32 dans GDN/SWA/normes (Tasks 3-5), loss scaling et clipping (Task 11), repli fp32 documenté (Task 12). §6.5 checkpoints et reprise : Tasks 10, 11, 12. §10 structure : respectée. §11 S1 : Task 13 lance le pré-entraînement. Hors périmètre de ce plan, volontairement : §4.3, §4.7, §4.8 (contrôleur), §4.9, §6.2, §6.3, §6.6, §7, §8, §9.

**Placeholders.** Aucun « TBD », chaque étape de code contient le code ; les essais réseau et GPU sont marqués manuels avec un résultat attendu précis.

**Cohérence des types.** `GatedDeltaNet.forward(x, S, conv_state)` et `.step` (Task 4) sont appelés ainsi par `Block` (Task 7) ; `SlidingWindowAttention.forward(x, cache)` (Task 5) idem ; `State.gdn[i]` est un tuple `(S, conv_state)` partout ; `SlotWrite.apply_blocks(h, slots)` (Task 6) reçoit exactement `block` positions dans Task 7 ; `RelisModel.forward(x, mode, state)` avec `mode` int ou tenseur `(B,L)` est utilisé de la même façon dans `loop._loss`, `bench.throughput` et les tests ; `RelisConfig.to_dict()` (Task 3) est utilisé par `save_checkpoint` via `loop.train` (Tasks 10-11).
