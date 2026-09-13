# RELIS — Plan 4 : négatifs et historiques longs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire dépendre le STOP de la question et non d'un appât de surface, et faire tenir CONT sur des centaines de segments : deux défauts de la **distribution d'entraînement**, prouvés au pas 500 par la sonde `relis.eval.distractor` (le modèle s'arrête sur le premier fait venu) et par l'aiguille (0,95 à 1 ko, 0 à 16 ko). Rien à changer au modèle ni au contrôleur : ce plan produit des données, un rapport d'évaluation qui distingue les familles, une configuration longue, et le mode d'emploi des deux sessions GPU qui reprennent depuis le pas 500.

**Architecture:** Les générateurs d'épisodes gagnent des **tours-appâts** (faits d'un autre sujet, mentions du sujet sans valeur) tirés sur des indices libres de l'historique, sans toucher à l'oracle : un appât n'est jamais nécessaire, donc il est lu et suivi de CONT. Une famille `long_history` étire l'historique jusqu'au budget de ruban, qui devient un paramètre (`max_tape_bytes`) du générateur et de l'empaquetage. Le curriculum sur la longueur se fait en deux étapes : 15 ko dans 16 384, puis 60 ko dans 65 536, ce que le traitement par morceaux de 512 positions rend possible sans changer le modèle.

**Tech Stack:** Python ≥ 3.10, PyTorch ≥ 2.2, aucune dépendance nouvelle. Code existant : `relis/data/episodes.py`, `relis/data/pack.py`, `relis/eval/autonomy.py`, `relis/eval/harness.py`, `configs/tape_t4.yaml`, `notebooks/KAGGLE.md`, `notebooks/COLAB_EVAL.md`.

**Spec:** `docs/superpowers/specs/2026-09-06-relis-design.md` — §6.1 (curriculum sur la longueur), §6.2 (« Négatifs (plan 4) », « Historiques longs (plan 4) »), §9 (aiguille, autonomie), §12 (extrapolation).

## Global Constraints

- **L'oracle ne change pas.** `oracle_pass` reste identique octet pour octet ; les appâts entrent dans `injections` de `_history` et ne figurent jamais dans `needed_hist`. Les tests `tests/test_episodes.py`, `tests/test_cases.py`, `tests/test_pack.py`, `tests/test_tape_train.py` restent verts **sans modification**.
- **Pas de fait du sujet demandé comme appât.** `SUJETS` croise `DOC_TOPICS` sur « la réunion » et « le serveur » : pour `what_did_i_say` sur X, aucun fait `DOC_TOPICS` de sujet X n'est tiré ; pour `doc_lookup`/`absent` sur un sujet, aucun fait de ce sujet ; pour les familles `FACTS`, aucun fait dont le libellé est celui de la question (les deux libellés pour `two_facts`).
- **Budget de ruban paramétrable.** `MAX_TAPE_BYTES = 15_000` reste la valeur par défaut ; `generate_case`, `iter_cases`, `iter_episodes`, `build_shards` et la CLI `relis.data.pack` acceptent `max_tape_bytes`. Le garde-fou de redessin (`MAX_REDRAWS`) s'applique au budget passé.
- **Le mélange imprimé par `pack` doit être relu** avant tout lancement (spec §6.3) : la famille longue ajoute beaucoup de CONT à poids 5 ; si la part « décisions » dépasse 0,40, réduire `KIND_WEIGHTS[long_history]` plutôt que toucher aux poids de perte.
- **Rien dans `relis/model/`, `relis/infer/`, `relis/train/loop.py`.** Ce plan ne touche pas au modèle, au contrôleur ni à la boucle.
- Tests sur processeur, suite complète sous 90 secondes (elle est à 65 s). Un test échoue avant le code qui le fait passer. Un commit par tâche au minimum.
- Branche `plan4-donnees`, créée depuis `main` à `dad096e` ou plus récent. Aucun entraînement ne tourne : fusion libre.

---

## Structure des fichiers

```
relis/data/episodes.py       (modifier) Case.negatives ; NEG_* ; _mention, _other_fact, _negatives, _neg_indices ;
                                        appâts dans les sept familles ; _long_history ; max_tape_bytes partout
relis/data/pack.py           (modifier) build_shards(..., max_tape_bytes) ; --max_tape_bytes
relis/eval/autonomy.py       (modifier) arrêts sur appât ; exactitude par famille
configs/tape_long_t4.yaml    (créer)    séquences de 65 536, batch 1 × accum 4
notebooks/KAGGLE.md          (modifier) section « Plan 4 » : deux empaquetages, deux sessions
notebooks/COLAB_EVAL.md      (modifier) lecture des nouvelles lignes du rapport ; critères de succès
docs/CARNET.md               (modifier) section plan 4
tests/test_negatives.py      (créer)
tests/test_long_history.py   (créer)
tests/test_autonomy_report.py (créer)
tests/test_configs.py        (créer)
```

---

### Task 1 : tours-appâts dans les sept familles

**Files:**
- Modify: `relis/data/episodes.py` (imports ligne 14 ; `Case` lignes 21-35 ; constantes après `FILLER_DOC_NAMES` ligne 91 ; générateurs lignes 183-297)
- Test: `tests/test_negatives.py`

**Interfaces:**
- Consumes : `_history(rng, n_pairs, injections)`, `_de(sujet)`, `FACTS`, `DOC_TOPICS`, `SUJETS`, `Case`, `case_to_spec`, `generate_case`.
- Produces : `Case.negatives: list[int]` (indices de segments dans l'ordre du balayage, récent → ancien, comme `stop_index`) ; `NEG_RATE: float = 0.7` ; `NEG_MENTIONS: list[str]` ; `DOC_LABELS: dict[str, str]` ; `_mention(rng, label) -> str` ; `_other_fact(rng, exclude: set) -> str` ; `_negatives(rng, n_pairs, taken, label, exclude, k_max=3) -> dict[int, str]` ; `_neg_indices(n_pairs, injections, keep) -> list[int]`. Task 3 lit `case.negatives`.

- [ ] **Step 1 : tests**

`tests/test_negatives.py` :
```python
import random
import re
from collections import Counter
from relis.tape import codes as C
from relis.data.episodes import (KINDS, FACTS, DOC_TOPICS, DOC_LABELS, NEG_MENTIONS, NEG_RATE, SUJETS,
                                 _negatives, _mention, _other_fact, _neg_indices, _de,
                                 generate_case, case_to_spec)


def _template_regexes():
    """Toutes les phrases de fait, {v} remplacé par .+ : reconnaît un « fait d'un autre sujet »."""
    tpls = [f[0] for f in FACTS] + [t[1] for t in DOC_TOPICS]
    return [re.compile("^" + re.escape(t).replace(re.escape("{v}"), ".+") + "$") for t in tpls]


def _is_fact(text: str) -> bool:
    return any(r.match(text) for r in _template_regexes())


def _is_mention(text: str, label: str) -> bool:
    return label in text and not _is_fact(text)


def test_negatives_avoid_taken_indices_and_excluded_subject():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(3, 14); taken = {rng.randrange(n)}
        neg = _negatives(rng, n, taken, "votre ville", {"votre ville"})
        assert not (set(neg) & taken) and 0 <= len(neg) <= 3
        for txt in neg.values():
            assert not txt.startswith("J'habite à"), txt        # le fait exclu ne sort jamais
            assert _is_fact(txt) or _is_mention(txt, "votre ville"), txt


def test_negatives_respect_rate_and_k_max():
    rng = random.Random(1)
    with_neg = sum(bool(_negatives(rng, 14, set(), "votre ville", {"votre ville"})) for _ in range(1000))
    assert 620 <= with_neg <= 780                             # NEG_RATE = 0,7
    big = [len(_negatives(rng, 200, set(), None, set(), k_max=6)) for _ in range(200)]
    assert max(big) == 6 and min(big) == 0


def test_mention_has_label_and_no_value():
    rng = random.Random(2)
    for label in ("votre code postal", "la réunion", "le mot de passe du wifi"):
        m = _mention(rng, label)
        assert m in {t.format(label=label, de_label=_de(label)) for t in NEG_MENTIONS}
        assert not any(ch.isdigit() for ch in m)


def test_other_fact_excludes_by_label_and_topic():
    rng = random.Random(3)
    for _ in range(200):
        txt = _other_fact(rng, {"votre prénom", "serveur"})
        assert not txt.startswith("Je m'appelle") and "adresse du serveur" not in txt
        assert _is_fact(txt)


def test_neg_indices_are_scan_order_user_segments():
    # n=5 paires ; injections aux paires 0 et 3 ; on garde 3 (le fait) → l'appât 0 est le plus ancien :
    # segment user de la paire 0 = 2*(5-1-0)+1 = 9
    assert _neg_indices(5, {0: "a", 3: "b"}, {3}) == [9]


def test_every_kind_still_builds_and_oracle_ignores_negatives():
    rng = random.Random(4)
    seen = Counter()
    for kind in KINDS:
        for _ in range(30):
            c = generate_case(rng, kind)
            spec = case_to_spec(c)
            seen[kind] += bool(c.negatives)
            labels = [f[3] for f in FACTS] + list(DOC_LABELS.values()) + ["la valeur de x"] + list(SUJETS)
            for i in c.negatives:
                assert 0 <= i < len(c.history)
                seg = c.history[i]
                assert seg.header.startswith("role=user")
                txt = seg.content.decode("utf-8")
                assert _is_fact(txt) or any(lbl in txt for lbl in labels), txt
                if i == c.stop_index:
                    # seul « absent » sans document s'arrête au dernier segment, qui peut être un appât
                    assert kind == "absent" and not c.docs, (kind, i)
                    continue
                # un appât lu avant le STOP est suivi de CONT dans le ruban
                hist = spec.passes[0].history
                if i < len(hist):
                    assert hist[i].after == C.CONT, (kind, i)
    for kind in KINDS:
        assert seen[kind] >= 10, (kind, seen[kind])            # ≈ 70 % des épisodes en ont


def test_what_did_i_say_never_gets_a_doc_fact_on_its_own_subject():
    rng = random.Random(5)
    for _ in range(300):
        c = generate_case(rng, "what_did_i_say")
        sujet = c.query.decode("utf-8")[len("Qu'ai-je dit sur "):-2]
        for i in c.negatives:
            txt = c.history[i].content.decode("utf-8")
            for topic, tpl, *_ in DOC_TOPICS:
                if DOC_LABELS[topic] == sujet:
                    assert not re.match("^" + re.escape(tpl).replace(re.escape("{v}"), ".+") + "$", txt), txt


def test_absent_and_doc_lookup_histories_are_longer_and_carry_mentions():
    rng = random.Random(6)
    mentions = 0
    for kind in ("absent", "doc_lookup"):
        for _ in range(100):
            c = generate_case(rng, kind)
            assert 4 <= len(c.history) <= 16
            for i in c.negatives:
                txt = c.history[i].content.decode("utf-8")
                mentions += any(lbl in txt for lbl in DOC_LABELS.values()) and not _is_fact(txt)
    assert mentions >= 20
```

- [ ] **Step 2 : lancer, vérifier l'échec**

Run : `python -m pytest tests/test_negatives.py -q`
Expected : ImportError (`DOC_LABELS`, `_negatives`… inexistants).

- [ ] **Step 3 : implémentation**

Dans `relis/data/episodes.py` :

Import : `from dataclasses import dataclass, field`.

`Case` : ajouter après `stop_index: int = -1` :
```python
    negatives: list = field(default_factory=list)   # indices (ordre du balayage) des tours-appâts, plan 4
```

Après `FILLER_DOC_NAMES` :
```python
# --- tours-appâts (spec §6.2 « Négatifs (plan 4) ») ---------------------------------------------
NEG_RATE = 0.7                     # part des épisodes qui reçoivent au moins un appât
NEG_MENTIONS = ["Parlons {de_label}.", "As-tu bien noté {label} ?", "Il faudra vérifier {label}.",
                "On reparlera {de_label} plus tard.", "Je te redonnerai {label} demain."]
DOC_LABELS = {"contrat": "le contrat", "facture": "la facture", "reunion": "la réunion",
              "serveur": "le serveur", "recette": "la recette"}


def _mention(rng, label: str) -> str:
    """Le sujet de la question, sans sa valeur : « Parlons de la réunion. »"""
    return rng.choice(NEG_MENTIONS).format(label=label, de_label=_de(label))


def _other_fact(rng, exclude: set) -> str:
    """Une phrase en forme de fait sur un autre sujet. `exclude` contient des libellés FACTS
    (f[3]) et/ou des sujets DOC_TOPICS (t[0]) à ne jamais tirer."""
    pool = [f for f in FACTS if f[3] not in exclude]
    topics = [t for t in DOC_TOPICS if t[0] not in exclude]
    k = rng.randrange(len(pool) + len(topics))
    if k < len(pool):
        f = pool[k]
        return f[0].format(v=f[4](rng))
    t = topics[k - len(pool)]
    return t[1].format(v=t[4](rng))


def _negatives(rng, n_pairs: int, taken, label, exclude, k_max: int = 3) -> dict:
    """0 à k_max tours-appâts sur des indices de paires libres (pas dans `taken`). Moitié
    mentions du sujet sans valeur (si `label` est donné), moitié faits d'un autre sujet.
    Ne touche jamais l'oracle : aucun appât n'est nécessaire, il est lu et suivi de CONT."""
    free = [i for i in range(n_pairs) if i not in taken]
    if not free or rng.random() > NEG_RATE:
        return {}
    k = min(len(free), rng.randint(1, k_max))
    out = {}
    for i in rng.sample(free, k):
        if label is not None and rng.random() < 0.5:
            out[i] = _mention(rng, label)
        else:
            out[i] = _other_fact(rng, exclude)
    return out


def _neg_indices(n_pairs: int, injections: dict, keep) -> list:
    """Indices de segment (récent → ancien, comme stop_index) des injections qui ne sont pas dans `keep`."""
    return sorted(2 * (n_pairs - 1 - i) + 1 for i in injections if i not in keep)
```

Générateurs, un par un (le reste de chaque fonction est inchangé) :

`_fact_recall` :
```python
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(3, 14); at = rng.randint(0, n - 1)
    inj = {at: f[0].format(v=v)}
    inj.update(_negatives(rng, n, {at}, f[3], {f[3]}))
    hist = _history(rng, n, inj)
    idx = 2 * (n - 1 - at) + 1
    return Case(kind="fact_recall", query=f[1].encode("utf-8"), history=hist, docs=[],
                doc_relevant=[], passes_needed=[({idx}, None)],
                answer_parts=[f[2].format(v=v).encode()], notes=[], answer_value=str(v),
                negatives=_neg_indices(n, inj, {at}))
```

`_variable_tracking` : après la boucle qui remplit `ops`,
```python
    ops.update(_negatives(rng, n, set(at), "la valeur de x", set()))
    hist = _history(rng, n, ops)
    needed = {2 * (n - 1 - a) + 1 for a in at}
    return Case(kind="variable_tracking", query=b"Combien vaut x maintenant ?", history=hist, docs=[],
                doc_relevant=[], passes_needed=[(needed, None)],
                answer_parts=[f"x vaut {val}.".encode()], notes=[], answer_value=str(val),
                negatives=_neg_indices(n, ops, set(at)))
```

`_what_did_i_say` :
```python
    sujet = rng.choice(SUJETS); v = rng.choice(["c'est urgent", "c'est reporté", "c'est terminé", "il faut un budget"])
    n = rng.randint(3, 12); at = rng.randint(0, n - 1)
    inj = {at: f"À propos {_de(sujet)} : {v}."}
    exclude = {t for t, lbl in DOC_LABELS.items() if lbl == sujet}    # « la réunion », « le serveur »
    inj.update(_negatives(rng, n, {at}, sujet, exclude))
    hist = _history(rng, n, inj)
    idx = 2 * (n - 1 - at) + 1
    return Case(kind="what_did_i_say", query=f"Qu'ai-je dit sur {sujet} ?".encode(), history=hist, docs=[],
                doc_relevant=[], passes_needed=[({idx}, None)],
                answer_parts=[f"Vous avez dit : « {v} ».".encode()], notes=[], answer_value=v,
                negatives=_neg_indices(n, inj, {at}))
```

`_doc_lookup` :
```python
    topic, tpl, q, a, gen = rng.choice(DOC_TOPICS); v = gen(rng)
    n = rng.randint(2, 8)
    inj = _negatives(rng, n, set(), DOC_LABELS[topic], {topic})
    hist = _history(rng, n, inj)
    docs, relevant, target = _docs_for(rng, topic, tpl.format(v=v), rng.randint(1, 8), noise=True)
    return Case(kind="doc_lookup", query=q.encode(), history=hist, docs=docs,
                doc_relevant=relevant, passes_needed=[(set(), target)],
                answer_parts=[a.format(v=v).encode()], notes=[], answer_value=str(v),
                negatives=_neg_indices(n, inj, set()))
```

`_absent` :
```python
    topic, tpl, q, a, gen = rng.choice(DOC_TOPICS)
    n = rng.randint(2, 8)
    inj = _negatives(rng, n, set(), DOC_LABELS[topic], {topic})
    hist = _history(rng, n, inj)
    n_docs = rng.randint(0, 6)
    docs = [_filler_doc(rng, j) for j in range(n_docs)]
    return Case(kind="absent", query=q.encode(), history=hist, docs=docs,
                doc_relevant=[False] * n_docs, passes_needed=[(set(), None)],
                answer_parts=[b"Je ne trouve pas cette information dans notre \xc3\xa9change ni dans les documents."],
                notes=[], answer_value="", negatives=_neg_indices(n, inj, set()))
```

`_two_facts` :
```python
    inj = {at_a: fa[0].format(v=va), at_b: fb[0].format(v=vb)}
    inj.update(_negatives(rng, n, {at_a, at_b}, rng.choice([fa[3], fb[3]]), {fa[3], fb[3]}))
    hist = _history(rng, n, inj)
    ...
                notes=[f"il manque {fb[3]}".encode()], answer_value=str(vb),
                negatives=_neg_indices(n, inj, {at_a, at_b}))
```

`_long_answer` :
```python
    inj = {at: f[0].format(v=v)}
    inj.update(_negatives(rng, n, {at}, f[3], {f[3]}))
    hist = _history(rng, n, inj)
    ...
                answer_parts=parts, notes=notes, answer_value=str(v),
                negatives=_neg_indices(n, inj, {at}))
```

- [ ] **Step 4 : lancer**

Run : `python -m pytest tests/test_negatives.py tests/test_episodes.py tests/test_cases.py tests/test_pack.py -q`
Expected : tous verts. Si `test_generated_tapes_never_exceed_budget` échoue, c'est que le garde-fou de redessin ne suffit plus : réduire `n` maximal de `_two_facts` à 12, ne pas toucher au budget.

- [ ] **Step 5 : commit**

```bash
git add relis/data/episodes.py tests/test_negatives.py
git commit -m "data: tours-appâts dans les sept familles (faits d'un autre sujet, mentions sans valeur) ; Case.negatives"
```

---

### Task 2 : famille `long_history` et budget de ruban paramétrable

**Files:**
- Modify: `relis/data/episodes.py` (`KINDS`/`KIND_WEIGHTS` ligne 47-48 ; `_GEN`/`generate_case`/`iter_*` lignes 300-330)
- Modify: `relis/data/pack.py` (`build_shards` ligne 267 ; `main` ligne 313)
- Test: `tests/test_long_history.py`

**Interfaces:**
- Consumes : `_negatives(rng, n_pairs, taken, label, exclude, k_max)`, `_neg_indices`, `_history`, `Case`.
- Produces : `KINDS` contient `"long_history"` ; `BUDGETED_KINDS = ("long_history",)` ; `_long_history(rng, max_tape_bytes=MAX_TAPE_BYTES) -> Case` ; `generate_case(rng, kind=None, max_tape_bytes=MAX_TAPE_BYTES)` ; `generate_episode(rng, kind=None, max_tape_bytes=MAX_TAPE_BYTES)` ; `iter_cases(seed, n, max_tape_bytes=MAX_TAPE_BYTES)` ; `iter_episodes(seed, n, max_tape_bytes=MAX_TAPE_BYTES)` ; `build_shards(..., max_tape_bytes=MAX_TAPE_BYTES)` ; option CLI `--max_tape_bytes` (défaut 15000).

- [ ] **Step 1 : tests**

`tests/test_long_history.py` :
```python
import os
import random
from relis.tape import codes as C
from relis.tape.tape import build_tape, validate_spec
from relis.data.episodes import (KINDS, MAX_TAPE_BYTES, generate_case, case_to_spec, iter_episodes,
                                 _long_history)
from relis.data.pack import build_shards, TapeWindows
from relis.data.shards import ShardWriter


def test_long_history_is_registered():
    assert "long_history" in KINDS


def test_long_history_reaches_hundreds_of_segments_within_default_budget():
    rng = random.Random(0)
    lengths, tapes = [], []
    for _ in range(40):
        c = generate_case(rng, "long_history")
        spec = case_to_spec(c); validate_spec(spec)
        t = build_tape(spec)
        assert t.decisions()[-1] == C.END and len(t) <= MAX_TAPE_BYTES
        lengths.append(len(c.history)); tapes.append(len(t))
        assert len(c.history) >= 40                       # ≥ 20 paires
        assert c.stop_index >= len(c.history) // 2 - 1    # le fait est dans la moitié ancienne
    assert max(lengths) >= 160 and max(tapes) >= 8_000


def test_long_history_scales_with_budget():
    rng = random.Random(1)
    lengths, tapes = [], []
    for _ in range(20):
        c = generate_case(rng, "long_history", max_tape_bytes=60_000)
        t = build_tape(case_to_spec(c))
        assert len(t) <= 60_000
        lengths.append(len(c.history)); tapes.append(len(t))
    assert max(lengths) >= 600 and max(tapes) >= 25_000


def test_long_history_has_negatives_and_oracle_stops_on_the_fact():
    rng = random.Random(2)
    with_neg = 0
    for _ in range(30):
        c = generate_case(rng, "long_history")
        with_neg += bool(c.negatives)
        assert max(c.negatives, default=-1) < len(c.history)
        spec = case_to_spec(c)
        assert spec.passes[0].history[-1].after == C.STOP
        assert c.answer_value in spec.passes[0].history[-1].content.decode("utf-8")
    assert with_neg >= 12


def test_iter_episodes_accepts_budget_and_stays_deterministic():
    a = [build_tape(s).data for s in iter_episodes(seed=3, n=20, max_tape_bytes=30_000)]
    b = [build_tape(s).data for s in iter_episodes(seed=3, n=20, max_tape_bytes=30_000)]
    assert a == b and all(len(x) <= 30_000 for x in a)


def test_build_shards_passes_budget_and_skips_nothing(tmp_path):
    pre = str(tmp_path / "pre.bin")
    w = ShardWriter(pre); w.add(("le chat dort. " * 1200).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    stats = build_shards(out, episodes=40, seed=0, seq_len=16384, block=512, replay=pre,
                         replay_frac=0.1, val_frac=0.1, max_tape_bytes=15_000)
    assert stats["train_skipped"] == 0 and stats["val_skipped"] == 0
    assert TapeWindows(os.path.join(out, "train")).seq_len == 16384
```

- [ ] **Step 2 : lancer, vérifier l'échec**

Run : `python -m pytest tests/test_long_history.py -q`
Expected : ImportError sur `_long_history`, puis TypeError sur `max_tape_bytes`.

- [ ] **Step 3 : implémentation**

`relis/data/episodes.py` :
```python
KINDS = ("fact_recall", "variable_tracking", "what_did_i_say", "doc_lookup", "absent", "two_facts",
         "long_answer", "long_history")
KIND_WEIGHTS = (20, 10, 12, 20, 8, 10, 8, 12)
BUDGETED_KINDS = ("long_history",)        # générateurs qui reçoivent max_tape_bytes
MAX_TAPE_BYTES = 15_000
MAX_REDRAWS = 8
BYTES_PER_PAIR = 125                      # ≈ 2 segments × 60 octets de ruban (mesuré)
```

Après `_long_answer` :
```python
def _long_history(rng, max_tape_bytes: int = MAX_TAPE_BYTES) -> Case:
    """Comme fact_recall, mais l'historique compte de 20 paires jusqu'au maximum que le budget de
    ruban permet, et le fait est dans la moitié ancienne pour que le balayage soit long (spec §6.2
    « Historiques longs »). Jusqu'à six appâts. C'est la famille qui apprend à maintenir CONT sur des
    centaines de segments."""
    f = rng.choice(FACTS); v = f[4](rng)
    n_max = max(21, (max_tape_bytes - 600) // BYTES_PER_PAIR)
    n = rng.randint(20, n_max); at = rng.randint(0, n // 2)
    inj = {at: f[0].format(v=v)}
    inj.update(_negatives(rng, n, {at}, f[3], {f[3]}, k_max=6))
    hist = _history(rng, n, inj)
    idx = 2 * (n - 1 - at) + 1
    return Case(kind="long_history", query=f[1].encode("utf-8"), history=hist, docs=[], doc_relevant=[],
                passes_needed=[({idx}, None)], answer_parts=[f[2].format(v=v).encode()], notes=[],
                answer_value=str(v), negatives=_neg_indices(n, inj, {at}))
```

Registre et générateurs :
```python
_GEN = {"fact_recall": _fact_recall, "variable_tracking": _variable_tracking, "what_did_i_say": _what_did_i_say,
        "doc_lookup": _doc_lookup, "absent": _absent, "two_facts": _two_facts, "long_answer": _long_answer,
        "long_history": _long_history}


def generate_case(rng: random.Random, kind: str | None = None, max_tape_bytes: int = MAX_TAPE_BYTES) -> Case:
    if kind is None:
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS, k=1)[0]
    for _ in range(MAX_REDRAWS + 1):
        case = _GEN[kind](rng, max_tape_bytes) if kind in BUDGETED_KINDS else _GEN[kind](rng)
        spec = case_to_spec(case)
        if len(build_tape(spec)) <= max_tape_bytes:
            return case
    raise RuntimeError(f"épisode « {kind} » toujours au-dessus de {max_tape_bytes} octets")


def generate_episode(rng: random.Random, kind: str | None = None, max_tape_bytes: int = MAX_TAPE_BYTES) -> TapeSpec:
    return case_to_spec(generate_case(rng, kind, max_tape_bytes))


def iter_cases(seed: int, n: int, max_tape_bytes: int = MAX_TAPE_BYTES):
    rng = random.Random(seed)
    for _ in range(n):
        yield generate_case(rng, max_tape_bytes=max_tape_bytes)


def iter_episodes(seed: int, n: int, max_tape_bytes: int = MAX_TAPE_BYTES):
    rng = random.Random(seed)
    for _ in range(n):
        yield case_to_spec(generate_case(rng, max_tape_bytes=max_tape_bytes))
```

`relis/data/pack.py`, `build_shards` : ajouter le paramètre `max_tape_bytes: int = 15_000` en fin de signature et remplacer la boucle :
```python
    for i, spec in enumerate(iter_episodes(seed, episodes, max_tape_bytes)):
        packers["val" if i % val_every == 0 else "train"].add(build_tape(spec))
```
`main` : `ap.add_argument("--max_tape_bytes", type=int, default=15_000, help="budget d'un ruban ; doit rester ≤ seq_len − block, sinon le ruban est ignoré (compteur skipped)")` et le passer à `build_shards(..., max_tape_bytes=args.max_tape_bytes)`. Après `print(stats)`, ajouter :
```python
    if stats.get("train_skipped", 0) or stats.get("val_skipped", 0):
        print("[pack] ATTENTION : des rubans dépassent seq_len et ont été ignorés ; "
              "baisser --max_tape_bytes ou monter --seq_len", file=sys.stderr)
```

- [ ] **Step 4 : lancer**

Run : `python -m pytest tests/test_long_history.py tests/test_negatives.py tests/test_episodes.py tests/test_cases.py tests/test_pack.py tests/test_eval.py -q`
Expected : tous verts. `test_every_kind_builds_a_valid_tape` (plafond `< 16_000`) et `test_generated_tapes_never_exceed_budget` couvrent la nouvelle famille au budget par défaut.

- [ ] **Step 5 : commit**

```bash
git add relis/data/episodes.py relis/data/pack.py tests/test_long_history.py
git commit -m "data: famille long_history ; budget de ruban max_tape_bytes paramétrable jusqu'à l'empaquetage"
```

---

### Task 3 : rapport d'autonomie par famille et arrêts sur appât

**Files:**
- Modify: `relis/eval/autonomy.py` (`evaluate` lignes 75-92 ; `format_report` lignes 93-101)
- Test: `tests/test_autonomy_report.py`

**Interfaces:**
- Consumes : `Case.negatives`, `Case.kind`, `run_case`, `classify`, `judge`, `budget_for`.
- Produces : `evaluate(...)` renvoie en plus `"arrets_appat": int` et `"par_famille": dict[str, tuple[int, int]]` (exacts, effectif) ; `format_report` imprime deux lignes de plus : `arrêts sur appât : N` et `par famille : kind a/b · …`.

- [ ] **Step 1 : tests**

`tests/test_autonomy_report.py` :
```python
from relis.eval import autonomy


def _res(**kw):
    base = {"n": 3, "classes": {k: 0 for k in autonomy.CLASSES}, "effectifs": {k: 0 for k in autonomy.CLASSES},
            "exactitude": 1.0, "octets_lus_moyen": 10, "economie_moyenne": 0.5, "arrets_budget": 0,
            "arrets_appat": 0, "par_famille": {}}
    base.update(kw)
    return base


def test_report_mentions_bait_stops_and_per_family():
    r = _res(arrets_appat=2, par_famille={"fact_recall": (3, 3), "long_history": (1, 4)})
    txt = autonomy.format_report(r)
    assert "arrêts sur appât : 2" in txt
    assert "fact_recall 3/3" in txt and "long_history 1/4" in txt


def test_evaluate_counts_bait_stops_and_groups_by_kind(monkeypatch):
    class Case:
        def __init__(self, kind, negatives, stop_index):
            self.kind, self.negatives, self.stop_index = kind, negatives, stop_index
            self.docs = []; self.passes_needed = [(set(), None)]; self.answer_value = "42"
            self.history = []; self.query = b"q"
    cases = [Case("a", [1], 3), Case("a", [], 3), Case("b", [0], 2)]
    results = iter([{"stop_at": 1, "doc_actions": [], "answer": "42", "read": 1, "saved": 0.5, "stopped_by_budget": False},
                    {"stop_at": 3, "doc_actions": [], "answer": "42", "read": 1, "saved": 0.5, "stopped_by_budget": False},
                    {"stop_at": 0, "doc_actions": [], "answer": "x", "read": 1, "saved": 0.5, "stopped_by_budget": False}])
    monkeypatch.setattr(autonomy, "run_case", lambda ctl, case, budget=None: next(results))
    monkeypatch.setattr(autonomy, "budget_for", lambda case, max_gen: None)
    monkeypatch.setattr(autonomy, "judge", lambda case, answer: answer == case.answer_value)

    class Ctl:
        class budget:
            max_gen_bytes = 8
    res = autonomy.evaluate(Ctl(), cases)
    assert res["arrets_appat"] == 2                      # cas 1 (stop 1 ∈ {1}) et cas 3 (stop 0 ∈ {0})
    assert res["par_famille"] == {"a": (1, 2), "b": (0, 1)}
    assert res["effectifs"]["exact"] == 1 and res["effectifs"]["trop_tot"] == 2
```

- [ ] **Step 2 : lancer, vérifier l'échec**

Run : `python -m pytest tests/test_autonomy_report.py -q`
Expected : KeyError `arrets_appat` / AssertionError sur le texte.

- [ ] **Step 3 : implémentation**

`evaluate` :
```python
def evaluate(ctl, cases, max_gen: int | None = None) -> dict:
    max_gen = ctl.budget.max_gen_bytes if max_gen is None else max_gen
    counts, bons, lus, saved, arrets_budget, arrets_appat = Counter(), 0, 0, 0.0, 0, 0
    par_famille = {}
    for case in cases:
        r = run_case(ctl, case, budget=budget_for(case, max_gen))
        cls = classify(case, r)
        counts[cls] += 1
        bons += int(judge(case, r["answer"]))
        lus += r["read"]
        saved += r["saved"]
        arrets_budget += int(r["stopped_by_budget"])
        # arrêt sur appât : STOP tombé sur un tour-appât (plan 4) — la preuve directe qu'un
        # raccourci de surface a tranché à la place de la question
        arrets_appat += int(r["stop_at"] is not None and r["stop_at"] in getattr(case, "negatives", []))
        ex, tot = par_famille.get(case.kind, (0, 0))
        par_famille[case.kind] = (ex + int(cls == "exact"), tot + 1)
    n = max(1, sum(counts.values()))
    return {"n": sum(counts.values()),
            "classes": {k: counts.get(k, 0) / n for k in CLASSES},
            "effectifs": {k: counts.get(k, 0) for k in CLASSES},
            "exactitude": bons / n, "octets_lus_moyen": lus // n,
            "economie_moyenne": round(saved / n, 3),
            "arrets_budget": arrets_budget, "arrets_appat": arrets_appat,
            "par_famille": par_famille}
```

`format_report` : conserver la première ligne telle quelle et ajouter :
```python
    fam = " · ".join(f"{k} {ex}/{tot}" for k, (ex, tot) in sorted(res["par_famille"].items()))
    return (first_line + f"\n  arrêts sur appât : {res['arrets_appat']}"
            + (f"\n  par famille (arrêts exacts) : {fam}" if fam else ""))
```
où `first_line` est la chaîne actuellement renvoyée.

- [ ] **Step 4 : lancer**

Run : `python -m pytest tests/test_autonomy_report.py tests/test_eval.py -q`
Expected : verts. `tests/test_eval.py` construit un vrai `Case` : il possède `negatives` depuis la Task 1.

- [ ] **Step 5 : commit**

```bash
git add relis/eval/autonomy.py tests/test_autonomy_report.py
git commit -m "eval(autonomy): arrêts sur appât et exactitude par famille"
```

---

### Task 4 : configuration longue, mode d'emploi des deux sessions, carnet

**Files:**
- Create: `configs/tape_long_t4.yaml`
- Modify: `notebooks/KAGGLE.md` (ajouter une section « Plan 4 » après « Entraînement ruban (plan 2a) »)
- Modify: `notebooks/COLAB_EVAL.md` (section « Comment lire chaque sortie », sous-section autonomie ; critères de succès)
- Modify: `docs/CARNET.md` (section « ⬜ Plan 2b » : ajouter une section « 🟠 Plan 4 » avant)
- Test: `tests/test_configs.py`

**Interfaces:**
- Consumes : `TrainConfig` (champs `seq_len`, `batch_size`, `grad_accum`), `RelisConfig` (`block`).
- Produces : `configs/tape_long_t4.yaml` chargeable par `relis.train.tape_train` ; aucune interface Python nouvelle.

- [ ] **Step 1 : test**

`tests/test_configs.py` :
```python
import yaml
from relis.model.config import RelisConfig
from relis.train.loop import TrainConfig


def _load(path):
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    return RelisConfig(**raw["model"]), TrainConfig(**raw["train"]), raw["data"]


def test_tape_long_config_matches_65536_sequences_and_keeps_bytes_per_step():
    m, t, _ = _load("configs/tape_long_t4.yaml")
    assert (t.seq_len + 1) == 65536 and 65536 % m.block == 0
    per_step = 2 * t.batch_size * t.grad_accum * (t.seq_len + 1)      # 2 GPU
    assert 450_000 <= per_step <= 600_000                            # ≈ 0,5 M octets, comme tape_t4
    assert m.grad_checkpoint and t.amp


def test_tape_long_and_tape_t4_share_the_model():
    m_long, _, _ = _load("configs/tape_long_t4.yaml")
    m_ref, _, _ = _load("configs/tape_t4.yaml")
    assert m_long.to_dict() == m_ref.to_dict()          # init_from strict : même architecture
```

- [ ] **Step 2 : lancer, vérifier l'échec**

Run : `python -m pytest tests/test_configs.py -q`
Expected : FileNotFoundError.

- [ ] **Step 3 : configuration**

`configs/tape_long_t4.yaml` :
```yaml
model:                 # identique à tape_t4.yaml : init_from est strict
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
  seq_len: 65535         # séquences empaquetées de 65 536 octets (rubans ≤ 60 000), décalées d'un
  batch_size: 1          # par GPU : le modèle traite la séquence par morceaux de 512, la mémoire
                         # d'activations par morceau ne dépend pas de la longueur ; à vérifier par bench
  grad_accum: 4          # 2 GPU × 1 × 4 × 65 535 ≈ 0,5 M octets par pas, comme tape_t4.yaml
  lr: 5.0e-5             # reprise depuis un modèle déjà entraîné aux rubans : moitié du taux initial
  warmup_frac: 0.05
  max_steps: 150         # ≈ 6 h à 140 s/pas ; 150 × 8 = 1 200 séquences sur ≈ 3 500 : 0,35 passage
  weight_decay: 0.1
  grad_clip: 1.0
  amp: true
  ckpt_every_minutes: 30
  log_every: 10
  val_batches: 4         # 4 lots × 65 536 octets ≈ 260 k octets de validation par point
  hub_repo: null         # ex. jaafar2022/relis-v2-long
  hub_every_minutes: 120
  time_budget_hours: 11.5
  init_from: null        # ex. jaafar2022/relis-v2-tape (fin de l'étape A)
  decision_balance: 0.0
  compile: false
data:
  train_dir: /kaggle/input/relis-tapes-long/train
  val_dir: /kaggle/input/relis-tapes-long/val
```

- [ ] **Step 4 : `notebooks/KAGGLE.md`, section « Plan 4 — négatifs et historiques longs »**

Ajouter après la section « Entraînement ruban (plan 2a) » :

````markdown
## Plan 4 — négatifs et historiques longs (reprise depuis le pas 500)

Deux étapes. Chacune : empaqueter sur CPU, téléverser en Kaggle Dataset, une session Kaggle, puis les
évaluations Colab (`notebooks/COLAB_EVAL.md`). Créer d'abord les dépôts Hub `relis-v2-tape` et
`relis-v2-long` (jeton WRITE), comme pour `relis-v1-tape`.

### Étape A — appâts et historiques jusqu'à 115 paires (15 ko dans 16 384)
```bash
python -m relis.data.pack --out shards/tapes_v2 --episodes 100000 --seed 1 --seq_len 16384 --block 512 \
    --max_tape_bytes 15000 --replay shards/v1/train.bin --replay_frac 0.2 --val_frac 0.02
```
Relire le mélange imprimé : la famille `long_history` (12 % des épisodes, rubans de 3 à 13 ko) ajoute
beaucoup de CONT à poids 5. **Si `parts décisions` dépasse 0,40**, refaire l'empaquetage après avoir
baissé `KIND_WEIGHTS[-1]` (12 → 8) dans `relis/data/episodes.py` ; ne pas toucher aux poids de perte.
Le compteur `skipped` doit valoir 0. Téléverser `shards/tapes_v2/` comme Kaggle Dataset `relis-tapes-v2`.

```python
%cd /kaggle/working/relis
!git pull -q && pip install -q -e .
!torchrun --nproc_per_node=2 -m relis.train.tape_train \
    --config configs/tape_t4.yaml --run_dir /kaggle/working/runs/tape2 --static_graph \
    --override train.hub_repo=<utilisateur>/relis-v2-tape train.init_from=<utilisateur>/relis-v1-tape \
               train.max_steps=300 train.lr=5e-5 \
               data.train_dir=/kaggle/input/datasets/<utilisateur>/relis-tapes-v2/train \
               data.val_dir=/kaggle/input/datasets/<utilisateur>/relis-tapes-v2/val
```
`init_from` charge les poids du pas 500 avec un optimiseur neuf ; `lr=5e-5` (moitié du taux initial)
parce que le modèle sait déjà lire ; 300 pas ≈ 12 h, donc une session pleine et un début de seconde.
La ligne `[val]` doit rester ≥ 0,90 en équilibrée dès les premiers points : les anciennes décisions ne
doivent pas régresser pendant que les appâts s'apprennent.

### Étape B — historiques jusqu'à 475 paires (60 ko dans 65 536)
D'abord mesurer la mémoire, sur une session courte (2 min) :
```python
!python -m relis.train.bench --config configs/tape_long_t4.yaml
```
Si `bench` imprime `[bench] OOM à batch_size=1`, l'étape B n'est pas possible telle quelle sur T4 :
revenir à `--seq_len 32768 --max_tape_bytes 28000` (≈ 220 paires) et `train.seq_len=32767
train.grad_accum=8`, et le noter dans le carnet.

```bash
python -m relis.data.pack --out shards/tapes_long --episodes 100000 --seed 2 --seq_len 65536 --block 512 \
    --max_tape_bytes 60000 --replay shards/v1/train.bin --replay_frac 0.2 --val_frac 0.02
```
Téléverser comme `relis-tapes-long`. Puis :
```python
%cd /kaggle/working/relis
!git pull -q && pip install -q -e .
!torchrun --nproc_per_node=2 -m relis.train.tape_train \
    --config configs/tape_long_t4.yaml --run_dir /kaggle/working/runs/tape_long --static_graph \
    --override train.hub_repo=<utilisateur>/relis-v2-long train.init_from=<utilisateur>/relis-v2-tape \
               data.train_dir=/kaggle/input/datasets/<utilisateur>/relis-tapes-long/train \
               data.val_dir=/kaggle/input/datasets/<utilisateur>/relis-tapes-long/val
```
150 pas ≈ 6 h. Le pas est plus lent qu'à l'étape A si la validation (4 lots de 65 536) prend plus
d'une minute : c'est attendu.

### Critères de succès (évaluations Colab sur chaque checkpoint)
| Mesure | Pas 500 (avant) | Après A | Après B |
|---|---|---|---|
| `distractor` A et B : `stop_at` | 3 (l'appât) | 7 (le fait) | 7 |
| `autonomy` : arrêts sur appât | à mesurer sur les rubans v2 | 0 à 3 sur 200 | idem |
| `autonomy` : arrêts exacts | 200/200 sur rubans v1 | ≥ 190/200 sur rubans v2 | idem |
| `needle` 4 ko | 0,35 | ≥ 0,90 | ≥ 0,90 |
| `needle` 16 ko | 0,00 | à mesurer (hors distribution ×4) | ≥ 0,80 |
| `needle` 64 ko | 0,05 | — | à mesurer (extrapolation ×2,5) |
| CLI sur `demo/conv.json`, question salle avec documents | STOP sur le serveur | documents lus, B12 | idem |
````

- [ ] **Step 5 : `notebooks/COLAB_EVAL.md`**

Dans la sous-section `relis.eval.autonomy`, ajouter après la description de la ligne principale :
```markdown
Deux lignes suivent depuis le plan 4 : `arrêts sur appât : N` compte les STOP tombés sur un
tour-appât (fait d'un autre sujet, mention du sujet sans valeur) — c'est la mesure directe du
raccourci « s'arrêter sur la première phrase en forme de fait » ; `par famille (arrêts exacts)`
donne, pour chacune des huit familles, le nombre d'arrêts exacts sur l'effectif, ce qui isole
`long_history` (des centaines de segments) des familles courtes. Comparer un checkpoint entraîné
sur les rubans v1 et un checkpoint v2 sur **les mêmes cas** : `iter_cases(777, 200)` tire depuis
le plan 4 des cas avec appâts ; le pas 500 y perdra des arrêts exacts, c'est la mesure « avant ».
```
Et dans la cellule 3, remplacer la commande `needle` par :
```python
!python -m relis.eval.needle --repo <utilisateur>/relis-v2-long --sizes 1000,4000,16000,64000,200000 --n 20
```
en précisant en dessous que 200 000 octets est au-delà de toute longueur d'entraînement (≈ 8 fois la
plus longue), donc mesure l'extrapolation de la récurrence, spec §12.

- [ ] **Step 6 : `docs/CARNET.md`**

Insérer avant « ### ⬜ Plan 2b » :
```markdown
### 🟠 Plan 4 — Négatifs et historiques longs

Deux défauts de la distribution d'entraînement, prouvés au pas 500 (voir plan 3) : le STOP suit la
forme d'un fait plutôt que la question, et CONT ne tient pas au-delà de 14 paires de tours. Le plan
n'ajoute rien au modèle : des tours-appâts dans les sept familles, une famille `long_history`, un
budget de ruban paramétrable, et deux sessions d'entraînement qui reprennent depuis le pas 500
(15 ko dans 16 384, puis 60 ko dans 65 536). Le rapport d'autonomie compte désormais les « arrêts sur
appât » et l'exactitude par famille.

**Chiffres réels : à venir.** Les critères de succès et le protocole sont dans `notebooks/KAGGLE.md`
(section « Plan 4 »). Aucun chiffre n'est inventé en attendant.
```
Et retirer de la liste du plan 2b les deux puces « Négatifs, en premier » et « Historiques longs, en même temps », désormais couvertes.

- [ ] **Step 7 : lancer toute la suite**

Run : `python -m pytest -q`
Expected : tous verts, sous 90 s.

- [ ] **Step 8 : commit**

```bash
git add configs/tape_long_t4.yaml tests/test_configs.py notebooks/KAGGLE.md notebooks/COLAB_EVAL.md docs/CARNET.md
git commit -m "plan 4: configuration 65 536, protocole des deux sessions, critères de succès, carnet"
```

---

## Exécution GPU (côté utilisateur, hors sous-agents)

1. Étape A : empaqueter `tapes_v2`, relire le mélange, téléverser, une session Kaggle (300 pas, `init_from` = pas 500).
2. Évaluations Colab sur `relis-v2-tape` : `distractor`, `autonomy`, `needle` (1 000 à 64 000), `adaptive`, CLI sur `demo/conv.json` avec les deux documents. Reporter dans le carnet.
3. `bench` avec `configs/tape_long_t4.yaml`. Si OOM à batch 1 : variante 32 768 décrite dans KAGGLE.md.
4. Étape B : empaqueter `tapes_long`, téléverser, une session Kaggle (150 pas, `init_from` = fin de A).
5. Évaluations Colab sur `relis-v2-long`, `needle` jusqu'à 200 000. Reporter dans le carnet, avec la mention explicite de ce qui est extrapolation.

## Auto-revue du plan

- **Couverture de la spec.** §6.2 « Négatifs » → Task 1 (appâts, indices conservés, exclusion des sujets croisés). §6.2 « Historiques longs » → Task 2 (famille, budget paramétrable) et Task 4 (deux étapes, 65 536). §9 autonomie « arrêts sur appât » → Task 3. §12 extrapolation → Task 4 (needle 200 000, formulation). §6.3 « le mélange doit être relu » → Global Constraints et Task 4.
- **Types.** `Case.negatives: list[int]` (Task 1) lu par `evaluate` via `getattr(case, "negatives", [])` (Task 3) ; `_negatives(rng, n_pairs, taken, label, exclude, k_max=3)` appelé avec `k_max=6` par `_long_history` (Task 2) ; `generate_case(rng, kind, max_tape_bytes)` positionnel dans `generate_episode`, nommé dans `iter_cases` ; `build_shards(..., max_tape_bytes)` nommé dans le test et la CLI.
- **Ce que le plan ne fait pas.** Il ne change pas l'oracle, le modèle, le contrôleur ni la boucle ; il ne promet aucun chiffre ; il ne traite pas les 200 ko en distribution (extrapolation mesurée, non entraînée) — si l'étape B échoue à extrapoler, la suite est l'entraînement par fenêtres successives avec Mémoire reportée (plan à part, touche `pack.py` et `loop.py`).
