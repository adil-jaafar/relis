"""Rembourrage et empaquetage des rubans (spec §6.1 étape 2).

Quatre tableaux parallèles par position : octet, mode, classe de poids, drapeaux
(bit 0 : réinitialisation Mémoire, bit 1 : réinitialisation slots). Tout début de
ruban tombe sur un multiple de `block` et porte les deux bits.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from relis.tape import codes as C
from relis.tape.tape import Tape, WEIGHTS, W_LOW, W_ONE, W_HIGH, build_tape
from .episodes import iter_episodes

PAD = 0x00
REF_SEQS_PER_STEP = 32   # 2 GPU × batch 2 × accum 8 (configs/tape_t4.yaml)
REF_MAX_STEPS = 1500
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
        stats.setdefault("tape_bytes", 0)
    cur = []; used = 0

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
            stats["packed"] += 1; stats["tape_bytes"] += len(t)
    if cur:
        yield flush()


def replay_tape(bin_path: str, seq_len: int, rng: random.Random, wclass: int = W_LOW) -> Tape:
    """Fenêtre brute du corpus de pré-entraînement, en ruban-document (mode SCAN).

    Le poids par défaut est `W_LOW` (0,1), celui du texte parcouru dans la spec :
    au poids 1 le rappel écraserait l'objectif (16 384 positions à 1,0 contre
    ~3 900 pour un ruban empaqueté).
    """
    data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    if len(data) < seq_len + 1:
        raise ValueError("fichier de rappel trop court pour seq_len")
    start = rng.randint(0, len(data) - seq_len)
    t = Tape()
    for b in data[start:start + seq_len].tobytes():
        t.put(b, int(C.Mode.SCAN), wclass)
    return t


def implied_epochs(n_seqs: int, batch_per_step: int, max_steps: int) -> float:
    """Nombre de passages sur le jeu de rubans pour `max_steps` pas de `batch_per_step` séquences."""
    return float(batch_per_step) * float(max_steps) / max(1.0, float(n_seqs))


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


class _Mix:
    """Somme des poids (WEIGHTS) par classe sur les positions écrites ; rappel compté à part."""

    def __init__(self):
        self.counts = np.zeros(len(WEIGHTS), dtype=np.int64)
        self.sum_w_replay = 0.0
        self.tape_seqs = 0
        self.replay_seqs = 0

    def add_tape_seq(self, wclass: np.ndarray) -> None:
        self.counts += np.bincount(np.asarray(wclass, dtype=np.int64), minlength=len(WEIGHTS))
        self.tape_seqs += 1

    def add_replay_seq(self, n: int, wclass: int) -> None:
        self.sum_w_replay += float(n) * WEIGHTS[wclass]
        self.replay_seqs += 1

    def report(self, prefix: str, tape_bytes: int, packed: int) -> dict:
        sw = {"decisions": float(self.counts[W_HIGH]) * WEIGHTS[W_HIGH],
              "answer": float(self.counts[W_ONE]) * WEIGHTS[W_ONE],
              "low": float(self.counts[W_LOW]) * WEIGHTS[W_LOW],
              "replay": self.sum_w_replay}
        total = max(1e-9, sum(sw.values()))
        out = {f"{prefix}tape_seqs": self.tape_seqs, f"{prefix}replay_seqs": self.replay_seqs,
               f"{prefix}mean_tape_bytes": (tape_bytes / packed) if packed else 0.0}
        for k, v in sw.items():
            out[f"{prefix}sum_w_{k}"] = v
            out[f"{prefix}share_{k}"] = v / total
        return out


def _print_mix(name: str, r: dict, prefix: str, seqs_per_step: int, max_steps: int) -> None:
    n = r[f"{prefix}tape_seqs"] + r[f"{prefix}replay_seqs"]
    print(f"[{name}] séquences {n} (rubans {r[f'{prefix}tape_seqs']}, "
          f"rappel {r[f'{prefix}replay_seqs']}) ; ruban moyen "
          f"{r[f'{prefix}mean_tape_bytes']:.0f} octets")
    print(f"[{name}] somme des poids : décisions {r[f'{prefix}sum_w_decisions']:.0f} "
          f"réponse {r[f'{prefix}sum_w_answer']:.0f} "
          f"texte {r[f'{prefix}sum_w_low']:.0f} "
          f"rappel {r[f'{prefix}sum_w_replay']:.0f}")
    print(f"[{name}] parts décisions {r[f'{prefix}share_decisions']:.3f} "
          f"réponse {r[f'{prefix}share_answer']:.3f} "
          f"texte {r[f'{prefix}share_low']:.3f} "
          f"rappel {r[f'{prefix}share_replay']:.3f}")
    print(f"[{name}] epochs_for({seqs_per_step} séq/pas × {max_steps} pas) = "
          f"{implied_epochs(n, seqs_per_step, max_steps):.2f}")


def build_shards(out: str, episodes: int, seed: int, seq_len: int, block: int,
                 replay: str | None, replay_frac: float, val_frac: float, extra_tapes=None,
                 replay_wclass: int = W_LOW, seqs_per_step: int = REF_SEQS_PER_STEP,
                 max_steps: int = REF_MAX_STEPS) -> dict:
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
        mix = _Mix()
        n = 0
        for s in pack_tapes(ts, seq_len, block, st):
            wr.add(s); mix.add_tape_seq(s["wclass"]); n += 1
        if replay and replay_frac > 0 and n > 0:
            n_replay = int(round(n * replay_frac / max(1e-9, 1 - replay_frac)))
            for _ in range(n_replay):
                t = replay_tape(replay, seq_len, rng, replay_wclass)
                d, m, w, f = _to_arrays(t)
                wr.add({"data": d, "mode": m, "wclass": w, "flags": f, "n_tapes": 1})
                mix.add_replay_seq(len(t), replay_wclass); n += 1
        wr.close()
        prefix = f"{name}_"
        stats[f"{name}_seqs"] = n
        stats[f"{name}_skipped"] = st.get("skipped", 0)
        r = mix.report(prefix, st.get("tape_bytes", 0), st.get("packed", 0))
        stats.update(r)
        _print_mix(name, r, prefix, seqs_per_step, max_steps)
    return stats


def main():
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq_len", type=int, default=16384)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--replay", default=None, help="shard de pré-entraînement pour le rappel")
    ap.add_argument("--replay_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--replay_wclass", type=int, default=W_LOW, choices=(1, 2, 3),
                    help="classe de poids du rappel : 1=0,1 2=1,0 3=5,0")
    ap.add_argument("--seqs_per_step", type=int, default=REF_SEQS_PER_STEP,
                    help="séquences par pas d'optimisation, pour la ligne epochs_for")
    args = ap.parse_args()
    stats = build_shards(args.out, args.episodes, args.seed, args.seq_len, args.block,
                         args.replay, args.replay_frac, args.val_frac,
                         replay_wclass=args.replay_wclass, seqs_per_step=args.seqs_per_step)
    print(stats)


if __name__ == "__main__":
    main()
