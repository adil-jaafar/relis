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
