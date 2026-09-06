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
