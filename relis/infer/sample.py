"""Échantillonnage de texte depuis un checkpoint RELIS (pré-entraînement octets).

    python -m relis.infer.sample --repo jaafar2022/relis-v1-pretrain --prompt "La France est"
    python -m relis.infer.sample --ckpt runs/v1/last.pt --prompt "def fibonacci(n):" --header "src=code;lang=python"

Le modèle a été pré-entraîné sur des documents de la forme `SEG en-tête HDR contenu` ;
le prompt est donc préfixé de la même façon (§6.1). Les codes de contrôle sont masqués
à la génération pour ne produire que du texte. Le décodage UTF-8 contraint (spec §7.2)
arrive avec le contrôleur du plan 3 ; ici les séquences invalides sont remplacées.
"""
import argparse
import os
import sys

import torch

from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape import codes as C

_CONTROL_MASK = torch.zeros(256, dtype=torch.bool)
for _b in C.CONTROL_CODES:
    _CONTROL_MASK[_b] = True


def build_prompt(text: str, header: str = "src=wiki_fr") -> bytes:
    """Préfixe le texte comme un document d'entraînement : SEG + en-tête + HDR + texte."""
    return bytes([C.SEG]) + header.encode("utf-8") + bytes([C.HDR]) + C.sanitize(text.encode("utf-8"))


def load_model(ckpt_path: str, device: str = "cpu"):
    """Charge `last.pt` (local) et reconstruit le modèle depuis la config sauvegardée."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = RelisConfig(**payload["cfg"])
    model = RelisModel(cfg)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload["step"]


def load_from_hub(repo_id: str, device: str = "cpu", local_dir: str = "hub_ckpt"):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, "last.pt", token=os.environ.get("HF_TOKEN"), local_dir=local_dir)
    return load_model(path, device)


@torch.no_grad()
def generate(model, prompt: bytes, max_new: int = 200, temperature: float = 0.8, top_k: int = 40,
             seed: int | None = None, device: str = "cpu", mode: int = int(C.Mode.SCAN)) -> bytes:
    """Renvoie prompt + max_new octets générés, sans codes de contrôle."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    state = model.new_state(1, device)
    x = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    logits, state = model(x, mode, state)               # lecture du prompt en mode bloc
    last = logits[0, -1]
    out = bytearray(prompt)
    mask = _CONTROL_MASK.to(device)
    for _ in range(max_new):
        lg = last.float().masked_fill(mask, float("-inf"))
        if temperature <= 0:
            nxt = int(lg.argmax())
        else:
            lg = lg / temperature
            if top_k and top_k > 0:
                kth = torch.topk(lg, min(top_k, lg.numel())).values[-1]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        out.append(nxt)
        last, state = model.step(torch.tensor([nxt], device=device), mode, state)
        last = last[0]
    return bytes(out)


def main():
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="dépôt Hugging Face contenant last.pt")
    src.add_argument("--ckpt", help="chemin local vers last.pt")
    ap.add_argument("--prompt", default="La France est")
    ap.add_argument("--header", default="src=wiki_fr", help="en-tête de document, ex. src=code;lang=python")
    ap.add_argument("--max_new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.repo:
        model, step = load_from_hub(args.repo, args.device)
    else:
        model, step = load_model(args.ckpt, args.device)
    print(f"[sample] checkpoint au pas {step} ; {sum(p.numel() for p in model.parameters())/1e6:.1f} M paramètres")
    prompt = build_prompt(args.prompt, args.header)
    out = generate(model, prompt, args.max_new, args.temperature, args.top_k, args.seed, args.device)
    text = out[len(prompt):].decode("utf-8", errors="replace")
    print("-" * 60)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
