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
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=amp and device.startswith("cuda"))

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


def _measure_with_fallback(model, seq_len, batch_size, device, amp):
    """Mesure le débit en divisant le batch par deux à chaque OOM (jusqu'à 1)."""
    bs = batch_size
    while True:
        try:
            print(f"[bench] essai batch_size={bs}")
            return throughput(model, seq_len, bs, device, amp), bs
        except torch.cuda.OutOfMemoryError:
            print(f"[bench] OOM à batch_size={bs}")
            model.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_gb", type=float, default=2.0)
    ap.add_argument("--compile", action="store_true",
                    help="mesurer avec torch.compile sur le cœur GDN (même effet que train.compile=true)")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mcfg = RelisConfig(**raw["model"]); t = raw["train"]
    if args.compile:
        from relis.train.loop import enable_compile
        enable_compile()
    model = RelisModel(mcfg)
    asked_bs, asked_accum = t["batch_size"], t["grad_accum"]
    bps, bs = _measure_with_fallback(model, t["seq_len"], asked_bs, args.device, t["amp"])
    if args.compile:
        print("[bench] mesure avec torch.compile ; la chauffe inclut la compilation, "
              "le débit affiché est donc prudent")
    accum = max(1, asked_bs * asked_accum // bs)          # batch effectif conservé
    per_step = bs * accum * t["seq_len"]
    steps_needed = args.target_gb * 1e9 / per_step
    print(f"débit : {bps:,.0f} octets/s par GPU (batch_size={bs})")
    print(f"octets par pas (1 GPU) : {per_step:,} ; pas pour {args.target_gb} Go : {steps_needed:,.0f}")
    print(f"heures pour {args.target_gb} Go sur 1 GPU : {args.target_gb*1e9/bps/3600:.1f}")
    if args.device.startswith("cuda"):
        print(f"mémoire GPU max : {torch.cuda.max_memory_allocated()/1e9:.2f} Go")
    if bs != asked_bs:
        print(f"[bench] batch réduit {asked_bs} -> {bs} ; recopier cette option au lancement :")
        print(f"    --override train.batch_size={bs} train.grad_accum={accum}")


if __name__ == "__main__":
    main()
