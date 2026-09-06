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
