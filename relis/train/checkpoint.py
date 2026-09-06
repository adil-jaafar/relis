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
        saved_cuda_count = len(rng["cuda"])
        current_device_count = torch.cuda.device_count()
        if saved_cuda_count == current_device_count:
            torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            print(f"[checkpoint] état RNG CUDA ignoré ({saved_cuda_count} états sauvegardés, {current_device_count} GPU visibles)")
    return {"step": payload["step"], "cfg": payload["cfg"], "extra": payload.get("extra", {})}


def push_to_hub(dir, repo_id):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=os.path.join(dir, "last.pt"), path_in_repo="last.pt", repo_id=repo_id)
    # last.pt pèse ~2 Go : sans squash, l'historique LFS épuise le quota en un jour.
    try:
        api.super_squash_history(repo_id=repo_id)
    except Exception as e:
        print(f"[hub] squash d'historique échoué : {e}")


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
