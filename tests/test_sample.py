import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape import codes as C
from relis.train.checkpoint import save_checkpoint
from relis.infer.sample import generate, load_model, build_prompt


def test_build_prompt_mimics_training_layout():
    p = build_prompt("Bonjour", header="src=wiki_fr")
    assert p[0] == C.SEG and p[1:12] == b"src=wiki_fr" and p[12] == C.HDR
    assert p.endswith(b"Bonjour")


def test_generate_returns_text_bytes_without_control_codes():
    m = RelisModel(RelisConfig.tiny())
    prompt = build_prompt("le chat", header="src=t")
    out = generate(m, prompt, max_new=50, temperature=1.0, top_k=0, seed=0, device="cpu")
    assert out[:len(prompt)] == prompt
    assert len(out) == len(prompt) + 50
    assert all(b not in C.CONTROL_CODES for b in out[len(prompt):])


def test_generate_is_deterministic_with_seed():
    m = RelisModel(RelisConfig.tiny())
    prompt = build_prompt("abc", header="src=t")
    a = generate(m, prompt, max_new=20, temperature=0.8, top_k=10, seed=1, device="cpu")
    b = generate(m, prompt, max_new=20, temperature=0.8, top_k=10, seed=1, device="cpu")
    assert a == b


def test_load_model_from_checkpoint_dir(tmp_path):
    cfg = RelisConfig.tiny()
    m = RelisModel(cfg)
    save_checkpoint(str(tmp_path), m, None, None, step=3, cfg_dict=cfg.to_dict(), extra={})
    m2, step = load_model(str(tmp_path / "last.pt"), device="cpu")
    assert step == 3
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
