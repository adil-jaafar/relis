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
