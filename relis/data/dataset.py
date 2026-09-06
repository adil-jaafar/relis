import numpy as np
import torch


class ByteWindows:
    """Fenêtres aléatoires de seq_len+1 octets dans un fichier binaire (memmap)."""

    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError("fichier trop court pour seq_len")

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator)
        rows = np.stack([self.data[s:s + self.seq_len + 1] for s in starts.tolist()])
        t = torch.from_numpy(rows.astype(np.int64))
        return t[:, :-1], t[:, 1:]
