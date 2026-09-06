import torch
import pytest


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    yield
