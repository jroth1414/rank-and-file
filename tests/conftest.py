import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

@pytest.fixture
def tiny_cfg():
    from rankfile.model import ModelConfig
    return ModelConfig(vocab_size=512, n_layer=2, d_model=64, n_head=4, n_kv_head=2,
                       head_dim=16, d_ff=128, max_seq_len=64)
