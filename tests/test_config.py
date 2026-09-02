from dataclasses import dataclass
import pytest
from rankfile.config import load_yaml, apply_overrides, from_dict, to_yaml

@dataclass
class Cfg:
    lr: float = 1e-3
    name: str = "x"
    compile: bool = True

def test_roundtrip_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    to_yaml(Cfg(lr=2e-3), p)
    d = load_yaml(p)
    assert d == {"lr": 2e-3, "name": "x", "compile": True}

def test_overrides_parse_yaml_scalars():
    d = apply_overrides({"lr": 1e-3, "compile": True}, ["lr=5e-4", "compile=false"])
    assert d["lr"] == 5e-4 and d["compile"] is False

def test_override_unknown_key_rejected():
    with pytest.raises(KeyError):
        apply_overrides({"lr": 1.0}, ["nope=1"])

def test_from_dict_rejects_unknown():
    assert from_dict(Cfg, {"lr": 0.5}).lr == 0.5
    with pytest.raises(KeyError):
        from_dict(Cfg, {"bogus": 1})
