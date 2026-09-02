# configs

YAML configs. A run is `model config + train (or finetune) config + seed`.
Every run directory stores the fully resolved config it ran with.

- `model/`     architecture: `m30.yaml` (smoke test only), `m124.yaml`
- `train/`     pretraining: `adamw.yaml`, `muon.yaml`, `sweep_adamw.yaml`, `sweep_muon.yaml`, `p3_matched.yaml`
- `finetune/`  fine-tuning methods (`full.yaml`, `lora_r{4,16,64}.yaml`) and `tasks/`

Rule: never edit a config that a completed run used. Copy it to a new file.
