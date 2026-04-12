# Simple LLM Training (EC2 Ready)

This repository contains a single training script that trains a small GPT-like language model on WikiText-2.

## 1. Launch EC2

Use an Ubuntu GPU instance (recommended: g4dn.xlarge or better), with at least 30 GB EBS.

## 2. Clone and Setup

```bash
git clone <your-repo-url>
cd model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train

```bash
python untitled41.py --output-dir artifacts --epochs 5 --batch-size 16 --max-length 128
```

Optional W&B logging:

```bash
export WANDB_API_KEY=<your_key>
python untitled41.py --use-wandb --wandb-project simple-llm
```

## 4. Outputs

Artifacts are saved under `artifacts/`:
- `model_checkpoint_epoch_*.pt`
- `simple_llm_final.pt`
- tokenizer files in `artifacts/tokenizer/`

## Notes for EC2

- The script auto-selects CUDA when available.
- First run downloads WikiText-2 from Hugging Face.
- If your instance runs out of memory, reduce `--batch-size`.
