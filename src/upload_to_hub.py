"""Upload the trained LoRA adapter and the model card to the Hugging Face Hub.

Checks first, so a half-finished upload does not happen:
  - HF_TOKEN is set
  - the adapter folder holds adapter_config.json and adapter weights
  - MODEL_CARD.md has no "TODO" left (override with --allow-todo)

The card is uploaded as the repo's README.md. Training checkpoints and optimizer
files are not uploaded.

Usage:
    python src/upload_to_hub.py --adapter outputs/qlora_main --repo-id your-name/cyber-slm-qlora
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
# Files that belong in the Hub repo. Everything else (checkpoints, optimizer state) is skipped.
IGNORE = ["checkpoint-*", "optimizer*", "scheduler*", "rng_state*", "trainer_state.json", "training_args.bin", "README.md"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/qlora_main", help="adapter folder (relative to the repo root)")
    p.add_argument("--repo-id", required=True, help="e.g. your-name/cyber-slm-qlora")
    p.add_argument("--card", default="MODEL_CARD.md")
    p.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    p.add_argument("--allow-todo", action="store_true", help="upload even if the card still has TODO markers")
    args = p.parse_args()

    # ---- Checks ----
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the HF_TOKEN environment variable (a Hugging Face token with write access).")

    adapter = ROOT / args.adapter
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter found in {adapter} (adapter_config.json is missing). Train first.")
    if not any((adapter / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin")):
        raise SystemExit(f"No adapter weights in {adapter}.")

    card = ROOT / args.card
    if not card.exists():
        raise SystemExit(f"{card} not found.")
    todos = card.read_text(encoding="utf-8").count("TODO")
    if todos and not args.allow_todo:
        raise SystemExit(f"{card.name} still has {todos} TODO markers. Fill them with real values, or pass --allow-todo.")

    # ---- Upload ----
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=not args.public, exist_ok=True)  # private unless --public
    api.upload_folder(folder_path=str(adapter), repo_id=args.repo_id, ignore_patterns=IGNORE)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=args.repo_id)
    print(f"Uploaded to https://huggingface.co/{args.repo_id} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
