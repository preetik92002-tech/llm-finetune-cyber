"""Fine-tune a small chat model on the severity dataset with 4-bit QLoRA.

Reads a YAML config, loads chat-format JSONL (system / user / assistant messages),
trains with TRL's SFTTrainer + PEFT (LoRA), and saves only the adapter.
Loss is computed on the assistant answer only, not on the system/user text.

Needs a CUDA GPU (bitsandbytes 4-bit does not run on CPU). Use Colab for the
smoke test: notebooks/run_on_colab.ipynb.

Usage:
    python src/train.py --config configs/smoke_test.yaml
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_examples(path, max_samples, seed):
    """Read chat JSONL and turn each example into prompt + completion.

    prompt     = the system and user messages
    completion = the assistant message
    With this prompt/completion layout, SFTTrainer can compute the loss on the
    completion only. (assistant_only_loss would need a chat template with special
    generation markers, which Qwen's template does not have.)
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            if messages[-1]["role"] != "assistant":
                raise ValueError("Last message of every example must be the assistant answer.")
            rows.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    if not rows:
        raise SystemExit(
            f"{path} has no examples. Run stages 02 to 04 first (the severity files are built from the NVD data)."
        )
    random.Random(seed).shuffle(rows)  # seeded, so the same 50 are picked every time
    return Dataset.from_list(rows[:max_samples])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a YAML config")
    cfg = load_config(parser.parse_args().config)

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. 4-bit QLoRA needs one. Use Google Colab (GPU runtime).")

    # bf16 needs a newer GPU (Ampere or later). Older ones (Colab T4) use fp16.
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    output_dir = ROOT / cfg["output_dir"]
    train_file = ROOT / cfg["train_file"]

    # ---- Data ----
    dataset = load_examples(train_file, cfg["max_samples"], cfg["seed"])
    print(f"Training examples: {len(dataset)}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:  # batches need a padding token
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model in 4-bit (the "Q" in QLoRA) ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # store the frozen base weights in 4 bits
        bnb_4bit_quant_type="nf4",  # NormalFloat4: a 4-bit format suited to model weights
        bnb_4bit_use_double_quant=True,  # also compress the scaling constants (saves a little more memory)
        bnb_4bit_compute_dtype=compute_dtype,  # math is done in 16-bit, only storage is 4-bit
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map={"": 0},  # put the whole model on GPU 0
        torch_dtype=compute_dtype,
    )
    # Gets a 4-bit model ready for training (casts some layers, enables gradient checkpointing hooks).
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA: small trainable adapters on top of the frozen model ----
    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],  # adapter rank (size)
        lora_alpha=lora["alpha"],  # adapter scaling
        lora_dropout=lora["dropout"],  # randomly drops adapter outputs while training to reduce overfitting
        target_modules=lora["target_modules"],  # which layers get adapters
        bias="none",  # do not train bias terms
        task_type="CAUSAL_LM",  # next-token prediction
    )

    # ---- Weights & Biases only if a key is present ----
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "cyber-slm"))
    print(f"Weights & Biases logging: {'on' if use_wandb else 'off (no WANDB_API_KEY)'}")

    # ---- Training arguments ----
    args = SFTConfig(
        output_dir=str(output_dir),  # where checkpoints/logs go
        max_steps=cfg["max_steps"],  # total optimizer steps; overrides number of epochs
        per_device_train_batch_size=cfg["per_device_train_batch_size"],  # examples per forward pass
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],  # add up gradients over this many passes: effective batch = batch x this
        learning_rate=cfg["learning_rate"],  # step size of weight updates
        lr_scheduler_type="cosine",  # learning rate falls smoothly toward 0
        warmup_steps=cfg["warmup_steps"],  # start with a tiny learning rate for this many steps
        optim="paged_adamw_8bit",  # memory-light optimizer from bitsandbytes
        max_length=cfg["max_length"],  # longer examples are cut to this many tokens
        completion_only_loss=True,  # loss only on the assistant answer (the completion)
        packing=False,  # one example per sequence (packing would blur the loss mask)
        gradient_checkpointing=True,  # recompute activations to save GPU memory (slower)
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the recommended mode
        bf16=use_bf16,  # 16-bit format for newer GPUs
        fp16=not use_bf16,  # 16-bit format for older GPUs such as the Colab T4
        logging_steps=1,  # print the loss every step (the run is short)
        save_strategy="no",  # no mid-run checkpoints; we save the adapter at the end
        seed=cfg["seed"],  # fixes weight init and shuffling
        data_seed=cfg["seed"],  # fixes the data order
        report_to="wandb" if use_wandb else "none",  # log to W&B only with a key
        run_name=output_dir.name,  # name shown in W&B
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,  # applies the chat template and tokenizes
        peft_config=peft_config,  # wraps the model with LoRA adapters
    )
    trainer.train()

    # Saves only the adapter (a few MB), not the whole base model.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
