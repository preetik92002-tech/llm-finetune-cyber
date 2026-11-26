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
from collections import Counter
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


def load_rows(path, max_samples, seed):
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
    random.Random(seed).shuffle(rows)  # seeded, so the same examples are picked every time
    # max_samples: None (or missing) means use everything in the file.
    return rows if max_samples is None else rows[:max_samples]


def build_dataset(cfg):
    """Severity data, optionally mixed with ATT&CK data.

    Config keys:
      train_file          severity chat JSONL (required)
      max_samples         cap on severity examples (null = all)
      attack_file         ATT&CK chat JSONL (null or missing = severity only)
      max_attack_samples  cap on ATT&CK examples, so the two tasks stay balanced
                          (null = all; ATT&CK files can be much larger or smaller
                          than the severity file)
    """
    seed = cfg["seed"]
    severity = load_rows(ROOT / cfg["train_file"], cfg.get("max_samples"), seed)
    rows = list(severity)
    print(f"Severity examples: {len(severity)}")

    if cfg.get("attack_file"):
        attack = load_rows(ROOT / cfg["attack_file"], cfg.get("max_attack_samples"), seed)
        print(f"ATT&CK examples:   {len(attack)}")
        rows += attack
        # Mix the two tasks so every batch can contain both.
        random.Random(seed).shuffle(rows)

    print(f"Total training examples: {len(rows)}")
    return Dataset.from_list(rows)


def dtype_summary(model, title):
    """Print parameter counts by dtype, so a wrong dtype is visible BEFORE training.

    - trainable: parameters that get gradients (the LoRA adapters, and anything else
      left trainable). With fp16 mixed precision these must be float32.
    - not quantized: embeddings, norms, lm_head and similar. This is the dtype of the
      base model's non-quantized layers.
    - 4-bit packed: the quantized weights. bitsandbytes stores them as uint8.
    Returns the trainable counts by dtype name.
    """
    trainable, not_quantized, packed = Counter(), Counter(), 0
    for _, p in model.named_parameters():
        name = str(p.dtype).replace("torch.", "")
        if p.requires_grad:
            trainable[name] += p.numel()
        if p.dtype == torch.uint8:
            packed += p.numel()
        else:
            not_quantized[name] += p.numel()
    print(f"[dtypes] {title}")
    print(f"  trainable parameters by dtype:       {dict(trainable) or 'none'}")
    print(f"  not-quantized parameters by dtype:   {dict(not_quantized) or 'none'}")
    print(f"  4-bit packed weights (uint8):        {packed:,}")
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a YAML config")
    cfg = load_config(parser.parse_args().config)

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. 4-bit QLoRA needs one. Use Google Colab (GPU runtime).")

    # ---- Precision: choose it from the GPU's compute capability ----
    # bf16 needs hardware support, which starts at compute capability 8.0 (Ampere).
    # Older GPUs such as the Colab T4 (7.5) must use fp16.
    # Do NOT use torch.cuda.is_bf16_supported() for this: newer PyTorch versions
    # return True on older GPUs too, because bf16 can be emulated there. The
    # emulated path then fails inside some kernels. That caused the error "gemm
    # input type BFloat16 and output type float is only supported for CUDA devices
    # with compute capability 8.0 or higher" on a T4.
    major, minor = torch.cuda.get_device_capability()
    use_bf16 = major >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name()} (compute capability {major}.{minor}). "
          f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    output_dir = ROOT / cfg["output_dir"]

    # ---- Hub push needs a token; fail now, not after hours of training ----
    push_to_hub = bool(cfg.get("push_to_hub"))
    if push_to_hub and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise SystemExit("push_to_hub is true but HF_TOKEN is not set.")
    if push_to_hub and not cfg.get("hub_model_id"):
        raise SystemExit("push_to_hub is true but hub_model_id is missing (for example 'your-name/cyber-slm-qlora').")

    # ---- Data ----
    dataset = build_dataset(cfg)

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
        dtype=compute_dtype,  # dtype of the non-quantized weights; "torch_dtype" is the deprecated name
    )
    # On a GPU without bf16 the base model must be loaded as float16, never "auto":
    # "auto" would take the dtype saved in the checkpoint, which is bfloat16 for Qwen.
    print(f"Model config dtype: {getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)}")
    dtype_summary(model, "base model right after loading (non-quantized layers show the load dtype)")
    if not use_bf16 and any(p.dtype == torch.bfloat16 for p in model.parameters()):
        raise SystemExit("The base model has bfloat16 parameters although this GPU has no bf16 support.")

    # Gets a 4-bit model ready for training (freezes the base weights, may cast some
    # layers, enables gradient checkpointing hooks). The summaries below show the result.
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
        # Length of training: either a fixed number of steps (smoke test) or full
        # passes over the data. max_steps = -1 means "use epochs instead".
        max_steps=cfg.get("max_steps") or -1,
        num_train_epochs=cfg.get("num_train_epochs", 1),  # passes over the data (ignored if max_steps is set)
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
        # Loss: plain negative log-likelihood. Setting it explicitly keeps TRL off its
        # chunked log-prob kernel (trl/kernels/chunked_logprob.py), which needs bf16
        # matrix math that a T4 does not have.
        loss_type="nll",
        bf16=use_bf16,  # bf16 only on GPUs with compute capability 8 or higher (see above)
        fp16=not use_bf16,  # fp16 on older GPUs such as the Colab T4
        logging_steps=1,  # print the loss every step (the run is short)
        # Checkpoints: every save_steps steps if set, otherwise none (the adapter
        # is still saved at the end).
        save_strategy="steps" if cfg.get("save_steps") else "no",
        save_steps=cfg.get("save_steps") or 500,  # only used when save_strategy is "steps"
        save_total_limit=cfg.get("save_total_limit"),  # keep only the newest N checkpoints locally (null = all)
        # Hub upload: "every_save" pushes each checkpoint as it is saved, so a
        # crashed or timed-out Colab session loses little. Needs HF_TOKEN.
        push_to_hub=push_to_hub,
        hub_model_id=cfg.get("hub_model_id"),  # e.g. "your-name/cyber-slm-qlora"
        hub_strategy=cfg.get("hub_strategy", "every_save"),
        hub_private_repo=cfg.get("hub_private_repo", True),  # keep the repo private unless the config says otherwise
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

    # ---- Check and fix the dtype of the trainable parameters ----
    # The trainer has now created the PEFT model (base model + LoRA adapters).
    # With fp16 mixed precision, PyTorch's gradient scaler must unscale the gradients
    # before clipping them, and its kernel only works on float32 or float16 gradients
    # of float32 parameters. A trainable parameter left in bfloat16 fails there with
    # '"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for BFloat16'.
    # The usual rule for fp16 + LoRA is: keep the frozen base in fp16, train the
    # small adapters in float32.
    # TRL's model_init_kwargs only matters when the model is passed as a string. Here a
    # loaded model is passed, so it is unused. Check anyway that it holds no bfloat16.
    init_kwargs = getattr(args, "model_init_kwargs", None) or {}
    print(f"SFTConfig model_init_kwargs: {init_kwargs or 'not set (unused: the model is already loaded)'}")
    print(f"Trainer precision flags: fp16={args.fp16}, bf16={args.bf16}")
    if not use_bf16 and "bfloat16" in str(init_kwargs):
        raise SystemExit("model_init_kwargs asks for bfloat16 on a GPU without bf16 support.")

    dtype_summary(trainer.model, "BEFORE the float32 cast (after the PEFT model was created)")
    if not use_bf16:
        for p in trainer.model.parameters():
            if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.to(torch.float32)  # same Parameter object, so the optimizer will see it
        trainable = dtype_summary(trainer.model, "AFTER the float32 cast")
        # Stop here, not after minutes of training, if any trainable parameter is still 16-bit.
        assert "bfloat16" not in trainable, f"Trainable bfloat16 parameters remain: {dict(trainable)}"
        assert "float16" not in trainable, f"Trainable float16 parameters remain: {dict(trainable)}"

    train_result = trainer.train()

    # Save the loss history and run time so docs and figures can use them
    # (src/make_figures.py draws the loss curve from this file). On Colab, copy
    # results/train_<name>.json back to the repo.
    log_file = ROOT / "results" / f"train_{output_dir.name}.json"
    log_file.parent.mkdir(exist_ok=True)
    log_file.write_text(json.dumps({
        "config": cfg,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "log_history": trainer.state.log_history,  # loss, learning rate and step for each logged step
    }, indent=2), encoding="utf-8")

    # Saves only the adapter (a few MB), not the whole base model.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")
    if push_to_hub:
        trainer.push_to_hub()  # final adapter and tokenizer
        print(f"Pushed to the Hub: {cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
