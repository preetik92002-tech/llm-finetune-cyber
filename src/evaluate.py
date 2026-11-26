"""Evaluate a base model, optionally with a LoRA adapter, on the test sets.

Modes
  zero-shot   base model, just the test prompt
  few-shot    base model, plus k worked examples from the TRAIN files in the prompt
  fine-tuned  base model + LoRA adapter (--adapter is required)

Metrics
  severity : accuracy, macro-F1, confusion matrix, unparseable count.
             Also reported separately for samples with cvss_source = nvd and = cna
             (results JSON key "severity_by_source", with sample counts).
  ATT&CK   : exact-match accuracy on the technique id, parent-technique accuracy
             (T1566 for T1566.001), unparseable count, ids that do not exist
  halluc.  : how often the model gives a severity for CVE ids that do not exist

Outputs (in results/)
  <name>.json                       all metrics and the run settings
  <name>_predictions.jsonl          gold answer, parsed prediction and raw output per example
  <name>_severity_confusion.png     confusion matrix plot
  <name>_attack_accuracy.png        ATT&CK accuracy bars

Examples
  python src/evaluate.py --mode zero-shot --model Qwen/Qwen2.5-1.5B-Instruct
  python src/evaluate.py --mode few-shot  --model Qwen/Qwen2.5-1.5B-Instruct --few-shot-k 4
  python src/evaluate.py --mode fine-tuned --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/qlora_main

Needs a CUDA GPU for reasonable speed (use Colab). It also runs on CPU, very slowly.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEVERITY_LABELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
UNPARSEABLE = "UNPARSEABLE"  # extra "predicted" label for outputs we cannot read
MAX_PROMPT_TOKENS = 2048  # prompts longer than this are cut from the left

# Used for the hallucination check if no severity test file is available.
DEFAULT_SEVERITY_SYSTEM = (
    "You are a cybersecurity analyst. Given a CVE description, "
    "classify its CVSS v3 base severity as one of: CRITICAL, HIGH, MEDIUM, LOW."
)

# Word match, case-insensitive, so "Severity: **High**." still parses.
SEVERITY_RE = re.compile(r"\b(CRITICAL|HIGH|MEDIUM|LOW)\b", re.IGNORECASE)
# ATT&CK id: T + 4 digits, optionally .NNN for a sub-technique.
TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
# Words that show the model declined or said it did not know.
REFUSAL_RE = re.compile(
    r"\b(unknown|not (?:a )?(?:valid|real|found|available|exist)|does not exist|doesn't exist|"
    r"cannot|can't|unable|no information|not aware|don't know|do not know|invalid)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ data
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_test_rows(path, max_samples, seed):
    """Test file -> list of {"prompt": [system, user], "gold": answer text}."""
    # "meta" is a separate field next to "messages" (written by build_cve_dataset.py).
    # It holds the cvss_source ("nvd" or "cna"). Older test files do not have it,
    # so the source is then None.
    rows = [{"prompt": r["messages"][:-1],
             "gold": r["messages"][-1]["content"].strip(),
             "source": (r.get("meta") or {}).get("cvss_source")} for r in read_jsonl(path)]
    if not rows:
        raise SystemExit(f"{path} has no examples.")
    if max_samples and len(rows) > max_samples:
        random.Random(seed).shuffle(rows)  # seeded random subset, not just the first N
        rows = rows[:max_samples]
    return rows


def pick_shots(train_path, k, seed, balance_by_label):
    """Choose k worked examples (user + assistant messages) from the train file.

    Severity: spread over the four classes. ATT&CK: random.
    """
    rows = read_jsonl(train_path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if balance_by_label:
        groups = defaultdict(list)
        for r in rows:
            groups[r["messages"][-1]["content"].strip()].append(r)
        chosen, queues = [], [groups[l] for l in sorted(groups)]
        while len(chosen) < k and any(queues):  # take one from each class in turn
            for q in queues:
                if q and len(chosen) < k:
                    chosen.append(q.pop())
        rng.shuffle(chosen)
    else:
        chosen = rows[:k]
    shots = []
    for r in chosen:
        shots += [r["messages"][-2], r["messages"][-1]]  # user, assistant
    return shots


def build_messages(prompt, shots):
    """System message, then the worked examples (few-shot only), then the real question."""
    system = [m for m in prompt if m["role"] == "system"]
    user = [m for m in prompt if m["role"] != "system"]
    return system + shots + user


# ------------------------------------------------------------------ model
def load_model(args):
    """Load the base model (4-bit by default on GPU) and attach the adapter if given."""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation needs padding on the left
    tokenizer.truncation_side = "left"  # if a prompt is too long, keep its end

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: no CUDA GPU found. Evaluation will be very slow.")
    kwargs = {}
    if use_cuda and not args.no_4bit:
        # bf16 only on compute capability 8.0 or higher. is_bf16_supported() can say True
        # on a T4 (7.5) because bf16 is emulated there, and that path fails in some
        # kernels. Same rule as src/train.py.
        major, minor = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"GPU compute capability {major}.{minor}. Precision: {'bf16' if major >= 8 else 'fp16'}")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kwargs["dtype"] = dtype
        kwargs["device_map"] = {"": 0}
    elif use_cuda:
        kwargs["dtype"] = torch.float16  # "torch_dtype" is the deprecated name
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    if args.mode == "fine-tuned":
        model = PeftModel.from_pretrained(model, args.adapter)  # adds the LoRA weights
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, message_lists, batch_size, max_new_tokens):
    """Greedy generation for a list of chat prompts; returns only the new text."""
    # Turn each chat into one prompt string using the model's own chat template.
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_PROMPT_TOKENS).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[:, enc["input_ids"].shape[1]:]  # drop the prompt part
        outputs += [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outputs


# ------------------------------------------------------------------ parsing
def parse_severity(text):
    """First severity word in the output, upper-cased; None if there is none."""
    m = SEVERITY_RE.search(text)
    return m.group(1).upper() if m else None


def parse_technique(text):
    """First ATT&CK id in the output, upper-cased; None if there is none."""
    m = TECHNIQUE_RE.search(text)
    return m.group(0).upper() if m else None


def parent_of(technique_id):
    """T1566.001 -> T1566."""
    return technique_id.split(".")[0]


# ------------------------------------------------------------------ metrics
def severity_by_source(golds, preds, sources):
    """The same severity metrics three times: all samples, only NVD labels, only CNA labels.

    Why: the label source shifts between train and test (train is mostly NVD scores,
    the natural test file mostly CNA scores). A single number would mix the two. Each
    part has its own sample count "n" so small groups are visible.
    """
    def pick(source):
        idx = [i for i, s in enumerate(sources) if source is None or s == source]
        return severity_metrics([golds[i] for i in idx], [preds[i] for i in idx])

    out = {"all": pick(None), "nvd": pick("nvd"), "cna": pick("cna")}
    known = out["nvd"]["n"] + out["cna"]["n"]
    out["samples_without_source"] = len(golds) - known  # rows with no cvss_source in the test file
    if known == 0:
        out["note"] = ("The test file has no cvss_source metadata, so the NVD and CNA parts are empty. "
                       "Run src/build_cve_dataset.py again to add it.")
    return out


def severity_metrics(golds, preds):
    """Accuracy, macro-F1 and confusion matrix. Unparseable outputs count as wrong."""
    n = len(golds)
    if n == 0:  # an empty group (for example no CNA samples): report that instead of dividing by zero
        return {"n": 0, "accuracy": None, "accuracy_on_parseable": None, "macro_f1": None,
                "unparseable": 0, "unparseable_rate": None, "per_class": {}, "confusion_matrix": None}
    columns = SEVERITY_LABELS + [UNPARSEABLE]
    matrix = {g: {c: 0 for c in columns} for g in SEVERITY_LABELS}
    for g, p in zip(golds, preds):
        matrix[g][p if p in SEVERITY_LABELS else UNPARSEABLE] += 1

    correct = sum(matrix[l][l] for l in SEVERITY_LABELS)
    unparseable = sum(matrix[l][UNPARSEABLE] for l in SEVERITY_LABELS)

    f1s, per_class = [], {}
    for l in SEVERITY_LABELS:
        support = sum(matrix[l].values())  # how many gold examples have this label
        predicted = sum(matrix[g][l] for g in SEVERITY_LABELS)  # how often the model said it
        tp = matrix[l][l]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[l] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:  # a class with no gold examples is left out of the macro average
            f1s.append(f1)

    parseable = n - unparseable
    return {
        "n": n,
        "accuracy": correct / n,  # over all examples; unparseable counts as wrong
        "accuracy_on_parseable": correct / parseable if parseable else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "unparseable": unparseable,
        "unparseable_rate": unparseable / n,
        "per_class": per_class,
        "confusion_matrix": {"rows_gold": SEVERITY_LABELS, "columns_predicted": columns,
                             "counts": [[matrix[g][c] for c in columns] for g in SEVERITY_LABELS]},
    }


def attack_metrics(golds, preds, known_ids):
    """Exact and parent-technique accuracy. Unparseable outputs count as wrong."""
    n = len(golds)
    exact = parent = unparseable = invalid_id = 0
    for g, p in zip(golds, preds):
        if p is None:
            unparseable += 1
            continue
        exact += p == g
        parent += parent_of(p) == parent_of(g)  # T1566 vs T1566.001 counts here
        invalid_id += p not in known_ids  # parsed, but no such technique in our data
    return {
        "n": n,
        "exact_match_accuracy": exact / n,
        "parent_accuracy": parent / n,
        "unparseable": unparseable,
        "unparseable_rate": unparseable / n,
        # Only ids found in attack_{train,val,test}.jsonl count as known, so this can
        # over-count real techniques that never appear in our data.
        "ids_not_in_our_data": invalid_id,
    }


# --------------------------------------------------------------- hallucination
def fake_cve_ids(n, seed):
    """Made-up CVE ids with years 2090 to 2099, which cannot exist yet."""
    rng = random.Random(seed)
    ids = set()
    while len(ids) < n:
        ids.add(f"CVE-{rng.randint(2090, 2099)}-{rng.randint(10000, 99999)}")
    return sorted(ids)


def hallucination_check(model, tokenizer, system_prompt, n, seed, batch_size, max_new_tokens):
    """Ask for the severity of CVEs that do not exist.

    A good answer says it does not know. Giving a severity label is a
    hallucination: the model invented a fact about a CVE it has never seen.
    """
    ids = fake_cve_ids(n, seed)
    chats = [[{"role": "system", "content": system_prompt},
              {"role": "user", "content": f"What is the CVSS base severity of {cve}?"}] for cve in ids]
    outputs = generate(model, tokenizer, chats, batch_size, max_new_tokens)
    gave_label = [parse_severity(o) is not None for o in outputs]
    refused = [REFUSAL_RE.search(o) is not None for o in outputs]
    return {
        "n": n,
        "gave_a_severity": sum(gave_label),
        "hallucination_rate": sum(gave_label) / n,  # lower is better
        "looked_like_refusal": sum(refused),
        "examples": [{"cve": c, "output": o} for c, o in list(zip(ids, outputs))[:10]],
    }


# ------------------------------------------------------------------ plots
def plot_confusion(metrics, path, title):
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    cm = metrics["confusion_matrix"]
    counts = cm["counts"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(cm["columns_predicted"])), cm["columns_predicted"], rotation=30, ha="right")
    ax.set_yticks(range(len(cm["rows_gold"])), cm["rows_gold"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    top = max(max(r) for r in counts) or 1
    for i, row in enumerate(counts):  # write the number in each cell
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > top / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attack(metrics, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Exact match", "Parent technique", "Unparseable"]
    values = [metrics["exact_match_accuracy"], metrics["parent_accuracy"], metrics["unparseable_rate"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of test examples")
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="base model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", required=True, choices=["zero-shot", "few-shot", "fine-tuned"])
    p.add_argument("--adapter", help="path or Hub id of the LoRA adapter (fine-tuned mode)")
    p.add_argument("--task", default="both", choices=["severity", "attack", "both"])
    p.add_argument("--severity-test", default=str(PROCESSED / "severity_test.jsonl"))
    p.add_argument("--attack-test", default=str(PROCESSED / "attack_test.jsonl"))
    p.add_argument("--severity-train", default=str(PROCESSED / "severity_train.jsonl"), help="source of few-shot examples")
    p.add_argument("--attack-train", default=str(PROCESSED / "attack_train.jsonl"), help="source of few-shot examples")
    p.add_argument("--few-shot-k", type=int, default=4, help="worked examples per prompt (few-shot mode)")
    p.add_argument("--max-samples", type=int, default=None, help="limit test examples per task (quick runs)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--hallucination-n", type=int, default=50, help="made-up CVE ids to ask about (0 = skip)")
    p.add_argument("--no-4bit", action="store_true", help="load in 16-bit instead of 4-bit")
    p.add_argument("--name", help="output name; default is <mode>_<model>")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ---- Check the mode and its inputs before loading anything big ----
    if args.mode == "fine-tuned" and not args.adapter:
        raise SystemExit("--mode fine-tuned needs --adapter.")
    if args.mode != "fine-tuned" and args.adapter:
        raise SystemExit("--adapter is only used with --mode fine-tuned.")
    name = args.name or f"{args.mode}_{args.model.split('/')[-1]}"
    RESULTS.mkdir(exist_ok=True)

    # ---- Load the model ----
    model, tokenizer = load_model(args)
    result = {"run": {k: v for k, v in vars(args).items()}}
    predictions = []

    # ---- Severity ----
    if args.task in ("severity", "both"):
        print("Severity test set")
        rows = load_test_rows(args.severity_test, args.max_samples, args.seed)
        shots = pick_shots(args.severity_train, args.few_shot_k, args.seed, True) if args.mode == "few-shot" else []
        outputs = generate(model, tokenizer, [build_messages(r["prompt"], shots) for r in rows],
                           args.batch_size, args.max_new_tokens)
        golds = [r["gold"].upper() for r in rows]
        preds = [parse_severity(o) for o in outputs]
        sources = [r["source"] for r in rows]  # "nvd", "cna" or None
        # "severity" keeps the all-samples metrics (other scripts read it).
        # "severity_by_source" holds the same metrics for all, nvd and cna.
        result["severity"] = severity_metrics(golds, preds)
        result["severity_by_source"] = severity_by_source(golds, preds, sources)
        plot_confusion(result["severity"], RESULTS / f"{name}_severity_confusion.png", f"Severity: {name}")
        predictions += [{"task": "severity", "cvss_source": src, "gold": g, "pred": pr, "raw": o}
                        for g, pr, o, src in zip(golds, preds, outputs, sources)]

    # ---- ATT&CK ----
    if args.task in ("attack", "both"):
        print("ATT&CK test set")
        rows = load_test_rows(args.attack_test, args.max_samples, args.seed)
        shots = pick_shots(args.attack_train, args.few_shot_k, args.seed, False) if args.mode == "few-shot" else []
        outputs = generate(model, tokenizer, [build_messages(r["prompt"], shots) for r in rows],
                           args.batch_size, args.max_new_tokens)
        golds = [parse_technique(r["gold"]) for r in rows]  # gold is "T1566.001 Name"; keep the id
        preds = [parse_technique(o) for o in outputs]
        # Technique ids that appear anywhere in our ATT&CK files count as "real".
        known = {parse_technique(r["messages"][-1]["content"])
                 for f in PROCESSED.glob("attack_*.jsonl") for r in read_jsonl(f)}
        result["attack"] = attack_metrics(golds, preds, known)
        plot_attack(result["attack"], RESULTS / f"{name}_attack_accuracy.png", f"ATT&CK: {name}")
        predictions += [{"task": "attack", "gold": g, "pred": pr, "raw": o} for g, pr, o in zip(golds, preds, outputs)]

    # ---- Hallucination check (made-up CVE ids) ----
    if args.hallucination_n > 0:
        print("Hallucination check")
        system = DEFAULT_SEVERITY_SYSTEM
        if Path(args.severity_test).exists() and Path(args.severity_test).stat().st_size:
            system = read_jsonl(args.severity_test)[0]["messages"][0]["content"]  # same system prompt as training
        result["hallucination"] = hallucination_check(model, tokenizer, system, args.hallucination_n,
                                                      args.seed, args.batch_size, args.max_new_tokens)

    # ---- Save ----
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(RESULTS / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Short summary ----
    if "severity" in result:
        s = result["severity"]
        print(f"Severity: accuracy {s['accuracy']:.3f}, macro-F1 {s['macro_f1']:.3f}, unparseable {s['unparseable']}/{s['n']}")
        for part in ("nvd", "cna"):  # the same numbers per label source
            p = result["severity_by_source"][part]
            if p["n"]:
                print(f"  {part.upper()} only: n={p['n']}, accuracy {p['accuracy']:.3f}, macro-F1 {p['macro_f1']:.3f}")
            else:
                print(f"  {part.upper()} only: n=0")
        if result["severity_by_source"].get("note"):
            print("  " + result["severity_by_source"]["note"])
    if "attack" in result:
        a = result["attack"]
        print(f"ATT&CK: exact {a['exact_match_accuracy']:.3f}, parent {a['parent_accuracy']:.3f}, unparseable {a['unparseable']}/{a['n']}")
    if "hallucination" in result:
        h = result["hallucination"]
        print(f"Hallucination: gave a severity for {h['gave_a_severity']}/{h['n']} fake CVE ids")
    print(f"Saved {RESULTS / (name + '.json')}")


if __name__ == "__main__":
    main()
