"""Run LoRA-rank and training-size ablations, then build a results table and plot.

For each planned run this script
  1. writes a config (base config + new rank / size) to outputs/ablations/<run>/config.yaml
  2. trains with src/train.py
  3. evaluates with src/evaluate.py (metrics go to results/ablation_<run>.json)
Then it writes:
  results/ablations.md                 markdown table of all runs
  results/ablations_data_size.png      performance vs training-set size

A run whose metrics file already exists is skipped, so an interrupted session
can be restarted. Use --force to redo runs.

Usage:
    python src/run_ablations.py --config configs/ablations.yaml
    python src/run_ablations.py --config configs/ablations.yaml --dry-run      # only print the plan
    python src/run_ablations.py --config configs/ablations.yaml --only-report  # rebuild table and plot

Needs a CUDA GPU (it calls train.py). Use Colab.
"""

import argparse
import copy
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RUNS_DIR = ROOT / "outputs" / "ablations"  # adapters and generated configs (git-ignored)


def plan_runs(plan):
    """List of (rank, size) pairs to run, from the plan's mode."""
    if plan["mode"] == "grid":
        return list(product(plan["ranks"], plan["sizes"]))
    if plan["mode"] != "one_at_a_time":
        raise SystemExit("mode must be 'one_at_a_time' or 'grid'")
    runs = [(r, plan["default_size"]) for r in plan["ranks"]]  # change rank, keep size
    runs += [(plan["default_rank"], s) for s in plan["sizes"]]  # change size, keep rank
    return list(dict.fromkeys(runs))  # drop the repeated centre run, keep order


def count_lines(rel_path):
    """Examples available in a JSONL file (0 if it is missing or empty)."""
    path = ROOT / rel_path
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def make_config(base, plan, rank, size):
    """Copy the base config and change only what this run varies."""
    cfg = copy.deepcopy(base)
    cfg["lora"]["r"] = rank
    cfg["lora"]["alpha"] = rank * plan["alpha_multiplier"]
    cfg["max_samples"] = size  # severity examples (train.py uses fewer if the file is smaller)
    attack = round(size * plan["attack_ratio"])
    if attack > 0 and cfg.get("attack_file"):
        cfg["max_attack_samples"] = attack
    else:
        cfg["attack_file"] = None  # severity only
    cfg["max_steps"] = None  # train by epochs, as in the base config
    # Ablation runs stay local: no Hub uploads, no mid-run checkpoints.
    cfg["push_to_hub"] = False
    cfg["save_steps"] = None
    cfg["output_dir"] = f"outputs/ablations/{run_name(rank, size)}"
    return cfg


def run_name(rank, size):
    return f"r{rank}_n{size}"


def run_one(base, plan, rank, size, force):
    """Train and evaluate one (rank, size) run. Returns True on success."""
    name = run_name(rank, size)
    metrics_file = RESULTS / f"ablation_{name}.json"
    if metrics_file.exists() and not force:
        print(f"[{name}] already done, skipping")
        return True

    cfg = make_config(base, plan, rank, size)
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    py = sys.executable  # the same Python that runs this script
    print(f"[{name}] training")
    train = subprocess.run([py, str(ROOT / "src" / "train.py"), "--config", str(cfg_path)])
    if train.returncode != 0:
        print(f"[{name}] TRAINING FAILED (exit {train.returncode})")
        return False

    print(f"[{name}] evaluating")
    evaluate = subprocess.run([
        py, str(ROOT / "src" / "evaluate.py"),
        "--mode", "fine-tuned",
        "--model", cfg["model_name"],
        "--adapter", str(ROOT / cfg["output_dir"]),
        "--name", f"ablation_{name}",
        "--hallucination-n", "0",  # skip the hallucination check; it is not part of this comparison
    ])
    if evaluate.returncode != 0:
        print(f"[{name}] EVALUATION FAILED (exit {evaluate.returncode})")
        return False
    return True


# ------------------------------------------------------------------ report
def collect(plan, base):
    """Read each planned run's metrics file. Missing runs get metrics=None."""
    rows = []
    sev_available = count_lines(base["train_file"])
    atk_available = count_lines(base["attack_file"]) if base.get("attack_file") else 0
    for rank, size in plan_runs(plan):
        f = RESULTS / f"ablation_{run_name(rank, size)}.json"
        metrics = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        attack_asked = round(size * plan["attack_ratio"])
        rows.append({
            "rank": rank,
            "size": size,
            # Examples really used: the request, or fewer if the file is smaller.
            "severity_used": min(size, sev_available),
            "attack_used": min(attack_asked, atk_available) if attack_asked else 0,
            "metrics": metrics,
        })
    return rows


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_table(rows, path):
    """Markdown table; runs that have not been run say so instead of showing numbers."""
    lines = [
        "# Ablation results", "",
        "Numbers come from `results/ablation_<run>.json`. `not run` means no metrics file exists yet.", "",
        "| Run | LoRA rank | Severity train examples | ATT&CK train examples | Severity accuracy | Severity macro-F1 "
        "| Severity unparseable | ATT&CK exact | ATT&CK parent | ATT&CK unparseable |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        head = f"| {run_name(r['rank'], r['size'])} | {r['rank']} | {r['severity_used']} | {r['attack_used']} |"
        if m is None:
            lines.append(head + " not run | not run | not run | not run | not run | not run |")
            continue
        s, a = m.get("severity"), m.get("attack")
        lines.append(
            head
            + f" {fmt(s and s['accuracy'])} | {fmt(s and s['macro_f1'])} | {s['unparseable'] if s else 'n/a'} |"
            + f" {fmt(a and a['exact_match_accuracy'])} | {fmt(a and a['parent_accuracy'])} | {a['unparseable'] if a else 'n/a'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_data_size(rows, path):
    """Performance against training-set size, one line per LoRA rank."""
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    done = [r for r in rows if r["metrics"] is not None]
    panels = [("Severity macro-F1", "severity", "macro_f1"), ("ATT&CK exact match", "attack", "exact_match_accuracy")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, task, key) in zip(axes, panels):
        for rank in sorted({r["rank"] for r in done}):
            pts = sorted((r["severity_used"], r["metrics"][task][key]) for r in done
                         if r["rank"] == rank and r["metrics"].get(task))
            if pts:  # a single point still shows, as a marker
                ax.plot(*zip(*pts), marker="o", label=f"rank {rank}")
        if done and all(r["severity_used"] > 0 for r in done):
            ax.set_xscale("log")  # sizes are 500, 2000, 5000: log spacing reads better
        ax.set_xlabel("Severity training examples")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ablations.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("--only-report", action="store_true", help="skip training; rebuild the table and plot")
    p.add_argument("--force", action="store_true", help="redo runs that already have metrics")
    args = p.parse_args()

    plan = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / plan["base_config"]).read_text(encoding="utf-8"))
    runs = plan_runs(plan)

    print(f"{len(runs)} runs:")
    for rank, size in runs:
        print(f"  {run_name(rank, size)}  rank={rank}  severity examples={size}")
    if args.dry_run:
        return

    RESULTS.mkdir(exist_ok=True)
    failed = []
    if not args.only_report:
        for rank, size in runs:
            if not run_one(base, plan, rank, size, args.force):
                failed.append(run_name(rank, size))  # keep going; one failure should not lose the others

    rows = collect(plan, base)
    write_table(rows, RESULTS / "ablations.md")
    plot_data_size(rows, RESULTS / "ablations_data_size.png")
    print(f"Wrote {RESULTS / 'ablations.md'} and {RESULTS / 'ablations_data_size.png'}")
    if failed:
        print(f"Failed runs: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
