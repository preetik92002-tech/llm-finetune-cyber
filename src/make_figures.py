"""Draw the charts used in the docs from the stats and results files.

Reads  data/processed/stats_clean.json, stats_cve.json, stats_attack.json
       results/train_*.json, results/*.json (evaluation runs)
Writes docs/images/*.png

A chart whose input file is missing, empty or all zeros is skipped with a
message. The script never invents numbers and never fails because of a missing file.

Usage:
    python src/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
IMAGES = ROOT / "docs" / "images"

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
COLORS = {"CRITICAL": "#b2182b", "HIGH": "#ef8a62", "MEDIUM": "#fddb7a", "LOW": "#67a9cf"}


def load(path):
    """Read a JSON file, or return None if it does not exist or is unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=150)
    plt.close(fig)
    print(f"saved docs/images/{name}")


def label_bars(ax, bars, values):
    for b, v in zip(bars, values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:,}", va="center", fontsize=8)


# ------------------------------------------------------------------ charts
def funnel_cves():
    """(1) How many CVEs are left after each cleaning and building step."""
    clean = load(PROCESSED / "stats_clean.json")
    build = load(PROCESSED / "stats_cve.json")
    steps = []  # (label, count)
    if clean:
        f = clean.get("cleaning_funnel", {})
        for key, label in [("read", "Read from raw files"), ("not_rejected", "Not rejected"),
                           ("has_cvss_v3_severity", "Has CVSS v3 severity"),
                           ("severity_not_none", "Severity is not NONE"),
                           ("description_long_enough", "Description long enough")]:
            if f.get(key):  # skip steps that are Pending (null)
                steps.append((label, f[key]))
    if build and build.get("steps", {}).get("0_loaded", {}).get("count"):
        for key, label in [("1_exact_dedup", "After exact dedup"), ("2_near_dedup", "After near-dedup")]:
            if build["steps"].get(key, {}).get("count"):
                steps.append((label, build["steps"][key]["count"]))
    if not steps:
        return print("skip: CVE funnel (no stats_clean.json)")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(steps) + 1.5))
    labels, values = zip(*steps)
    bars = ax.barh(labels[::-1], values[::-1], color="#4c78a8")
    label_bars(ax, bars, values[::-1])
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("CVE records")
    ax.set_title("CVE data funnel")
    save(fig, "cve_funnel.png")


def severity_per_split():
    """(2) Class counts per split. The first bar is all cleaned CVEs before dedup."""
    groups = {}  # name -> {class: count}
    clean = load(PROCESSED / "stats_clean.json")
    if clean and clean.get("severity_distribution"):
        groups["all cleaned\n(before dedup)"] = clean["severity_distribution"]
    build = load(PROCESSED / "stats_cve.json")
    if build:
        for key, name in [("5_final_train", "train"), ("5_final_val", "val"),
                          ("5_final_test", "test (natural)"), ("5_final_test_balanced", "test (balanced)")]:
            classes = build.get("steps", {}).get(key, {}).get("classes")
            if classes:
                groups[name] = classes
    if not groups:
        return print("skip: severity per split (no data)")
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 4.5))
    bottoms = [0] * len(groups)
    shares = []
    for g in groups.values():
        total = sum(g.get(s, 0) for s in SEVERITIES) or 1
        shares.append([100 * g.get(s, 0) / total for s in SEVERITIES])
    for i, sev in enumerate(SEVERITIES):  # stacked bars of percent per class
        vals = [s[i] for s in shares]
        ax.bar(list(groups), vals, bottom=bottoms, label=sev, color=COLORS[sev])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Share of records (%)")
    ax.set_title("Severity class balance")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "severity_per_split.png")


def source_per_split():
    """(3) Where the severity label came from (NVD or CNA), per split."""
    groups = {}
    clean = load(PROCESSED / "stats_clean.json")
    if clean and clean.get("cvss_source"):
        groups["all cleaned"] = clean["cvss_source"]
    build = load(PROCESSED / "stats_cve.json")
    if build:
        for key, name in [("5_final_train", "train"), ("5_final_val", "val"),
                          ("5_final_test", "test (natural)"), ("5_final_test_balanced", "test (balanced)")]:
            src = build.get("steps", {}).get(key, {}).get("cvss_source")
            if src:
                groups[name] = src
    if not groups:
        return print("skip: label source per split (no data)")
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 4.5))
    totals = [sum(g.values()) or 1 for g in groups.values()]
    bottoms = [0] * len(groups)
    for src, color in [("nvd", "#4c78a8"), ("cna", "#f58518")]:
        vals = [100 * g.get(src, 0) / t for g, t in zip(groups.values(), totals)]
        ax.bar(list(groups), vals, bottom=bottoms, label=src.upper(), color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("Share of records (%)")
    ax.set_title("Label source: NVD vs CNA")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "label_source_per_split.png")


def attack_splits():
    """(4) ATT&CK examples and source groups per split."""
    stats = load(PROCESSED / "stats_attack.json")
    if not stats or "splits" not in stats:
        return print("skip: ATT&CK splits (no stats_attack.json)")
    names = ["train", "val", "test"]
    examples = [stats["splits"][n]["examples"] for n in names]
    sources = [stats["splits"][n]["sources"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, vals, title in [(axes[0], examples, "Examples"), (axes[1], sources, "Groups / software (sources)")]:
        bars = ax.bar(names, vals, color="#54a24b")
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("ATT&CK split sizes (a source is in one split only)")
    save(fig, "attack_splits.png")


def training_loss():
    """(5) Training loss curves, one line per results/train_*.json."""
    files = sorted(RESULTS.glob("train_*.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = 0
    for f in files:
        data = load(f)
        points = [(h["step"], h["loss"]) for h in (data or {}).get("log_history", []) if "loss" in h]
        if points:
            ax.plot(*zip(*points), label=f.stem.replace("train_", ""))
            drawn += 1
    if not drawn:
        plt.close(fig)
        return print("skip: training loss (no results/train_*.json yet)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss")
    ax.legend()
    save(fig, "training_loss.png")


def evaluation_scores():
    """(6) Severity macro-F1 and ATT&CK exact match per evaluation run."""
    runs = []
    for f in sorted(RESULTS.glob("*.json")):
        d = load(f)
        if isinstance(d, dict) and "run" in d and ("severity" in d or "attack" in d):
            runs.append((f.stem, d))
    if not runs:
        return print("skip: evaluation scores (no evaluation results yet)")
    names = [n for n, _ in runs]
    sev = [d.get("severity", {}).get("macro_f1") for _, d in runs]
    atk = [d.get("attack", {}).get("exact_match_accuracy") for _, d in runs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title in [(axes[0], sev, "Severity macro-F1"), (axes[1], atk, "ATT&CK exact match")]:
        shown = [(n, v) for n, v in zip(names, vals) if v is not None]
        if shown:
            ax.bar(*zip(*shown))
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    save(fig, "evaluation_scores.png")


if __name__ == "__main__":
    for chart in (funnel_cves, severity_per_split, source_per_split, attack_splits, training_loss, evaluation_scores):
        try:
            chart()
        except Exception as exc:  # one broken chart must not stop the others
            print(f"skip: {chart.__name__} ({type(exc).__name__}: {exc})")
