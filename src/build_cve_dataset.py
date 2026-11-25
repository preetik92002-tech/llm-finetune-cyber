"""Build the CVE severity dataset (chat-format JSONL) from cves_clean.jsonl.

Input : data/processed/cves_clean.jsonl
Output: data/processed/severity_{train,val,test}.jsonl
        data/processed/stats.json   (counts + class distribution after each step)
Config: configs/dataset.json (optional; defaults below are used for missing keys)

Steps:
  1. normalize text, drop exact duplicates
  2. near-duplicate removal (MinHash LSH on word 3-shingles), one record kept per cluster
  3. split by published date: train / val / test
  4. drop val/test records that are near-duplicates of a train record
  5. train: class-balanced sample (cap per severity); val/test: natural
     distribution, random sample capped at eval_cap

Memory: only id, date, text and label are held (roughly 100 MB for ~80k CVEs).
MinHash signatures are freed as soon as they are no longer needed.

Usage:
    python src/build_cve_dataset.py
"""

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "processed" / "cves_clean.jsonl"
OUT_DIR = ROOT / "data" / "processed"
CONFIG_FILE = ROOT / "configs" / "dataset.json"

# Defaults; any key can be overridden in configs/dataset.json.
DEFAULTS = {
    "train_cap_per_class": 2000,  # max train records per severity class
    "eval_cap": 1000,  # max records for val and for test
    "near_dup_threshold": 0.85,  # Jaccard similarity above which = near-duplicate
    "num_perm": 128,  # MinHash size: more = more accurate, more memory
    "shingle_size": 3,  # words per shingle
    "seed": 42,
    "train_start": "2022-01-01",
    "train_end": "2024-09-30",
    "val_start": "2024-10-01",
    "val_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": "2025-12-31",
}

SYSTEM_PROMPT = (
    "You are a cybersecurity analyst. Given a CVE description, "
    "classify its CVSS v3 base severity as one of: CRITICAL, HIGH, MEDIUM, LOW."
)


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    return cfg


# ------------------------------------------------------------------ helpers
def normalize(text):
    """Unicode-normalize, lowercase and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def minhash_of(norm_text, num_perm, k):
    """MinHash signature over word k-shingles of the text."""
    words = re.findall(r"\w+", norm_text)
    if len(words) < k:  # very short text: use the words themselves
        shingles = {" ".join(words)}
    else:
        shingles = {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def distribution(records):
    """Severity counts for a list of records."""
    return dict(Counter(r["label"] for r in records))


def snapshot(stats, step, records):
    """Record the size and class distribution after a step, and print it."""
    stats["steps"][step] = {"count": len(records), "classes": distribution(records)}
    print(f"{step:<40}{len(records):>8}  {stats['steps'][step]['classes']}")


def to_chat(record):
    """One training example in chat format for the severity task."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CVE description: {record['text']}"},
            {"role": "assistant", "content": record["label"]},
        ]
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- main
def main():
    cfg = load_config()
    rng = random.Random(cfg["seed"])
    stats = {"config": cfg, "steps": {}}

    # Load only the fields we need (streamed line by line).
    records = []
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append(
                {
                    "id": r["id"],
                    "published": r["published"][:10],  # YYYY-MM-DD
                    "text": r["description"],
                    "label": r["cvss_severity"],
                }
            )
    # Oldest first, so when duplicates exist the earliest record is the one kept.
    # That favors train and avoids test records "explaining" train ones.
    records.sort(key=lambda r: (r["published"], r["id"]))
    snapshot(stats, "0_loaded", records)

    # ---- Step 1: exact duplicates (compare a hash of the normalized text) ----
    seen, unique = set(), []
    for r in records:
        r["norm"] = normalize(r["text"])
        h = hashlib.md5(r["norm"].encode("utf-8")).digest()
        if h not in seen:
            seen.add(h)
            unique.append(r)
    records = unique
    del seen, unique
    snapshot(stats, "1_exact_dedup", records)

    # ---- Step 2: near-duplicates via MinHash LSH ----
    # Records arrive oldest first. Each one is checked against those already
    # kept; if any is >= threshold similar it is dropped, else it is indexed.
    lsh = MinHashLSH(threshold=cfg["near_dup_threshold"], num_perm=cfg["num_perm"])
    kept, signatures = [], {}
    for r in records:
        m = minhash_of(r["norm"], cfg["num_perm"], cfg["shingle_size"])
        if lsh.query(m):
            continue
        lsh.insert(r["id"], m)
        signatures[r["id"]] = m  # kept for step 4
        kept.append(r)
    records = kept
    del lsh, kept
    snapshot(stats, "2_near_dedup", records)

    # ---- Step 3: split by published date ----
    def in_range(r, a, b):
        return cfg[a] <= r["published"] <= cfg[b]

    splits = {
        "train": [r for r in records if in_range(r, "train_start", "train_end")],
        "val": [r for r in records if in_range(r, "val_start", "val_end")],
        "test": [r for r in records if in_range(r, "test_start", "test_end")],
    }
    del records
    for name, rs in splits.items():
        snapshot(stats, f"3_split_{name}", rs)

    # ---- Step 4: remove val/test records that resemble a train record ----
    # Step 2 already removed cross-split near-duplicates, so this is normally 0.
    # It stays as an explicit leakage check, using an index of train only.
    train_lsh = MinHashLSH(threshold=cfg["near_dup_threshold"], num_perm=cfg["num_perm"])
    for r in splits["train"]:
        train_lsh.insert(r["id"], signatures[r["id"]])
    for name in ("val", "test"):
        before = len(splits[name])
        splits[name] = [r for r in splits[name] if not train_lsh.query(signatures[r["id"]])]
        stats.setdefault("leakage_removed", {})[name] = before - len(splits[name])
        snapshot(stats, f"4_no_train_leak_{name}", splits[name])
    del train_lsh, signatures

    # ---- Step 5: sampling ----
    # Train: up to train_cap_per_class per severity (smaller classes keep all).
    by_class = defaultdict(list)
    for r in splits["train"]:
        by_class[r["label"]].append(r)
    train = []
    for label in sorted(by_class):
        rs = by_class[label]
        if len(rs) > cfg["train_cap_per_class"]:
            rs = rng.sample(rs, cfg["train_cap_per_class"])
        train.extend(rs)
    rng.shuffle(train)
    splits["train"] = train

    # Val/test: a plain random sample keeps the natural class mix.
    for name in ("val", "test"):
        rs = splits[name]
        if len(rs) > cfg["eval_cap"]:
            rs = rng.sample(rs, cfg["eval_cap"])
        rs.sort(key=lambda r: r["published"])
        splits[name] = rs

    for name, rs in splits.items():
        snapshot(stats, f"5_final_{name}", rs)
        write_jsonl(OUT_DIR / f"severity_{name}.jsonl", rs)

    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSaved severity_train/val/test.jsonl and stats.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
