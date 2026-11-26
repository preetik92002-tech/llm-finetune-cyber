"""A simple baseline for the severity task: TF-IDF features + logistic regression.

Trains on data/processed/severity_train.jsonl and tests on severity_test.jsonl and
severity_test_balanced.jsonl. No GPU needed: it runs on a laptop CPU with 8 GB RAM
(about 7,000 short texts; the vectors are sparse).

A language model should beat this baseline to justify fine-tuning.

Output: results/baseline_tfidf.json with, for each test file, accuracy and macro-F1
for all samples, for samples with cvss_source = nvd, and for cvss_source = cna (with
sample counts).

Usage:
    python src/baseline_tfidf.py
"""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PREFIX = "CVE description: "  # build_cve_dataset.py puts this in front of every description
LABELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def read_split(path):
    """Read a severity chat file -> (texts, labels, sources).

    text   = the CVE description (the user message without its fixed prefix)
    label  = the assistant answer
    source = meta.cvss_source ("nvd" or "cna"), or None if the file has no meta field
    """
    texts, labels, sources = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            texts.append(user[len(PREFIX):] if user.startswith(PREFIX) else user)
            labels.append(row["messages"][-1]["content"].strip().upper())
            sources.append((row.get("meta") or {}).get("cvss_source"))
    return texts, labels, sources


def scores(gold, pred):
    """Accuracy and macro-F1 of one group. An empty group gives n = 0 and no scores."""
    if not gold:
        return {"n": 0, "accuracy": None, "macro_f1": None}
    present = [l for l in LABELS if l in set(gold)]  # macro average over classes that occur, as in evaluate.py
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=present, average="macro", zero_division=0),
    }


def evaluate(model_predict, path):
    """Metrics for one test file: all samples, NVD labels only, CNA labels only."""
    texts, gold, sources = read_split(path)
    pred = list(model_predict(texts))
    out = {"file": Path(path).name}
    for part in (None, "nvd", "cna"):
        idx = [i for i, s in enumerate(sources) if part is None or s == part]
        out["all" if part is None else part] = scores([gold[i] for i in idx], [pred[i] for i in idx])
    if out["nvd"]["n"] + out["cna"]["n"] == 0:
        out["note"] = "This file has no cvss_source metadata. Run src/build_cve_dataset.py again to add it."
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "baseline_tfidf.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 1. Load the training data ----
    train_path = PROCESSED / "severity_train.jsonl"
    if not train_path.exists() or not train_path.stat().st_size:
        raise SystemExit(f"{train_path} is missing or empty. Run stages 02 to 04 first.")
    train_texts, train_labels, _ = read_split(train_path)
    print(f"Training examples: {len(train_texts)}")

    # ---- 2. Turn text into numbers: TF-IDF ----
    # Each description becomes a sparse vector of word weights. Words that are frequent in
    # one text but rare overall get a high weight. Word pairs (1 to 2 words) capture
    # phrases like "buffer overflow". min_df=2 drops words seen only once. sublinear_tf
    # dampens very repeated words. max_features keeps memory small.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                 sublinear_tf=True, lowercase=True)
    x_train = vectorizer.fit_transform(train_texts)  # fit on TRAIN only, so test words cannot leak in
    print(f"TF-IDF features: {x_train.shape[1]}")

    # ---- 3. Train logistic regression ----
    # class_weight="balanced" gives the smaller classes (LOW) more weight. max_iter is high
    # enough to converge on this size. lbfgs handles several classes at once.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    clf.fit(x_train, train_labels)

    # ---- 4. Evaluate on both test files ----
    def predict(texts):
        return clf.predict(vectorizer.transform(texts))  # same vocabulary as in training

    result = {"model": "TF-IDF (1-2 grams) + logistic regression", "train_examples": len(train_texts),
              "features": x_train.shape[1], "tests": {}}
    for name in ("severity_test.jsonl", "severity_test_balanced.jsonl"):
        path = PROCESSED / name
        if not path.exists() or not path.stat().st_size:
            result["tests"][name] = {"note": "file is missing or empty"}
            print(f"{name}: missing or empty, skipped")
            continue
        r = evaluate(predict, path)
        result["tests"][name] = r
        print(f"{name}: n={r['all']['n']} accuracy {r['all']['accuracy']:.3f} macro-F1 {r['all']['macro_f1']:.3f}")
        for part in ("nvd", "cna"):
            p = r[part]
            print(f"  {part.upper()} only: n={p['n']}" + (f" accuracy {p['accuracy']:.3f} macro-F1 {p['macro_f1']:.3f}" if p["n"] else ""))
        if r.get("note"):
            print("  " + r["note"])

    # ---- 5. Save ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
