"""Build the ATT&CK technique-mapping dataset (chat-format JSONL) from STIX.

Input : data/raw/mitre_enterprise_attack.json
Output: data/processed/attack_{train,val,test}.jsonl
        data/processed/stats.json   (an "attack" section is added; other keys are kept)

Each example:
  user      = text describing how a group/software behaves
  assistant = "<technique id> <technique name>", e.g. "T1566.001 Spearphishing Attachment"

Steps:
  1. keep "uses" relationships: (intrusion-set | malware | tool) -> attack-pattern,
     with a non-empty description, not revoked or deprecated
  2. clean the text (remove citations and markdown links)
  3. drop duplicates
  4. split by SOURCE (group/software), about 80/10/10, so one source never
     appears in two splits (stops the model memorising a group's wording)

Usage:
    python src/build_attack_dataset.py
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "raw" / "mitre_enterprise_attack.json"
OUT_DIR = ROOT / "data" / "processed"
STATS_FILE = OUT_DIR / "stats.json"

SEED = 42
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SOURCE_PREFIXES = ("intrusion-set--", "malware--", "tool--")  # who "uses"
TARGET_PREFIX = "attack-pattern--"  # a technique or sub-technique

SYSTEM_PROMPT = (
    "You are a cybersecurity analyst. Given a description of adversary behaviour, "
    "name the MITRE ATT&CK technique as '<technique id> <technique name>'."
)

CITATION_RE = re.compile(r"\s*\(Citation:[^)]*\)")  # "(Citation: Some Report 2020)"
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text
TAG_RE = re.compile(r"</?code>")  # STIX text uses <code> for commands


# ------------------------------------------------------------------ helpers
def clean_text(text):
    """Keep the readable sentences: no citations, no link URLs, no code tags."""
    text = CITATION_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def attack_id(obj):
    """The public ATT&CK id (e.g. T1566.001) from the external references."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def is_active(obj):
    """Skip objects MITRE has revoked or deprecated."""
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def to_chat(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def split_sources(source_sizes, rng):
    """Assign whole sources to splits so example counts are about 80/10/10.

    Sources are taken largest first (random order among equal sizes). Each one
    goes to the split that is furthest below its target size. Placing the big
    sources first keeps one huge group from throwing the ratios off.
    """
    sources = sorted(source_sizes)
    rng.shuffle(sources)  # random tie-break
    sources.sort(key=lambda s: -source_sizes[s])  # stable: keeps the shuffle for ties
    total = sum(source_sizes.values())
    target = {k: f * total for k, f in SPLIT_FRACTIONS.items()}
    filled = dict.fromkeys(SPLIT_FRACTIONS, 0)
    assignment = {}
    for s in sources:
        best = max(target, key=lambda k: target[k] - filled[k])  # biggest gap
        assignment[s] = best
        filled[best] += source_sizes[s]
    return assignment


# --------------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    stats = {}

    bundle = json.loads(IN_FILE.read_text(encoding="utf-8"))
    objects = bundle["objects"]
    del bundle

    # Lookup tables for the objects that relationships point to.
    techniques = {}  # stix id -> "T1566.001 Spearphishing Attachment"
    source_names = {}  # stix id -> group/software name (for stats only)
    for o in objects:
        if not is_active(o):
            continue
        if o["type"] == "attack-pattern" and attack_id(o):
            techniques[o["id"]] = f"{attack_id(o)} {o['name']}"
        elif o["id"].startswith(SOURCE_PREFIXES):
            source_names[o["id"]] = o.get("name", o["id"])

    # ---- Step 1: pick the "uses" relationships ----
    rels = [o for o in objects if o["type"] == "relationship"]
    stats["relationships_total"] = len(rels)
    uses = [r for r in rels if r.get("relationship_type") == "uses" and is_active(r)]
    stats["uses_active"] = len(uses)
    del objects, rels

    wanted = [
        r
        for r in uses
        if r["source_ref"] in source_names and r["target_ref"] in techniques
    ]
    stats["uses_source_and_target_ok"] = len(wanted)

    with_desc = [r for r in wanted if r.get("description", "").strip()]
    stats["with_description"] = len(with_desc)
    del uses, wanted

    # ---- Step 2: clean the text ----
    examples = []
    for r in with_desc:
        text = clean_text(r["description"])
        if text:  # cleaning can leave nothing (a description that was only a citation)
            examples.append(
                {"source": r["source_ref"], "text": text, "answer": techniques[r["target_ref"]]}
            )
    stats["after_cleaning_non_empty"] = len(examples)
    del with_desc

    # ---- Step 3: remove duplicates (same text and same technique) ----
    seen, unique = set(), []
    for e in examples:
        key = (e["text"].lower(), e["answer"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    examples = unique
    stats["after_dedup"] = len(examples)

    # ---- Step 4: split by source so no group/software crosses splits ----
    sizes = Counter(e["source"] for e in examples)
    assignment = split_sources(sizes, rng)
    splits = defaultdict(list)
    for e in examples:
        splits[assignment[e["source"]]].append(e)

    train_techniques = {e["answer"] for e in splits["train"]}
    stats["splits"] = {}
    for name in ("train", "val", "test"):
        rs = splits[name]
        rng.shuffle(rs)
        techs = {e["answer"] for e in rs}
        stats["splits"][name] = {
            "examples": len(rs),
            "sources": len({e["source"] for e in rs}),
            "unique_techniques": len(techs),
            # techniques never seen in train: the model can only get these by generalising
            "techniques_not_in_train": len(techs - train_techniques) if name != "train" else 0,
        }
        with (OUT_DIR / f"attack_{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in rs:
                f.write(json.dumps(to_chat(e), ensure_ascii=False) + "\n")

    # ---- Save: merge into stats.json so the CVE stats are kept ----
    all_stats = {}
    if STATS_FILE.exists():
        all_stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    all_stats["attack"] = stats
    STATS_FILE.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
