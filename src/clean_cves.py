"""Clean the raw NVD files into one JSONL file.

Input : data/raw/nvd_cves_<year>.json   (full NVD "cve" objects, from collect.py)
Output: data/processed/cves_clean.jsonl (one JSON record per line)

Each output record:
  id, published, description, cvss_severity, cvss_score,
  cvss_source ("nvd" or "cna"), cvss_source_id, cwe_ids

Filters, in order (the count left after each one is printed):
  1. drop rejected CVEs
  2. drop CVEs with no CVSS v3.x severity
  3. drop CVEs whose severity is NONE (only LOW, MEDIUM, HIGH, CRITICAL remain)
  4. drop descriptions that are too short (after HTML is stripped)

Memory: one year is loaded at a time, and records are written to disk right away.

Usage:
    python src/clean_cves.py
"""

import html
import json
import re
from collections import Counter
from pathlib import Path

YEARS = range(2022, 2026)
MIN_DESC_CHARS = 30  # shorter descriptions carry too little signal to learn from

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "cves_clean.jsonl"

VALID_SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")  # "NONE" (score 0.0) is dropped
NVD_SOURCE = "nvd@nist.gov"  # NVD's own analysts; this is the "primary" source
CVSS_V3_KEYS = ("cvssMetricV31", "cvssMetricV30")  # newest first
TAG_RE = re.compile(r"<[^>]+>")
CWE_RE = re.compile(r"^CWE-\d+$")  # skips placeholders like NVD-CWE-Other


def clean_text(text):
    """Remove HTML tags, decode entities like &amp;, and collapse whitespace."""
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def english_description(cve):
    """Return the first English description, or an empty string."""
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value", "")
    return ""


def is_rejected(cve, description):
    """A CVE is rejected if NVD says so or the text carries the reject marker."""
    if cve.get("vulnStatus", "").lower() == "rejected":
        return True
    return description.startswith(("** REJECT **", "Rejected reason:"))


def pick_cvss_v3(cve):
    """Pick one CVSS v3.x metric: NVD primary first, then fall back to a CNA.

    Returns (severity, score, "nvd" | "cna", source_id) or None.
    """
    metrics = cve.get("metrics", {})
    candidates = []  # all v3.x metrics, newest version first
    for key in CVSS_V3_KEYS:
        candidates.extend(metrics.get(key, []))

    # 1) NVD's own score.
    for m in candidates:
        if m.get("source") == NVD_SOURCE:
            return _unpack(m, "nvd")
    # 2) Otherwise a score from the CNA (the vendor / CVE numbering authority).
    for m in candidates:
        return _unpack(m, "cna")
    return None


def _unpack(metric, label):
    data = metric.get("cvssData", {})
    severity = data.get("baseSeverity")
    if not severity:  # malformed metric: treat as missing
        return None
    return severity.upper(), data.get("baseScore"), label, metric.get("source")


def cwe_ids(cve):
    """Collect unique real CWE ids (e.g. CWE-79), keeping their order."""
    seen = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            value = d.get("value", "")
            if CWE_RE.match(value) and value not in seen:
                seen.append(value)
    return seen


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()  # how many records pass each stage
    severities = Counter()
    sources = Counter()

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for year in YEARS:
            path = RAW_DIR / f"nvd_cves_{year}.json"
            if not path.exists():
                print(f"{year}: {path.name} missing, skipping")
                continue

            # Load only this year, then let it go before the next one.
            with path.open(encoding="utf-8") as f:
                cves = json.load(f)

            for cve in cves:
                counts["0 read"] += 1

                description = clean_text(english_description(cve))
                if is_rejected(cve, description):
                    continue
                counts["1 not rejected"] += 1

                cvss = pick_cvss_v3(cve)
                if cvss is None:
                    continue
                counts["2 has CVSS v3 severity"] += 1

                # CVSS v3 gives "NONE" to a score of 0.0. It is not a class we
                # train on, so only LOW, MEDIUM, HIGH and CRITICAL are kept.
                if cvss[0] not in VALID_SEVERITIES:
                    continue
                counts["3 severity is not NONE"] += 1

                if len(description) < MIN_DESC_CHARS:
                    continue
                counts["4 description long enough"] += 1

                severity, score, source, source_id = cvss
                record = {
                    "id": cve["id"],
                    "published": cve.get("published"),
                    "description": description,
                    "cvss_severity": severity,
                    "cvss_score": score,
                    "cvss_source": source,
                    "cvss_source_id": source_id,
                    "cwe_ids": cwe_ids(cve),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                severities[severity] += 1
                sources[source] += 1

            del cves  # free the year before loading the next
            print(f"{year}: done")

    print("\nRecords remaining after each step:")
    for step, n in sorted(counts.items()):
        print(f"  {step:<28}{n:>8}")

    print("\nSeverity distribution:")
    total = sum(severities.values()) or 1
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if sev in severities:
            print(f"  {sev:<10}{severities[sev]:>8}  {severities[sev] / total:6.1%}")

    print(f"\nCVSS source used: {dict(sources)}")
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
