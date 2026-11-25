"""Collect raw data: NVD CVEs (2022-2025) and the MITRE ATT&CK STIX bundle.

Output (in data/raw/):
  - nvd_cves_<year>.json   one file per year, slim records
  - mitre_enterprise_attack.json   full STIX bundle

Usage:
    python src/collect.py
Optional: set NVD_API_KEY in .env (or the environment) for a higher rate limit.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()  # read NVD_API_KEY from .env if present
except ImportError:
    pass

# ---------------------------------------------------------------- settings
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
MITRE_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
YEARS = range(2022, 2026)  # 2022..2025 inclusive
PAGE_SIZE = 2000  # NVD maximum results per page
MAX_WINDOW_DAYS = 120  # NVD rejects date ranges longer than 120 days
MAX_RETRIES = 6
TIMEOUT = 60  # seconds per HTTP request

API_KEY = os.getenv("NVD_API_KEY")
# NVD limits: 5 requests / 30 s without a key, 50 / 30 s with a key.
# Sleep a little more than the minimum gap between calls to stay under it.
MIN_GAP = 0.7 if API_KEY else 6.5

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_last_call = 0.0


def get_with_retry(url, params=None, headers=None):
    """GET with pacing, plus retries on rate limits, server errors, timeouts."""
    global _last_call
    for attempt in range(1, MAX_RETRIES + 1):
        # Pacing: wait so calls are at least MIN_GAP seconds apart.
        wait = MIN_GAP - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as exc:
            print(f"  network error ({exc}); retry {attempt}/{MAX_RETRIES}")
            time.sleep(min(2**attempt, 60))
            continue

        # 403/429 = rate limited (NVD uses 403); 5xx = temporary server trouble.
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt * 5, 120)
            print(f"  HTTP {resp.status_code}; waiting {delay}s, retry {attempt}/{MAX_RETRIES}")
            time.sleep(delay)
            continue

        resp.raise_for_status()  # other 4xx: a real bug, do not retry
        return resp
    raise RuntimeError(f"Gave up after {MAX_RETRIES} tries: {url}")


def slim_cve(item):
    """Keep only id, dates, English description and CVSS base severity."""
    cve = item["cve"]
    description = next(
        (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), ""
    )

    # Prefer the newest CVSS version available. v3.x keeps baseSeverity inside
    # cvssData; v2 keeps it at the metric level.
    severity, score = None, None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            metric = metrics[key][0]
            data = metric["cvssData"]
            severity = data.get("baseSeverity") or metric.get("baseSeverity")
            score = data.get("baseScore")
            break

    return {
        "id": cve["id"],
        "published": cve.get("published"),
        "description": description,
        "cvss_base_severity": severity,
        "cvss_base_score": score,
    }


def date_windows(year):
    """Split a year into <=120-day windows, as the NVD API requires."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    while start <= year_end:
        end = min(start + timedelta(days=MAX_WINDOW_DAYS - 1, hours=23, minutes=59, seconds=59), year_end)
        yield start, end
        start = end + timedelta(seconds=1)


def fmt(dt):
    """NVD wants ISO-8601 with milliseconds."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000")


def collect_nvd():
    """Download CVEs year by year, page by page, and save one file per year."""
    headers = {"apiKey": API_KEY} if API_KEY else {}
    if not API_KEY:
        print("No NVD_API_KEY set: using the slow public rate limit.")

    for year in YEARS:
        out_file = RAW_DIR / f"nvd_cves_{year}.json"
        if out_file.exists():  # resume support: skip finished years
            print(f"{year}: already downloaded, skipping")
            continue

        records = []
        for start, end in date_windows(year):
            index = 0
            while True:
                params = {
                    "pubStartDate": fmt(start),
                    "pubEndDate": fmt(end),
                    "resultsPerPage": PAGE_SIZE,
                    "startIndex": index,
                }
                data = get_with_retry(NVD_URL, params, headers).json()
                records.extend(slim_cve(v) for v in data.get("vulnerabilities", []))
                index += data.get("resultsPerPage", 0)
                total = data.get("totalResults", 0)
                print(f"{year} {start:%Y-%m-%d}: {min(index, total)}/{total}")
                if index >= total or data.get("resultsPerPage", 0) == 0:
                    break

        out_file.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"{year}: saved {len(records)} CVEs -> {out_file}")


def collect_mitre():
    """Download the full ATT&CK Enterprise STIX bundle (one JSON file)."""
    out_file = RAW_DIR / "mitre_enterprise_attack.json"
    if out_file.exists():
        print("MITRE ATT&CK: already downloaded, skipping")
        return
    resp = get_with_retry(MITRE_URL)
    out_file.write_bytes(resp.content)
    print(f"MITRE ATT&CK saved -> {out_file}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    collect_mitre()
    collect_nvd()
