"""Collect raw data: NVD CVEs (2022-2025) and the MITRE ATT&CK STIX bundle.

Output (in data/raw/):
  - nvd_cves_<year>.json   one file per year, full NVD cve objects
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
    """Download CVEs year by year and save one file per year.

    Full NVD records are kept (source, CWE, status, all CVSS metrics) because
    clean_cves.py needs them. Records are streamed to disk as each page
    arrives, so memory stays low on small machines.
    """
    headers = {"apiKey": API_KEY} if API_KEY else {}
    if not API_KEY:
        print("No NVD_API_KEY set: using the slow public rate limit.")

    for year in YEARS:
        out_file = RAW_DIR / f"nvd_cves_{year}.json"
        if out_file.exists():  # resume support: skip finished years
            print(f"{year}: already downloaded, skipping")
            continue

        part_file = out_file.with_suffix(".json.part")  # renamed when complete
        count = 0
        with part_file.open("w", encoding="utf-8") as f:
            f.write("[")
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
                    for item in data.get("vulnerabilities", []):
                        f.write(",\n" if count else "\n")
                        json.dump(item["cve"], f)
                        count += 1
                    index += data.get("resultsPerPage", 0)
                    total = data.get("totalResults", 0)
                    print(f"{year} {start:%Y-%m-%d}: {min(index, total)}/{total}")
                    if index >= total or data.get("resultsPerPage", 0) == 0:
                        break
            f.write("\n]")

        part_file.replace(out_file)
        print(f"{year}: saved {count} CVEs -> {out_file}")

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
