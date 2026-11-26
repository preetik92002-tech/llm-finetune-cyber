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
import re
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
MAX_RETRIES = 8  # attempts per request before giving up
BACKOFF_BASE = 10  # first wait in seconds
BACKOFF_MAX = 300  # never wait longer than 5 minutes
TIMEOUT = 60  # seconds per HTTP request

def clean_api_key(raw):
    """Return a usable NVD key, or None if the value is empty or a placeholder.

    THE BUG THIS FIXES: the NVD API answers HTTP 404 when a request carries an
    `apiKey` header whose value is not a valid key. A `.env` line such as
    `NVD_API_KEY=your-key-here` (the placeholder from the setup docs) or
    `NVD_API_KEY=` with stray quotes or spaces looks "set" to the script, so it
    sent that junk as the header. NVD rejected it with 404, while `curl` with
    the same URL and no header returned 200. A request with no key is accepted.

    Real NVD keys are UUIDs (for example 12345678-90ab-cdef-1234-567890abcdef),
    so anything else is treated as "no key" and the header is never sent.
    """
    if raw is None:
        return None
    value = raw.strip().strip("\"'").strip()  # drop spaces and quotes pasted into .env
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value):
        return value
    if raw.strip():  # something was set but it is not a key: say so, never print it
        print("NVD_API_KEY is set but is not a valid key (empty or a placeholder). Ignoring it and sending no apiKey header.")
    return None


API_KEY = clean_api_key(os.getenv("NVD_API_KEY"))  # None means: send no apiKey header
# NVD limits: 5 requests / 30 s without a key, 50 / 30 s with a key.
# Sleep a little more than the minimum gap between calls to stay under it.
MIN_GAP = 0.7 if API_KEY else 6.5

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_last_call = 0.0


def backoff_delay(attempt):
    """Exponential backoff: 10, 20, 40, 80, 160, 300, 300, ... seconds."""
    return min(BACKOFF_BASE * 2 ** (attempt - 1), BACKOFF_MAX)


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
            time.sleep(backoff_delay(attempt))
            continue

        # On any 404, show what was really sent so the cause is visible.
        # Header NAMES only: values (such as the API key) are never printed.
        if resp.status_code == 404:
            print(f"  HTTP 404 for URL: {resp.url}")
            print(f"  Headers sent (names only): {', '.join(resp.request.headers.keys())}")
            print(f"  Response body: {resp.text[:1000]!r}")

        # Retryable statuses:
        #   403/429 = rate limited (NVD answers 403 when over the limit)
        #   5xx     = temporary server trouble (e.g. 503)
        #   404     = NVD sometimes returns a temporary 404 on a valid query.
        #             Only retried for the NVD endpoint: a 404 from any other
        #             URL (such as the MITRE file) means the URL is really wrong.
        is_nvd = url.startswith(NVD_URL)
        retryable = (
            resp.status_code in (403, 429)
            or resp.status_code >= 500
            or (is_nvd and resp.status_code == 404)
        )
        if retryable:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)  # the server told us how long to wait
            else:
                delay = backoff_delay(attempt)
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
            try:
                n = validate_year_file(out_file)  # but only if the file is readable
                print(f"{year}: already downloaded ({n} CVEs, checked), skipping")
                continue
            except (ValueError, OSError) as exc:
                bad = out_file.with_suffix(".json.corrupt")
                print(f"{year}: existing file is not valid ({exc}). Moving it to {bad.name} and downloading again.")
                out_file.replace(bad)  # keep it for inspection; never delete data silently

        # Write to nvd_cves_<year>.json.part first and rename to the final name
        # only after the whole year is done. If the run is interrupted or fails,
        # only the .part file exists, so the next run does not mistake a partial
        # year for a finished one (it is overwritten and restarted).
        part_file = out_file.with_suffix(".json.part")
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

        # Check the finished .part file BEFORE it gets its final name. A file
        # that fails the check keeps the .part name, so it is never skipped later.
        checked = validate_year_file(part_file)
        if checked != count:
            raise RuntimeError(f"{part_file.name}: wrote {count} CVEs but found {checked} when checking it")
        part_file.replace(out_file)
        print(f"{year}: saved {count} CVEs (checked) -> {out_file}")


def validate_year_file(path):
    """Check a yearly NVD file without loading it all: return its number of CVEs.

    Why: one yearly file once turned out to be unreadable (JSONDecodeError deep
    inside the file). It happened while two collector runs were active, which is
    my guess and not proven. Checking each file right after the download, and
    again before it is skipped, finds such a file early instead of in stage 03.

    The file layout is written by collect_nvd(): a line "[", one JSON record per
    line (all but the last end with a comma), then a line "]". So each line can
    be parsed on its own and memory stays low.
    """
    count = 0
    with path.open(encoding="utf-8") as f:
        first = f.readline().strip()
        if first != "[":
            raise ValueError(f"{path.name}: does not start with '['")
        last = ""
        for line in f:
            line = line.strip()
            if line == "]":
                last = line
                continue
            if last:  # something after the closing bracket
                raise ValueError(f"{path.name}: data after the closing ']'")
            json.loads(line.rstrip(","))  # raises if this record is broken
            count += 1
        if last != "]":
            raise ValueError(f"{path.name}: missing the closing ']' (the download was cut short)")
    return count


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
