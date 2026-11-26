# Metrics ledger

All metrics of the project in one place, by stage. Real values come from files in this repo. "Pending" means it has not run yet.

Home: [README](../README.md) | Docs index: [README](README.md) | Journal: [journal.md](journal.md)

> Update rule: after every run, update the affected table here, with the file the number came from.

## Stage 02: Data collection

Source: `src/collect.py` output. Each yearly file was checked by `validate_year_file()` and gives the same count. Also in `data/processed/stats_clean.json`.

| Year | CVEs downloaded |
|---|---|
| 2022 | 26,431 |
| 2023 | 30,949 |
| 2024 | 40,704 |
| 2025 | 49,972 |
| Total | 148,056 |

MITRE ATT&CK Enterprise (STIX): downloaded, 1 file.

## Stage 03: Data cleaning (CVE funnel)

Source: `data/processed/stats_clean.json`.

| Step | Records left |
|---|---|
| Read from raw files | 148,056 |
| Not rejected | 141,992 |
| Has CVSS v3 severity | 138,891 |
| Severity is not NONE | Pending (this filter was added after the run) |
| Description long enough | 138,874 |

Severity before dedup (138,874 records; re-counted from `cves_clean.jsonl`):

| Severity | Count | Share |
|---|---|---|
| CRITICAL | 19,911 | 14.3% |
| HIGH | 51,750 | 37.3% |
| MEDIUM | 64,383 | 46.4% |
| LOW | 2,828 | 2.0% |
| NONE | 2 | Dropped by the new filter (Pending re-run) |

Label source (who gave the CVSS score):

| Source | Count |
|---|---|
| NVD | 90,275 |
| CNA | 48,599 |

## Stage 04: Dataset building

### CVE severity dataset

Source: `data/processed/stats_cve.json` (run on the full cleaned data).

| Step | Records | CRITICAL | HIGH | MEDIUM | LOW |
|---|---|---|---|---|---|
| Loaded | 138,874 | 19,911 | 51,750 | 64,383 | 2,828 |
| Only LOW, MEDIUM, HIGH, CRITICAL | 138,872 | 19,911 | 51,750 | 64,383 | 2,828 |
| After exact dedup | 131,326 | 19,539 | 48,150 | 60,919 | 2,718 |
| After near-dedup | 122,752 | 18,517 | 44,738 | 56,949 | 2,548 |

| Split | Before sampling | Final | CRITICAL | HIGH | MEDIUM | LOW | Removed by train-leak check |
|---|---|---|---|---|---|---|---|
| train | 72,517 | 7,343 | 2,000 | 2,000 | 2,000 | 1,343 | n/a |
| val | 9,378 | 1,000 | 127 | 386 | 470 | 17 | 0 |
| test (natural) | 40,857 | 1,000 | 127 | 337 | 510 | 26 | 0 |
| test (balanced) | n/a | 600 | 150 | 150 | 150 | 150 | n/a |

Label source (NVD / CNA):

| Split | Before sampling | Final |
|---|---|---|
| train | 61,647 / 10,870 | 6,170 / 1,173 |
| val | 4,634 / 4,744 | 496 / 504 |
| test (natural) | 15,634 / 25,223 | 379 / 621 |
| test (balanced) | n/a | 218 / 382 |

CNA share of each split (computed from the counts above): train 15.0% before and 16.0% after sampling; val 50.6% and 50.4%; natural test 61.7% and 62.1%; balanced test 63.7% after sampling.

Records removed by each CVE step (from the table above): NONE 2 (both CNA); exact duplicates 7,546; near-duplicates 8,574 (HIGH 3,412, MEDIUM 3,970, CRITICAL 1,022, LOW 170); train-leak check 0 from val and 0 from test.

Configured values (`configs/dataset.json`): train cap 2000 per class, val and test cap 1000, balanced test 150 per class, near-duplicate threshold 0.85, 128 permutations, 3-word shingles, seed 42. Train 2022-01-01 to 2024-09-30, validation 2024-10-01 to 2024-12-31, test 2025.

### ATT&CK dataset

Source: `data/processed/stats_attack.json`.

| Step | Count |
|---|---|
| Relationships in the STIX file | 21,822 |
| Active "uses" relationships | 18,975 |
| Source is a group, malware or tool and target is a technique | 16,423 |
| With a description | 16,423 |
| After cleaning (text not empty) | 16,423 |
| After removing duplicates (usable examples) | 16,423 |

| Split | Examples | Sources (groups / software) | Different techniques | Techniques not in train |
|---|---|---|---|---|
| train | 13,139 | 594 | 592 | 0 |
| val | 1,642 | 211 | 286 | 9 |
| test | 1,642 | 211 | 293 | 9 |

## Stage 05: Smoke test

| Item | Value |
|---|---|
| Training loss at the first and last step | Pending |
| Run time | Pending |
| Adapter saved | Pending |

## Stage 06: QLoRA training (main run)

Planned in `configs/qlora_05b.yaml` (free Colab T4). The data selection and length check were run on the laptop with `train.py`'s functions; the training itself has not run.

| Item | Value |
|---|---|
| Severity examples (up to 1,000 per class) | 4,000 |
| ATT&CK examples | 2,000 |
| Total training examples | 6,000 |
| Steps per epoch (batch 4 x accumulation 4) | 375 |
| Steps in the full run (2 epochs) | 750 |
| Examples longer than 512 tokens (Qwen tokenizer) | 50 of 6,000 (0.8%) |
| Seconds per step on the T4 | Pending |
| Estimated total time | Pending |

| Item | Value |
|---|---|
| Final training loss | Pending |
| Training time | Pending |
| GPU | Pending |
| Examples used (severity / ATT&CK) | Pending |

## Stage 07: Evaluation

| Metric | Zero-shot | Few-shot | Fine-tuned |
|---|---|---|---|
| Severity accuracy | Pending | Pending | Pending |
| Severity macro-F1 | Pending | Pending | Pending |
| Severity unparseable outputs | Pending | Pending | Pending |
| Severity accuracy, NVD labels only (with n) | Pending | Pending | Pending |
| Severity macro-F1, NVD labels only | Pending | Pending | Pending |
| Severity accuracy, CNA labels only (with n) | Pending | Pending | Pending |
| Severity macro-F1, CNA labels only | Pending | Pending | Pending |
| ATT&CK exact match | Pending | Pending | Pending |
| ATT&CK parent-technique accuracy | Pending | Pending | Pending |
| ATT&CK unparseable outputs | Pending | Pending | Pending |
| Fake CVE ids answered with a severity | Pending | Pending | Pending |

## Stage 08: Ablations

Planned in `configs/ablations.yaml` (5 runs, one change at a time).

| Run | LoRA rank | Severity examples | Severity macro-F1 | ATT&CK exact match |
|---|---|---|---|---|
| r8_n2000 | 8 | 2000 | Pending | Pending |
| r16_n2000 | 16 | 2000 | Pending | Pending |
| r64_n2000 | 64 | 2000 | Pending | Pending |
| r16_n500 | 16 | 500 | Pending | Pending |
| r16_n5000 | 16 | 5000 | Pending | Pending |

## Stage 09: Results and model card

Pending. The model card draft has `TODO` markers for these numbers.
