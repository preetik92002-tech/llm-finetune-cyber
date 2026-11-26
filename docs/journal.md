# Journal: how I am learning LLM fine-tuning

This is the story of the project, one entry per work session, oldest first. I write it for myself and for other beginners. Every number comes from a file in this repo (see [metrics.md](metrics.md)). Where something has not run yet, I say "Pending".

Home: [README](../README.md) | Docs index: [README](README.md) | Errors: [errors-and-fixes.md](errors-and-fixes.md)

> Update rule: after every change to code or data, add an entry here and update the affected docs.

Times come from my git history. I did not record the exact time of events that have no commit.

---

## Session 1: Project skeleton and the first collector

Commit: `b15056c` (2026-11-26 02:21); first commit `0976962` on 2026-11-25.

**Goal.** Make a project structure and download the raw data: CVEs from the NVD (2022 to 2025) and the MITRE ATT&CK file.

**What I tried.** I created the folders (`configs`, `data/raw`, `data/processed`, `src`, `notebooks`, `results`), a `.gitignore` for raw data, model weights and `.env`, and `src/collect.py`. The collector cuts each year into windows of at most 120 days (an NVD rule) and reads 2000 CVEs per page.

**What went wrong.**
- I ran my first commands from the wrong folder and got "No such file or directory" ([E01](errors-and-fixes.md#e01)).
- My first collector saved only a short summary of each CVE (id, description, severity). The cleaner later needed the CVSS source, the CWE ids and the rejected status, which were gone ([E02](errors-and-fixes.md#e02)).

**How I found the cause.** For E01 the error named a file that existed in the project root, so the folder had to be wrong. For E02 I looked at what the cleaner needed and compared it with what I had saved.

**What fixed it.** `cd` into the project root. A collector that saves the full NVD record and streams it to disk page by page (low memory on an 8 GB laptop).

**What I learned.** Save raw data in full. Decide what later stages need first. See [stage 02](stage-02-data-collection/README.md).

---

## Session 2: Cleaning the CVEs

Commit: `2d83b40` (02:34).

**Goal.** Turn the raw NVD files into one clean file: id, date, English description, CVSS v3 severity and score, source of the score, CWE ids.

**What I tried.** I wrote `src/clean_cves.py`. It reads one year at a time, drops rejected CVEs, drops CVEs with no CVSS v3 severity, drops very short descriptions and strips HTML. It prefers the NVD score and falls back to the CNA score, and it records which one was used.

**What went wrong.** I ran the cleaner before the download had finished. It printed "missing, skipping" and wrote empty outputs ([E08](errors-and-fixes.md#e08)).

**How I found the cause.** The printed message named the missing files, and `data/raw` had no finished yearly file.

**What fixed it.** I waited for the download and ran the cleaner again. From the full data it read 148,056 records and kept 138,874.

**What I learned.** Run stages in order and look at each stage's output. The funnel also showed me the data: 6,064 rejected CVEs, 3,101 more with no v3 severity and 17 with a very short description. And the labels are very uneven: LOW is 2.0% of the records. See [stage 03](stage-03-data-cleaning/README.md).

---

## Session 3: Building the datasets

Commits: `6d59d79` (02:37), `78e31f1` (02:38), `08c3d91` (02:58).

**Goal.** Make train, validation and test files in chat format for two tasks: CVE severity, and ATT&CK technique naming.

**What I tried.**
- CVE: normalize text and drop exact duplicates, then drop near-duplicates with MinHash LSH. Split by date (train until 2024-09-30, validation in Q4 2024, test 2025). Cap each severity class in train.
- ATT&CK: take "uses" relationships (group, malware or tool to technique) that have a description, remove citations and links, drop duplicates, and split by group or software.

**What went wrong.**
- The ATT&CK script wrote into the same `stats.json` as the CVE script, so one overwrote the other ([E11](errors-and-fixes.md#e11)).
- The dataset builder stopped with `ModuleNotFoundError: No module named 'datasketch'` ([E07](errors-and-fixes.md#e07)).
- I ran the CVE builder on an empty input, so every CVE count in its stats is 0.

**How I found the cause.** I opened the stats file and saw the missing section. For the empty counts I checked the input: `cves_clean.jsonl` was 0 bytes at that time.

**What fixed it.** Separate stats files (`stats_cve.json`, `stats_attack.json`), and installing the requirements. The ATT&CK dataset worked: 16,423 examples, split 13,139 / 1,642 / 1,642.

**What I learned.** Splitting by time (CVEs) and by group (ATT&CK) is how I keep test data from leaking into training. See [stage 04](stage-04-dataset-building/README.md). The first CVE run was on empty input. I re-ran the builder on the full cleaned data later (session 7).

---

## Session 4: The docs learning path

Commits: `3644303` (02:58), `031ce41` (03:03).

**What I tried.** A docs folder with a glossary and ten stage pages, so I can learn as I go.

**What went wrong.** My `README.md` looked garbled on GitHub ([E12](errors-and-fixes.md#e12)).

**How I found the cause and fixed it.** The garbled text had a space between every letter, which is typical for a file saved as UTF-16. I saved it again as UTF-8.

**What I learned.** Check the encoding of text files.

---

## Session 5: Training, evaluation and ablation scripts

Commits: `ce10b85` (03:05), `cb97658` (03:06), `40c2c41` (03:09), `780209d` (03:12), `e68c9d1` (03:17), `afb125f` (03:21).

**Goal.** Write the code for the later stages before I have a GPU: a smoke test, QLoRA training, evaluation, ablations and a model card.

**What I tried.** `src/train.py` (4-bit QLoRA with TRL and PEFT, loss only on the assistant answer), `src/evaluate.py` (zero-shot, few-shot and fine-tuned modes, accuracy, macro-F1, confusion matrix, ATT&CK exact and parent-technique accuracy, and a hallucination check with made-up CVE ids), `src/run_ablations.py`, a model card draft and an upload script.

**What went wrong.**
- On my laptop these scripts failed with `ModuleNotFoundError` for `peft` and `datasets` ([E09](errors-and-fixes.md#e09)).
- `run_ablations.py` wrote a results table and an empty plot even though no run had worked ([E10](errors-and-fixes.md#e10)).

**How I found the cause.** The laptop has no GPU, so training belongs on Google Colab. The empty table was a script that always wrote its report.

**What fixed it.** I split the packages into `requirements-data.txt` (laptop) and `requirements-train.txt` (Colab, no `torch` because Colab has it). I deleted the placeholder table and plot, and the script now writes nothing when no run has results.

**What I learned.** Plan where each stage runs. Never publish a table of placeholders. None of this training code has run yet, so there are no results: all training and evaluation numbers are Pending.

---

## Session 6: The NVD download fights back

Commits: `cb901ef` (03:30) and `91c012b` (12:47, the same fix committed again).

**Goal.** Finish the real download of all four years.

**What went wrong.**
- NVD answered HTTP 404 while `curl` with the same URL answered 200 ([E03](errors-and-fixes.md#e03)).
- Rate limits: HTTP 429 ([E04](errors-and-fixes.md#e04)).
- A `.part` file was left behind and a file was locked, because two collectors were running at once ([E05](errors-and-fixes.md#e05)).
- `nvd_cves_2022.json` could not be read: `JSONDecodeError` at char 12,065,647 ([E06](errors-and-fixes.md#e06)). I think the two parallel runs caused it, but I could not prove it.
- While debugging, a secret showed on my screen ([E13](errors-and-fixes.md#e13)).

**How I found the cause.** For the 404 I repeated the request with `curl`, first with the header and then without it. It worked without. My `.env` held a placeholder key, and the script had sent it as a real `apiKey` header. For the corrupt file I looked at which processes were running when it was written.

**What fixed it.**
- `clean_api_key()`: an empty or placeholder value means no key, and no header is sent.
- On any 404 the script prints the URL, the header names (never the values) and the response body.
- Retry with exponential backoff.
- Write to a `.part` file and rename it only when complete, and run one collector at a time.
- Validate every yearly file after download, and again before skipping it. All four pass: 26,431, 30,949, 40,704 and 49,972 CVEs.
- For the exposed secret: I revoked it and changed my habits (see the lesson).

**What I learned.** Compare a failing request with a working one. Never print or paste secrets. Use `.env` on my computer and Colab Secrets in the cloud, and revoke anything that was exposed.

---

## Session 7: Looking at the data, and the NONE class

Commit: `f004ff8` (14:45).

**Goal.** Understand the labels before training.

**What I found.** From the cleaned data (138,874 records): CRITICAL 19,911 (14.3%), HIGH 51,750 (37.3%), MEDIUM 64,383 (46.4%), LOW 2,828 (2.0%), and NONE 2. LOW is rare, so a model that never says LOW would still look good on plain accuracy. The score source is mixed too: 90,275 labels come from the NVD and 48,599 from a CNA.

**What I changed.**
- Drop the 2 NONE records, so only LOW, MEDIUM, HIGH and CRITICAL remain.
- Add a class-balanced test file with up to 150 examples per class next to the natural test file.
- Write per-split class counts and the NVD vs CNA counts into `stats_cve.json`.

**The re-run.** I ran the builder again on the full cleaned data (`stats_cve.json`, 2026-11-26 14:52 by file time).
- 138,874 records loaded, 2 NONE dropped, 7,546 exact and 8,574 near-duplicates removed, 122,752 left.
- Splits before sampling: train 72,517, validation 9,378, test 40,857. The train-leak check removed 0.
- Final files: train 7,343 (2,000 each of CRITICAL, HIGH, MEDIUM and 1,343 LOW), validation 1,000, natural test 1,000, balanced test 600.
- The natural validation and test files hold only 17 and 26 LOW examples.

**What surprised me.** The label source shifts with time. Final train has 6,170 NVD and 1,173 CNA labels (84.0% NVD). The natural test file has 379 NVD and 621 CNA labels (62.1% CNA). The model will learn mostly from NVD-style labels and be tested mostly on CNA-style ones. I have not found out why the mix shifts.

**What I learned.** I will report macro-F1 next to accuracy, because it treats every class equally, and I will look at NVD and CNA test examples separately. The ATT&CK dataset was rebuilt at the same time and has the same counts as before. The cleaner was not re-run, so its stats file still shows the earlier run without the NONE step.

---

## Session 8: Redesigning the documentation

**Goal.** Make the docs a living story: every claim backed by a file, every pending result marked "Pending".

**What I did.** Rewrote the root README and all stage pages; added this journal, [errors-and-fixes.md](errors-and-fixes.md), [metrics.md](metrics.md) and `src/make_figures.py`, which draws the charts in `docs/images/` from the stats files. I also made `clean_cves.py` write `data/processed/stats_clean.json`, so the funnel numbers have a file behind them. `collect.py` got the file check from E06.

**A note on provenance.** The current `stats_clean.json` was filled by hand from the console output of my earlier cleaning run, because that run did not write a stats file. I re-counted the severity and source numbers from `cves_clean.jsonl` and they match. Running `clean_cves.py` again replaces the file with script-written numbers.

**What I learned.** Docs rot unless updating them is a rule. The rule now sits at the top of [docs/README.md](README.md).

---

## Next

- Re-run `clean_cves.py` so `stats_clean.json` is written by the script and includes the NONE step.
- Understand the NVD/CNA shift in the labels.
- Smoke test on Colab ([stage 05](stage-05-smoke-test/README.md)), then the main run.
