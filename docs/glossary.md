# Glossary

Simple definitions for the terms used in this repo. Home: [README](../README.md)

## Ablation
An experiment where you change one thing at a time (for example, remove deduplication) and compare results, to learn which part of the pipeline matters. Used in [stage 08](stage-08-ablations/README.md).

## API key
A secret string that identifies you to a web service. The NVD gives out a free one that allows more requests per minute. Keep it in `.env`, never in code.

## Base severity
The label CVSS gives a vulnerability: `CRITICAL`, `HIGH`, `MEDIUM` or `LOW` (`NONE` also exists for score 0). It comes from the [CVSS](#cvss) base score. This is what the severity task asks the model to predict.

## Baseline
The result you get before any fine-tuning (or from a very simple method). A fine-tuned model is only useful if it beats it.

## Chat format
Training examples written as a list of messages with roles: `system` (instructions), `user` (the input) and `assistant` (the answer the model should give). Here every line of the dataset files is `{"messages": [...]}`.

## Class balancing
Making each label appear a similar number of times in the training data, so the model does not just learn to always say the most common label. Here: a cap per severity class in the train set.

## CNA
CVE Numbering Authority. An organisation (often the software vendor) allowed to assign CVE ids. A CNA can also give its own CVSS score. See [CVE](#cve).

## CVE
Common Vulnerabilities and Exposures. A public id (like `CVE-2024-12345`) plus a short description for one known security flaw.

## CVSS
Common Vulnerability Scoring System. A way to score how serious a flaw is, from 0 to 10. Version 3.x scores give a [base severity](#base-severity). This repo uses v3.x only.

## CWE
Common Weakness Enumeration. A list of kinds of software mistakes, such as `CWE-79` (cross-site scripting). A CVE can be linked to one or more CWE ids.

## Data leakage
When information from the test data gets into the training data (for example, the same text in both). Scores then look better than they really are.

## Data split
Dividing data into three parts: **train** (the model learns from it), **validation** (used to tune choices) and **test** (used once at the end to judge the model).

## Exponential backoff
After a failed request, wait a short time, then wait twice as long after each next failure, up to a limit. This repo waits 10, 20, 40, 80, 160 seconds, then 300 seconds at most.

## Environment variable (`.env`)
A named setting kept outside the code. A `.env` file holds them for local use. Here: `NVD_API_KEY`. The file is git-ignored.

## Fine-tuning
Training an already-trained model a bit more on your own examples so it gets better at your task.

## Hallucination
When a model gives a confident answer that is wrong or made up, such as a technique id that does not exist.

## Intrusion set
MITRE's name for a group of attackers (a "threat group"), such as APT29. In the data it is the source of some `uses` relationships.

## Jaccard similarity
A score from 0 to 1 for how much two sets overlap: size of what they share divided by size of everything in either. 1 means identical sets.

## JSONL
JSON Lines. A text file with one JSON object per line. Easy to read one line at a time, so it saves memory.

## LoRA
A fine-tuning method that trains small extra matrices instead of changing all the model's weights. It needs far less memory.

## LSH
Locality-Sensitive Hashing. A trick to find similar items fast without comparing every pair. See [MinHash](#minhash).

## MinHash
A short fingerprint of a set (here, a set of word [shingles](#shingle)). Two similar texts have similar fingerprints, so [Jaccard similarity](#jaccard-similarity) can be estimated quickly.

## MITRE ATT&CK
A public knowledge base of how attackers behave, organised into tactics (goals) and techniques (ways to reach the goals). This repo uses the Enterprise matrix.

## Model card
A short document describing a model: what it was trained on, how it was tested, what it is good and bad at, and its limits. See [stage 09](stage-09-results-and-model-card/README.md).

## Near-duplicate
Two texts that are not identical but almost the same (for example, one word changed). Removed so the model does not memorise and so the test set stays fair.

## NVD
National Vulnerability Database. The US government site that adds scores and details to CVE entries and offers them through a web API.

## Overfitting
When a model learns its training examples too closely and does badly on new ones.

## QLoRA
[LoRA](#lora) on top of a model that is stored in a compressed (4-bit) form. It lets you fine-tune a model on a laptop-sized GPU.

## Quantization
Storing a model's numbers with fewer bits (for example 4 instead of 16) to save memory, at a small cost in accuracy.

## Rate limit
A cap on how many requests you may send in a time window. The NVD allows 5 requests per 30 seconds without a key and 50 with one.

## Shingle
A small piece of text used for comparing documents. Word 3-shingles are every run of 3 words in a row: "the quick brown fox" gives "the quick brown" and "quick brown fox".

## SLM
Small language model. A language model small enough to run on modest hardware.

## Smoke test
A tiny, fast run that only checks the whole pipeline works from start to end, before spending hours on the real run. See [stage 05](stage-05-smoke-test/README.md).

## STIX
Structured Threat Information Expression. A JSON format for sharing threat data. The ATT&CK download is a STIX "bundle": one file with a list of objects (techniques, groups, relationships and so on).

## Technique / sub-technique
An ATT&CK entry for one way of attacking, with an id like `T1566` (technique) or `T1566.001` (sub-technique, a more specific version).

## Virtual environment
A private folder of Python packages for one project, so its versions do not clash with other projects. Created with `python -m venv .venv`.

## Class imbalance
When some labels are much rarer than others. Here LOW is about 2% of the cleaned CVEs. Plain accuracy can look good even if the model never predicts the rare class. See [macro-F1](#macro-f1).

## Confusion matrix
A table that shows, for each true label, how often the model predicted each label. The diagonal holds the right answers.

## Epoch
One full pass over the training data. Training for 2 epochs means the model sees every example twice.

## Label source
Who gave a CVE its CVSS score: the NVD's own analysts (`nvd`) or the CNA, often the vendor (`cna`). This repo prefers the NVD score and stores which one it used.

## Macro-F1
An average of the F1 score of each class. F1 mixes precision (how many predicted items were right) and recall (how many true items were found). Every class counts the same, so a rare class is not hidden.
