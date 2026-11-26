---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
language:
  - en
tags:
  - cybersecurity
  - qlora
  - lora
  - cve
  - mitre-attack
---

# cyber-slm (LoRA adapter)

> **Status: draft. No model has been trained or evaluated yet.** Every `TODO` below must be replaced with a real value before publishing. `src/upload_to_hub.py` refuses to upload a card that still contains `TODO`.

A LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned with 4-bit QLoRA on two small cybersecurity tasks:

1. **CVE severity.** Given a CVE description, answer one of `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` (the CVSS v3 base severity).
2. **ATT&CK technique naming.** Given a description of attacker behaviour, answer the technique as `<id> <name>`, for example `T1566.001 Spearphishing Attachment`.

Code and docs: https://github.com/preetik92002-tech/llm-finetune-cyber

## Intended use

- **Intended:** defensive and educational work. Examples: triaging public vulnerability text, labelling threat reports with ATT&CK techniques, learning how CVE severity and ATT&CK labelling work, and research on small models.
- **Data:** public data only. Do not send private, confidential or customer data to this model.
- **Not intended:**
  - the only basis for a patching or risk decision
  - replacing the official CVSS score or analyst judgement
  - generating exploits, attack tooling or instructions for attacking systems

## Data

| Source | What was used | Link |
|---|---|---|
| NVD (National Vulnerability Database) | CVEs published 2022 to 2025: English description, CVSS v3.x base severity (NVD score first, CNA score if NVD has none), CWE ids | https://nvd.nist.gov |
| MITRE ATT&CK Enterprise (STIX) | "uses" relationships from groups, malware and tools to techniques, with their description text | https://attack.mitre.org |

ATT&CK® is a registered trademark of The MITRE Corporation. The ATT&CK data is used under MITRE's terms of use.

Processing (see `docs/stage-03-data-cleaning` and `docs/stage-04-dataset-building` in the repo):

- CVEs: rejected CVEs, CVEs without a v3 severity and descriptions under 30 characters were removed. HTML was stripped. Exact and near-duplicates (MinHash LSH on word 3-shingles, threshold 0.85) were removed.
- CVE split by publication date: train 2022-01-01 to 2024-09-30, validation 2024-10-01 to 2024-12-31, test 2025. Train is capped per severity class; validation and test keep the natural mix.
- ATT&CK: citations and link addresses were removed. The split is by group or software, so none appears in two splits.

Dataset sizes:

| Dataset | Train | Validation | Test |
|---|---|---|---|
| ATT&CK | 13,139 | 1,642 | 1,642 |
| Severity | 7,343 (2,000 each CRITICAL, HIGH, MEDIUM; 1,343 LOW) | 1,000 | 1,000 natural, plus 600 balanced (150 per class) |

The ATT&CK numbers come from `data/processed/stats.json`. Of the techniques in validation and test, 9 each do not appear in train. The severity numbers come from `data/processed/stats_cve.json`. CNA scores are 16.0% of the final train set but 62.1% of the natural test set (see Limitations).

## Training setup

Planned settings, from `configs/qlora_main.yaml`. Confirm each against the real run before publishing.

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Method | QLoRA: 4-bit NF4 base weights, LoRA adapters on attention and MLP projection layers |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | 2e-4, cosine schedule, 30 warmup steps |
| Epochs | 2 |
| Max sequence length | 768 tokens |
| Effective batch size | 16 (4 per device x 4 accumulation) |
| Loss | On the assistant answer only |
| ATT&CK examples mixed in | up to 4000 (`max_attack_samples`) |
| Seed | 42 |
| Hardware | TODO |
| Libraries and versions | TODO |

## Evaluation

Run with `src/evaluate.py`. Metrics are on the 2025 severity test set and the ATT&CK test set (split by group). Parsing: the first severity word or ATT&CK id in the output; outputs with none are counted as wrong and reported as unparseable.

| Metric | Zero-shot base | Few-shot base | This adapter |
|---|---|---|---|
| Severity accuracy | TODO | TODO | TODO |
| Severity macro-F1 | TODO | TODO | TODO |
| Severity unparseable outputs | TODO | TODO | TODO |
| Severity macro-F1, NVD labels only | TODO | TODO | TODO |
| Severity macro-F1, CNA labels only | TODO | TODO | TODO |
| ATT&CK exact-match accuracy | TODO | TODO | TODO |
| ATT&CK parent-technique accuracy | TODO | TODO | TODO |
| ATT&CK unparseable outputs | TODO | TODO | TODO |
| Fake CVE ids answered with a severity (hallucination check) | TODO | TODO | TODO |

Ablations (LoRA rank 8, 16, 64; 500, 2000, 5000 training examples): TODO: no runs yet. Summarise `results/ablations.md` once it exists.

## Limitations

- Labels are CVSS scores, which are partly a judgement and can differ between NVD and CNAs. The model learns those labels, not the truth about a flaw.
- A CVE description does not always contain what the score is based on, so some severity errors cannot be avoided from text alone.
- The label source shifts between splits: CNA scores are 16.0% of the final train set but 62.1% of the natural test set. A lower test score may come from this shift and not from the model. Per-source metrics (NVD only, CNA only) are reported next to the overall ones.
- The natural validation and test sets hold only 17 and 26 LOW examples, so LOW scores there are noisy. The balanced test set has 150 per class.
- CVEs without a v3 score were dropped, which may bias the data.
- The model was trained to always answer with a label. It may give a confident severity or technique for input it cannot know, including made-up ids. See the hallucination row above.
- ATT&CK covers the Enterprise matrix only. Each example has a single technique as the answer, while real behaviour often matches several.
- Technique ids and names in the answers can be wrong or outdated as ATT&CK changes between versions.
- Small model (1.5B parameters) and a small, partly capped training set. Results on other data, languages or newer CVEs may differ.
- English only.

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "TODO-your-hf-username/TODO-repo-name")

messages = [
    {"role": "system", "content": "You are a cybersecurity analyst. Given a CVE description, classify its CVSS v3 base severity as one of: CRITICAL, HIGH, MEDIUM, LOW."},
    {"role": "user", "content": "CVE description: <paste a public CVE description>"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```
