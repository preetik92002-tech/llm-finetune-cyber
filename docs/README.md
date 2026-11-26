# Documentation index

> **Update rule: after every future change to code or data, update the affected docs section and add an entry to [journal.md](journal.md).** Numbers must come from files in this repo (`data/processed/stats_*.json`, `results/*.json`, `configs/`). If something has not run yet, write "Pending". Never put secrets, tokens, e-mail addresses or local paths with a user name in the docs.

Home: [project README](../README.md)

## Start here

| Page | What it is |
|---|---|
| [journal.md](journal.md) | The learning story, one entry per work session: what I tried, what went wrong, the cause, the fix, what I learned |
| [errors-and-fixes.md](errors-and-fixes.md) | Table of every error: symptom, text, cause, fix, check, lesson |
| [metrics.md](metrics.md) | All metrics by stage. Real values only, otherwise "Pending" |
| [glossary.md](glossary.md) | Simple definitions of the terms used |
| [interview-qa.md](interview-qa.md) | Interview questions (empty for now) |

## Stages

| Stage | Topic | Status |
|---|---|---|
| [01](stage-01-setup/README.md) | Setup | Done |
| [02](stage-02-data-collection/README.md) | Data collection | Done |
| [03](stage-03-data-cleaning/README.md) | Data cleaning | Done (re-run Pending) |
| [04](stage-04-dataset-building/README.md) | Dataset building | Done |
| [05](stage-05-smoke-test/README.md) | Smoke test | In progress |
| [06](stage-06-qlora-training/README.md) | QLoRA training | In progress |
| [07](stage-07-evaluation/README.md) | Evaluation | In progress |
| [08](stage-08-ablations/README.md) | Ablations | In progress |
| [09](stage-09-results-and-model-card/README.md) | Results and model card | In progress |
| [10](stage-10-extensions/README.md) | Extensions | Todo |

Each stage page has the same sections: goal, concepts, flowchart, what the code does, how to run, results and metrics, errors faced, what I learned, navigation.

## Charts

Charts in `docs/images/` are made by `python src/make_figures.py` from the stats and results files. A chart is skipped when its input file does not exist yet. Run the script again after every run that changes a stats or results file.

| Image | Shows |
|---|---|
| `images/cve_funnel.png` | CVEs left after each cleaning step |
| `images/severity_per_split.png` | Severity class balance |
| `images/label_source_per_split.png` | NVD vs CNA label source |
| `images/attack_splits.png` | ATT&CK split sizes |
| `images/training_loss.png` | Training loss (appears after a training run) |
| `images/evaluation_scores.png` | Evaluation scores (appears after an evaluation run) |
