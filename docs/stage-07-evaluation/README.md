# Stage 07: Evaluation

**Status:** In progress. `src/evaluate.py` is written. I have not run it on a model yet.

## Goal

I measure the base model and the fine-tuned model on the test files, and compare them.
Fine-tuning only counts if it beats the [baseline](../glossary.md#baseline).

## Concepts

- Three modes: **zero-shot** (just the question), **few-shot** (a few worked examples from the train files in the prompt) and **fine-tuned** (base model plus the adapter).
- **Accuracy** is the share of right answers. With uneven classes it can hide problems (LOW is about 2% of the data, see [stage 03](../stage-03-data-cleaning/README.md)).
- [Macro-F1](../glossary.md#macro-f1) averages the F1 score of each class, so a rare class counts as much as a common one.
- A [confusion matrix](../glossary.md#confusion-matrix) shows which class is mistaken for which.
- For ATT&CK I report exact-match accuracy on the technique id and parent-technique accuracy (`T1566` for `T1566.001`).
- Outputs that I cannot read as a label count as wrong and are counted separately as "unparseable".
- The [hallucination](../glossary.md#hallucination) check asks for the severity of CVE ids that do not exist (years 2090 to 2099). A good model says it does not know.
- I will run the severity test twice: on the natural test file and on `severity_test_balanced.jsonl`.

## Flowchart

```mermaid
flowchart LR
    A["Test files"] --> B["Zero-shot base model"]
    A --> C["Few-shot base model"]
    A --> D["Base + adapter"]
    B --> E["Parse outputs"]
    C --> E
    D --> E
    E --> F["Accuracy, macro-F1, confusion matrix, unparseable count"]
    E --> G["ATT&CK exact and parent accuracy"]
    D --> H["Hallucination check with fake CVE ids"]
    F --> I["results/*.json and plots"]
    G --> I
    H --> I
```

## What the code does

[src/evaluate.py](../../src/evaluate.py):

- `load_model()`: loads the base model in 4-bit and attaches the adapter in fine-tuned mode.
- `pick_shots()` and `build_messages()`: build few-shot prompts from the train files.
- `generate()`: greedy generation in batches with the model's chat template.
- `parse_severity()` and `parse_technique()`: take the first severity word or technique id in the output, ignoring case.
- `severity_metrics()`: accuracy, accuracy on parseable outputs, macro-F1, per-class scores and the confusion matrix.
- `attack_metrics()`: exact-match and parent-technique accuracy, and how many predicted ids do not exist in my data.
- `hallucination_check()`: asks about made-up CVE ids and counts how often the model gives a severity anyway.
- Saves `results/<name>.json`, a predictions file and the plots.

## How to run

On Colab:

```
python src/evaluate.py --mode zero-shot --model Qwen/Qwen2.5-1.5B-Instruct
python src/evaluate.py --mode few-shot --model Qwen/Qwen2.5-1.5B-Instruct --few-shot-k 4
python src/evaluate.py --mode fine-tuned --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/qlora_main
```

To use the balanced test file: add `--severity-test data/processed/severity_test_balanced.jsonl --name <new name>`. For a first check without severity data, add `--task attack`.

## Results and metrics

| Metric | Zero-shot | Few-shot | Fine-tuned |
|---|---|---|---|
| Severity accuracy | Pending | Pending | Pending |
| Severity macro-F1 | Pending | Pending | Pending |
| Severity unparseable | Pending | Pending | Pending |
| ATT&CK exact match | Pending | Pending | Pending |
| ATT&CK parent-technique | Pending | Pending | Pending |
| ATT&CK unparseable | Pending | Pending | Pending |
| Fake CVE ids answered with a severity | Pending | Pending | Pending |

The chart `docs/images/evaluation_scores.png` is created by `python src/make_figures.py` after evaluation runs exist.

## Errors faced in this stage

- [E09](../errors-and-fixes.md#e09): the evaluation script also needs the Colab packages.

## What I learned

- Pick the metric before looking at results, and choose one that fits uneven classes.
- Count what I cannot parse. Hiding it would flatter the model.
- A baseline is the only way to know if fine-tuning helped.

## Navigation

Previous: [Stage 06: QLoRA training](../stage-06-qlora-training/README.md) | [Home](../../README.md) | Next: [Stage 08: Ablations](../stage-08-ablations/README.md)
