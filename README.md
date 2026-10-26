# IntentScope

**Fine-grained banking intent classification with an auditable data-quality and annotation workflow**

IntentScope classifies short banking-support messages into 77 operational intents. Queries about a delayed card, a declined card payment, and a pending transfer can use similar language while requiring different responses. The project combines a DistilBERT classification experiment with a separate workflow for inspecting dataset quality and recording human annotation decisions.

## Project at a glance

| Area | Approach |
| --- | --- |
| Dataset | [PolyAI Banking77](https://huggingface.co/datasets/PolyAI/banking77): 10,003 official training queries and 3,080 held-out test queries across 77 English intents |
| Validation | 15% stratified split drawn from the official training set with seed 42 |
| Model | `distilbert-base-uncased` with a 77-class classification head |
| Selection | Best checkpoint chosen by validation macro F1 |
| Evaluation | Independent metrics, per-intent analysis, confusion analysis, and a TF-IDF/logistic-regression baseline |
| Data review | Reproducible audit, targeted sampling, reviewer-specific local judgments, revision history, and traceable cleaning candidates |

## Model experiment

The completed GPU run used a 25-epoch budget. Validation macro F1 peaked at epoch 14, so the epoch-14 checkpoint was retained for final evaluation. The official test set was used only after that selection.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Accuracy | 0.9227 | 0.9182 |
| Macro F1 | 0.9250 | 0.9183 |

Macro F1 gives each intent equal weight, making it more informative than accuracy alone when inspecting fine-grained classes. The published [results](results/) retain training history, per-class metrics, predictions, errors, and an experiment summary. These scores describe the existing model run; the data-quality extension has not retrained the model or changed its results.

## Dataset quality and annotation design

The audit examines only the official training data and the validation subset derived from it. It checks blank text, missing or invalid labels, duplicate records, conservative whitespace/case matches, conflicting labels, train/validation overlap, class distribution, and text-length outliers. Stable record IDs retain original row provenance. Findings remain review candidates until interpreted; a normalization match or long query is not automatically a defect.

On the current Banking77 assets, the audit examined 10,003 training-source rows. It flagged 930 word-length outliers under the 1.5×IQR rule and four normalization-match groups affecting eight rows. It found no exact duplicate groups, identical-text label conflicts, or train/validation text overlap. The held-out test set was excluded from audit-driven decisions.

The review interface presents the original query and a searchable intent taxonomy without revealing the source label before a reviewer saves an initial judgment. Each reviewer enters a self-selected ID and receives a **separate local workspace** for their labels, ambiguity decisions, confidence, issue categories, rationales, progress, and revision history. The source label appears after that reviewer's first save, allowing a documented comparison or revision. Reviewer workspaces use local SQLite files; IDs separate records but are not authenticated accounts.

Sampling combines label coverage with audit-flagged examples using a fixed seed and records why each item was selected. Cleaning is an explicit export that preserves raw text and labels, logs every proposed change, and leaves unresolved findings visible. Reports distinguish automatic audit findings from completed human judgments. Source-label disagreement is treated as a review candidate rather than a confirmed labeling error.

## What this project demonstrates

- Validation-led model selection and a held-out final evaluation.
- Reproducible dataset inspection with split and row provenance.
- Reviewer-specific annotation storage, resumable decisions, and an audit trail.
- Conservative, traceable cleaning that does not silently relabel or delete data.
- Reporting that separates measured model results, automated findings, and human conclusions.

## Scope

Banking77 is an English, single-domain benchmark. The reported scores do not establish performance for other banks, languages, or unsupported intents. One training seed does not quantify run-to-run variation, and softmax confidence has not been calibrated. The review workflow supports human validation but does not claim annotation improvements, corrected labels, time savings, or model gains without completed reviews and a separate evaluation.

Implementation details and reproducibility commands are in [WORKFLOW.md](WORKFLOW.md). The [proposed annotation guidelines](ANNOTATION_GUIDELINES.md) and [offline Kaggle notes](OFFLINE_KAGGLE.md) provide supporting context.
