# IntentScope

## Fine-grained banking-intent classification with DistilBERT

IntentScope is an offline-first machine-learning experiment for routing short banking-support queries to one of 77 operational intents. It fine-tunes DistilBERT and treats model quality as more than a single headline number: checkpoint selection is validation-led, final predictions are evaluated independently, and error analysis is retained as a first-class artifact.

The project addresses a practical product problem. Support utterances such as “my card was declined,” “my cash withdrawal is pending,” and “I do not recognise this payment” can be lexically similar while requiring different downstream workflows. Fine-grained intent classification tests whether a compact Transformer can learn those domain-specific distinctions from limited labelled data.

## Objective

> How effectively can a pretrained Transformer separate highly similar banking-support intents after supervised fine-tuning on a small, balanced domain dataset?

## Experiment design

| Component | Decision |
| --- | --- |
| Dataset | [Banking77](https://huggingface.co/datasets/PolyAI/banking77), 77 English banking intents |
| Data split | Official train/test split retained; 15% stratified validation split drawn only from training data |
| Model | `distilbert-base-uncased` with a 77-class sequence-classification head |
| Optimizer | AdamW, learning rate `2e-5`, weight decay `0.01` |
| Training | Dynamic padding, gradient clipping, linear warmup/decay, fixed seed |
| Checkpoint policy | Save the model with the highest validation macro F1 |
| Evaluation | Manual NumPy metrics verified exactly against scikit-learn |

The experiment design supports a 100-epoch maximum when configured. The completed GPU run used a 25-epoch budget and selected the best validation checkpoint rather than assuming the final epoch was optimal.

## Reported result

The fresh GPU run reached its best validation result at epoch 14:

| Metric | Value |
| --- | ---: |
| Validation accuracy | **0.9227** |
| Validation macro F1 | **0.9250** |

This result is credible for this setting because validation is stratified across all 77 intents, model selection uses validation macro F1 rather than training loss, and performance rises steadily before plateauing. Training loss continued to decrease after epoch 14 while validation F1 did not; this is evidence of emerging overfitting and is why the epoch-14 checkpoint is retained.

| Final test metric | Value |
| --- | ---: |
| Accuracy | **0.9182** |
| Macro F1 | **0.9183** |
| Weighted F1 | **0.9183** |

The final test evaluation used the epoch-14 checkpoint once, after validation-based selection. The test set was not used to select epochs or hyperparameters. The test macro F1 is within 0.0067 of the held-out validation macro F1, supporting the conclusion that performance transferred rather than being specific to one split.

### Training-efficiency finding

For this experiment, validation macro F1 peaked at epoch 14 (**0.9250**) and did not improve during the remaining epochs through the 25-epoch budget. This makes 25 epochs a practical upper bound for the current model, data split, and hyperparameters: training longer would add GPU time and cost without an observed validation benefit. This conclusion is specific to this configuration; a different model, dataset, or optimization schedule should be re-evaluated rather than assumed to follow the same curve.

## Why macro F1 is the primary metric

Accuracy reports the fraction of correctly routed queries. Macro F1 is the primary model-selection metric because it computes F1 separately for every intent and averages them equally. It prevents strong performance on easy intents from hiding weak performance on difficult or easily confused ones. Because Banking77’s official test split has 40 examples per intent, macro and weighted F1 are expected to be very similar.

## Evaluation and error analysis

The notebook writes auditable artifacts for:

- per-intent precision, recall, F1, and support;
- the full confusion matrix and high-frequency directional confusions;
- all test predictions and misclassified examples;
- confidence distributions for correct and incorrect classifications; and
- a TF-IDF + logistic-regression baseline.

Keeping measured outputs separate from later interpretation helps make the reported score reproducible and appropriately cautious.

## Run on Kaggle with Internet disabled

1. Upload [IntentScope_Kaggle.ipynb](IntentScope_Kaggle.ipynb) to Kaggle.
2. Attach a Kaggle Dataset containing `banking77/` CSV assets and `distilbert-base-uncased/` model assets.
3. Select a GPU accelerator and keep Internet disabled.
4. Run all cells. Results are written to `/kaggle/working/`.
5. Copy the CSV, JSON, and figure artifacts into `results/` before publishing.

The current notebook configuration uses batch size 64, 25 epochs, and `num_workers=0` for stable Kaggle execution. To use a 100-epoch maximum, change `epochs` in `Config`; the best validation checkpoint remains the one used for final evaluation.

## Repository layout

```text
IntentScope_Kaggle.ipynb        Kaggle-ready training and evaluation notebook
scripts/build_notebook.py       Auditable notebook generator
scripts/prepare_offline_bundle.py
requirements-kaggle.txt         Runtime compatibility guidance
OFFLINE_KAGGLE.md               Offline asset preparation notes
results/                        Published metrics, predictions, and figures
```

## Limitations and next steps

- Banking77 is a narrow English-language benchmark; results do not establish production reliability across banks, regions, or languages.
- Softmax confidence is a ranking signal, not calibrated uncertainty.
- One split and one seed give a useful experiment, not an uncertainty estimate.
- The most direct next improvements are multi-seed evaluation, calibration analysis, targeted collection for confused intent pairs, and stronger offline-compatible encoders.
