# Published run artifacts

Copy the following files from `/kaggle/working/` after the final test evaluation:

- `experiment_summary.json`
- `training_history.csv`
- `per_class_metrics.csv`
- `model_comparison.csv`
- `test_predictions.csv`
- `misclassified_examples.csv`

Do not commit `best_model/` or the pretrained offline assets. These files are large, can exceed GitHub limits, and are reproducible from the listed model source and Kaggle Dataset attachment.
