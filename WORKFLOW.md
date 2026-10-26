# IntentScope reproducibility and review workflow

This guide contains setup and command details for the project. The [README](README.md) describes the research question, design, and measured results.

## Dataset assets

The quality workflow uses `categories.json` and `train.csv` from PolyAI Banking77. Local assets are stored by default in `intentscope-offline-assets/banking77/` and ignored by Git. Existing assets can be reused. On an internet-connected machine, the dataset-only preparation command is:

```bash
python -m pip install datasets
python scripts/prepare_quality_data.py
```

The training notebook additionally needs the offline model bundle described in [OFFLINE_KAGGLE.md](OFFLINE_KAGGLE.md). It remains separate from the quality workflow.

## Quality workflow

```bash
python -m pip install -r requirements-quality.txt
python scripts/quality_cli.py audit
python scripts/quality_cli.py sample --target 200 --flagged-fraction 0.25 --seed 42
streamlit run quality/review_app.py
```

The audit writes `quality-output/audit.json`, `records.json`, and `audit.md`. It mirrors the notebook's default seed 42 and 15% stratified validation split over original CSV row positions. `test.csv` is never loaded by the quality workflow. `--dataset-dir` and `--output-dir` can change the input and output locations.

Sampling writes `quality-output/batch.json` with unique record IDs, selection reasons, and counts. `--output quality-output/batch-special.json` creates another batch selectable in the app. Small datasets yield a smaller batch without repeating rows.

The app requires a self-selected reviewer ID. The same ID resumes that reviewer's workspace; a different ID opens a separate SQLite database under `quality-output/reviewers/`. These IDs organize local work and do not provide authentication or access control. Source labels are hidden until the current reviewer saves an initial judgment. The app saves revision history and exports only current, non-demo judgments.

For a reviewer ID such as `analyst-1`, exports and reports can be created with:

```bash
python scripts/quality_cli.py export --reviewer-id analyst-1
python scripts/quality_cli.py report --reviewer-id analyst-1
```

The outputs are stored inside that reviewer's local directory. The optional `--db` and `--output` arguments support a specific database or destination, including a pre-existing shared `quality-output/reviews.sqlite`. A report without `--reviewer-id` uses that legacy shared path. Human review results are counted only when saved; demo records are excluded.

## Cleaning

The default cleaning export makes no changes. The one optional text rule trims leading and trailing whitespace without changing interior wording:

```bash
python scripts/quality_cli.py clean --trim-outer-whitespace
```

The export writes `quality-output/cleaned/candidate.csv` and `changes.json`. It retains the original text, original label, source row, and unresolved flags. It does not relabel or remove examples. Rerun `report` after a review or cleaning export to refresh the evidence summary.

## Verification

```bash
python -m pytest tests/test_quality.py -q --basetemp .pytest-tmp
```

The tests cover duplicate and conflicting-label detection, matching the training notebook's validation split, stable sampling, held-out test exclusion, reviewer persistence and separation, export/report counts, demo exclusion, and original-data preservation.
