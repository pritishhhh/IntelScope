# Running IntentScope on Kaggle with Internet Off

Offline execution requires all remote assets to be attached before the session starts. Kaggle's preinstalled packages usually cover most dependencies, but the exact image changes over time, so the notebook begins with a version audit.

## 1. Prepare the asset bundle

On an internet-connected **Linux** machine using the same Python minor version as Kaggle:

```bash
python -m pip install -r requirements-kaggle.txt
python scripts/prepare_offline_bundle.py --output intentscope_offline_bundle
```

Do not prepare Linux wheels on Windows or macOS; compiled wheels are platform-specific. PyTorch/CUDA wheels are large and Kaggle normally already includes a compatible PyTorch build, so it is preferable to verify and retain Kaggle's preinstalled `torch` rather than replace it.

## 2. Upload and attach

Upload `intentscope_offline_bundle/` as a private Kaggle Dataset and attach it to the notebook. The expected content is:

```text
intentscope_offline_bundle/
├── banking77/
├── distilbert-base-uncased/
└── wheels/
```

The notebook searches recursively below `/kaggle/input`, so the Kaggle Dataset slug does not matter.

## 3. Verify packages

Run only the first notebook cell. If everything prints `[OK]`, continue. If a package is missing, set `INSTALL_MISSING_FROM_WHEELS = True`, rerun that cell, restart the session after installation, reset the flag, and run all cells.

Never use plain `pip install` in an offline session. The notebook uses `--no-index --find-links` so pip cannot contact PyPI.

## 4. GPU settings and outputs

Select a GPU accelerator in Kaggle Notebook settings. The notebook prints the actual GPU name; do not assume a particular accelerator from the notebook label. All outputs are written to `/kaggle/working`, which can be downloaded or committed as a Kaggle Notebook version.

