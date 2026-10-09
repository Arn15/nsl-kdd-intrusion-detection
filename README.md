# Multi-Class Network Intrusion Detection on NSL-KDD

Hybrid CNN-LSTM (C-LSTM) intrusion detection on NSL-KDD, compared against Random Forest, Logistic Regression and gradient-boosted baselines, with a focus on rare-attack detection and adversarial robustness.

> **Status:** being rebuilt into a single reproducible pipeline. Results below will be filled in from the re-run.

## Background
Originally built as a team project for NC State's Neural Networks & Deep Learning course (Spring 2026) and presented at the NC State AI Student Symposium 2026. Team: Asavari Hangal, Nidhi Grover, Arnav Yennam.
This repository is my independent rebuild and extension of that work. The original course notebooks are kept unchanged in `notebooks/*_original.ipynb` for reference.

## Planned work (this repo)
- [ ] Single pipeline in `src/` (data → train → eval); every number in the results table comes from one run
- [ ] Evaluation on the official **KDDTest+** split: 5-category (Normal/DoS/Probe/R2L/U2R) and binary
- [ ] Model selection on validation only; test set used once
- [ ] Baselines: Logistic Regression, Random Forest, XGBoost/LightGBM
- [ ] 3–5 seeds, mean ± std; per-class recall for R2L/U2R
- [ ] Novel-attack (zero-day) analysis on KDDTest+ attack types unseen in training
- [ ] Adversarial robustness (feature-constrained FGSM/PGD) and adversarial training
- [ ] SHAP explanations per attack category

## Quickstart
```bash
pip install -r requirements.txt
bash data/download_data.sh
```

## Results
_To be added after the re-run._

## Repository layout
```
data/        download script (data not committed)
notebooks/   original course notebooks + new analysis notebooks
src/         pipeline code
results/     figures and metrics
docs/        symposium poster
```
