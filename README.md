# Multi-Class Network Intrusion Detection on NSL-KDD

Hybrid CNN-LSTM (C-LSTM) intrusion detection on NSL-KDD, compared against Random Forest, Logistic Regression and gradient-boosted baselines, with a focus on rare-attack detection and adversarial robustness.

> **Status:** being rebuilt into a single reproducible pipeline. Results below will be filled in from the re-run.

## Background
Originally built as a team project for NC State's Neural Networks & Deep Learning course (Spring 2026) and presented at the NC State AI Student Symposium 2026. Team: Asavari Hangal, Nidhi Grover, Arnav Yennam.
This repository is my independent rebuild and extension of that work. The original course notebooks are kept unchanged in `notebooks/*_original.ipynb` for reference.

## Evaluation protocol
- **Train:** KDDTrain+ (stratified 10% held out for validation). **Test:** the official **KDDTest+** split, used once per seed.
- **Tasks:** 5-category (Normal / DoS / Probe / R2L / U2R) and binary (attack vs. normal). KDDTest+ contains 17 attack types that never appear in training, so 23-class labels can't be evaluated on it.
- **Preprocessing** (one-hot, log1p on byte/duration features, scaling) and capped SMOTE are fit on training data only.
- **Model selection:** each model's hyperparameters are chosen on validation macro-F1 only; the chosen config is then re-trained with 3 seeds and reported as mean ± std.
- **Metrics:** accuracy, macro-F1, weighted F1, per-class precision/recall, attack detection rate, false-positive rate, attack ROC-AUC, and **novel-attack detection rate** (share of test attacks unseen in training that are still flagged as attacks).

## Quickstart
```bash
pip install -r requirements.txt
bash data/download_data.sh
python -m src.run --task 5class --quick     # ~5 min smoke test
python -m src.run --task 5class             # full run (GPU recommended for C-LSTM)
python -m src.run --task binary
```
Results land in `results/<task>/` (`results.md`, `summary.csv`, `per_seed.json`, `chosen_configs.json`). On Colab, use `notebooks/03_run_pipeline.ipynb`.

## Roadmap
- [x] Single pipeline in `src/`; every reported number comes from one run
- [x] Official KDDTest+ evaluation, 5-category and binary
- [x] Validation-only model selection; multi-seed reporting
- [x] Baselines: Logistic Regression, Random Forest, XGBoost
- [x] Novel-attack (zero-day) detection rate
- [ ] Adversarial robustness (feature-constrained FGSM/PGD) and adversarial training
- [ ] SHAP explanations per attack category

## Results
_To be added after the re-run._

## Repository layout
```
data/        download script (data not committed)
notebooks/   original course notebooks + new analysis notebooks
src/         pipeline: data.py, models.py, evaluate.py, run.py
results/     figures and metrics
docs/        symposium poster
```
