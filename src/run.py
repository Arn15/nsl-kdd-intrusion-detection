"""End-to-end experiment runner.

Protocol
  1. Train on KDDTrain+ (minus a stratified validation split), test on the official KDDTest+.
  2. For each model, pick hyperparameters on the validation split only (seed 0).
  3. Re-train the chosen config with several seeds; evaluate each on KDDTest+ once.
  4. Report mean +/- std across seeds.

Usage
  python -m src.run --task 5class
  python -m src.run --task binary --models lr rf xgb
  python -m src.run --task 5class --quick          # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import models as M
from .data import capped_smote, make_split
from .evaluate import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
HEADLINE = ["accuracy", "macro_f1", "weighted_f1", "detection_rate", "false_positive_rate",
            "attack_auc", "novel_attack_detection_rate"]


def prepare(task, seed, imbalance, quick):
    s = make_split(task, seed=seed)
    X, y = s.X_train, s.y_train
    if quick:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=len(X) // 5, replace=False)
        X, y = X[idx], y[idx]
    if imbalance == "smote":
        X, y = capped_smote(X, y, seed=seed)
    return s, X.astype(np.float32), y


def select_config(name, s, X, y, epochs, verbose):
    best = None
    for params in M.GRIDS[name]:
        t0 = time.time()
        model = M.fit(M.build(name, params, len(s.class_names), seed=0, epochs=epochs, verbose=verbose),
                      X, y, s.X_val, s.y_val)
        val = compute_metrics(s.y_val, model.predict(s.X_val), model.predict_proba(s.X_val), s.class_names)
        print(f"  [select] {name} {params} -> val macroF1 {val['macro_f1']:.4f} ({time.time()-t0:.0f}s)")
        if best is None or val["macro_f1"] > best[1]:
            best = (params, val["macro_f1"])
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["5class", "binary"], default="5class")
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "clstm"], choices=list(M.GRIDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imbalance", choices=["smote", "none"], default="smote")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="20%% of training data, 3 epochs, 1 seed")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.quick:
        a.epochs, a.seeds = 3, a.seeds[:1]

    out = Path(a.out or ROOT / "results" / (a.task + ("_quick" if a.quick else "")))
    out.mkdir(parents=True, exist_ok=True)
    print(f"device for C-LSTM: {M.get_device()} | task={a.task} seeds={a.seeds} -> {out}")

    # ---- 1. hyperparameter selection on validation (seed 0) ----
    s0, X0, y0 = prepare(a.task, 0, a.imbalance, a.quick)
    print(f"train {X0.shape} (after resampling) | val {s0.X_val.shape} | test {s0.X_test.shape} "
          f"| novel-attack test rows {int(s0.test_is_novel.sum())}")
    chosen = {}
    for name in a.models:
        params, val_f1 = select_config(name, s0, X0, y0, a.epochs, a.verbose)
        chosen[name] = {"params": params, "val_macro_f1": val_f1}

    # ---- 2. multi-seed evaluation on KDDTest+ ----
    rows, per_class = [], []
    for seed in a.seeds:
        s, X, y = (s0, X0, y0) if seed == 0 else prepare(a.task, seed, a.imbalance, a.quick)
        for name in a.models:
            t0 = time.time()
            model = M.fit(M.build(name, chosen[name]["params"], len(s.class_names), seed=seed,
                                  epochs=a.epochs, verbose=a.verbose), X, y, s.X_val, s.y_val)
            pred, proba = model.predict(s.X_test), model.predict_proba(s.X_test)
            m = compute_metrics(s.y_test, pred, proba, s.class_names, s.test_is_novel)
            m.update(model=M.NAMES[name], seed=seed, train_seconds=round(time.time() - t0, 1))
            rows.append(m)
            print(f"  [test] seed {seed} {M.NAMES[name]:<20} acc {m['accuracy']:.4f} "
                  f"macroF1 {m['macro_f1']:.4f} novel-det {m.get('novel_attack_detection_rate', float('nan')):.3f}")

    # ---- 3. save ----
    (out / "per_seed.json").write_text(json.dumps(rows, indent=2))
    (out / "chosen_configs.json").write_text(json.dumps(chosen, indent=2))
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in rows])
    metric_cols = [c for c in df.columns if c not in ("model", "seed", "train_seconds")]
    summary = df.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")

    cols = [c for c in HEADLINE if c in df.columns]
    recall_cols = [f"recall_{c}" for c in s0.class_names]
    def fmt(model, c):
        mu, sd = summary.loc[model, (c, "mean")], summary.loc[model, (c, "std")]
        return f"{mu:.3f}" if np.isnan(sd) else f"{mu:.3f} ± {sd:.3f}"
    lines = [f"# NSL-KDD results — {a.task} (KDDTest+, {len(a.seeds)} seed(s))", "",
             "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {m} | " + " | ".join(fmt(m, c) for c in cols) + " |" for m in summary.index]
    lines += ["", "## Per-class recall", "",
              "| Model | " + " | ".join(s0.class_names) + " |", "|---" * (len(recall_cols) + 1) + "|"]
    lines += [f"| {m} | " + " | ".join(fmt(m, c) for c in recall_cols) + " |" for m in summary.index]
    lines += ["", "Hyperparameters chosen on validation (seed 0): see `chosen_configs.json`."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
