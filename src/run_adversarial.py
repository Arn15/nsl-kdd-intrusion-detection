"""Adversarial robustness of the binary (attack vs normal) detectors on KDDTest+.

  python -m src.run_adversarial                 # full study (GPU recommended)
  python -m src.run_adversarial --quick         # small smoke test

For each attack budget eps (fraction of each modifiable feature's valid range):
  * white-box PGD on C-LSTM and on logistic regression
  * transfer of those adversarial examples to Random Forest and XGBoost
    (a sample counts as evading if either source's version evades)
  * a C-LSTM retrained with PGD adversarial training, attacked white-box
Reports attack detection rate (higher = more robust), overall and per attack category.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import adversarial as A
from . import models as M
from .data import ATTACK_CATEGORY, make_split
from .evaluate import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = {"lr": {"C": 10.0}, "rf": {"n_estimators": 200, "max_depth": None},
                  "xgb": {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.1},
                  "clstm": {"cnn_filters": 128, "lstm_hidden": 256, "dropout": 0.3, "lr": 1e-3}}


def chosen_params() -> dict:
    p = ROOT / "results" / "binary" / "chosen_configs.json"
    if p.exists():
        return {k: v["params"] for k, v in json.loads(p.read_text()).items()}
    return DEFAULT_PARAMS


def detection(model, X) -> np.ndarray:
    return model.predict(X) != 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--train-eps", type=float, default=0.1, help="budget used for adversarial training")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--query-iters", type=int, default=300, help="black-box random-search iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.quick:   # smoke test: tiny data, small C-LSTM, few steps
        a.epochs, a.steps, a.eps, a.query_iters = 1, 5, [0.0, 0.1, 0.3], 20
    out = Path(a.out or ROOT / "results" / ("adversarial_quick" if a.quick else "adversarial"))
    out.mkdir(parents=True, exist_ok=True)
    device = M.get_device()
    print(f"device {device} | eps {a.eps} | train-eps {a.train_eps} | PGD steps {a.steps}")

    # Binary task: classes are already balanced, so no resampling (training rows keep their
    # attack labels, which the adversarial-training constraints need).
    s = make_split("binary", seed=a.seed)
    X, y, names = s.X_train, s.y_train, s.train_attack_names
    rng = np.random.default_rng(a.seed)
    if a.quick:
        idx = rng.choice(len(X), len(X) // 20, replace=False)
        X, y, names = X[idx], y[idx], names[idx]
    fs = A.build_feature_space(s)

    att = np.where(s.y_test == 1)[0]
    if a.quick:
        att = rng.choice(att, 500, replace=False)
    X_att = s.X_test[att]
    cats = np.array([ATTACK_CATEGORY[n] for n in s.test_attack_names[att]])
    mask = A.row_masks(fs, s.test_attack_names[att])

    params = chosen_params()
    if a.quick:
        params["clstm"] = {"cnn_filters": 16, "lstm_hidden": 32, "dropout": 0.3, "lr": 1e-3}
        params["rf"] = {"n_estimators": 50, "max_depth": None}
        params["xgb"] = {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1}
    models = {}
    for name in ("lr", "rf", "xgb", "clstm"):
        t0 = time.time()
        models[name] = M.fit(M.build(name, params[name], 2, seed=a.seed, epochs=a.epochs),
                             X, y, s.X_val, s.y_val)
        print(f"  trained {M.NAMES[name]} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    train_mask = A.row_masks(fs, names)
    models["clstm_at"] = A.train_clstm_adversarial(2, params["clstm"], X, y, train_mask, s.X_val, s.y_val, fs,
                                                   eps=a.train_eps, epochs=a.epochs, seed=a.seed)
    print(f"  trained adversarially-trained C-LSTM ({time.time()-t0:.0f}s)")

    clean = {}
    for name, m in models.items():
        met = compute_metrics(s.y_test, m.predict(s.X_test), m.predict_proba(s.X_test), s.class_names, s.test_is_novel)
        clean[name] = {k: met[k] for k in ("accuracy", "macro_f1", "detection_rate", "false_positive_rate",
                                           "attack_auc", "novel_attack_detection_rate")}

    rows = []
    label = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
             "clstm": "C-LSTM", "clstm_at": "C-LSTM + adversarial training"}
    diff = ("lr", "clstm", "clstm_at")                      # differentiable: white-box PGD possible
    logit = {"lr": lambda: A.lr_logit_fn(models["lr"], device),
             "clstm": lambda: A.clstm_logit_fn(models["clstm"]),
             "clstm_at": lambda: A.clstm_logit_fn(models["clstm_at"])}

    def check(name, Z):
        chk = A.check_constraints(fs, Z, X_att, mask)
        assert chk["illegal_changes"] == chk["out_of_range"] == chk["non_integer"] == 0, (name, chk)

    for eps in a.eps:
        t0 = time.time()
        # 1) white-box PGD (random start and zero start) on every differentiable model
        pgd = {}
        if eps:
            for src in diff:
                for rs in (True, False):
                    Z = A.pgd_to_normal(logit[src](), X_att, mask, fs, eps, a.steps, device=device,
                                        seed=a.seed, random_start=rs)
                    check(f"pgd-{src}", Z)
                    pgd[(src, rs)] = Z
        for name in models:
            if eps == 0:
                d = detection(models[name], X_att)
                per = {"white-box": d, "black-box": d, "transfer": d}
            else:
                # 2) black-box score-based queries on the model itself
                bb = A.random_search_to_normal(models[name].predict_proba, X_att, mask, fs, eps,
                                               a.query_iters, seed=a.seed)
                check(f"bb-{name}", bb)
                per = {"black-box": detection(models[name], bb)}
                if name in diff:
                    per["white-box"] = detection(models[name], pgd[(name, True)]) & detection(models[name], pgd[(name, False)])
                # 3) transfer: examples crafted on the other differentiable models
                tr = np.ones(len(X_att), bool)
                for (src, rs), Z in pgd.items():
                    if src != name:
                        tr &= detection(models[name], Z)
                per["transfer"] = tr
            worst = np.logical_and.reduce(list(per.values()))   # evaded if ANY attack evades
            r = {"eps": eps, "model": label[name], "detection_worst": float(worst.mean())}
            for k, v in per.items():
                r[f"detection_{k.replace('-', '_')}"] = float(v.mean())
            for c in A.CATEGORIES:
                r[f"det_{c}"] = float(worst[cats == c].mean()) if (cats == c).any() else float("nan")
            rows.append(r)
        print(f"  eps {eps:.2f} worst-case detection: " +
              "  ".join(f"{r['model'][:22]} {r['detection_worst']:.3f}"
                        for r in rows if r["eps"] == eps) + f"  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "detection_vs_eps.csv", index=False)
    (out / "clean_metrics.json").write_text(json.dumps(clean, indent=2))
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    # ---- report ----
    piv = df.pivot_table(index="model", columns="eps", values="detection_worst", sort=False)
    lines = ["# Adversarial robustness — binary detectors on KDDTest+", "",
             f"Attack detection rate on the {len(X_att)} attack connections in KDDTest+ (higher is better) under "
             f"constrained evasion. eps = max change per modifiable feature as a fraction of its valid range. "
             f"**Worst case**: a connection counts as evading if any attack succeeds. Attacks: white-box PGD "
             f"({a.steps} steps, random and zero start) on differentiable models, black-box score queries "
             f"({a.query_iters} iterations) on every model, and transfer from the other differentiable models. "
             f"Adversarial training used eps = {a.train_eps}.", "",
             "## Worst-case detection rate", "",
             "| Model | " + " | ".join(f"eps={e:g}" for e in piv.columns) + " |",
             "|---|" + "---|" * len(piv.columns)]
    for m in piv.index:
        lines.append(f"| {m} | " + " | ".join(f"{v:.3f}" for v in piv.loc[m]) + " |")
    pos = [e for e in a.eps if e > 0]
    for show in pos[1:3] if len(pos) > 2 else pos[-1:]:
        sub = df[df.eps == show]
        lines += ["", f"## By attack at eps = {show:g}", "",
                  "| Model | White-box PGD | Black-box queries | Transfer | Worst case |", "|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            wb = r.get("detection_white_box")
            wb = "–" if wb is None or np.isnan(wb) else f"{wb:.3f}"
            lines.append(f"| {r.model} | {wb} | {r.detection_black_box:.3f} | {r.detection_transfer:.3f} | "
                         f"{r.detection_worst:.3f} |")
    show = pos[len(pos) // 2] if pos else 0
    sub = df[df.eps == show]
    lines += ["", f"## Worst-case detection by attack category at eps = {show:g}", "",
              "| Model | " + " | ".join(c.upper() for c in A.CATEGORIES) + " |", "|---|" + "---|" * 4]
    for _, r in sub.iterrows():
        lines.append(f"| {r.model} | " + " | ".join(f"{r[f'det_{c}']:.3f}" for c in A.CATEGORIES) + " |")
    lines += ["", "## Clean performance (no attack)", "",
              "| Model | Accuracy | Macro-F1 | Detection | FPR | Novel-attack detection |", "|---|---|---|---|---|---|"]
    for n, c in clean.items():
        lines.append(f"| {label[n]} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} | {c['detection_rate']:.3f} | "
                     f"{c['false_positive_rate']:.3f} | {c['novel_attack_detection_rate']:.3f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in piv.index:
        ax.plot(piv.columns, piv.loc[m], marker="o", label=m)
    ax.set_xlabel("attack budget eps (fraction of feature range)")
    ax.set_ylabel("worst-case attack detection rate")
    ax.set_title("Detection under constrained evasion (KDDTest+)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "detection_vs_eps.png", dpi=150)


if __name__ == "__main__":
    main()
