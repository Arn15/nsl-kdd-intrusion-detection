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
             "rf_bb": "Random Forest ", "xgb_bb": "XGBoost ",          # trailing space: separate rows
             "clstm": "C-LSTM", "clstm_at": "C-LSTM + adversarial training"}
    for eps in a.eps:
        t0 = time.time()
        if eps == 0:
            adv = {"clstm": X_att, "lr": X_att, "clstm_at": X_att}
        else:
            adv = {"clstm": A.pgd_to_normal(A.clstm_logit_fn(models["clstm"]), X_att, mask, fs, eps, a.steps, device=device),
                   "lr": A.pgd_to_normal(A.lr_logit_fn(models["lr"], device), X_att, mask, fs, eps, a.steps, device=device),
                   "clstm_at": A.pgd_to_normal(A.clstm_logit_fn(models["clstm_at"]), X_att, mask, fs, eps, a.steps, device=device)}
        detected = {
            "lr": (detection(models["lr"], adv["lr"]), "white-box"),
            "clstm": (detection(models["clstm"], adv["clstm"]), "white-box"),
            "clstm_at": (detection(models["clstm_at"], adv["clstm_at"]), "white-box"),
        }
        for tree in ("rf", "xgb"):
            # transfer: evades if either source's example evades
            d = detection(models[tree], adv["clstm"]) & detection(models[tree], adv["lr"])
            detected[tree] = (d, "transfer")
            # black-box: query the tree model's scores directly
            adv_bb = (X_att if eps == 0 else
                      A.random_search_to_normal(models[tree].predict_proba, X_att, mask, fs, eps, a.query_iters, seed=a.seed))
            if eps:
                chk = A.check_constraints(fs, adv_bb, X_att, mask)
                assert chk["illegal_changes"] == chk["out_of_range"] == chk["non_integer"] == 0, (tree, chk)
            detected[tree + "_bb"] = (detection(models[tree], adv_bb), "black-box (queries)")
        for name, (d, kind) in detected.items():
            r = {"eps": eps, "model": label[name], "attack": "none" if eps == 0 else kind,
                 "detection_rate": float(d.mean())}
            for c in A.CATEGORIES:
                r[f"det_{c}"] = float(d[cats == c].mean()) if (cats == c).any() else float("nan")
            rows.append(r)
        if eps:
            for src in adv:
                chk = A.check_constraints(fs, adv[src], X_att, mask)
                assert chk["illegal_changes"] == chk["out_of_range"] == chk["non_integer"] == 0, (src, chk)
        changed = np.abs(adv["clstm"] - X_att).max(1) > 1e-6 if eps else np.zeros(len(X_att), bool)
        print(f"  eps {eps:.2f}: " + "  ".join(f"{label[n].strip()[:12]}/{k[:5]} {d.mean():.3f}" for n, (d, k) in detected.items())
              + f"  | rows changed {changed.mean():.2f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "detection_vs_eps.csv", index=False)
    (out / "clean_metrics.json").write_text(json.dumps(clean, indent=2))
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    # ---- report ----
    piv = df.pivot_table(index="model", columns="eps", values="detection_rate", sort=False)
    lines = ["# Adversarial robustness — binary detectors on KDDTest+", "",
             f"Attack detection rate on the {len(X_att)} attack connections in KDDTest+ (higher is better) "
             f"under constrained evasion attacks. eps = max change per modifiable feature as a fraction of its "
             f"valid range. White-box PGD ({a.steps} steps) for C-LSTM and logistic regression; transfer from "
             f"both to the tree models. Adversarial training used eps = {a.train_eps}.", "",
             "| Model | Attack | " + " | ".join(f"eps={e:g}" for e in piv.columns) + " |",
             "|---|---|" + "---|" * len(piv.columns)]
    kind_of = {m: df[(df.model == m) & (df.eps > 0)].attack.iloc[0] if (df.eps > 0).any() else "-" for m in piv.index}
    for m in piv.index:
        lines.append(f"| {m.strip()} | {kind_of[m]} | " + " | ".join(f"{v:.3f}" for v in piv.loc[m]) + " |")
    show_eps = [e for e in a.eps if e > 0][len([e for e in a.eps if e > 0]) // 2] if any(e > 0 for e in a.eps) else 0
    sub = df[df.eps == show_eps]
    lines += ["", f"## Per attack category at eps = {show_eps:g}", "",
              "| Model | " + " | ".join(c.upper() for c in A.CATEGORIES) + " |", "|---|" + "---|" * 4]
    for _, r in sub.iterrows():
        lines.append(f"| {r.model.strip()} ({r.attack}) | " + " | ".join(f"{r[f'det_{c}']:.3f}" for c in A.CATEGORIES) + " |")
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
        ax.plot(piv.columns, piv.loc[m], marker="o", label=f"{m.strip()} ({kind_of[m]})",
                linestyle={"transfer": ":", "black-box (queries)": "--"}.get(kind_of[m], "-"))
    ax.set_xlabel("attack budget eps (fraction of feature range)")
    ax.set_ylabel("attack detection rate")
    ax.set_title("Detection under constrained evasion (KDDTest+)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "detection_vs_eps.png", dpi=150)


if __name__ == "__main__":
    main()
