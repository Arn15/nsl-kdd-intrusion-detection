"""Model definitions: classical baselines and the C-LSTM."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- C-LSTM ----
class CLSTM(nn.Module):
    """1D-CNN -> LSTM -> MLP head over the feature vector treated as a sequence.

    Same architecture as the original course notebook's baseline C-LSTM.
    """

    def __init__(self, n_features: int, n_classes: int, cnn_filters: int = 64,
                 lstm_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Linear(64, n_classes))

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TorchClassifier:
    """sklearn-style wrapper: fit with early stopping on validation macro-F1."""

    def __init__(self, n_classes: int, epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 512, patience: int = 5, seed: int = 0,
                 verbose: bool = True, **arch):
        self.n_classes, self.epochs, self.lr = n_classes, epochs, lr
        self.batch_size, self.patience, self.seed = batch_size, patience, seed
        self.verbose, self.arch = verbose, arch
        self.device = get_device()

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = CLSTM(X.shape[1], self.n_classes, **self.arch).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        loss_fn = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                            batch_size=self.batch_size, shuffle=True, generator=g)
        best, best_state, bad = -1.0, None, 0
        self.history = []
        t0 = time.time()
        for ep in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(bx), by)
                loss.backward()
                opt.step()
                total += loss.item() * len(bx)
            val_f1 = f1_score(y_val, self.predict(X_val), average="macro")
            sched.step(val_f1)
            self.history.append({"epoch": ep, "train_loss": total / len(X), "val_macro_f1": val_f1})
            if val_f1 > best:
                best, best_state, bad = val_f1, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
            if self.verbose:
                print(f"    ep {ep:2d} loss {total/len(X):.4f} val macroF1 {val_f1:.4f}")
            if bad >= self.patience:
                break
        self.model.load_state_dict(best_state)
        if self.verbose:
            print(f"    best val macroF1 {best:.4f} ({(time.time()-t0)/60:.1f} min)")
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        out = []
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(self.device)
            out.append(torch.softmax(self.model(xb), 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# -------------------------------------------------------------- registry ----
# Small hyperparameter grids; the best config per model is chosen on the
# validation split only.
GRIDS = {
    "lr":  [{"C": c} for c in (0.1, 1.0, 10.0)],
    "rf":  [{"n_estimators": 200, "max_depth": d} for d in (20, None)],
    "xgb": [{"n_estimators": n, "max_depth": d, "learning_rate": 0.1}
            for n, d in ((300, 6), (500, 8))],
    "clstm": [{"cnn_filters": 64, "lstm_hidden": 128, "dropout": 0.3, "lr": 1e-3},
              {"cnn_filters": 128, "lstm_hidden": 256, "dropout": 0.3, "lr": 1e-3}],
}
NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "clstm": "C-LSTM"}


def build(name: str, params: dict, n_classes: int, seed: int, epochs: int = 20, verbose: bool = False):
    if name == "lr":
        return LogisticRegression(max_iter=2000, random_state=seed, **params)
    if name == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed,
                             eval_metric="mlogloss", **params)
    if name == "clstm":
        p = dict(params)
        lr = p.pop("lr")
        return TorchClassifier(n_classes, epochs=epochs, lr=lr, seed=seed, verbose=verbose, **p)
    raise ValueError(name)


def fit(model, X, y, X_val, y_val):
    if isinstance(model, TorchClassifier):
        return model.fit(X, y, X_val, y_val)
    return model.fit(X, y)
