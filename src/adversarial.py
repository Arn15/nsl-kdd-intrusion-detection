"""Constrained evasion attacks on NSL-KDD intrusion detectors.

Threat model
------------
An attacker sends malicious traffic and wants it classified as *normal*. They can change
properties of their own connections, but:
  * they must not change the features that make the attack work ("functional" features).
    Following IDSGAN (Lin et al., 2022), per attack category:
        DoS, Probe : intrinsic + time-based traffic features are functional
        R2L, U2R   : intrinsic + content features are functional
    so only the remaining groups can be modified (e.g. host-based traffic features for DoS).
  * categorical features (protocol, service, flag) and binary flags are never modified
  * every modified value must stay valid: rates in [0, 1], counts/bytes non-negative
    integers no larger than seen in training.
The budget eps is an L-infinity bound expressed as a fraction of each feature's valid range.

White-box attacks (PGD) are run on the differentiable models (C-LSTM, logistic regression);
tree models (RF, XGBoost) are attacked by transferring the C-LSTM adversarial examples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import ATTACK_CATEGORY

# NSL-KDD feature groups (Tavallaee et al. 2009)
INTRINSIC = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
             "wrong_fragment", "urgent"]
CONTENT = ["hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
           "is_host_login", "is_guest_login"]
TIME_TRAFFIC = ["count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"]
HOST_TRAFFIC = ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
FUNCTIONAL = {
    "dos": INTRINSIC + TIME_TRAFFIC,
    "probe": INTRINSIC + TIME_TRAFFIC,
    "r2l": INTRINSIC + CONTENT,
    "u2r": INTRINSIC + CONTENT,
}
BINARY = {"land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"}
CATEGORIES = ["dos", "probe", "r2l", "u2r"]


@dataclass
class FeatureSpace:
    """Maps between raw feature values and the model's scaled input space."""
    names: list            # raw feature name per model column (None for one-hot columns)
    kind: list             # 'cat' | 'log' | 'num'
    mean: np.ndarray
    scale: np.ndarray
    lo: np.ndarray         # valid range, in scaled space
    hi: np.ndarray
    integer: np.ndarray    # bool: raw value must be an integer
    masks: dict            # category -> bool array of modifiable columns

    def to_raw(self, Z):
        R = Z * self.scale + self.mean
        log = np.array([k == "log" for k in self.kind])
        R[:, log] = np.expm1(R[:, log])
        return R

    def to_scaled(self, R):
        R = R.copy()
        log = np.array([k == "log" for k in self.kind])
        R[:, log] = np.log1p(np.clip(R[:, log], 0, None))
        return (R - self.mean) / self.scale

    def realise(self, Z_adv, Z_orig, mask_rows):
        """Snap perturbed rows to valid raw values (integers, ranges); untouched columns keep exact values."""
        R = self.to_raw(Z_adv)
        R[:, self.integer] = np.round(R[:, self.integer])
        R = np.clip(R, self.to_raw(np.tile(self.lo, (1, 1)))[0], self.to_raw(np.tile(self.hi, (1, 1)))[0])
        Z = self.to_scaled(R)
        return np.where(mask_rows, Z, Z_orig).astype(np.float32)


def build_feature_space(split) -> FeatureSpace:
    pre = split.preprocessor
    names, kind, mean, scale = [], [], [], []
    for name, trans, cols in pre.transformers_:
        if name == "cat":
            n_out = sum(len(c) for c in trans.categories_)
            names += [None] * n_out; kind += ["cat"] * n_out
            mean += [0.0] * n_out; scale += [1.0] * n_out
        elif name == "log":
            sc = trans.named_steps["scale"]
            names += list(cols); kind += ["log"] * len(cols)
            mean += list(sc.mean_); scale += list(sc.scale_)
        elif name == "num":
            names += list(cols); kind += ["num"] * len(cols)
            mean += list(trans.mean_); scale += list(trans.scale_)
    mean, scale = np.array(mean), np.array(scale)
    raw_lo = np.zeros(len(names)); raw_hi = np.ones(len(names))
    integer = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        if n is None:
            continue
        if n.endswith("_rate"):
            raw_lo[i], raw_hi[i] = 0.0, 1.0
        else:
            raw_lo[i], raw_hi[i] = 0.0, split.train_raw_max[n]
            integer[i] = True
    log = np.array([k == "log" for k in kind])
    lo, hi = raw_lo.copy(), raw_hi.copy()
    lo[log], hi[log] = np.log1p(lo[log]), np.log1p(hi[log])
    lo, hi = (lo - mean) / scale, (hi - mean) / scale

    masks = {}
    for cat in CATEGORIES:
        m = np.array([n is not None and n not in FUNCTIONAL[cat] and n not in BINARY
                      and split.train_raw_max.get(n, 0) > 0 for n in names])
        masks[cat] = m
    return FeatureSpace(names, kind, mean, scale, lo, hi, integer, masks)


def row_masks(fs: FeatureSpace, attack_names) -> np.ndarray:
    """Per-row modifiable-feature mask from each row's attack type (normal rows: nothing)."""
    out = np.zeros((len(attack_names), len(fs.names)), dtype=bool)
    for cat in CATEGORIES:
        rows = np.array([ATTACK_CATEGORY[a] == cat for a in attack_names])
        out[rows] = fs.masks[cat]
    return out


def pgd_to_normal(logit_fn, X, mask, fs: FeatureSpace, eps: float, steps: int = 40,
                  step_frac: float = 0.1, device="cpu", batch: int = 4096, seed: int = 0):
    """L-inf PGD that pushes samples toward class 0 (normal) within the constraints.

    logit_fn maps a float tensor [B, F] to logits [B, C] and must be differentiable.
    Returns realised (valid raw-value) adversarial inputs in scaled space.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    span = torch.tensor(fs.hi - fs.lo, dtype=torch.float32, device=device)
    lo = torch.tensor(fs.lo, dtype=torch.float32, device=device)
    hi = torch.tensor(fs.hi, dtype=torch.float32, device=device)
    out = np.empty_like(X)
    for i in range(0, len(X), batch):
        x0 = torch.tensor(X[i:i + batch], device=device)
        m = torch.tensor(mask[i:i + batch], dtype=torch.float32, device=device)
        radius = eps * span * m
        delta = (torch.rand(x0.shape, generator=g).to(device) * 2 - 1) * radius
        for _ in range(steps):
            delta.requires_grad_(True)
            logits = logit_fn(x0 + delta)
            other = logits[:, 1:].max(1).values
            loss = (logits[:, 0] - other).sum()          # margin toward "normal"
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = delta + step_frac * radius * grad.sign()
                delta = torch.max(torch.min(delta, radius), -radius)
                delta = torch.max(torch.min(x0 + delta, hi), lo) - x0
        out[i:i + batch] = (x0 + delta).detach().cpu().numpy()
    return fs.realise(out, X, mask)


def random_search_to_normal(proba_fn, X, mask, fs: FeatureSpace, eps: float, iters: int = 300,
                            frac: float = 0.3, seed: int = 0):
    """Score-based black-box attack (random search, in the spirit of Square Attack).

    Needs only predict_proba access. Each iteration proposes, for every sample, moving a random
    subset of its modifiable features to a corner of the eps-box; the proposal is snapped to
    valid raw values and kept if it raises P(normal). Works on tree models.
    """
    rng = np.random.default_rng(seed)
    span = (fs.hi - fs.lo)[None, :]
    lo = np.maximum(fs.lo[None, :], X - eps * span)
    hi = np.minimum(fs.hi[None, :], X + eps * span)
    cur = X.copy()
    cur_p = proba_fn(cur)[:, 0]
    for _ in range(iters):
        pick = (rng.random(X.shape) < frac) & mask
        corner = np.where(rng.random(X.shape) < 0.5, lo, hi)
        prop = fs.realise(np.where(pick, corner, cur), X, mask)
        prop_p = proba_fn(prop)[:, 0]
        better = prop_p > cur_p
        cur[better], cur_p[better] = prop[better], prop_p[better]
    return cur


def check_constraints(fs: FeatureSpace, X_adv, X_orig, mask, tol: float = 1e-5) -> dict:
    """Verify adversarial rows: only modifiable columns changed, and every changed value is
    valid in raw space (inside its range, integer where required)."""
    changed = np.abs(X_adv - X_orig) > tol
    illegal = changed & ~mask
    R = fs.to_raw(X_adv.astype(np.float64))
    raw_lo, raw_hi = fs.to_raw(fs.lo[None, :])[0], fs.to_raw(fs.hi[None, :])[0]
    tol_abs = 1e-4 * np.maximum(1.0, np.abs(raw_hi))     # float32 storage error
    out_of_range = changed & ((R < raw_lo - tol_abs) | (R > raw_hi + tol_abs))
    non_int = changed & fs.integer[None, :] & (np.abs(R - np.round(R)) > 1e-3 * np.maximum(1, np.abs(R)))
    return {"illegal_changes": int(illegal.sum()), "out_of_range": int(out_of_range.sum()),
            "non_integer": int(non_int.sum()), "rows_changed": float(changed.any(1).mean())}


def lr_logit_fn(lr_model, device="cpu"):
    """Differentiable logits for a fitted sklearn LogisticRegression (binary or multinomial)."""
    W = torch.tensor(lr_model.coef_, dtype=torch.float32, device=device)
    b = torch.tensor(lr_model.intercept_, dtype=torch.float32, device=device)
    if W.shape[0] == 1:                                  # binary: logit for class 1
        return lambda x: torch.cat([torch.zeros(len(x), 1, device=device), x @ W.T + b], 1)
    return lambda x: x @ W.T + b


def clstm_logit_fn(clf):
    clf.model.eval()
    return lambda x: clf.model(x)


def train_clstm_adversarial(n_classes, params, X, y, mask, X_val, y_val, fs: FeatureSpace,
                            eps: float, pgd_steps: int = 7, epochs: int = 20, seed: int = 0,
                            patience: int = 5, verbose: bool = True):
    """PGD adversarial training (Madry et al.): each batch trains on clean + attacked samples.

    Only attack rows are perturbed (an attacker does not disguise normal traffic), with the
    same per-category constraints as the evaluation attack.
    """
    import copy
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from .models import CLSTM, TorchClassifier, get_device

    device = get_device()
    p = dict(params); lr = p.pop("lr")
    clf = TorchClassifier(n_classes, epochs=epochs, lr=lr, seed=seed, verbose=False, **p)
    torch.manual_seed(seed); np.random.seed(seed)
    model = CLSTM(X.shape[1], n_classes, **clf.arch).to(device)
    clf.model = model
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    span = torch.tensor(fs.hi - fs.lo, dtype=torch.float32, device=device)
    lo = torch.tensor(fs.lo, dtype=torch.float32, device=device)
    hi = torch.tensor(fs.hi, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long(),
                                      torch.from_numpy(mask.astype(np.float32))),
                        batch_size=512, shuffle=True, generator=g)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        total = 0.0
        for bx, by, bm in loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            radius = eps * span * bm
            model.eval()                                  # fixed BatchNorm stats while attacking
            delta = (torch.rand_like(bx) * 2 - 1) * radius
            for _ in range(pgd_steps):
                delta.requires_grad_(True)
                logits = model(bx + delta)
                l = (logits[:, 0] - logits[:, 1:].max(1).values).sum()
                grad, = torch.autograd.grad(l, delta)
                with torch.no_grad():
                    delta = delta + 0.25 * radius * grad.sign()
                    delta = torch.max(torch.min(delta, radius), -radius)
                    delta = torch.max(torch.min(bx + delta, hi), lo) - bx
            model.train()
            opt.zero_grad()
            loss = 0.5 * loss_fn(model(bx), by) + 0.5 * loss_fn(model(bx + delta.detach()), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(bx)
        val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        if verbose:
            print(f"    [adv-train] ep {ep:2d} loss {total/len(X):.4f} val macroF1 {val_f1:.4f}")
        if val_f1 > best:
            best, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return clf
