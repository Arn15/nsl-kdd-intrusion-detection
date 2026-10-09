"""NSL-KDD loading and preprocessing.

Uses the official KDDTrain+ / KDDTest+ split. All preprocessing (one-hot
categories, scaling) is fit on the training portion only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes"]  # log1p before scaling

# Standard NSL-KDD attack -> category mapping (includes attacks that only
# appear in KDDTest+).
ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "mailbomb": "dos", "processtable": "dos",
    "udpstorm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "named": "r2l", "sendmail": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "worm": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}
CLASSES = {
    "binary": ["normal", "attack"],
    "5class": ["normal", "dos", "probe", "r2l", "u2r"],
}


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_is_novel: np.ndarray   # True for KDDTest+ rows whose attack type never appears in KDDTrain+
    test_attack_names: np.ndarray
    class_names: list[str]
    feature_names: list[str]


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [data_dir / "KDDTrain+.txt", data_dir / "KDDTest+.txt"]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: bash data/download_data.sh")
    train, test = (pd.read_csv(p, header=None, names=COLUMNS) for p in paths)
    unknown = set(train.label) | set(test.label)
    unknown -= set(ATTACK_CATEGORY)
    if unknown:
        raise ValueError(f"Unmapped attack labels: {sorted(unknown)}")
    return train, test


def encode_labels(labels: pd.Series, task: str) -> np.ndarray:
    cat = labels.map(ATTACK_CATEGORY)
    if task == "binary":
        cat = cat.where(cat == "normal", "attack")
    elif task != "5class":
        raise ValueError(f"task must be one of {list(CLASSES)}")
    index = {c: i for i, c in enumerate(CLASSES[task])}
    return cat.map(index).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    numeric = [c for c in COLUMNS[:-2] if c not in CATEGORICAL]
    log_cols = [c for c in numeric if c in HEAVY_TAILED]
    lin_cols = [c for c in numeric if c not in HEAVY_TAILED]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), log_cols),
        ("num", StandardScaler(), lin_cols),
    ])


def make_split(task: str = "5class", val_size: float = 0.1, seed: int = 0,
               data_dir: Path = DATA_DIR) -> Split:
    train, test = load_raw(data_dir)
    y_all = encode_labels(train.label, task)
    tr, va = train_test_split(train, test_size=val_size, random_state=seed, stratify=y_all)

    pre = build_preprocessor()
    X_tr = pre.fit_transform(tr).astype(np.float32)   # fit on train only
    X_va = pre.transform(va).astype(np.float32)
    X_te = pre.transform(test).astype(np.float32)

    seen = set(train.label)
    return Split(
        X_train=X_tr, y_train=encode_labels(tr.label, task),
        X_val=X_va, y_val=encode_labels(va.label, task),
        X_test=X_te, y_test=encode_labels(test.label, task),
        test_is_novel=(~test.label.isin(seen)).to_numpy(),
        test_attack_names=test.label.to_numpy(),
        class_names=CLASSES[task],
        feature_names=list(pre.get_feature_names_out()),
    )


def capped_smote(X: np.ndarray, y: np.ndarray, cap_frac: float = 0.10,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Upsample minority classes to at most cap_frac * majority count (training data only)."""
    from imblearn.over_sampling import SMOTE

    classes, counts = np.unique(y, return_counts=True)
    cap = int(counts.max() * cap_frac)
    target = {c: max(cap, n) for c, n in zip(classes, counts)}
    k = max(1, min(5, counts.min() - 1))
    return SMOTE(sampling_strategy=target, k_neighbors=k, random_state=seed).fit_resample(X, y)
