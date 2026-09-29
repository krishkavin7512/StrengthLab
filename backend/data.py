"""Loading, cleaning, splitting and feature engineering for the concrete data."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# Linear models see engineered features: Abrams' law says strength falls with the
# water/cement ratio, and strength gain slows with age, so ln(age) is closer to linear.
LINEAR_FEATURES = ["cement", "slag", "flyash", "water", "sp", "coarse", "fine", "log_age", "wc", "wb"]
# Kernel models see the raw recipe (with ln(age)); the kernel supplies the non-linearity.
KERNEL_FEATURES = ["cement", "slag", "flyash", "water", "sp", "coarse", "fine", "log_age"]
BOUNDARY_FEATURES = ["wb", "log_age"]  # the 2-D view used to draw decision boundaries


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_FILE)
    df.columns = config.COLUMN_NAMES
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate lab records so no mix sits in both the training and test sets."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_age"] = np.log(out["age"])
    out["wc"] = out["water"] / out["cement"]
    out["wb"] = out["water"] / (out["cement"] + out["slag"] + out["flyash"])
    return out


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split(df: pd.DataFrame, seed: int = config.SEED) -> Split:
    idx = np.random.default_rng(seed).permutation(len(df))
    n_tr = int(round(len(df) * config.SPLIT[0]))
    n_va = int(round(len(df) * config.SPLIT[1]))
    parts = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return Split(*(df.iloc[p].reset_index(drop=True) for p in parts))


class Standardizer:
    """z = (x - mean) / std, with mean and std estimated on the training set only."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        X = df[self.columns].to_numpy(dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (df[self.columns].to_numpy(dtype=float) - self.mean) / self.std

    def to_dict(self):
        return {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["columns"])
        s.mean, s.std = np.array(d["mean"]), np.array(d["std"])
        return s


def summary_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1])
        m = x.mean()
        s = x.std(ddof=1)
        rows.append({"column": col, "label": config.LABEL.get(col, col), "mean": float(m), "std": float(s),
                     "min": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]),
                     "max": float(q[4]), "zeros": int((x == 0).sum()), "unique": int(len(np.unique(x))),
                     "skewness": float(((x - m) ** 3).mean() / x.std() ** 3)})
    return rows


def preview(df: pd.DataFrame, offset: int = 0, limit: int = 25) -> dict:
    page = df.iloc[offset:offset + limit]
    return {"columns": list(df.columns), "rows": page.round(2).to_numpy().tolist(),
            "offset": offset, "limit": limit, "total": int(len(df))}
