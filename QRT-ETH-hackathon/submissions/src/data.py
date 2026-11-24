"""Load the QRT x ETH data."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PRICE_COL = "Price TS"
TARGET_COL = "Close return"
N_BARS = 96


def prices_to_matrix(series: pd.Series, n_bars: int = N_BARS) -> np.ndarray:
    """List column of prices -> (n, n_bars) float matrix, right-aligned on the cutoff
    (last column = cutoff price). Missing values = NaN."""
    out = np.full((len(series), n_bars), np.nan, dtype=np.float64)
    for i, p in enumerate(series.to_numpy()):
        if p is None:
            continue
        a = np.asarray(p, dtype=np.float64)[-n_bars:]
        out[i, n_bars - len(a):] = a
    return out


def load(split: str = "train"):
    """Returns (df_meta, prices matrix, y or None)."""
    x = pd.read_parquet(DATA_DIR / f"x_{split}.parquet")
    P = prices_to_matrix(x[PRICE_COL])
    meta = x.drop(columns=[PRICE_COL]).reset_index(drop=True)
    y = None
    if split == "train":
        yt = pd.read_csv(DATA_DIR / "y_train.csv")
        y = meta[["ID"]].merge(yt, on="ID", how="left")[TARGET_COL].to_numpy()
    return meta, P, y
