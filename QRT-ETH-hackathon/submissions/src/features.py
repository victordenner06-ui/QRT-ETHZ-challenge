"""Features built from the (n, 96) price matrix. Ratios only (arbitrary price level)."""
import numpy as np
import pandas as pd

COUNTRIES = ["DEU", "ESP", "FRA", "GBR", "ITA", "NLD"]


def _ffill(P: np.ndarray) -> np.ndarray:
    return pd.DataFrame(P).T.ffill().bfill().T.to_numpy()


def benchmark(P: np.ndarray) -> np.ndarray:
    """Official benchmark: minus the last 5-min return."""
    Pf = _ffill(P)
    r = np.diff(np.log(Pf), axis=1)
    pred = -r[:, -1]
    # If the cutoff price is missing, ffill gives a 0 return -> use the last non-zero return
    for k in range(2, r.shape[1] + 1):
        z = pred == 0
        if not z.any():
            break
        pred[z] = -r[z, -k]
    pred = np.nan_to_num(pred, nan=0.0)  # row with no price at all
    return np.where(pred == 0, 1e-9, pred)  # an exact 0 always counts as wrong


def build_features(meta: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    Pf = _ffill(P)
    logp = np.log(Pf)
    r = np.diff(logp, axis=1)  # 5-min log returns (n, 95)
    last = logp[:, -1]

    f = {}
    for k in (1, 2, 3, 6, 12, 24, 48):
        f[f"ret_{k}"] = last - logp[:, -1 - k]
    f["ret_day"] = last - logp[:, 0]
    f["vol_day"] = np.nanstd(r, axis=1)
    f["vol_last12"] = np.nanstd(r[:, -12:], axis=1)
    f["hi_dist"] = last - np.nanmax(logp, axis=1)
    f["lo_dist"] = last - np.nanmin(logp, axis=1)
    # Separate the real missing prices from the NaN padding of shorter days
    n_bars = P.shape[1] - np.argmax(~np.isnan(P), axis=1)  # length from the first available price
    f["n_bars"] = n_bars
    f["n_missing"] = np.isnan(P).sum(axis=1) - (P.shape[1] - n_bars)
    for k in (1, 3, 6, 12):
        f[f"ret_{k}_z"] = f[f"ret_{k}"] / (f["vol_day"] + 1e-8)

    df = pd.DataFrame(f)
    # Fixed coding, identical across train and test
    df["COUNTRY"] = pd.Categorical(meta["COUNTRY"], categories=COUNTRIES).codes
    df["STOCK_ID"] = meta["STOCK_ID"].to_numpy()
    df["open_min"] = meta["OPEN_TIME"].str.slice(0, 2).astype(int) * 60 + meta["OPEN_TIME"].str.slice(3, 5).astype(int)

    # Cross-sectional features: relative to the country on the same day (target is country-relative)
    grp = [meta["DATE_ID"].to_numpy(), meta["COUNTRY"].to_numpy()]
    for c in ["ret_1", "ret_3", "ret_6", "ret_12", "ret_day", "vol_day"]:
        g = df[c].groupby(grp)
        df[f"{c}_cs"] = df[c] - g.transform("mean")
        df[f"{c}_rank"] = g.rank(pct=True)
        if c.startswith("ret"):
            df[f"{c}_mkt"] = g.transform("mean")
    return df
