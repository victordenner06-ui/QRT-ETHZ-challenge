"""Ridge on relative return paths + LightGBM on features + MLP trained with time-shift augmentation.
Writes two submissions: the augmented MLP alone (CV ~0.540) and the equal-weight ensemble (CV ~0.539).
Predictions are centred within each (DATE_ID, COUNTRY) group, like the target.

Time-shift augmentation: we pretend the cutoff is s bars earlier (s = 6..48) and use the observed
country-relative return over the next 6 bars as target. Only pre-cutoff prices are used, so no label leaks.

Usage (from the project root):  python -m src.train_v2
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from torch import nn

from src.data import load
from src.features import _ffill, benchmark, build_features
from src.metric import weighted_sign_accuracy

ROOT = Path(__file__).resolve().parents[1]
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  n_estimators=600, verbose=-1)
MLP_SEEDS = 5
SHIFTS = [6, 12, 18, 24, 30, 36, 42, 48]  # pseudo-cutoffs, in 5-min bars before the real cutoff
PSEUDO_WEIGHT = 0.3  # weight of the pseudo samples in the loss (the last 30 min behave a bit differently)


def groups(meta):
    return pd.factorize(meta["DATE_ID"].astype(str) + "_" + meta["COUNTRY"])[0]


def folds_of(meta, n):
    return list(GroupKFold(n_splits=4).split(np.zeros(n), groups=meta["DATE_ID"]))


def rel_returns(meta, P):
    """5-min log returns minus the (DATE_ID, COUNTRY) mean of the same bar, scaled by the stock's day vol.
    Returns (stock relative path, country path), both (n, 95)."""
    r = np.diff(np.log(_ffill(P)), axis=1)
    g = groups(meta)
    sums = np.zeros((g.max() + 1, r.shape[1]))
    np.add.at(sums, g, np.nan_to_num(r))
    mkt = sums[g] / np.bincount(g)[g, None]
    vol = np.nanstd(r, axis=1, keepdims=True) + 1e-6
    return np.nan_to_num((r - mkt) / vol).astype(np.float32), np.nan_to_num(mkt / vol).astype(np.float32)


def shifted(P, s):
    """Drop the last s bars and right-align again (cutoff moved s bars earlier)."""
    out = np.full_like(P, np.nan)
    out[:, s:] = P[:, :P.shape[1] - s]
    return out


def pseudo_target(meta, P, s):
    """Country-demeaned return from the pseudo-cutoff to 6 bars later."""
    Pf = _ffill(P)
    R = np.nan_to_num(Pf[:, -1 - s + 6] / Pf[:, -1 - s] - 1)
    g = groups(meta)
    return R - (np.bincount(g, R) / np.bincount(g))[g]


def demean(p, g):
    s = pd.Series(p)
    return (s - s.groupby(g).transform("mean")).to_numpy()


def zscore_fit(A):
    """Clip to the 1-99% quantiles of A and z-score; returns the transform."""
    A = np.where(np.isfinite(A), A, np.nan)
    lo, hi = np.nanquantile(A, 0.01, axis=0), np.nanquantile(A, 0.99, axis=0)
    mu, sd = np.nanmean(np.clip(A, lo, hi), axis=0), np.nanstd(np.clip(A, lo, hi), axis=0) + 1e-9
    return lambda B: np.nan_to_num((np.clip(np.where(np.isfinite(B), B, np.nan), lo, hi) - mu) / sd).astype(np.float32)


def fit_mlp(Z, t, w, seed):
    """One epoch, loss = smooth version of the metric: maximise mean(w * y * tanh(f))."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(Z)
    yt = torch.from_numpy((t / np.abs(t).mean()).astype(np.float32))
    wt = torch.from_numpy(w.astype(np.float32))
    net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=len(Z) // 1024 + 1)
    net.train()
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(Z)))
    for i in range(0, len(Z), 1024):
        b = perm[i:i + 1024]
        loss = -(wt[b] * yt[b] * torch.tanh(net(Xt[b]).squeeze(1))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()

    def predict(A):
        with torch.no_grad():
            return net(torch.from_numpy(A)).squeeze(1).numpy()
    return predict


def mlp_part():
    """Augmented MLP, OOF + test predictions. Runs in its own (spawned) process, single-threaded:
    on macOS torch crashes/deadlocks in OpenMP when LightGBM/sklearn's libomp is also loaded."""
    torch.set_num_threads(1)
    meta_tr, P_tr, y = load("train")
    meta_te, P_te, _ = load("test")
    X_tr, X_te = build_features(meta_tr, P_tr), build_features(meta_te, P_te)
    feats = [c for c in X_tr.columns if c != "STOCK_ID"]
    F_tr, F_te = X_tr[feats].to_numpy(np.float64), X_te[feats].to_numpy(np.float64)
    rel_tr, mkt_tr = rel_returns(meta_tr, P_tr)
    rel_te, mkt_te = rel_returns(meta_te, P_te)
    pseudo = []
    for s in SHIFTS:
        Ps = shifted(P_tr, s)
        rel, mkt = rel_returns(meta_tr, Ps)
        Fs = build_features(meta_tr, Ps)[feats].to_numpy(np.float32)
        pseudo.append((rel, mkt, Fs, pseudo_target(meta_tr, P_tr, s), s))

    oof, test = np.zeros(len(y)), np.zeros(len(meta_te))
    folds = folds_of(meta_tr, len(y))
    for tr, va in folds:
        zf = zscore_fit(F_tr[tr])
        Z_tr = np.hstack([rel_tr, mkt_tr, zf(F_tr), np.zeros((len(y), 1), np.float32)])
        Z_te = np.hstack([rel_te, mkt_te, zf(F_te), np.zeros((len(meta_te), 1), np.float32)])
        # real rows first, then one block per shift (preallocated: the machine has 8 GB)
        n = len(tr)
        Za = np.empty(((1 + len(pseudo)) * n, Z_tr.shape[1]), np.float32)
        ta, wa = np.empty(len(Za)), np.full(len(Za), PSEUDO_WEIGHT)
        Za[:n], ta[:n], wa[:n] = Z_tr[tr], y[tr], 1.0
        for k, (rel, mkt, Fs, ys, s) in enumerate(pseudo, start=1):  # indicator s/48: real vs pseudo cutoff
            Za[k * n:(k + 1) * n] = np.hstack([rel[tr], mkt[tr], zf(Fs[tr]), np.full((n, 1), s / 48, np.float32)])
            ta[k * n:(k + 1) * n] = ys[tr]
        for seed in range(MLP_SEEDS):
            f = fit_mlp(Za, ta, wa, seed)
            oof[va] += f(Z_tr[va]) / MLP_SEEDS
            test += f(Z_te) / (MLP_SEEDS * len(folds))
    return oof, test


def write_sub(name, meta_tr, oof, meta_te, test):
    """Train IDs (OOF) + test IDs, never an exact 0."""
    sub = pd.concat([pd.DataFrame({"ID": meta_tr["ID"], "Close return": oof}),
                     pd.DataFrame({"ID": meta_te["ID"], "Close return": test})])
    sub["Close return"] = sub["Close return"].replace(0, 1e-9)
    sub = sub.sort_values("ID").reset_index(drop=True)
    assert len(sub) == 219_157 and sub["ID"].is_unique and np.isfinite(sub["Close return"]).all()
    assert (sub["Close return"] != 0).all()
    out = ROOT / "submissions" / f"{name}_{datetime.now():%H%M}.csv"
    sub.to_csv(out, index=False)
    print("Saved", out)


def main():
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        mlp_job = ex.submit(mlp_part)

        meta_tr, P_tr, y = load("train")
        meta_te, P_te, _ = load("test")
        print("Benchmark CV:", round(weighted_sign_accuracy(y, benchmark(P_tr)), 4), flush=True)
        X_tr, X_te = build_features(meta_tr, P_tr), build_features(meta_te, P_te)
        feats = [c for c in X_tr.columns if c != "STOCK_ID"]
        rel_tr, _ = rel_returns(meta_tr, P_tr)
        rel_te, _ = rel_returns(meta_te, P_te)
        g_tr, g_te = groups(meta_tr), groups(meta_te)

        mask = y != 0
        label, w = (y > 0).astype(int), np.abs(y)
        models = ["ridge", "lgbm", "mlp"]
        oof = {m: np.zeros(len(y)) for m in models}
        test = {m: np.zeros(len(X_te)) for m in models}
        folds = folds_of(meta_tr, len(y))
        for tr, va in folds:
            tr = tr[mask[tr]]
            # Ridge on the relative return path
            m = Ridge(alpha=100.0).fit(rel_tr[tr], y[tr] / y[tr].std())
            oof["ridge"][va] = m.predict(rel_tr[va])
            test["ridge"] += m.predict(rel_te) / len(folds)
            # LightGBM classifier on the hand-made features (same as train.py)
            m = lgb.LGBMClassifier(**LGB_PARAMS)
            m.fit(X_tr.iloc[tr][feats], label[tr], sample_weight=w[tr] / w[tr].mean())
            oof["lgbm"][va] = m.predict_proba(X_tr.iloc[va][feats])[:, 1] - 0.5
            test["lgbm"] += (m.predict_proba(X_te[feats])[:, 1] - 0.5) / len(folds)
        print("ridge + lgbm done, waiting for the MLP...", flush=True)
        oof["mlp"], test["mlp"] = mlp_job.result()

    for m in models:
        print(f"{m:6s} folds: " + " ".join(f"{weighted_sign_accuracy(y[va], demean(oof[m], g_tr)[va]):.4f}" for _, va in folds)
              + f"   CV (centred): {weighted_sign_accuracy(y, demean(oof[m], g_tr)):.4f}")

    # 1) augmented MLP alone
    write_sub("mlpaug", meta_tr, demean(oof["mlp"], g_tr), meta_te, demean(test["mlp"], g_te))
    # 2) equal weights on standardised predictions (stacking was tested: it overfits), centred per group
    oof_ens = demean(sum(oof[m] / oof[m].std() for m in models), g_tr)
    test_ens = demean(sum(test[m] / test[m].std() for m in models), g_te)
    print("ENSEMBLE folds: " + " ".join(f"{weighted_sign_accuracy(y[va], oof_ens[va]):.4f}" for _, va in folds)
          + f"   CV: {weighted_sign_accuracy(y, oof_ens):.4f}")
    write_sub("ens3aug", meta_tr, oof_ens, meta_te, test_ens)


if __name__ == "__main__":
    main()
