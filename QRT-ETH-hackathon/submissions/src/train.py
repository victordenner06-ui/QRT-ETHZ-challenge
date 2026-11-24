"""CV GroupKFold(DATE_ID) + LightGBM sign classifier weighted by |y| + submission.

Usage (from the project root):  python -m src.train
"""
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import load
from src.features import benchmark, build_features
from src.metric import weighted_sign_accuracy

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              n_estimators=600, verbose=-1)


def main():
    meta_tr, P_tr, y = load("train")
    meta_te, P_te, _ = load("test")

    print("Benchmark CV:", round(weighted_sign_accuracy(y, benchmark(P_tr)), 4))

    X_tr = build_features(meta_tr, P_tr)
    X_te = build_features(meta_te, P_te)
    feats = [c for c in X_tr.columns if c != "STOCK_ID"]  # STOCK_ID: try target encoding inside each fold

    mask = y != 0
    label, w = (y > 0).astype(int), np.abs(y)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_te))
    gkf = GroupKFold(n_splits=4)
    for k, (tr, va) in enumerate(gkf.split(X_tr, y, groups=meta_tr["DATE_ID"])):
        tr = tr[mask[tr]]
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X_tr.iloc[tr][feats], label[tr], sample_weight=w[tr] / w[tr].mean())
        oof[va] = m.predict_proba(X_tr.iloc[va][feats])[:, 1] - 0.5
        test_pred += (m.predict_proba(X_te[feats])[:, 1] - 0.5) / gkf.n_splits
        print(f"fold {k}: {weighted_sign_accuracy(y[va], oof[va]):.4f}")
    print("LGBM CV:", round(weighted_sign_accuracy(y, oof), 4))

    # Submission: train IDs (OOF) + test IDs, never an exact 0
    sub = pd.concat([pd.DataFrame({"ID": meta_tr["ID"], "Close return": oof}),
                     pd.DataFrame({"ID": meta_te["ID"], "Close return": test_pred})])
    sub["Close return"] = sub["Close return"].replace(0, 1e-9)
    # (sample_submission.csv is not in the Kaggle zip: IDs = train + test)
    sub = sub.sort_values("ID").reset_index(drop=True)
    assert len(sub) == 219_157 and sub["ID"].is_unique and np.isfinite(sub["Close return"]).all()
    out = ROOT / "submissions" / f"lgbm_{datetime.now():%H%M}.csv"
    sub.to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
