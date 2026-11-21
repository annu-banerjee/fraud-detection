"""
Train a compact fraud model for deployment.

Why a separate script instead of the notebook?
Notebooks are for exploring. Production models are trained by a script
that anyone can re-run from scratch and get the same result.

Run from the project folder:
    python src/train.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models"

TARGET = "isFraud"
DROP = ["TransactionID", "TransactionDT", "day", TARGET]
N_FEATURES = 30          # keep the API input small
FP_COST = 20             # cost of a false alarm
FN_FEE = 15              # extra cost on top of the amount for a missed fraud

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
    # same data + same code = same model, every time
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}


def set_categories(train, valid, cols):
    """Learn category lists from training only; unseen values in valid become NaN."""
    cats = {}
    for c in cols:
        train[c] = train[c].astype("category")
        known = train[c].cat.categories
        valid[c] = pd.Categorical(valid[c].where(valid[c].isin(known)), categories=known)
        cats[c] = [str(x) for x in known]
    return cats


def fit(fit_part, stop_part, features):
    d1 = lgb.Dataset(fit_part[features], fit_part[TARGET])
    d2 = lgb.Dataset(stop_part[features], stop_part[TARGET], reference=d1)
    return lgb.train(
        PARAMS, d1, num_boost_round=3000, valid_sets=[d2],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def total_cost(y, p, amt, t):
    flag = p >= t
    fp = (flag & (y == 0)).sum() * FP_COST
    fn = (((~flag) & (y == 1)) * (amt + FN_FEE)).sum()
    return fp + fn


def main():
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    valid = pd.read_parquet(DATA / "valid.parquet")

    cat_cols = train.select_dtypes(exclude="number").columns.tolist()
    set_categories(train, valid, cat_cols)

    all_features = [c for c in train.columns if c not in DROP]
    fit_part = train[train["day"] <= 120]
    stop_part = train[train["day"] > 120]

    # 1. full model, only used to rank features
    print(f"Training full model on {len(all_features)} features...")
    full_model = fit(fit_part, stop_part, all_features)
    gain = pd.Series(full_model.feature_importance("gain"), index=all_features)
    features = gain.sort_values(ascending=False).head(N_FEATURES).index.tolist()

    # 2. compact model on the top features
    print(f"Training compact model on top {N_FEATURES} features...")
    model = fit(fit_part, stop_part, features)

    # 3. threshold chosen on the stopping set (never on validation)
    p_stop = model.predict(stop_part[features], num_iteration=model.best_iteration)
    thresholds = np.linspace(0.01, 0.9, 90)
    costs = [total_cost(stop_part[TARGET].values, p_stop,
                        stop_part["TransactionAmt"].values, t) for t in thresholds]
    threshold = float(thresholds[int(np.argmin(costs))])

    # reference score distribution, so the API can tell when live scores drift
    edges = np.unique(np.quantile(p_stop, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = 0.0, 1.0
    ref_props = np.histogram(p_stop, edges)[0] / len(p_stop)

    # 4. honest evaluation on validation
    p_full = full_model.predict(valid[all_features], num_iteration=full_model.best_iteration)
    p_valid = model.predict(valid[features], num_iteration=model.best_iteration)
    y = valid[TARGET].values
    metrics = {
        "full_model_pr_auc": round(average_precision_score(y, p_full), 4),
        "compact_pr_auc": round(average_precision_score(y, p_valid), 4),
        "compact_roc_auc": round(roc_auc_score(y, p_valid), 4),
        "threshold": round(threshold, 2),
        "valid_cost_at_threshold": int(total_cost(y, p_valid, valid["TransactionAmt"].values, threshold)),
    }

    # 5. save the model plus everything the API needs to use it correctly
    OUT.mkdir(exist_ok=True)
    model.save_model(str(OUT / "model.txt"), num_iteration=model.best_iteration)
    feature_cats = {c: cats for c, cats in
                    set_categories(train.copy(), valid.copy(), [c for c in features if c in cat_cols]).items()}
    metadata = {
        "features": features,
        "categorical": feature_cats,
        "threshold": threshold,
        "metrics": metrics,
        "reference_scores": {"edges": edges.tolist(), "proportions": ref_props.tolist()},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nRESULTS")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nFEATURES USED")
    print(" ", ", ".join(features))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
