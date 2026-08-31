
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MODEL_FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "tenure_days",
    "interpurchase_mean_days",
    "interpurchase_std_days",
    "category_diversity",
]


def train_repurchase_model(features: pd.DataFrame, config: dict) -> dict:

    scoring_cfg = config["scoring"]

   
    trainable = features[features["has_supervised_label"]].copy()

    feature_cols = [c for c in MODEL_FEATURE_COLS if c in trainable.columns]
    X = trainable[feature_cols].to_numpy()
    y = trainable["days_to_next_purchase"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=scoring_cfg["test_size"],
        random_state=scoring_cfg["random_state"],
    )

    model = RandomForestRegressor(
        n_estimators=scoring_cfg["n_estimators"],
        max_depth=scoring_cfg["max_depth"],
        random_state=scoring_cfg["random_state"],
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_mae": round(float(test_mae), 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_repurchase_window(features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    X = features[feature_cols].to_numpy()
    preds = model.predict(X)

    out = features.copy()
    out["predicted_days_to_next_purchase"] = np.round(preds, 1)
    return out


if __name__ == "__main__":
   
    from config_loader import load_config
    from synthetic_data import generate_synthetic_orders
    from cleaning import clean_orders
    from feature_engineering import build_customer_features
    from segmentation import segment_customers

    cfg = load_config()
    raw = generate_synthetic_orders(cfg)
    clean = clean_orders(raw, cfg)
    features = build_customer_features(clean, cfg)
    segmented = segment_customers(features, cfg)

    bundle = train_repurchase_model(segmented, cfg)
    print(f"Trained on {bundle['n_train']} customers, "
          f"tested on {bundle['n_test']}, MAE = {bundle['test_mae']} days")

    scored = predict_repurchase_window(segmented, bundle)
    print(scored[["customer_id", "segment_name",
                   "predicted_days_to_next_purchase"]].head())
