

from __future__ import annotations

import numpy as np
import pandas as pd


def build_customer_features(orders: pd.DataFrame, config: dict) -> pd.DataFrame:
  
    fe_cfg = config["feature_engineering"]
    min_orders_for_label = fe_cfg["min_orders_for_supervised_label"]

  
    anchor_date = fe_cfg.get("anchor_date")
    if anchor_date is None:
        anchor_date = orders["order_date"].max() + pd.Timedelta(days=1)
    else:
        anchor_date = pd.Timestamp(anchor_date)

    rows = []

  
    for customer_id, group in orders.groupby("customer_id"):
        dates = group["order_date"].sort_values().to_numpy()
        values = group["order_value"].to_numpy()
        n_orders = len(dates)

        recency_days = (anchor_date - pd.Timestamp(dates[-1])).days
        frequency = n_orders
        monetary_total = float(values.sum())
        monetary_avg = float(values.mean())
        tenure_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
        category_diversity = group["category"].nunique()

        if n_orders >= 2:
       
            gaps = np.diff(dates).astype("timedelta64[D]").astype(float)
            interpurchase_mean = float(gaps.mean())
            interpurchase_std = float(gaps.std()) if n_orders >= 3 else 0.0
        else:
         
            interpurchase_mean = np.nan
            interpurchase_std = np.nan

     
        if n_orders >= min_orders_for_label:
            has_label = True
            days_to_next_purchase = float(
                (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[-2])).days
            )
        else:
            has_label = False
            days_to_next_purchase = np.nan

        rows.append(
            {
                "customer_id": customer_id,
                "recency_days": recency_days,
                "frequency": frequency,
                "monetary_total": round(monetary_total, 2),
                "monetary_avg": round(monetary_avg, 2),
                "tenure_days": tenure_days,
                "interpurchase_mean_days": interpurchase_mean,
                "interpurchase_std_days": interpurchase_std,
                "category_diversity": category_diversity,
                "has_supervised_label": has_label,
                "days_to_next_purchase": days_to_next_purchase,
            }
        )

    features = pd.DataFrame(rows)

    for col in ["interpurchase_mean_days", "interpurchase_std_days"]:
        median_val = features[col].median()
        features[col] = features[col].fillna(median_val)

    return features


if __name__ == "__main__":
   
    from config_loader import load_config
    from synthetic_data import generate_synthetic_orders
    from cleaning import clean_orders

    cfg = load_config()
    raw = generate_synthetic_orders(cfg)
    clean = clean_orders(raw, cfg)
    features = build_customer_features(clean, cfg)
    print(f"Built features for {len(features):,} customers")
    print(features.describe(include="all"))
    print(f"\nCustomers with a supervised label: "
          f"{features['has_supervised_label'].sum():,} "
          f"({features['has_supervised_label'].mean():.1%})")
