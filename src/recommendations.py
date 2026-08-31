

from __future__ import annotations

import pandas as pd


def generate_recommendations(scored: pd.DataFrame, config: dict) -> pd.DataFrame:
  
    rec_cfg = config["recommendations"]
    overdue_mult = rec_cfg["at_risk_overdue_multiplier"]

    out = scored.copy()

    actions = []
    urgencies = []

    for _, row in out.iterrows():
        segment = row["segment_name"]
        recency = row["recency_days"]
        typical_gap = row["interpurchase_mean_days"]
        predicted_gap = row["predicted_days_to_next_purchase"]

       
        is_overdue = typical_gap > 0 and (recency / typical_gap) >= overdue_mult

        if segment == "one_time":
            action = (
                f"No repeat purchase yet — the model expects one around "
                f"day {predicted_gap:.0f}. Send a first-repeat nudge close "
                f"to that window instead of a generic 30-day email."
            )
            urgency = "medium"

        elif segment == "vip":
            if is_overdue:
                action = (
                    "High-value customer who's gone quiet relative to their "
                    "own rhythm — personal outreach this week, not a "
                    "generic campaign."
                )
                urgency = "high"
            else:
                action = (
                    "High-value, on-rhythm customer — protect this "
                    "relationship (loyalty perk, early access), don't "
                    "spend acquisition budget chasing them."
                )
                urgency = "low"

        elif segment == "at_risk":
            action = (
                f"Overdue relative to their own repurchase rhythm "
                f"(usually every {typical_gap:.0f} days) — send a win-back "
                f"offer now, before they're gone for good."
            )
            urgency = "high"

        else:  # regular
            action = (
                f"On track — expected back around day "
                f"{predicted_gap:.0f}. No action needed yet."
            )
            urgency = "low"

        actions.append(action)
        urgencies.append(urgency)

    out["recommended_action"] = actions
    out["urgency"] = urgencies
    return out


if __name__ == "__main__":
    
    from config_loader import load_config
    from synthetic_data import generate_synthetic_orders
    from cleaning import clean_orders
    from feature_engineering import build_customer_features
    from segmentation import segment_customers
    from scoring import train_repurchase_model, predict_repurchase_window

    cfg = load_config()
    raw = generate_synthetic_orders(cfg)
    clean = clean_orders(raw, cfg)
    features = build_customer_features(clean, cfg)
    segmented = segment_customers(features, cfg)
    bundle = train_repurchase_model(segmented, cfg)
    scored = predict_repurchase_window(segmented, bundle)
    recs = generate_recommendations(scored, cfg)
    print(recs["urgency"].value_counts())
    print(recs[["customer_id", "segment_name", "urgency",
                 "recommended_action"]].head(3).to_string())
