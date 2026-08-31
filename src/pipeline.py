
from __future__ import annotations

import pandas as pd

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_orders
from src.cleaning import clean_orders
from src.feature_engineering import build_customer_features
from src.segmentation import segment_customers
from src.scoring import train_repurchase_model, predict_repurchase_window
from src.recommendations import generate_recommendations


def run_pipeline(config: dict, data_mode: str = "demo") -> dict:
 
    if data_mode == "demo":
     
        csv_path = config["app"]["demo_csv_path"]
        raw_orders = pd.read_csv(csv_path, parse_dates=["order_date"])
    elif data_mode == "local":
    
        raw_orders = generate_synthetic_orders(config)
    else:
        raise ValueError(
            f"Unknown data_mode '{data_mode}'. Expected 'demo' or 'local'."
        )

    clean = clean_orders(raw_orders, config)
    features = build_customer_features(clean, config)
    segmented = segment_customers(features, config)
    model_bundle = train_repurchase_model(segmented, config)
    scored = predict_repurchase_window(segmented, model_bundle)
    result = generate_recommendations(scored, config)

    return {
        "result": result,
        "model_mae": model_bundle["test_mae"],
        "silhouette_avg": float(result["silhouette_avg"].iloc[0]),
        "n_customers": len(result),
    }


if __name__ == "__main__":
   
    cfg = load_config()
    output = run_pipeline(cfg, data_mode="local")
    print(f"Scored {output['n_customers']:,} customers")
    print(f"Model MAE: {output['model_mae']} days")
    print(f"Silhouette score: {output['silhouette_avg']}")
    print(output["result"]["segment_name"].value_counts())
