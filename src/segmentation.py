
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_customers(features: pd.DataFrame, config: dict) -> pd.DataFrame:

    seg_cfg = config["segmentation"]
    feature_cols = seg_cfg["clustering_features"]

    X = features[feature_cols].to_numpy()

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=seg_cfg["n_clusters"],
        random_state=seg_cfg["random_state"],
        n_init=seg_cfg["n_init"],
    )
    cluster_ids = kmeans.fit_predict(X_scaled)

    sample_size = min(2000, len(X_scaled))
    sample_idx = np.random.default_rng(seg_cfg["random_state"]).choice(
        len(X_scaled), size=sample_size, replace=False
    )
    sil_score = silhouette_score(X_scaled[sample_idx], cluster_ids[sample_idx])

    out = features.copy()
    out["cluster_id"] = cluster_ids
    out["silhouette_avg"] = round(float(sil_score), 3)

    out["segment_name"] = _name_segments(out, config)

    return out


def _name_segments(df: pd.DataFrame, config: dict) -> pd.Series:

    vip_floor = config["recommendations"]["vip_monetary_floor"]

    
    centroid_stats = df.groupby("cluster_id").agg(
        mean_monetary=("monetary_total", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_recency=("recency_days", "mean"),
    )

    names = pd.Series(index=df.index, dtype="object")

    for cluster_id, row in centroid_stats.iterrows():
        mask = df["cluster_id"] == cluster_id

        if row["mean_frequency"] <= 1.5:
          
            label = "one_time"
        elif row["mean_monetary"] >= vip_floor or (
            row["mean_frequency"] == centroid_stats["mean_frequency"].max()
            and row["mean_monetary"] == centroid_stats["mean_monetary"].max()
        ):
            label = "vip"
        elif row["mean_recency"] >= centroid_stats["mean_recency"].median():
            
            label = "at_risk"
        else:
            label = "regular"

        names[mask] = label

    return names


if __name__ == "__main__":
    
    from config_loader import load_config
    from synthetic_data import generate_synthetic_orders
    from cleaning import clean_orders
    from feature_engineering import build_customer_features

    cfg = load_config()
    raw = generate_synthetic_orders(cfg)
    clean = clean_orders(raw, cfg)
    features = build_customer_features(clean, cfg)
    segmented = segment_customers(features, cfg)
    print(segmented["segment_name"].value_counts())
    print(f"\nSilhouette score: {segmented['silhouette_avg'].iloc[0]}")
