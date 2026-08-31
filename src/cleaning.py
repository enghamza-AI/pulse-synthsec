

from __future__ import annotations

import pandas as pd


def clean_orders(orders: pd.DataFrame, config: dict) -> pd.DataFrame:
  
    rules = config["cleaning"]
    df = orders.copy()

    n_before = len(df)

   
    required_cols = ["customer_id", "order_id", "order_date", "order_value", "category"]
    df = df.dropna(subset=required_cols)

 
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])  # drops rows where the coercion failed

   
    df = df[df["order_value"] >= rules["min_order_value"]]

  
    df["order_value"] = df["order_value"].clip(upper=rules["max_order_value_cap"])

   
    df = df.drop_duplicates(subset=["order_id"], keep="first")

    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    n_after = len(df)
    n_dropped = n_before - n_after
    if n_dropped > 0:
  
        print(f"[cleaning.clean_orders] dropped {n_dropped:,} of {n_before:,} rows "
              f"({n_dropped / n_before:.1%})")

    return df


if __name__ == "__main__":
 
    from config_loader import load_config
    from synthetic_data import generate_synthetic_orders

    cfg = load_config()
    raw = generate_synthetic_orders(cfg)
    clean = clean_orders(raw, cfg)
    print(f"Raw rows: {len(raw):,} -> Clean rows: {len(clean):,}")
    print(clean.dtypes)
