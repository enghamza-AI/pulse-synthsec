

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_orders(config: dict) -> pd.DataFrame:

    sd = config["synthetic_data"]

 
    rng = np.random.default_rng(sd["random_seed"])

    n_customers = sd["n_customers"]
    sim_days = sd["simulation_days"]
    one_time_frac = sd["one_time_buyer_fraction"]
    categories = sd["categories"]

 
    sim_end_date = pd.Timestamp.today().normalize()
    sim_start_date = sim_end_date - pd.Timedelta(days=sim_days)

    is_one_time = rng.random(n_customers) < one_time_frac

    rows = []
    order_counter = 0

    for customer_id in range(1, n_customers + 1):
      
        first_order_offset = rng.integers(0, sim_days)
        first_order_date = sim_start_date + pd.Timedelta(days=int(first_order_offset))

        if is_one_time[customer_id - 1]:
           
            order_dates = [first_order_date]
        else:
         
            true_cadence_days = rng.lognormal(
                mean=sd["cadence_lognormal_mean"],
                sigma=sd["cadence_lognormal_sigma"],
            )

            order_dates = [first_order_date]
            current_date = first_order_date
            while True:
               
                jitter = rng.lognormal(mean=0.0, sigma=0.3)
                gap_days = true_cadence_days * jitter
                current_date = current_date + pd.Timedelta(days=float(gap_days))
                if current_date > sim_end_date:
                    break
                order_dates.append(current_date)

        for order_date in order_dates:
            order_counter += 1
            order_value = rng.lognormal(
                mean=sd["order_value_lognormal_mean"],
                sigma=sd["order_value_lognormal_sigma"],
            )
            category = categories[rng.integers(0, len(categories))]
            rows.append(
                {
                    "customer_id": customer_id,
                    "order_id": f"ord_{order_counter:08d}",
                    "order_date": order_date,
                    "order_value": round(float(order_value), 2),
                    "category": category,
                }
            )

    orders = pd.DataFrame(rows)
    orders = orders.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    return orders


if __name__ == "__main__":
   
    from config_loader import load_config

    cfg = load_config()
    df = generate_synthetic_orders(cfg)
    print(f"Generated {len(df):,} orders for {df['customer_id'].nunique():,} customers")
    print(df.head())
    orders_per_customer = df.groupby("customer_id").size()
    print("\nOrders-per-customer distribution:")
    print(orders_per_customer.describe())
