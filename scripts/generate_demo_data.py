

from __future__ import annotations

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_orders


def main() -> None:
    config = load_config()
    demo_max_rows = config["app"]["demo_max_rows"]
    demo_csv_path = config["app"]["demo_csv_path"]

   
    demo_config = {
        **config,
        "synthetic_data": {
            **config["synthetic_data"],
            "n_customers": demo_max_rows,
        },
    }

    orders = generate_synthetic_orders(demo_config)
    orders.to_csv(demo_csv_path, index=False)

    print(f"Wrote {len(orders):,} rows "
          f"({orders['customer_id'].nunique():,} customers) to {demo_csv_path}")


if __name__ == "__main__":
    main()
