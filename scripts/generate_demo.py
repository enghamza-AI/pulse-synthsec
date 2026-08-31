from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pickle
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import run_pipeline


def main():
    print("Running Pulse pipeline...")

    config = load_config()

    output = run_pipeline(config, data_mode="demo")

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "demo"
        / "pulse_results.pkl"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print()
    print("Pulse demo results saved successfully.")
    print(f"File: {output_path}")
    print(f"Customers: {output['n_customers']:,}")
    print(f"Model MAE: {output['model_mae']:.2f} days")
    print(f"Silhouette score: {output['silhouette_avg']:.3f}")


if __name__ == "__main__":
    main()