"""
compare_metrics.py — decides whether a freshly retrained price predictor
model is actually better than the one currently deployed, before the
retrain workflow is allowed to commit it.

This is the "safety gate" of the autonomous pipeline: retraining happens
automatically, but a worse model never silently replaces a good one.

Usage:
    python src/compare_metrics.py OLD_METRICS.json NEW_METRICS.json [--tolerance 0.02]

Exit code 0: new model is at least as good (or no old metrics exist yet) -> deploy it
Exit code 1: new model is meaningfully worse -> keep the old one, don't deploy
"""
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('old_metrics_path')
    parser.add_argument('new_metrics_path')
    parser.add_argument('--tolerance', type=float, default=0.02,
                         help="Allow the new model's MAE to be up to this fraction worse "
                              "and still deploy it (guards against noise from small data "
                              "changes causing an unnecessary rollback).")
    args = parser.parse_args()

    try:
        with open(args.new_metrics_path) as f:
            new_metrics = json.load(f)
    except FileNotFoundError:
        print("No new metrics file found — training must have failed. Blocking deployment.")
        sys.exit(1)

    try:
        with open(args.old_metrics_path) as f:
            old_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No previous metrics found (first run) — deploying new model unconditionally.")
        print(f"New model: MAE={new_metrics['mae']:.3f} Lakhs, R²={new_metrics['r2']:.4f}")
        sys.exit(0)

    old_mae = old_metrics['mae']
    new_mae = new_metrics['mae']
    threshold = old_mae * (1 + args.tolerance)

    print(f"Previously deployed model — MAE: {old_mae:.3f} Lakhs, R²: {old_metrics.get('r2', float('nan')):.4f}")
    print(f"Newly trained model       — MAE: {new_mae:.3f} Lakhs, R²: {new_metrics.get('r2', float('nan')):.4f}")

    if new_mae <= threshold:
        print(f"New model accepted (MAE {new_mae:.3f} <= threshold {threshold:.3f}). Deploying.")
        sys.exit(0)
    else:
        print(f"New model REJECTED — it's worse than tolerance allows "
              f"(MAE {new_mae:.3f} > threshold {threshold:.3f}). Keeping the previous model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
