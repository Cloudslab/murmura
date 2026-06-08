#!/usr/bin/env python3
"""Analyse Experiment 2 results (PAMAP2 multi-dataset validation).

Usage:
    python experiments/paper/dmtt/exp2_multidata/analyze_exp2.py

Outputs:
    - Console: honest-accuracy mean ± std at final round
    - results/exp2_table.csv — machine-readable table
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results"

CONDITION_ORDER  = ["c1_static", "dyn_fedavg", "c3_dmtt"]
CONDITION_LABELS = {
    "c1_static":  "FedAvg (static)",
    "dyn_fedavg": "FedAvg (dynamic)",
    "c3_dmtt":    "DMTT (proposed)",
}


def load_results(results_dir: Path) -> pd.DataFrame:
    parquets = sorted(results_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No Parquet files found in {results_dir}.\n"
            "Run  bash experiments/paper/dmtt/exp2_multidata/run_exp2.sh  first."
        )
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"Loaded {len(parquets)} files  ({len(df):,} rows)")
    return df


def print_table(df: pd.DataFrame, results_dir: Path) -> None:
    final_round = df["round"].max()
    final = df[df["round"] == final_round]

    rows = []
    for cond in CONDITION_ORDER:
        sub = final[final["condition"] == cond]
        if sub.empty:
            row = {"Method": CONDITION_LABELS.get(cond, cond), "PAMAP2 (30 % Byz)": "—"}
        else:
            m = sub["honest_accuracy"].mean()
            s = sub["honest_accuracy"].std()
            row = {
                "Method": CONDITION_LABELS.get(cond, cond),
                "PAMAP2 (30 % Byz)": f"{m:.3f} ± {s:.3f}",
            }
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Method")
    print(f"\n=== Honest-node accuracy at round {final_round} (mean ± std, 3 seeds) ===\n")
    print(table.to_string())
    print()

    csv_path = results_dir / "exp2_table.csv"
    table.to_csv(csv_path)
    print(f"Saved → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    print_table(df, args.results_dir)


if __name__ == "__main__":
    main()
