#!/usr/bin/env python3
"""Analyse Experiment 6 (communication overhead).

Reads Parquet results from C2 and C3 runs.  C3 includes bytes_sent_model
and bytes_sent_topo columns (from dmtt.log_comm_bytes: true).  C2 has only
MODEL_STATE traffic; TOPO_CLAIM bytes = 0.

Reports a small table:
    Condition | bytes/round/node (model) | bytes/round/node (topo) | total
    C2 FedAvg |  X kB                   |  0 kB                   |  X kB
    C3 DMTT   |  X kB                   |  Y kB                   |  X+Y kB

Usage:
    python experiments/paper/dmtt/exp6_comm_overhead/analyze_exp6.py
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results"


def load_results(results_dir: Path) -> pd.DataFrame:
    parquets = sorted(results_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No Parquet files found in {results_dir}.")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"Loaded {len(parquets)} files  ({len(df):,} rows)")
    return df


def print_table(df: pd.DataFrame) -> None:
    rows = []
    for cond, label in [("dyn_fedavg", "C2 — FedAvg (dynamic)"), ("c3_dmtt", "C3 — DMTT (proposed)")]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue

        model_col = "bytes_sent_model"
        topo_col  = "bytes_sent_topo"

        if model_col in sub.columns and sub[model_col].notna().any():
            m_mean = sub[model_col].mean() / 1024
            t_mean = sub[topo_col].mean() / 1024 if topo_col in sub.columns else 0.0
        else:
            # C2 — estimate from model size (can be derived from Parquet but not logged)
            m_mean = float("nan")
            t_mean = 0.0

        rows.append({
            "Condition":             label,
            "Model (kB/round/node)": f"{m_mean:.1f}" if not pd.isna(m_mean) else "N/A",
            "Topo (kB/round/node)":  f"{t_mean:.1f}",
            "Total (kB/round/node)": f"{m_mean + t_mean:.1f}" if not pd.isna(m_mean) else "N/A",
        })

    if not rows:
        print("No data found.")
        return

    table = pd.DataFrame(rows).set_index("Condition")
    print("\n=== Communication overhead (mean over rounds and honest nodes) ===\n")
    print(table.to_string())
    print(
        "\nNote: DMTT overhead = Topo (kB/round/node) column.\n"
        "      TOPO_CLAIM messages are small (pickle of {round_idx, neighbors list})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    print_table(df)


if __name__ == "__main__":
    main()
