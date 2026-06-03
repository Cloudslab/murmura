#!/usr/bin/env python3
"""Analyse Experiment 1 results and print the main comparison table.

Usage (from repo root, after experiments have run):
    python experiments/paper/dmtt/exp1_main_comparison/analyze_exp1.py

    # Custom results directory
    python experiments/paper/dmtt/exp1_main_comparison/analyze_exp1.py --results-dir path/to/results

Outputs:
    - Console: main comparison table (honest accuracy mean ± std at final round)
    - results/exp1_table.csv   — machine-readable version of the table
    - results/exp1_curves.png  — honest-accuracy learning curves per condition
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results"

CONDITION_ORDER = [
    "c1_static",
    "dyn_fedavg",
    "dyn_krum",
    "dyn_balance",
    "dyn_ubar",
    "c3_dmtt",
]

CONDITION_LABELS = {
    "c1_static":   "FedAvg (static)",
    "dyn_fedavg":  "FedAvg (dynamic)",
    "dyn_krum":    "Krum (dynamic)",
    "dyn_balance": "BALANCE (dynamic)",
    "dyn_ubar":    "UBAR (dynamic)",
    "c3_dmtt":     "DMTT (proposed)",
}

BYZ_LABELS = {0.1: "10 %", 0.2: "20 %", 0.3: "30 %"}


def load_results(results_dir: Path) -> pd.DataFrame:
    parquets = sorted(results_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No Parquet files found in {results_dir}.\n"
            "Run  bash experiments/paper/dmtt/exp1_main_comparison/run_exp1.sh  first."
        )
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"Loaded {len(parquets)} experiment files  ({len(df):,} rows)")
    return df


def print_table(df: pd.DataFrame, results_dir: Path) -> None:
    """Print honest-accuracy mean ± std at the final round for each condition × byz_pct."""
    final_round = df["round"].max()
    final = df[df["round"] == final_round].copy()

    rows = []
    for cond in CONDITION_ORDER:
        row = {"Method": CONDITION_LABELS.get(cond, cond)}
        for byz in sorted(BYZ_LABELS):
            sub = final[(final["condition"] == cond) & (final["byzantine_pct"] == byz)]
            if sub.empty:
                row[BYZ_LABELS[byz]] = "—"
            else:
                m = sub["honest_accuracy"].mean()
                s = sub["honest_accuracy"].std()
                row[BYZ_LABELS[byz]] = f"{m:.3f} ± {s:.3f}"
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Method")
    print("\n=== Honest-node accuracy at round", final_round, "(mean ± std, 3 seeds) ===\n")
    print(table.to_string())
    print()

    csv_path = results_dir / "exp1_table.csv"
    table.to_csv(csv_path)
    print(f"Saved → {csv_path}")


def plot_curves(df: pd.DataFrame, results_dir: Path) -> None:
    """Plot per-round honest accuracy curves for each condition at 30 % Byzantine."""
    byz = 0.3
    sub = df[df["byzantine_pct"] == byz]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    for i, cond in enumerate(CONDITION_ORDER):
        cdf = sub[sub["condition"] == cond]
        if cdf.empty:
            continue
        agg = cdf.groupby("round")["honest_accuracy"].agg(["mean", "std"])
        ax.plot(agg.index, agg["mean"], label=CONDITION_LABELS.get(cond, cond), color=colors[i])
        ax.fill_between(
            agg.index,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.15,
            color=colors[i],
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Honest-node accuracy")
    ax.set_title(f"Honest-node accuracy (30 % Byzantine, topology-liar + Gaussian)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = results_dir / "exp1_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved → {fig_path}")
    plt.close()


def print_convergence(df: pd.DataFrame) -> None:
    """Print the median round at which honest accuracy first crosses 0.80."""
    threshold = 0.80

    records = []
    for (cond, byz, seed), grp in df.groupby(["condition", "byzantine_pct", "seed"]):
        crossed = grp[grp["honest_accuracy"] >= threshold]["round"]
        conv = int(crossed.min()) if not crossed.empty else None
        records.append({"condition": cond, "byzantine_pct": byz, "seed": seed, "conv_round": conv})

    conv_df = pd.DataFrame(records)
    summary = (
        conv_df.groupby(["condition", "byzantine_pct"])["conv_round"]
        .median()
        .unstack("byzantine_pct")
        .reindex(CONDITION_ORDER)
        .rename(columns=BYZ_LABELS)
        .rename(index=CONDITION_LABELS)
    )
    print(f"\n=== Median convergence round (first honest acc ≥ {threshold}) ===\n")
    print(summary.to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    print_table(df, args.results_dir)
    print_convergence(df)
    plot_curves(df, args.results_dir)


if __name__ == "__main__":
    main()
