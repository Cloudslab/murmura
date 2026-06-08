#!/usr/bin/env python3
"""Analyse Experiment 3 (ablation study) results.

Usage:
    python experiments/paper/dmtt/exp3_ablation/analyze_exp3.py

Outputs:
    - Console: honest-accuracy mean ± std per ablation variant
    - results/exp3_table.csv
    - results/exp3_curves.png — learning curves for each variant
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results"

VARIANT_ORDER = [
    "full",
    "no_beta_trust",
    "no_topo_claims",
    "no_model_compat",
    "no_link_rel",
    "uniform_q",
]

VARIANT_LABELS = {
    "full":           "DMTT-full",
    "no_beta_trust":  "No Beta trust",
    "no_topo_claims": "No TOPO_CLAIM",
    "no_model_compat":"No model compat.",
    "no_link_rel":    "No link reliability",
    "uniform_q":      "Uniform weights",
}


def _variant_from_name(name: str) -> str:
    """Infer variant label from experiment_name column."""
    for v in VARIANT_ORDER:
        if v in name:
            return v
    return name


def load_results(results_dir: Path) -> pd.DataFrame:
    parquets = sorted(results_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No Parquet files found in {results_dir}.\n"
            "Run  bash experiments/paper/dmtt/exp3_ablation/run_exp3.sh  first."
        )
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    df["variant"] = df["experiment_name"].apply(_variant_from_name)
    print(f"Loaded {len(parquets)} files  ({len(df):,} rows)")
    return df


def print_table(df: pd.DataFrame, results_dir: Path) -> None:
    final_round = df["round"].max()
    final = df[df["round"] == final_round]

    rows = []
    for v in VARIANT_ORDER:
        sub = final[final["variant"] == v]
        if sub.empty:
            row = {"Variant": VARIANT_LABELS.get(v, v), "Honest acc (30 % Byz)": "—"}
        else:
            m = sub["honest_accuracy"].mean()
            s = sub["honest_accuracy"].std()
            row = {
                "Variant": VARIANT_LABELS.get(v, v),
                "Honest acc (30 % Byz)": f"{m:.3f} ± {s:.3f}",
            }
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Variant")
    print(f"\n=== Honest-node accuracy at round {final_round} (mean ± std, 3 seeds) ===\n")
    print(table.to_string())
    print()

    csv_path = results_dir / "exp3_table.csv"
    table.to_csv(csv_path)
    print(f"Saved → {csv_path}")


def plot_curves(df: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    for i, v in enumerate(VARIANT_ORDER):
        sub = df[df["variant"] == v]
        if sub.empty:
            continue
        agg = sub.groupby("round")["honest_accuracy"].agg(["mean", "std"])
        ax.plot(agg.index, agg["mean"], label=VARIANT_LABELS.get(v, v), color=colors[i])
        ax.fill_between(
            agg.index,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.15,
            color=colors[i],
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Honest-node accuracy")
    ax.set_title("Ablation study — honest-node accuracy (30 % Byzantine)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = results_dir / "exp3_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved → {fig_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    print_table(df, args.results_dir)
    plot_curves(df, args.results_dir)


if __name__ == "__main__":
    main()
