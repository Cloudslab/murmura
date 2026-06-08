#!/usr/bin/env python3
"""Analyse Experiment 5 (sensitivity analysis) results.

Usage:
    python experiments/paper/dmtt/exp5_sensitivity/analyze_exp5.py

Outputs:
    results/exp5a_heterogeneity.png  — honest acc vs. alpha for C2 and C3
    results/exp5b_comm_range.png     — honest acc vs. comm_range
    results/exp5b_max_speed.png      — honest acc vs. max_speed
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def _extract_alpha(name: str) -> float:
    import re
    m = re.search(r"alpha([0-9p]+)", name)
    if m:
        return float(m.group(1).replace("p", "."))
    return float("nan")


def _extract_param(name: str, key: str) -> float:
    import re
    m = re.search(rf"{key}([0-9]+)", name)
    return float(m.group(1)) if m else float("nan")


def plot_heterogeneity(df: pd.DataFrame, results_dir: Path) -> None:
    het = df[df["experiment_name"].str.contains("Exp5a", na=False)].copy()
    if het.empty:
        print("No Exp5a (heterogeneity) data found — skipping.")
        return

    final_round = het["round"].max()
    final = het[het["round"] == final_round].copy()
    final["alpha"] = final["experiment_name"].apply(_extract_alpha)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color, cond_substr in [
        ("C2 (dynamic FedAvg)", "steelblue", "C2"),
        ("C3 DMTT (proposed)", "tomato",     "C3"),
    ]:
        sub = final[final["experiment_name"].str.contains(cond_substr, na=False)]
        if sub.empty:
            continue
        agg = sub.groupby("alpha")["honest_accuracy"].agg(["mean", "std"]).sort_index()
        ax.plot(agg.index, agg["mean"], marker="o", label=label, color=color)
        ax.fill_between(
            agg.index,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Dirichlet α (smaller = more non-IID)")
    ax.set_ylabel("Honest-node accuracy (round 50)")
    ax.set_title("Heterogeneity sensitivity — 30 % Byzantine")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = results_dir / "exp5a_heterogeneity.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


def plot_mobility(df: pd.DataFrame, results_dir: Path, param: str, label: str, xlabel: str) -> None:
    mob = df[df["experiment_name"].str.contains("Exp5b", na=False)].copy()
    if mob.empty:
        print("No Exp5b (mobility) data found — skipping.")
        return

    final_round = mob["round"].max()
    final = mob[mob["round"] == final_round].copy()
    final["param"] = final["experiment_name"].apply(lambda n: _extract_param(n, param))

    # Filter to runs that swept this parameter (fix the other)
    if param == "cr":
        final = final[final["experiment_name"].str.contains("sp8", na=False)]
    else:
        final = final[final["experiment_name"].str.contains("cr40", na=False)]

    if final.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl, color, cond_substr in [
        ("C2 (dynamic FedAvg)", "steelblue", "C2"),
        ("C3 DMTT (proposed)", "tomato",     "C3"),
    ]:
        sub = final[final["experiment_name"].str.contains(cond_substr, na=False)]
        if sub.empty:
            continue
        agg = sub.groupby("param")["honest_accuracy"].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=lbl, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Honest-node accuracy (round 50)")
    ax.set_title(f"Mobility sensitivity ({label}) — 30 % Byzantine")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = results_dir / f"exp5b_{label}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    plot_heterogeneity(df, args.results_dir)
    plot_mobility(df, args.results_dir, "cr", "comm_range", "Communication range")
    plot_mobility(df, args.results_dir, "sp", "max_speed",  "Max speed per round")


if __name__ == "__main__":
    main()
