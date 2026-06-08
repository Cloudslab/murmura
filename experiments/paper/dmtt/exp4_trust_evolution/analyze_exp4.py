#!/usr/bin/env python3
"""Analyse Experiment 4 — trust-score evolution over training rounds.

Reads the per-round trust log produced when dmtt.trust_log_path is set and
generates a two-panel figure:
  Left:  T_ij^topo vs. round for honest→honest vs. honest→Byzantine edges
  Right: c_hat (link reliability EMA) vs. round for the same edge groups

Usage:
    python experiments/paper/dmtt/exp4_trust_evolution/analyze_exp4.py

Outputs:
    results/exp4_trust_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DEFAULT_LOG     = HERE / "results" / "trust_log.jsonl"
DEFAULT_RESULTS = HERE / "results"


def load_trust_log(log_path: Path) -> list:
    records = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_edge_series(
    records: list,
    byzantine_nodes: set,
    num_rounds: int,
) -> dict:
    """Return per-round mean ± std of T_topo and c_hat, split by edge type.

    Edge types from honest node i's perspective:
      honest→honest:    j not in byzantine_nodes
      honest→byzantine: j in byzantine_nodes
    """
    hon_to_hon_T  = {r: [] for r in range(num_rounds)}
    hon_to_byz_T  = {r: [] for r in range(num_rounds)}
    hon_to_hon_c  = {r: [] for r in range(num_rounds)}
    hon_to_byz_c  = {r: [] for r in range(num_rounds)}

    for record in records:
        node  = record["node"]
        round_idx = record["round"]
        if node in byzantine_nodes:
            continue  # only track honest nodes' trust state
        if round_idx >= num_rounds:
            continue
        for peer_str, state in record["peers"].items():
            peer = int(peer_str)
            T    = state.get("T_topo", 1.0)
            c    = state.get("c_hat",  0.5)
            if peer in byzantine_nodes:
                hon_to_byz_T[round_idx].append(T)
                hon_to_byz_c[round_idx].append(c)
            else:
                hon_to_hon_T[round_idx].append(T)
                hon_to_hon_c[round_idx].append(c)

    def agg(d):
        rounds = sorted(d)
        means  = [np.mean(d[r]) if d[r] else np.nan for r in rounds]
        stds   = [np.std(d[r])  if d[r] else 0.0    for r in rounds]
        return np.array(rounds), np.array(means), np.array(stds)

    return {
        "hon_to_hon_T":  agg(hon_to_hon_T),
        "hon_to_byz_T":  agg(hon_to_byz_T),
        "hon_to_hon_c":  agg(hon_to_hon_c),
        "hon_to_byz_c":  agg(hon_to_byz_c),
    }


def plot(series: dict, results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel — T_ij^topo
    ax = axes[0]
    for label, key, color in [
        ("Honest→honest",    "hon_to_hon_T", "steelblue"),
        ("Honest→Byzantine", "hon_to_byz_T", "tomato"),
    ]:
        rounds, means, stds = series[key]
        ax.plot(rounds + 1, means, label=label, color=color)
        ax.fill_between(rounds + 1, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel(r"$T_{ij}^{topo}$")
    ax.set_title("Topology trust score vs. round")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel — c_hat (link reliability)
    ax = axes[1]
    for label, key, color in [
        ("Honest→honest",    "hon_to_hon_c", "steelblue"),
        ("Honest→Byzantine", "hon_to_byz_c", "tomato"),
    ]:
        rounds, means, stds = series[key]
        ax.plot(rounds + 1, means, label=label, color=color)
        ax.fill_between(rounds + 1, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel(r"$\hat{c}_{ij}$ (link reliability)")
    ax.set_title("Link reliability EMA vs. round")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Trust-score evolution — 30 % Byzantine (topology-liar + Gaussian)", y=1.02)
    plt.tight_layout()

    fig_path = results_dir / "exp4_trust_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {fig_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log",         type=Path, default=DEFAULT_LOG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--byzantine-nodes",
        type=str,
        default=None,
        help=(
            "Comma-separated Byzantine node IDs.  If not given, the script infers them as the "
            "nodes whose T_topo trajectories decrease earliest.  Example: --byzantine-nodes 7,8,9"
        ),
    )
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(
            f"Trust log not found: {args.log}\n"
            "Run  bash experiments/paper/dmtt/exp4_trust_evolution/run_exp4.sh  first."
        )

    records = load_trust_log(args.log)
    print(f"Loaded {len(records)} trust-state records")

    if args.byzantine_nodes:
        byz = set(int(x) for x in args.byzantine_nodes.split(","))
    else:
        # Heuristic: Byzantine nodes are 30 % = 3 out of 10; round to int
        byz = _infer_byzantine_nodes(records, byz_fraction=0.3)
        print(f"Inferred Byzantine nodes: {sorted(byz)}")

    series = compute_edge_series(records, byz, num_rounds=args.rounds)
    plot(series, args.results_dir)


def _infer_byzantine_nodes(records: list, byz_fraction: float) -> set:
    """Identify Byzantine nodes as those with consistently low incoming T_topo."""
    from collections import defaultdict
    late_T: dict = defaultdict(list)
    all_rounds = sorted({r["round"] for r in records})
    cutoff = all_rounds[len(all_rounds) // 2]  # second half of training

    for record in records:
        if record["round"] < cutoff:
            continue
        node = record["node"]
        for peer_str, state in record["peers"].items():
            peer = int(peer_str)
            late_T[peer].append(state.get("T_topo", 1.0))

    # Nodes with lowest mean T_topo from honest nodes' perspective
    mean_T = {p: float(np.mean(vs)) for p, vs in late_T.items()}
    num_byz = round(byz_fraction * len(mean_T))
    sorted_nodes = sorted(mean_T, key=lambda n: mean_T[n])
    return set(sorted_nodes[:num_byz])


if __name__ == "__main__":
    main()
