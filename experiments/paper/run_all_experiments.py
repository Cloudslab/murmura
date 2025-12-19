#!/usr/bin/env python3
"""Run all paper experiments and collect results.

This script runs comprehensive experiments across three wearable datasets
(UCI HAR, PAMAP2, ExtraSensory) with six aggregation algorithms.

Usage:
    python experiments/paper/run_all_experiments.py

    # Run specific dataset only
    python experiments/paper/run_all_experiments.py --dataset uci_har

    # Run specific algorithm only
    python experiments/paper/run_all_experiments.py --algorithm evidential_trust
"""

import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


# Experiment configurations
DATASETS = ["uci_har", "pamap2", "extrasensory"]
ALGORITHMS = ["fedavg", "krum", "balance", "ubar", "sketchguard", "evidential_trust"]

PAPER_DIR = Path(__file__).parent
RESULTS_FILE = PAPER_DIR / "results.json"


def parse_experiment_output(output: str) -> Dict:
    """Parse the experiment output to extract metrics."""
    results = {
        "rounds": [],
        "final_accuracy": None,
        "final_std": None,
        "peak_accuracy": None,
        "convergence_round": None,  # Round when accuracy first exceeds 80%
        "final_vacuity": None,
        "final_entropy": None,
        "final_strength": None,
    }

    # Extract round-by-round metrics
    round_pattern = r"Round (\d+): Mean Accuracy = ([\d.]+) ± ([\d.]+)"
    uncertainty_pattern = r"Uncertainty: Vacuity=([\d.]+), Entropy=([\d.]+), Strength=([\d.]+)"

    rounds_data = []
    current_round = {}

    for line in output.split('\n'):
        round_match = re.search(round_pattern, line)
        if round_match:
            current_round = {
                "round": int(round_match.group(1)),
                "accuracy": float(round_match.group(2)),
                "std": float(round_match.group(3)),
            }

        uncertainty_match = re.search(uncertainty_pattern, line)
        if uncertainty_match and current_round:
            current_round["vacuity"] = float(uncertainty_match.group(1))
            current_round["entropy"] = float(uncertainty_match.group(2))
            current_round["strength"] = float(uncertainty_match.group(3))
            rounds_data.append(current_round)
            current_round = {}

    if rounds_data:
        results["rounds"] = rounds_data

        # Final metrics (last round)
        final = rounds_data[-1]
        results["final_accuracy"] = final["accuracy"]
        results["final_std"] = final["std"]
        results["final_vacuity"] = final.get("vacuity")
        results["final_entropy"] = final.get("entropy")
        results["final_strength"] = final.get("strength")

        # Peak accuracy
        results["peak_accuracy"] = max(r["accuracy"] for r in rounds_data)

        # Convergence round (first round with accuracy >= 0.80)
        for r in rounds_data:
            if r["accuracy"] >= 0.80:
                results["convergence_round"] = r["round"]
                break

    return results


def run_experiment(config_path: Path) -> Dict:
    """Run a single experiment and return parsed results."""
    print(f"\n{'='*60}")
    print(f"Running: {config_path.name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            ["uv", "run", "murmura", "run", str(config_path)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        output = result.stdout + result.stderr
        print(output)

        if result.returncode != 0:
            return {"error": f"Experiment failed with code {result.returncode}"}

        return parse_experiment_output(output)

    except subprocess.TimeoutExpired:
        return {"error": "Experiment timed out (30 minutes)"}
    except Exception as e:
        return {"error": str(e)}


def run_all_experiments(
    datasets: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
) -> Dict:
    """Run all experiments and collect results."""
    datasets = datasets or DATASETS
    algorithms = algorithms or ALGORITHMS

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "datasets": datasets,
            "algorithms": algorithms,
        },
        "results": {}
    }

    # Load existing results if available
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
            all_results["results"] = existing.get("results", {})

    for dataset in datasets:
        if dataset not in all_results["results"]:
            all_results["results"][dataset] = {}

        for algorithm in algorithms:
            config_path = PAPER_DIR / dataset / f"{algorithm}.yaml"

            if not config_path.exists():
                print(f"Warning: Config not found: {config_path}")
                continue

            # Skip if already have results (unless error)
            existing_result = all_results["results"][dataset].get(algorithm, {})
            if existing_result and "error" not in existing_result:
                print(f"\nSkipping {dataset}/{algorithm} (already have results)")
                continue

            result = run_experiment(config_path)
            all_results["results"][dataset][algorithm] = result

            # Save after each experiment
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    return all_results


def generate_summary_table(results: Dict) -> str:
    """Generate a markdown summary table from results."""
    lines = ["# Paper Experiment Results\n"]
    lines.append(f"Generated: {results['metadata']['timestamp']}\n")

    for dataset in results["metadata"]["datasets"]:
        lines.append(f"\n## {dataset.upper().replace('_', ' ')}\n")
        lines.append("| Algorithm | Final Acc | Std Dev | Conv. Round | Peak Acc | Vacuity | Entropy |")
        lines.append("|-----------|-----------|---------|-------------|----------|---------|---------|")

        dataset_results = results["results"].get(dataset, {})

        for algorithm in results["metadata"]["algorithms"]:
            r = dataset_results.get(algorithm, {})

            if "error" in r:
                lines.append(f"| {algorithm} | ERROR | - | - | - | - | - |")
            elif r.get("final_accuracy") is not None:
                acc = f"{r['final_accuracy']*100:.2f}%"
                std = f"{r['final_std']*100:.2f}%"
                conv = str(r.get('convergence_round', '-'))
                peak = f"{r['peak_accuracy']*100:.2f}%"
                vac = f"{r.get('final_vacuity', 0):.3f}"
                ent = f"{r.get('final_entropy', 0):.3f}"
                lines.append(f"| {algorithm} | {acc} | {std} | {conv} | {peak} | {vac} | {ent} |")
            else:
                lines.append(f"| {algorithm} | - | - | - | - | - | - |")

    # Generate LaTeX table
    lines.append("\n\n## LaTeX Table\n")
    lines.append("```latex")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Model Personalization Performance Across Wearable Datasets (Non-IID, α=0.1)}")
    lines.append("\\label{tab:personalization_all}")
    lines.append("\\begin{tabular}{l|cccc|cccc|cccc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{4}{c|}{\\textbf{UCI HAR}} & \\multicolumn{4}{c|}{\\textbf{PAMAP2}} & \\multicolumn{4}{c}{\\textbf{ExtraSensory}} \\\\")
    lines.append("\\textbf{Algorithm} & Acc & Std & Conv & Peak & Acc & Std & Conv & Peak & Acc & Std & Conv & Peak \\\\")
    lines.append("\\midrule")

    for algorithm in results["metadata"]["algorithms"]:
        row = [algorithm]
        for dataset in results["metadata"]["datasets"]:
            r = results["results"].get(dataset, {}).get(algorithm, {})
            if r.get("final_accuracy") is not None:
                row.extend([
                    f"{r['final_accuracy']*100:.1f}",
                    f"{r['final_std']*100:.1f}",
                    str(r.get('convergence_round', '-')),
                    f"{r['peak_accuracy']*100:.1f}",
                ])
            else:
                row.extend(["-", "-", "-", "-"])
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument("--dataset", choices=DATASETS, help="Run specific dataset only")
    parser.add_argument("--algorithm", choices=ALGORITHMS, help="Run specific algorithm only")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--summary-only", action="store_true", help="Only generate summary from existing results")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else None
    algorithms = [args.algorithm] if args.algorithm else None

    if args.summary_only:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        else:
            print("No results file found!")
            sys.exit(1)
    else:
        if args.force and RESULTS_FILE.exists():
            RESULTS_FILE.unlink()

        results = run_all_experiments(datasets, algorithms)

    # Generate and save summary
    summary = generate_summary_table(results)
    summary_file = PAPER_DIR / "RESULTS_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(summary)
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
