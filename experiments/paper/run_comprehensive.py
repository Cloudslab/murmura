#!/usr/bin/env python3
"""Run comprehensive paper experiments across all categories.

Categories:
- baseline: Basic experiments (no attacks, fully connected, α=0.5)
- heterogeneity: Varying Dirichlet α ∈ {0.1, 0.5, 1.0}
- attacks: Byzantine attacks (Gaussian, Directed Deviation) at various percentages
- topologies: Different network structures (ring, fully, erdos, k-regular)
- ablation: Hyperparameter sensitivity for evidential_trust

Usage:
    # Run all experiments
    python experiments/paper/run_comprehensive.py

    # Run specific category
    python experiments/paper/run_comprehensive.py --category heterogeneity

    # Run specific dataset within category
    python experiments/paper/run_comprehensive.py --category attacks --dataset ppg_dalia

    # Generate summary only
    python experiments/paper/run_comprehensive.py --summary-only
"""

import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

PAPER_DIR = Path(__file__).parent
CATEGORIES = ["baseline", "heterogeneity", "attacks", "topologies", "ablation"]
DATASETS = ["uci_har", "pamap2", "ppg_dalia"]

# Map baseline to dataset folders
BASELINE_FOLDERS = {
    "uci_har": PAPER_DIR / "uci_har",
    "pamap2": PAPER_DIR / "pamap2",
    "ppg_dalia": PAPER_DIR / "ppg_dalia",
}


def parse_experiment_output(output: str) -> Dict:
    """Parse the experiment output to extract metrics."""
    results = {
        "rounds": [],
        "final_accuracy": None,
        "final_std": None,
        "peak_accuracy": None,
        "convergence_round": None,
        "final_vacuity": None,
        "final_entropy": None,
        "final_strength": None,
    }

    round_pattern = r"Round (\d+): Mean Accuracy = ([\d.]+) ± ([\d.]+)"
    uncertainty_pattern = r"Uncertainty: Vacuity=([\d.]+), Entropy=([\d.]+), Strength=([\d.]+)"
    honest_pattern = r"Honest: ([\d.]+), Compromised: ([\d.]+)"

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

        honest_match = re.search(honest_pattern, line)
        if honest_match and current_round:
            current_round["honest_accuracy"] = float(honest_match.group(1))
            current_round["compromised_accuracy"] = float(honest_match.group(2))

        uncertainty_match = re.search(uncertainty_pattern, line)
        if uncertainty_match and current_round:
            current_round["vacuity"] = float(uncertainty_match.group(1))
            current_round["entropy"] = float(uncertainty_match.group(2))
            current_round["strength"] = float(uncertainty_match.group(3))
            rounds_data.append(current_round)
            current_round = {}

    if rounds_data:
        results["rounds"] = rounds_data
        final = rounds_data[-1]
        results["final_accuracy"] = final["accuracy"]
        results["final_std"] = final["std"]
        results["final_vacuity"] = final.get("vacuity")
        results["final_entropy"] = final.get("entropy")
        results["final_strength"] = final.get("strength")
        results["final_honest_accuracy"] = final.get("honest_accuracy")
        results["final_compromised_accuracy"] = final.get("compromised_accuracy")
        results["peak_accuracy"] = max(r["accuracy"] for r in rounds_data)

        for r in rounds_data:
            if r["accuracy"] >= 0.80:
                results["convergence_round"] = r["round"]
                break

    return results


def run_experiment(config_path: Path, timeout: int = 1800) -> Dict:
    """Run a single experiment and return parsed results."""
    print(f"\n{'='*60}")
    print(f"Running: {config_path.relative_to(PAPER_DIR)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            ["uv", "run", "murmura", "run", str(config_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)  # Print last 2000 chars

        if result.returncode != 0:
            return {"error": f"Experiment failed with code {result.returncode}"}

        return parse_experiment_output(output)

    except subprocess.TimeoutExpired:
        return {"error": f"Experiment timed out ({timeout}s)"}
    except Exception as e:
        return {"error": str(e)}


def get_configs_for_category(category: str, dataset: Optional[str] = None) -> List[Path]:
    """Get all config files for a category."""
    configs = []

    if category == "baseline":
        for ds, folder in BASELINE_FOLDERS.items():
            if dataset and ds != dataset:
                continue
            if folder.exists():
                configs.extend(sorted(folder.glob("*.yaml")))
    else:
        category_dir = PAPER_DIR / category
        if category_dir.exists():
            for ds in DATASETS:
                if dataset and ds != dataset:
                    continue
                ds_dir = category_dir / ds
                if ds_dir.exists():
                    configs.extend(sorted(ds_dir.glob("*.yaml")))

    return configs


def run_category(
    category: str,
    dataset: Optional[str] = None,
    results_file: Path = None,
    force: bool = False,
) -> Dict:
    """Run all experiments in a category."""
    results_file = results_file or PAPER_DIR / f"results_{category}.json"

    # Load existing results
    if results_file.exists() and not force:
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "metadata": {
                "category": category,
                "timestamp": datetime.now().isoformat(),
            },
            "results": {}
        }

    configs = get_configs_for_category(category, dataset)
    print(f"\nFound {len(configs)} configs for category '{category}'")

    for config_path in configs:
        # Extract experiment key
        rel_path = config_path.relative_to(PAPER_DIR)
        exp_key = str(rel_path).replace("/", "__").replace(".yaml", "")

        # Skip if already have successful results
        existing = all_results["results"].get(exp_key, {})
        if existing and "error" not in existing and not force:
            print(f"\nSkipping {exp_key} (already have results)")
            continue

        result = run_experiment(config_path)
        all_results["results"][exp_key] = result
        all_results["metadata"]["timestamp"] = datetime.now().isoformat()

        # Save after each experiment
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    return all_results


def generate_category_summary(category: str, results: Dict) -> str:
    """Generate markdown summary for a category."""
    lines = [f"# {category.title()} Experiments\n"]
    lines.append(f"Generated: {results['metadata']['timestamp']}\n")

    # Group by dataset
    by_dataset = {}
    for exp_key, result in results["results"].items():
        parts = exp_key.split("__")
        if category == "baseline":
            ds = parts[0]
            algo = parts[1]
        else:
            ds = parts[1]
            algo = parts[2] if len(parts) > 2 else parts[1]

        if ds not in by_dataset:
            by_dataset[ds] = {}
        by_dataset[ds][algo] = result

    for ds in DATASETS:
        if ds not in by_dataset:
            continue

        lines.append(f"\n## {ds.upper().replace('_', ' ')}\n")
        lines.append("| Experiment | Final Acc | Std | Peak | Vacuity | Entropy |")
        lines.append("|------------|-----------|-----|------|---------|---------|")

        for exp_name, r in sorted(by_dataset[ds].items()):
            if "error" in r:
                lines.append(f"| {exp_name} | ERROR | - | - | - | - |")
            elif r.get("final_accuracy") is not None:
                acc = f"{r['final_accuracy']*100:.2f}%"
                std = f"{r['final_std']*100:.2f}%"
                peak = f"{r['peak_accuracy']*100:.2f}%"
                vac = f"{r.get('final_vacuity', 0):.3f}"
                ent = f"{r.get('final_entropy', 0):.3f}"
                lines.append(f"| {exp_name} | {acc} | {std} | {peak} | {vac} | {ent} |")
            else:
                lines.append(f"| {exp_name} | - | - | - | - | - |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive paper experiments")
    parser.add_argument("--category", choices=CATEGORIES, help="Run specific category only")
    parser.add_argument("--dataset", choices=DATASETS, help="Run specific dataset only")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--summary-only", action="store_true", help="Generate summaries only")
    parser.add_argument("--list", action="store_true", help="List all configs without running")
    args = parser.parse_args()

    if args.list:
        for category in (args.category and [args.category]) or CATEGORIES:
            configs = get_configs_for_category(category, args.dataset)
            print(f"\n{category}: {len(configs)} configs")
            for c in configs[:5]:
                print(f"  {c.relative_to(PAPER_DIR)}")
            if len(configs) > 5:
                print(f"  ... and {len(configs) - 5} more")
        return

    categories_to_run = [args.category] if args.category else CATEGORIES

    for category in categories_to_run:
        results_file = PAPER_DIR / f"results_{category}.json"

        if args.summary_only:
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                summary = generate_category_summary(category, results)
                summary_file = PAPER_DIR / f"SUMMARY_{category.upper()}.md"
                with open(summary_file, 'w') as f:
                    f.write(summary)
                print(f"Summary saved to: {summary_file}")
            else:
                print(f"No results file for {category}")
            continue

        print(f"\n{'#'*60}")
        print(f"# Running {category.upper()} experiments")
        print(f"{'#'*60}")

        results = run_category(
            category=category,
            dataset=args.dataset,
            results_file=results_file,
            force=args.force,
        )

        # Generate summary
        summary = generate_category_summary(category, results)
        summary_file = PAPER_DIR / f"SUMMARY_{category.upper()}.md"
        with open(summary_file, 'w') as f:
            f.write(summary)

        print(f"\nResults saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
