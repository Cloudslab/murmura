#!/usr/bin/env python3
"""Run all experiments for the Evidential Trust paper evaluation.

This script runs experiments sequentially or in parallel, collecting
results for analysis and comparison.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse


# Experiment configurations organized by category
EXPERIMENTS = {
    "baseline": [
        "exp1_baseline_fedavg.yaml",
        "exp1_baseline_krum.yaml",
        "exp1_baseline_balance.yaml",
        "exp1_baseline_ubar.yaml",
        "exp1_baseline_sketchguard.yaml",
        "exp1_baseline_evidential.yaml",
    ],
    "attack_20": [
        "exp2_attack20_fedavg.yaml",
        "exp2_attack20_krum.yaml",
        "exp2_attack20_balance.yaml",
        "exp2_attack20_ubar.yaml",
        "exp2_attack20_sketchguard.yaml",
        "exp2_attack20_evidential.yaml",
    ],
    "attack_scaling": [
        "exp2_attack30_krum.yaml",
        "exp2_attack30_evidential.yaml",
        "exp2_attack40_krum.yaml",
        "exp2_attack40_evidential.yaml",
    ],
    "attack_directed": [
        "exp2_directed_krum.yaml",
        "exp2_directed_evidential.yaml",
    ],
    "heterogeneity": [
        "exp3_heterog_extreme_fedavg.yaml",
        "exp3_heterog_extreme_evidential.yaml",
        "exp3_heterog_mild_fedavg.yaml",
        "exp3_heterog_mild_evidential.yaml",
    ],
    "combined": [
        "exp3_heterog_extreme_attack_evidential.yaml",
        "exp3_heterog_extreme_attack_krum.yaml",
    ],
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def run_experiment(config_path: Path, results_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Run a single experiment using the murmura CLI.

    Args:
        config_path: Path to the experiment configuration file
        results_dir: Directory to save results
        verbose: Whether to print output

    Returns:
        Dictionary containing experiment results
    """
    config_name = config_path.stem
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")

    # Run the experiment
    cmd = ["murmura", "run", str(config_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=3600,  # 1 hour timeout
        )

        output = result.stdout
        error = result.stderr

        if verbose:
            print(output)
            if error:
                print(f"Stderr: {error}")

        # Parse results from output
        results = {
            "config": config_name,
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": output,
            "stderr": error,
            "timestamp": datetime.now().isoformat(),
        }

        # Save individual result
        result_file = results_dir / f"{config_name}_result.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    except subprocess.TimeoutExpired:
        print(f"Experiment {config_name} timed out!")
        return {
            "config": config_name,
            "status": "timeout",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error running {config_name}: {e}")
        return {
            "config": config_name,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def run_experiment_category(
    category: str,
    configs_dir: Path,
    results_dir: Path,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run all experiments in a category.

    Args:
        category: Category name from EXPERIMENTS
        configs_dir: Directory containing config files
        results_dir: Directory to save results
        verbose: Whether to print output

    Returns:
        List of result dictionaries
    """
    if category not in EXPERIMENTS:
        print(f"Unknown category: {category}")
        print(f"Available categories: {list(EXPERIMENTS.keys())}")
        return []

    results = []
    configs = EXPERIMENTS[category]

    print(f"\n{'#'*60}")
    print(f"Running category: {category} ({len(configs)} experiments)")
    print(f"{'#'*60}")

    for config_name in configs:
        config_path = configs_dir / config_name
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue

        result = run_experiment(config_path, results_dir, verbose)
        results.append(result)

    return results


def run_all_experiments(
    configs_dir: Path,
    results_dir: Path,
    categories: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all experiments across categories.

    Args:
        configs_dir: Directory containing config files
        results_dir: Directory to save results
        categories: List of categories to run (None = all)
        verbose: Whether to print output

    Returns:
        Dictionary mapping category to list of results
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    cats = categories or list(EXPERIMENTS.keys())

    for category in cats:
        results = run_experiment_category(category, configs_dir, results_dir, verbose)
        all_results[category] = results

    # Save summary
    summary_file = results_dir / "experiment_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "categories_run": cats,
        "total_experiments": sum(len(r) for r in all_results.values()),
        "successful": sum(
            1 for cat_results in all_results.values()
            for r in cat_results if r.get("status") == "success"
        ),
        "results": all_results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Results saved to: {results_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Evidential Trust experiments")
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Experiment category to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Run a specific config file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available experiments",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    configs_dir = project_root / "experiments" / "configs"
    results_dir = project_root / "experiments" / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.list:
        print("Available experiments:\n")
        for category, configs in EXPERIMENTS.items():
            print(f"  {category}:")
            for config in configs:
                print(f"    - {config}")
        return

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = configs_dir / args.config
        results_dir.mkdir(parents=True, exist_ok=True)
        run_experiment(config_path, results_dir, not args.quiet)
    elif args.category == "all":
        run_all_experiments(configs_dir, results_dir, verbose=not args.quiet)
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        run_experiment_category(args.category, configs_dir, results_dir, not args.quiet)


if __name__ == "__main__":
    main()
