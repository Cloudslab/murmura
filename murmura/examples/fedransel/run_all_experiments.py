#!/usr/bin/env python3
"""
FedRansel Experiment Runner
===========================

This script runs all experiments from the FedRansel paper:

Experiment 1: Convergence + Communication
- 4 datasets (CIFAR-10, FEMNIST, UCI HAR, PAMAP2)
- IID + non-IID settings
- Compare: FedRansel, FedAvg, DP-FedAvg (ε=4), Top-K (10%)

Experiment 2: Privacy-Utility Tradeoff
- Grid: T_l ∈ {0.3, 0.5, 0.7} × T_g ∈ {0.5, 0.7, 1.0} × τ ∈ {1, N/4}

Experiment 3: Byzantine Robustness
- Byzantine fraction f ∈ {0, 0.1, 0.2, 0.3}
- Attack types: gaussian, directed_deviation
- τ ∈ {1, 5, 15}

Experiment 4: Conjecture Validation
- Synthetic validation of sampling probabilities (Eqs. 3-6)

Usage:
    python -m murmura.examples.fedransel.run_all_experiments [OPTIONS]

Options:
    --exp1          Run Experiment 1 only
    --exp2          Run Experiment 2 only
    --exp3          Run Experiment 3 only
    --exp4          Run Experiment 4 only
    --quick         Run with reduced rounds for quick testing
    --output-dir    Directory to save results (default: ./fedransel_results)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Murmura imports
from murmura.config.schema import (
    Config, ExperimentConfig, TopologyConfig, AggregationConfig,
    AttackConfig, TrainingConfig, DataConfig, ModelConfig, NetworkConfig
)
from murmura.core.network import Network
from murmura.utils.seed import set_seed
from murmura.utils.device import get_device

console = Console()


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    config: Dict[str, Any]
    final_accuracy: float
    final_loss: float
    history: Dict[str, List[Any]]
    aggregator_stats: Optional[Dict[str, Any]] = None
    detailed_metrics: Optional[List[Dict[str, Any]]] = None
    aggregator_detailed_history: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0


# =============================================================================
# Configuration Builders
# =============================================================================

def build_base_config(
    name: str,
    dataset: str,
    algorithm: str,
    num_nodes: int = 50,
    rounds: int = 300,
    partition_method: str = "iid",
    alpha: float = 0.5,
    mode: str = "decentralized",
    agg_params: Optional[Dict] = None,
    attack_enabled: bool = False,
    attack_type: Optional[str] = None,
    attack_percentage: float = 0.0,
    attack_params: Optional[Dict] = None,
) -> Config:
    """Build a configuration object programmatically."""

    # Dataset-specific settings
    dataset_configs = {
        "cifar10": {
            "adapter": "cifar10",
            "data_params": {
                "data_path": "./data/cifar10",
                "partition_method": partition_method,
                "alpha": alpha,
                "normalize": True,
            },
            "model_factory": "examples.cifar10.resnet18",
            "model_params": {"num_classes": 10},
        },
        "femnist": {
            "adapter": "leaf.femnist",
            "data_params": {"data_path": "leaf/data/femnist/data"},
            "model_factory": "examples.leaf.LEAFFEMNISTModel",
            "model_params": {"num_classes": 62},
        },
        "uci_har": {
            "adapter": "wearables.uci_har",
            "data_params": {
                "data_path": "wearables_datasets/UCI HAR Dataset",
                "split": "train",
                "partition_method": partition_method,
                "alpha": alpha,
                "normalize": True,
            },
            "model_factory": "examples.wearables.uci_har",
            "model_params": {"input_dim": 561, "hidden_dims": [256, 128], "num_classes": 6},
        },
        "pamap2": {
            "adapter": "wearables.pamap2",
            "data_params": {
                "data_path": "wearables_datasets/PAMAP2_Dataset",
                "partition_method": partition_method,
                "alpha": alpha,
                "window_size": 100,
                "window_stride": 50,
                "normalize": True,
                "include_heart_rate": True,
            },
            "model_factory": "examples.wearables.pamap2",
            "model_params": {"input_dim": 4000, "hidden_dims": [512, 256], "num_classes": 12},
        },
    }

    ds_config = dataset_configs[dataset]

    return Config(
        experiment=ExperimentConfig(
            name=name,
            seed=42,
            rounds=rounds,
            verbose=False,
        ),
        topology=TopologyConfig(
            type="fully",
            num_nodes=num_nodes,
        ),
        network=NetworkConfig(mode=mode),
        aggregation=AggregationConfig(
            algorithm=algorithm,
            params=agg_params or {},
        ),
        attack=AttackConfig(
            enabled=attack_enabled,
            type=attack_type,
            percentage=attack_percentage,
            params=attack_params or {},
        ),
        training=TrainingConfig(
            local_epochs=2,
            batch_size=64,
            lr=0.01,
        ),
        data=DataConfig(
            adapter=ds_config["adapter"],
            params=ds_config["data_params"],
        ),
        model=ModelConfig(
            factory=ds_config["model_factory"],
            params=ds_config["model_params"],
        ),
    )


def run_single_experiment(config: Config, device: torch.device) -> ExperimentResult:
    """Run a single experiment and return results."""
    import time
    from murmura.cli import _load_dataset_adapter, _load_model_factory, _create_aggregator_factory

    start_time = time.time()

    # Set seed
    set_seed(config.experiment.seed)

    # Load components
    dataset_adapter = _load_dataset_adapter(config)
    model_factory = _load_model_factory(config)
    aggregator_factory = _create_aggregator_factory(config, model_factory, device)

    # Check for evidential models
    evidential = config.model.factory.startswith("examples.wearables.")
    criterion = None
    if evidential:
        from murmura.examples.wearables import get_evidential_loss
        num_classes = config.model.params.get("num_classes", 6)
        annealing_rounds = config.experiment.rounds // 2
        criterion = get_evidential_loss(
            num_classes=num_classes,
            annealing_epochs=annealing_rounds,
            lambda_weight=0.1,
        )

    # Create and train network
    network = Network.from_config(
        config=config,
        model_factory=model_factory,
        dataset_adapter=dataset_adapter,
        aggregator_factory=aggregator_factory,
        device=device,
        criterion=criterion,
        evidential=evidential,
    )

    history = network.train(
        rounds=config.experiment.rounds,
        local_epochs=config.training.local_epochs,
        lr=config.training.lr,
        verbose=False,
    )

    # Get aggregator statistics (from first node's aggregator)
    agg_stats = None
    if network.nodes and network.nodes[0].aggregator:
        agg_stats = network.nodes[0].aggregator.get_statistics()

    # Get detailed per-node metrics
    detailed_metrics = network.get_detailed_metrics()

    # Get aggregator-specific detailed history (FedRansel, DP-FedAvg)
    aggregator_detailed = network.get_aggregator_detailed_history()

    duration = time.time() - start_time

    # Extract final metrics
    final_acc = history["mean_accuracy"][-1] if history["mean_accuracy"] else 0.0
    final_loss = history["mean_loss"][-1] if history["mean_loss"] else 0.0

    return ExperimentResult(
        experiment_name=config.experiment.name,
        config=config.model_dump(),
        final_accuracy=final_acc,
        final_loss=final_loss,
        history=history,
        aggregator_stats=agg_stats,
        detailed_metrics=detailed_metrics,
        aggregator_detailed_history=aggregator_detailed,
        duration_seconds=duration,
    )


# =============================================================================
# Experiment 1: Convergence + Communication
# =============================================================================

def run_experiment_1(
    device: torch.device,
    output_dir: Path,
    quick: bool = False,
) -> List[ExperimentResult]:
    """Run Experiment 1: Convergence + Communication.

    Claims validated: FedRansel converges; reduces communication

    Setup:
    - All 4 datasets, IID + one non-IID setting each
    - N=50, T=300 rounds (or reduced for quick mode)
    - T_l ∈ {0.3, 0.5, 0.7}, T_g=1.0, τ=1

    Compare: FedRansel, FedAvg, DP-FedAvg (ε=4), Top-K (10%)
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  Experiment 1: Convergence + Communication[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    results = []

    datasets = ["cifar10", "femnist", "uci_har", "pamap2"]
    partitions = ["iid", "dirichlet"]
    algorithms = [
        ("fedavg", "decentralized", {}),
        ("dp_fedavg", "decentralized", {"epsilon": 4.0, "delta": 1e-5, "clip_norm": 1.0}),
        ("topk", "decentralized", {"k_ratio": 0.1}),
        ("fedransel", "centralized", {"T_l": 0.3, "T_g": 1.0, "tau": 1}),
        ("fedransel", "centralized", {"T_l": 0.5, "T_g": 1.0, "tau": 1}),
        ("fedransel", "centralized", {"T_l": 0.7, "T_g": 1.0, "tau": 1}),
    ]

    num_nodes = 50 if not quick else 10
    rounds = 300 if not quick else 10

    total_experiments = len(datasets) * len(partitions) * len(algorithms)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Experiment 1...", total=total_experiments)

        for dataset in datasets:
            for partition in partitions:
                for alg_name, mode, agg_params in algorithms:
                    # Skip FEMNIST non-IID (it's naturally non-IID)
                    if dataset == "femnist" and partition == "dirichlet":
                        progress.advance(task)
                        continue

                    name = f"exp1-{dataset}-{alg_name}-{partition}"
                    if alg_name == "fedransel":
                        name += f"-Tl{agg_params['T_l']}"

                    progress.update(task, description=f"[cyan]{name}")

                    try:
                        config = build_base_config(
                            name=name,
                            dataset=dataset,
                            algorithm=alg_name,
                            num_nodes=num_nodes,
                            rounds=rounds,
                            partition_method=partition,
                            mode=mode,
                            agg_params=agg_params,
                        )

                        result = run_single_experiment(config, device)
                        results.append(result)

                        console.print(f"  [green]✓[/green] {name}: acc={result.final_accuracy:.4f}")

                    except Exception as e:
                        console.print(f"  [red]✗[/red] {name}: {str(e)[:50]}")

                    progress.advance(task)

    # Save results
    save_results(results, output_dir / "experiment_1_results.json")
    display_exp1_summary(results)

    return results


def display_exp1_summary(results: List[ExperimentResult]):
    """Display summary table for Experiment 1."""
    table = Table(title="Experiment 1: Convergence + Communication Summary")
    table.add_column("Dataset")
    table.add_column("Partition")
    table.add_column("Algorithm")
    table.add_column("Acc", style="green")
    table.add_column("F1", style="yellow")
    table.add_column("AUC", style="cyan")
    table.add_column("Comm Cost", style="magenta")
    table.add_column("Time(s)")

    for r in results:
        parts = r.experiment_name.split("-")
        dataset = parts[1] if len(parts) > 1 else "?"
        alg = parts[2] if len(parts) > 2 else "?"
        partition = parts[3] if len(parts) > 3 else "iid"

        comm_cost = "N/A"
        if r.aggregator_stats and "total_communication" in r.aggregator_stats:
            comm_cost = f"{r.aggregator_stats['total_communication']:,}"

        # Get F1 and AUC from history
        final_f1 = r.history.get("mean_f1", [])
        final_auc = r.history.get("mean_auc", [])
        f1_str = f"{final_f1[-1]:.4f}" if final_f1 else "N/A"
        auc_str = f"{final_auc[-1]:.4f}" if final_auc else "N/A"

        table.add_row(
            dataset,
            partition,
            alg,
            f"{r.final_accuracy:.4f}",
            f1_str,
            auc_str,
            comm_cost,
            f"{r.duration_seconds:.1f}",
        )

    console.print(table)


# =============================================================================
# Experiment 2: Privacy-Utility Tradeoff
# =============================================================================

def run_experiment_2(
    device: torch.device,
    output_dir: Path,
    quick: bool = False,
) -> List[ExperimentResult]:
    """Run Experiment 2: Privacy-Utility Tradeoff.

    Claims validated: Theorem 1 (privacy bound), practical tradeoff characterization

    Setup:
    - Grid: T_l ∈ {0.3, 0.5, 0.7} × T_g ∈ {0.5, 0.7, 1.0} × τ ∈ {1, N/4}
    - Track empirical exposure rate
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  Experiment 2: Privacy-Utility Tradeoff[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    results = []

    T_l_values = [0.3, 0.5, 0.7]
    T_g_values = [0.5, 0.7, 1.0]
    num_nodes = 50 if not quick else 10
    tau_values = [1, num_nodes // 4]  # [1, N/4]
    rounds = 100 if not quick else 10

    total_experiments = len(T_l_values) * len(T_g_values) * len(tau_values)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Experiment 2...", total=total_experiments)

        for T_l in T_l_values:
            for T_g in T_g_values:
                for tau in tau_values:
                    name = f"exp2-privacy-Tl{T_l}-Tg{T_g}-tau{tau}"
                    progress.update(task, description=f"[cyan]{name}")

                    try:
                        config = build_base_config(
                            name=name,
                            dataset="cifar10",
                            algorithm="fedransel",
                            num_nodes=num_nodes,
                            rounds=rounds,
                            partition_method="iid",
                            mode="centralized",
                            agg_params={"T_l": T_l, "T_g": T_g, "tau": tau},
                        )

                        result = run_single_experiment(config, device)
                        results.append(result)

                        # Compute theoretical privacy bound
                        theoretical_bound = (1 + T_l) / 2 * T_g

                        console.print(
                            f"  [green]✓[/green] {name}: "
                            f"acc={result.final_accuracy:.4f}, "
                            f"bound={theoretical_bound:.4f}"
                        )

                    except Exception as e:
                        console.print(f"  [red]✗[/red] {name}: {str(e)[:50]}")

                    progress.advance(task)

    # Save results
    save_results(results, output_dir / "experiment_2_results.json")
    display_exp2_summary(results)

    return results


def display_exp2_summary(results: List[ExperimentResult]):
    """Display summary table for Experiment 2."""
    table = Table(title="Experiment 2: Privacy-Utility Tradeoff Summary")
    table.add_column("T_l")
    table.add_column("T_g")
    table.add_column("τ")
    table.add_column("Acc", style="green")
    table.add_column("F1", style="yellow")
    table.add_column("AUC", style="cyan")
    table.add_column("Theo Bound", style="magenta")
    table.add_column("Final Ratio")

    for r in results:
        agg_params = r.config.get("aggregation", {}).get("params", {})
        T_l = agg_params.get("T_l", 0)
        T_g = agg_params.get("T_g", 0)
        tau = agg_params.get("tau", 0)

        theoretical_bound = (1 + T_l) / 2 * T_g
        avg_final_ratio = r.aggregator_stats.get("avg_final_ratio", 0) if r.aggregator_stats else 0

        # Get F1 and AUC from history
        final_f1 = r.history.get("mean_f1", [])
        final_auc = r.history.get("mean_auc", [])
        f1_str = f"{final_f1[-1]:.4f}" if final_f1 else "N/A"
        auc_str = f"{final_auc[-1]:.4f}" if final_auc else "N/A"

        table.add_row(
            f"{T_l}",
            f"{T_g}",
            f"{tau}",
            f"{r.final_accuracy:.4f}",
            f1_str,
            auc_str,
            f"{theoretical_bound:.4f}",
            f"{avg_final_ratio:.4f}",
        )

    console.print(table)


# =============================================================================
# Experiment 3: Byzantine Robustness
# =============================================================================

def run_experiment_3(
    device: torch.device,
    output_dir: Path,
    quick: bool = False,
) -> List[ExperimentResult]:
    """Run Experiment 3: Byzantine Robustness.

    Claims validated: Theorem 2

    Setup:
    - N=50, Byzantine fraction f ∈ {0, 0.1, 0.2, 0.3}
    - Attack types: gaussian, directed_deviation
    - τ ∈ {1, 5, 15}
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  Experiment 3: Byzantine Robustness[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    results = []

    byzantine_fractions = [0.0, 0.1, 0.2, 0.3]
    attack_types = ["gaussian", "directed_deviation"]
    tau_values = [1, 5, 15]

    num_nodes = 50 if not quick else 10
    rounds = 100 if not quick else 10

    attack_params_map = {
        "gaussian": {"noise_std": 10.0},
        "directed_deviation": {"lambda_param": -5.0},
    }

    total_experiments = len(byzantine_fractions) * len(attack_types) * len(tau_values)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Experiment 3...", total=total_experiments)

        for byz_frac in byzantine_fractions:
            for attack_type in attack_types:
                for tau in tau_values:
                    name = f"exp3-byz{int(byz_frac*100)}-{attack_type}-tau{tau}"
                    progress.update(task, description=f"[cyan]{name}")

                    try:
                        config = build_base_config(
                            name=name,
                            dataset="cifar10",
                            algorithm="fedransel",
                            num_nodes=num_nodes,
                            rounds=rounds,
                            partition_method="iid",
                            mode="centralized",
                            agg_params={"T_l": 0.5, "T_g": 1.0, "tau": tau},
                            attack_enabled=(byz_frac > 0),
                            attack_type=attack_type if byz_frac > 0 else None,
                            attack_percentage=byz_frac,
                            attack_params=attack_params_map[attack_type] if byz_frac > 0 else None,
                        )

                        result = run_single_experiment(config, device)
                        results.append(result)

                        # Compute Theorem 2 bound
                        T_l = 0.5
                        min_honest = int(np.ceil(2 * tau / (1 + T_l)))
                        actual_honest = int(num_nodes * (1 - byz_frac))
                        bound_satisfied = actual_honest >= min_honest

                        console.print(
                            f"  [green]✓[/green] {name}: "
                            f"acc={result.final_accuracy:.4f}, "
                            f"honest={actual_honest}, min_needed={min_honest}, "
                            f"bound={'✓' if bound_satisfied else '✗'}"
                        )

                    except Exception as e:
                        console.print(f"  [red]✗[/red] {name}: {str(e)[:50]}")

                    progress.advance(task)

    # Save results
    save_results(results, output_dir / "experiment_3_results.json")
    display_exp3_summary(results, num_nodes)

    return results


def display_exp3_summary(results: List[ExperimentResult], num_nodes: int):
    """Display summary table for Experiment 3."""
    table = Table(title="Experiment 3: Byzantine Robustness Summary")
    table.add_column("Byz %")
    table.add_column("Attack")
    table.add_column("τ")
    table.add_column("Acc", style="green")
    table.add_column("F1", style="yellow")
    table.add_column("AUC", style="cyan")
    table.add_column("Honest Acc", style="blue")
    table.add_column("Bound OK", style="magenta")

    for r in results:
        attack_cfg = r.config.get("attack", {})
        agg_params = r.config.get("aggregation", {}).get("params", {})

        byz_pct = int(attack_cfg.get("percentage", 0) * 100)
        attack_type = attack_cfg.get("type", "none") or "none"
        tau = agg_params.get("tau", 1)
        T_l = agg_params.get("T_l", 0.5)

        # Check Theorem 2 bound
        min_honest = int(np.ceil(2 * tau / (1 + T_l)))
        actual_honest = int(num_nodes * (1 - attack_cfg.get("percentage", 0)))
        bound_ok = "✓" if actual_honest >= min_honest else "✗"

        honest_acc = r.history.get("honest_accuracy", [])
        honest_final = f"{honest_acc[-1]:.4f}" if honest_acc else "N/A"

        # Get F1 and AUC from history
        final_f1 = r.history.get("mean_f1", [])
        final_auc = r.history.get("mean_auc", [])
        f1_str = f"{final_f1[-1]:.4f}" if final_f1 else "N/A"
        auc_str = f"{final_auc[-1]:.4f}" if final_auc else "N/A"

        table.add_row(
            f"{byz_pct}%",
            attack_type[:10],
            f"{tau}",
            f"{r.final_accuracy:.4f}",
            f1_str,
            auc_str,
            honest_final,
            bound_ok,
        )

    console.print(table)


# =============================================================================
# Experiment 4: Conjecture Validation
# =============================================================================

def run_experiment_4(output_dir: Path, quick: bool = False) -> Dict[str, Any]:
    """Run Experiment 4: Conjecture Validation (synthetic, no training).

    Claims validated: Conjecture 1 (sampling probabilities)

    Setup:
    - m ∈ {10^4, 10^5, 10^6}
    - T_l ∈ {0.3, 0.5, 0.7}
    - 10,000 sampling iterations each
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  Experiment 4: Conjecture Validation[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    from murmura.examples.fedransel.exp4_synthetic_validation import (
        run_conjecture_validation,
        display_results,
    )

    m_values = [10_000, 100_000, 1_000_000] if not quick else [1_000, 10_000]
    T_l_values = [0.3, 0.5, 0.7]
    num_iterations = 10_000 if not quick else 1_000

    results = run_conjecture_validation(
        m_values=m_values,
        T_l_values=T_l_values,
        num_iterations=num_iterations,
        seed=42,
    )

    display_results(results)

    # Convert to dict for JSON serialization
    results_dict = {
        "m_values": m_values,
        "T_l_values": T_l_values,
        "num_iterations": num_iterations,
        "results": [asdict(r) for r in results],
    }

    # Save results
    with open(output_dir / "experiment_4_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    return results_dict


# =============================================================================
# Utilities
# =============================================================================

def save_results(results: List[ExperimentResult], filepath: Path, save_detailed: bool = True):
    """Save experiment results to JSON file.

    Args:
        results: List of experiment results
        filepath: Path to save results
        save_detailed: If True, save detailed per-node metrics (can be large)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = []
    for r in results:
        entry = {
            "experiment_name": r.experiment_name,
            "final_accuracy": r.final_accuracy,
            "final_loss": r.final_loss,
            "duration_seconds": r.duration_seconds,
            "config": r.config,
            "aggregator_stats": r.aggregator_stats,
            "history": {k: v for k, v in r.history.items() if v},  # Remove empty lists
        }

        if save_detailed:
            # Include detailed metrics (per-node per-round)
            if r.detailed_metrics:
                entry["detailed_metrics"] = r.detailed_metrics

            # Include aggregator-specific detailed history
            if r.aggregator_detailed_history:
                entry["aggregator_detailed_history"] = r.aggregator_detailed_history

        data.append(entry)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"\n[dim]Results saved to: {filepath}[/dim]")

    # Also save a summary file with just key metrics
    summary_filepath = filepath.parent / f"{filepath.stem}_summary.json"
    summary_data = []
    for r in results:
        summary_entry = {
            "experiment_name": r.experiment_name,
            "final_accuracy": r.final_accuracy,
            "final_loss": r.final_loss,
            "final_precision": r.history.get("mean_precision", [None])[-1] if r.history.get("mean_precision") else None,
            "final_recall": r.history.get("mean_recall", [None])[-1] if r.history.get("mean_recall") else None,
            "final_f1": r.history.get("mean_f1", [None])[-1] if r.history.get("mean_f1") else None,
            "final_auc": r.history.get("mean_auc", [None])[-1] if r.history.get("mean_auc") else None,
            "duration_seconds": r.duration_seconds,
            "algorithm": r.config.get("aggregation", {}).get("algorithm"),
        }
        summary_data.append(summary_entry)

    with open(summary_filepath, "w") as f:
        json.dump(summary_data, f, indent=2)

    console.print(f"[dim]Summary saved to: {summary_filepath}[/dim]")


def generate_summary_report(output_dir: Path):
    """Generate a summary report from all experiment results."""
    report_path = output_dir / "summary_report.md"

    with open(report_path, "w") as f:
        f.write("# FedRansel Experiment Results Summary\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Load and summarize each experiment's results
        for exp_num in [1, 2, 3, 4]:
            result_file = output_dir / f"experiment_{exp_num}_results.json"
            if result_file.exists():
                f.write(f"## Experiment {exp_num}\n\n")
                with open(result_file) as rf:
                    data = json.load(rf)
                    if isinstance(data, list):
                        f.write(f"- Total configurations: {len(data)}\n")

                        # Extract key metrics
                        accs = [d.get("final_accuracy", 0) for d in data]
                        f1_scores = []
                        auc_scores = []

                        for d in data:
                            hist = d.get("history", {})
                            f1_list = hist.get("mean_f1", [])
                            auc_list = hist.get("mean_auc", [])
                            if f1_list:
                                f1_scores.append(f1_list[-1])
                            if auc_list:
                                auc_scores.append(auc_list[-1])

                        if accs:
                            f.write(f"- Accuracy range: {min(accs):.4f} - {max(accs):.4f}\n")
                            f.write(f"- Mean accuracy: {np.mean(accs):.4f}\n")
                        if f1_scores:
                            f.write(f"- F1 range: {min(f1_scores):.4f} - {max(f1_scores):.4f}\n")
                            f.write(f"- Mean F1: {np.mean(f1_scores):.4f}\n")
                        if auc_scores:
                            f.write(f"- AUC range: {min(auc_scores):.4f} - {max(auc_scores):.4f}\n")
                            f.write(f"- Mean AUC: {np.mean(auc_scores):.4f}\n")

                        # Experiment-specific summaries
                        if exp_num == 1:
                            f.write("\n### Best performing configurations:\n")
                            sorted_data = sorted(data, key=lambda x: x.get("final_accuracy", 0), reverse=True)
                            for d in sorted_data[:5]:
                                name = d.get("experiment_name", "unknown")
                                acc = d.get("final_accuracy", 0)
                                f.write(f"- {name}: {acc:.4f}\n")

                        elif exp_num == 2:
                            f.write("\n### Privacy-Utility Tradeoff Summary:\n")
                            for d in data:
                                params = d.get("config", {}).get("aggregation", {}).get("params", {})
                                T_l = params.get("T_l", 0)
                                T_g = params.get("T_g", 0)
                                tau = params.get("tau", 0)
                                acc = d.get("final_accuracy", 0)
                                bound = (1 + T_l) / 2 * T_g
                                f.write(f"- T_l={T_l}, T_g={T_g}, τ={tau}: acc={acc:.4f}, bound={bound:.4f}\n")

                        elif exp_num == 3:
                            f.write("\n### Byzantine Robustness Summary:\n")
                            for d in data:
                                attack = d.get("config", {}).get("attack", {})
                                byz_pct = attack.get("percentage", 0) * 100
                                attack_type = attack.get("type", "none")
                                acc = d.get("final_accuracy", 0)
                                f.write(f"- {byz_pct:.0f}% {attack_type}: acc={acc:.4f}\n")

                    f.write("\n")

    console.print(f"\n[dim]Summary report saved to: {report_path}[/dim]")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run FedRansel experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp1", action="store_true", help="Run Experiment 1 only")
    parser.add_argument("--exp2", action="store_true", help="Run Experiment 2 only")
    parser.add_argument("--exp3", action="store_true", help="Run Experiment 3 only")
    parser.add_argument("--exp4", action="store_true", help="Run Experiment 4 only")
    parser.add_argument("--quick", action="store_true", help="Quick mode (reduced rounds)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fedransel_results",
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    console.print("\n[bold green]╔═══════════════════════════════════════════════════════════════╗[/bold green]")
    console.print("[bold green]║           FedRansel Experiment Runner                         ║[/bold green]")
    console.print("[bold green]╚═══════════════════════════════════════════════════════════════╝[/bold green]")
    console.print(f"\n[bold]Device:[/bold] {device}")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    console.print(f"[bold]Mode:[/bold] {'Quick' if args.quick else 'Full'}")

    # Determine which experiments to run
    run_all = not (args.exp1 or args.exp2 or args.exp3 or args.exp4)

    all_results = {}

    try:
        if run_all or args.exp1:
            all_results["exp1"] = run_experiment_1(device, output_dir, args.quick)

        if run_all or args.exp2:
            all_results["exp2"] = run_experiment_2(device, output_dir, args.quick)

        if run_all or args.exp3:
            all_results["exp3"] = run_experiment_3(device, output_dir, args.quick)

        if run_all or args.exp4:
            all_results["exp4"] = run_experiment_4(output_dir, args.quick)

        # Generate summary report
        generate_summary_report(output_dir)

        console.print("\n[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
        console.print("[bold green]  All experiments completed successfully![/bold green]")
        console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Experiments interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
