#!/usr/bin/env python3
"""Generate all experiment configurations for the paper.

This script generates comprehensive experiment configs covering:
1. Baseline experiments (no attacks, fully connected)
2. Heterogeneity study (α ∈ {0.1, 0.5, 1.0})
3. Byzantine attack study (Gaussian, Directed Deviation at 10%, 20%, 30%)
4. Topology study (ring, fully, erdos, k-regular)
5. Ablation study (self_weight, trust_threshold, accuracy_weight)
"""

import yaml
from pathlib import Path

PAPER_DIR = Path(__file__).parent
ALGORITHMS = ["fedavg", "krum", "balance", "ubar", "sketchguard", "evidential_trust"]

# Dataset configurations
DATASETS = {
    "uci_har": {
        "adapter": "wearables.uci_har",
        "data_path": "wearables_datasets/UCI HAR Dataset",
        "model_factory": "examples.wearables.uci_har",
        "input_dim": 561,
        "hidden_dims": [256, 128],
        "num_classes": 6,
        "num_nodes": 10,
    },
    "pamap2": {
        "adapter": "wearables.pamap2",
        "data_path": "wearables_datasets/PAMAP2_Dataset",
        "model_factory": "examples.wearables.pamap2",
        "input_dim": 4000,
        "hidden_dims": [512, 256, 128],
        "num_classes": 12,
        "num_nodes": 9,
    },
    "ppg_dalia": {
        "adapter": "wearables.ppg_dalia",
        "data_path": "wearables_datasets/PPG_FieldStudy",
        "model_factory": "examples.wearables.ppg_dalia",
        "input_dim": 192,
        "hidden_dims": [256, 128, 64],
        "num_classes": 7,
        "num_nodes": 15,
    },
}

# Aggregation algorithm parameters
AGG_PARAMS = {
    "fedavg": {},
    "krum": {"f": 3, "m": 5},
    "balance": {"threshold_multiplier": 2.0, "min_neighbors": 2},
    "ubar": {"rho": 0.5, "sample_fraction": 0.2},
    "sketchguard": {"num_buckets": 1000, "num_hashes": 5, "threshold_multiplier": 2.0},
    "evidential_trust": {
        "vacuity_threshold": 0.5,
        "accuracy_weight": 0.7,
        "trust_threshold": 0.1,
        "self_weight": 0.6,
    },
}


def create_config(
    dataset: str,
    algorithm: str,
    name_suffix: str = "",
    topology_type: str = "fully",
    topology_params: dict = None,
    alpha: float = 0.5,
    attack_enabled: bool = False,
    attack_type: str = "gaussian",
    attack_percentage: float = 0.2,
    attack_params: dict = None,
    agg_overrides: dict = None,
    rounds: int = 50,
) -> dict:
    """Create a single experiment configuration."""
    ds = DATASETS[dataset]
    topology_params = topology_params or {}
    attack_params = attack_params or {}
    agg_overrides = agg_overrides or {}

    # Build experiment name
    exp_name = f"{dataset.upper().replace('_', '')}-{algorithm.title()}"
    if name_suffix:
        exp_name += f"-{name_suffix}"

    # Merge aggregation params with overrides
    agg_params = {**AGG_PARAMS.get(algorithm, {}), **agg_overrides}

    config = {
        "experiment": {
            "name": exp_name,
            "seed": 42,
            "rounds": rounds,
            "verbose": True,
        },
        "topology": {
            "type": topology_type,
            "num_nodes": ds["num_nodes"],
            "seed": 12345,
            **topology_params,
        },
        "aggregation": {
            "algorithm": algorithm,
            "params": agg_params,
        },
        "attack": {
            "enabled": attack_enabled,
            "type": attack_type if attack_enabled else "gaussian",
            "percentage": attack_percentage if attack_enabled else 0.2,
            "params": attack_params if attack_enabled else {},
        },
        "training": {
            "local_epochs": 2,
            "batch_size": 32,
            "lr": 0.01,
            "max_samples": None,
        },
        "data": {
            "adapter": ds["adapter"],
            "params": {
                "data_path": ds["data_path"],
                "partition_method": "dirichlet",
                "alpha": alpha,
            },
        },
        "model": {
            "factory": ds["model_factory"],
            "params": {
                "input_dim": ds["input_dim"],
                "hidden_dims": ds["hidden_dims"],
                "num_classes": ds["num_classes"],
                "dropout": 0.3,
            },
        },
    }

    # Add dataset-specific params
    if dataset == "pamap2":
        config["data"]["params"]["window_size"] = 100
        config["data"]["params"]["window_stride"] = 50
    elif dataset == "ppg_dalia":
        config["data"]["params"]["window_size"] = 32
        config["data"]["params"]["window_stride"] = 16

    return config


def save_config(config: dict, filepath: Path):
    """Save configuration to YAML file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Created: {filepath}")


def generate_baseline_configs():
    """Generate baseline configs (already exist in dataset folders)."""
    print("\n=== Baseline Experiments ===")
    print("Already generated in: experiments/paper/{uci_har,pamap2,ppg_dalia}/")


def generate_heterogeneity_configs():
    """Generate heterogeneity study configs (α ∈ {0.1, 0.5, 1.0})."""
    print("\n=== Heterogeneity Study ===")

    for dataset in DATASETS:
        for alpha in [0.1, 0.5, 1.0]:
            alpha_str = str(alpha).replace(".", "")
            for algorithm in ALGORITHMS:
                config = create_config(
                    dataset=dataset,
                    algorithm=algorithm,
                    name_suffix=f"alpha{alpha_str}",
                    alpha=alpha,
                )
                filepath = PAPER_DIR / "heterogeneity" / dataset / f"{algorithm}_alpha{alpha_str}.yaml"
                save_config(config, filepath)


def generate_attack_configs():
    """Generate Byzantine attack configs."""
    print("\n=== Byzantine Attack Study ===")

    attack_configs = [
        ("gaussian", {"noise_std": 10.0}),
        ("directed_deviation", {"lambda_param": -5.0}),
    ]

    attack_percentages = [0.1, 0.2, 0.3]  # 10%, 20%, 30%

    for dataset in DATASETS:
        for attack_type, attack_params in attack_configs:
            for pct in attack_percentages:
                pct_str = f"{int(pct*100)}pct"
                for algorithm in ALGORITHMS:
                    config = create_config(
                        dataset=dataset,
                        algorithm=algorithm,
                        name_suffix=f"{attack_type}_{pct_str}",
                        alpha=0.5,  # Use moderate heterogeneity for attack study
                        attack_enabled=True,
                        attack_type=attack_type,
                        attack_percentage=pct,
                        attack_params=attack_params,
                    )
                    filepath = PAPER_DIR / "attacks" / dataset / f"{algorithm}_{attack_type}_{pct_str}.yaml"
                    save_config(config, filepath)


def generate_topology_configs():
    """Generate topology study configs."""
    print("\n=== Topology Study ===")

    # Only test key algorithms for topology study
    key_algorithms = ["fedavg", "evidential_trust", "krum", "sketchguard"]

    topology_configs = [
        ("ring", {}),
        ("fully", {}),
        ("erdos", {"p": 0.3}),
        ("k-regular", {"k": 4}),
    ]

    for dataset in DATASETS:
        for topo_type, topo_params in topology_configs:
            for algorithm in key_algorithms:
                config = create_config(
                    dataset=dataset,
                    algorithm=algorithm,
                    name_suffix=f"topo_{topo_type}",
                    topology_type=topo_type,
                    topology_params=topo_params,
                    alpha=0.5,
                )
                topo_name = topo_type.replace("-", "_")
                filepath = PAPER_DIR / "topologies" / dataset / f"{algorithm}_{topo_name}.yaml"
                save_config(config, filepath)


def generate_ablation_configs():
    """Generate ablation study configs for evidential_trust."""
    print("\n=== Ablation Study ===")

    # Only for evidential_trust algorithm
    algorithm = "evidential_trust"

    # Base parameters
    base_params = {
        "vacuity_threshold": 0.5,
        "accuracy_weight": 0.7,
        "trust_threshold": 0.1,
        "self_weight": 0.6,
    }

    # Parameter variations
    ablations = {
        "self_weight": [0.3, 0.5, 0.6, 0.7, 0.9],
        "trust_threshold": [0.05, 0.1, 0.2, 0.3],
        "accuracy_weight": [0.3, 0.5, 0.7, 0.9],
        "vacuity_threshold": [0.3, 0.5, 0.7, 0.9],
    }

    for dataset in DATASETS:
        for param_name, values in ablations.items():
            for value in values:
                # Create overrides
                overrides = {param_name: value}
                value_str = str(value).replace(".", "")

                config = create_config(
                    dataset=dataset,
                    algorithm=algorithm,
                    name_suffix=f"{param_name}_{value_str}",
                    alpha=0.5,
                    agg_overrides=overrides,
                )
                filepath = PAPER_DIR / "ablation" / dataset / f"{param_name}_{value_str}.yaml"
                save_config(config, filepath)


def count_configs():
    """Count total number of configs generated."""
    total = 0
    for category in ["heterogeneity", "attacks", "topologies", "ablation"]:
        category_dir = PAPER_DIR / category
        if category_dir.exists():
            count = sum(1 for _ in category_dir.rglob("*.yaml"))
            total += count
            print(f"  {category}: {count} configs")

    # Add baseline configs
    baseline_count = sum(1 for ds in DATASETS for _ in ALGORITHMS)
    print(f"  baseline: {baseline_count} configs")
    total += baseline_count

    print(f"\n  TOTAL: {total} experiment configurations")


def main():
    print("=" * 60)
    print("Generating Paper Experiment Configurations")
    print("=" * 60)

    generate_baseline_configs()
    generate_heterogeneity_configs()
    generate_attack_configs()
    generate_topology_configs()
    generate_ablation_configs()

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    count_configs()

    print("\n" + "=" * 60)
    print("To run all experiments:")
    print("  python experiments/paper/run_comprehensive.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
