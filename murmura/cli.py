"""Command-line interface for Murmura."""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import importlib

import torch

from murmura.config import load_config
from murmura.utils.seed import set_seed
from murmura.utils.device import get_device
from murmura.core.network import Network

app = typer.Typer(
    name="murmura",
    help="Murmura: Decentralized Federated Learning Framework",
    add_completion=False
)
console = Console()


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to configuration file (YAML/JSON)"),
    device_override: Optional[str] = typer.Option(None, "--device", help="Override device (cpu/cuda/mps)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Enable verbose output"),
):
    """Run a decentralized federated learning experiment from config file.

    Example:
        murmura run experiments/basic_fedavg.yaml
    """
    try:
        # Load configuration
        console.print(f"[bold blue]Loading configuration from:[/bold blue] {config_path}")
        config = load_config(config_path)

        # Set seed
        set_seed(config.experiment.seed)

        # Get device
        if device_override:
            device = torch.device(device_override)
        else:
            device = get_device()
        console.print(f"[bold green]Using device:[/bold green] {device}")

        # Display experiment info
        console.print(f"\n[bold]Experiment:[/bold] {config.experiment.name}")
        console.print(f"  Rounds: {config.experiment.rounds}")
        console.print(f"  Topology: {config.topology.type} ({config.topology.num_nodes} nodes)")
        console.print(f"  Aggregation: {config.aggregation.algorithm}")
        if config.attack.enabled:
            console.print(f"  [red]Attack: {config.attack.type} ({config.attack.percentage*100:.0f}% nodes)[/red]")

        # Load dataset adapter
        console.print(f"\n[bold]Loading dataset:[/bold] {config.data.adapter}")
        dataset_adapter = _load_dataset_adapter(config)

        # Load model factory
        console.print(f"[bold]Loading model:[/bold] {config.model.factory}")
        model_factory = _load_model_factory(config)

        # Create aggregator factory
        aggregator_factory = _create_aggregator_factory(config, model_factory, device)

        # Check if using evidential models (wearables)
        evidential = config.model.factory.startswith("examples.wearables.")
        criterion = None
        if evidential:
            from murmura.examples.wearables import get_evidential_loss
            num_classes = config.model.params.get("num_classes", 6)
            # Use longer annealing period for FL (rounds, not epochs)
            # and reduced KL weight for stability with aggregation
            annealing_rounds = config.experiment.rounds // 2  # Anneal over first half
            criterion = get_evidential_loss(
                num_classes=num_classes,
                annealing_epochs=annealing_rounds,  # In FL: rounds, not epochs
                lambda_weight=0.1,  # Reduced KL weight for FL stability
            )
            console.print("[bold cyan]Using evidential deep learning with uncertainty quantification[/bold cyan]")
            console.print(f"  KL annealing over {annealing_rounds} rounds, λ=0.1")

        # Create network
        console.print("\n[bold]Creating network...[/bold]")
        network = Network.from_config(
            config=config,
            model_factory=model_factory,
            dataset_adapter=dataset_adapter,
            aggregator_factory=aggregator_factory,
            device=device,
            criterion=criterion,
            evidential=evidential,
        )

        # Run training
        console.print("\n[bold green]Starting training...[/bold green]\n")
        history = network.train(
            rounds=config.experiment.rounds,
            local_epochs=config.training.local_epochs,
            lr=config.training.lr,
            verbose=verbose or config.experiment.verbose
        )

        # Display results
        _display_results(history, network)

        console.print("\n[bold green]✓ Training completed successfully![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def list_components(
    component_type: str = typer.Argument(..., help="Component type (topologies/aggregators/attacks)")
):
    """List available components.

    Example:
        murmura list-components topologies
        murmura list-components aggregators
    """
    if component_type == "topologies":
        console.print("[bold]Available Topologies:[/bold]")
        console.print("  • ring - Ring topology (each node connects to 2 neighbors)")
        console.print("  • fully - Fully connected (all-to-all)")
        console.print("  • erdos - Erdős-Rényi random graph (requires p parameter)")
        console.print("  • k-regular - k-regular graph (requires k parameter)")

    elif component_type == "aggregators":
        console.print("[bold]Available Aggregators:[/bold]")
        console.print("  • fedavg - Decentralized FedAvg (simple averaging)")
        console.print("  • krum - Multi-Krum Byzantine-resilient")
        console.print("  • balance - Distance-based filtering with adaptive thresholds")
        console.print("  • sketchguard - Count-Sketch compression for lightweight filtering")
        console.print("  • ubar - Two-stage Byzantine-resilient (distance + loss)")
        console.print("  • evidential_trust - Uncertainty-aware trust aggregation (EDL-based)")

    elif component_type == "attacks":
        console.print("[bold]Available Attacks:[/bold]")
        console.print("  • gaussian - Gaussian noise injection")
        console.print("  • directed_deviation - Directional parameter scaling")

    else:
        console.print(f"[red]Unknown component type: {component_type}[/red]")
        console.print("Available types: topologies, aggregators, attacks")


def _load_dataset_adapter(config):
    """Load dataset adapter from config."""
    adapter_name = config.data.adapter

    if adapter_name.startswith("leaf."):
        # LEAF dataset
        dataset_type = adapter_name.split(".")[1]
        from murmura.examples.leaf import load_leaf_adapter
        return load_leaf_adapter(
            dataset_type,
            num_nodes=config.topology.num_nodes,
            seed=config.experiment.seed,
            **config.data.params
        )
    elif adapter_name.startswith("wearables."):
        # Wearable dataset (uci_har, pamap2, ppg_dalia)
        dataset_type = adapter_name.split(".")[1]
        from murmura.examples.wearables import load_wearable_adapter
        return load_wearable_adapter(
            dataset_type=dataset_type,
            num_nodes=config.topology.num_nodes,
            seed=config.experiment.seed,
            **config.data.params
        )
    else:
        # Custom adapter - dynamically import
        module_path, class_name = adapter_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)
        return adapter_class(**config.data.params)


def _load_model_factory(config):
    """Load model factory from config."""
    factory_path = config.model.factory

    if factory_path.startswith("examples.leaf."):
        # LEAF model
        from murmura.examples.leaf import get_leaf_model_factory
        model_type = factory_path.split(".")[-1]
        return get_leaf_model_factory(model_type, **config.model.params)
    elif factory_path.startswith("examples.wearables."):
        # Wearable evidential model
        from murmura.examples.wearables import get_wearable_model_factory
        dataset_type = factory_path.split(".")[-1]
        return get_wearable_model_factory(dataset_type, **config.model.params)
    else:
        # Custom model factory
        module_path, factory_name = factory_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        factory = getattr(module, factory_name)
        return lambda: factory(**config.model.params)


def _create_aggregator_factory(config, model_factory, device):
    """Create aggregator factory function."""
    from murmura.aggregation import (
        FedAvgAggregator,
        KrumAggregator,
        BALANCEAggregator,
        SketchguardAggregator,
        UBARAggregator,
        EvidentialTrustAggregator,
    )
    from murmura.aggregation.base import calculate_model_dimension

    agg_type = config.aggregation.algorithm.lower()
    params = dict(config.aggregation.params)  # Make a copy to avoid mutation

    # Calculate model dimension for Sketchguard
    if agg_type == "sketchguard":
        sample_model = model_factory()
        model_dim = calculate_model_dimension(sample_model)
        params["model_dim"] = model_dim
        params["total_rounds"] = config.experiment.rounds

    # Common parameters for algorithms that need total_rounds
    if agg_type in ["balance", "ubar", "evidential_trust"]:
        params["total_rounds"] = config.experiment.rounds

    def factory(node_id: int):
        if agg_type == "fedavg":
            return FedAvgAggregator(**params)
        elif agg_type == "krum":
            return KrumAggregator(**params)
        elif agg_type == "balance":
            return BALANCEAggregator(**params)
        elif agg_type == "sketchguard":
            return SketchguardAggregator(**params)
        elif agg_type == "ubar":
            return UBARAggregator(**params)
        elif agg_type == "evidential_trust":
            return EvidentialTrustAggregator(**params)
        else:
            raise ValueError(f"Unknown aggregation algorithm: {agg_type}")

    return factory


def _display_results(history, network):
    """Display training results in a table."""
    # Check if we have evidential metrics
    has_uncertainty = len(history.get("mean_vacuity", [])) > 0

    table = Table(title="Training Results")
    table.add_column("Round", style="cyan")
    table.add_column("Mean Acc", style="green")
    table.add_column("Std Acc", style="yellow")
    table.add_column("Honest Acc", style="blue")
    table.add_column("Compromised Acc", style="red")

    if has_uncertainty:
        table.add_column("Vacuity", style="magenta")
        table.add_column("Entropy", style="cyan")
        table.add_column("Strength", style="white")

    for i in range(len(history["round"])):
        round_num = history["round"][i]
        mean_acc = history["mean_accuracy"][i]
        std_acc = history["std_accuracy"][i]

        honest_acc = history["honest_accuracy"][i] if i < len(history["honest_accuracy"]) else "-"
        comp_acc = history["compromised_accuracy"][i] if i < len(history["compromised_accuracy"]) else "-"

        row = [
            str(round_num),
            f"{mean_acc:.4f}",
            f"{std_acc:.4f}",
            f"{honest_acc:.4f}" if isinstance(honest_acc, float) else honest_acc,
            f"{comp_acc:.4f}" if isinstance(comp_acc, float) else comp_acc,
        ]

        if has_uncertainty:
            vacuity = history["mean_vacuity"][i] if i < len(history["mean_vacuity"]) else 0
            entropy = history["mean_entropy"][i] if i < len(history["mean_entropy"]) else 0
            strength = history["mean_strength"][i] if i < len(history["mean_strength"]) else 0
            row.extend([f"{vacuity:.4f}", f"{entropy:.4f}", f"{strength:.2f}"])

        table.add_row(*row)

    console.print(table)

    # Print interpretation for evidential metrics
    if has_uncertainty:
        console.print("\n[bold]Uncertainty Metrics:[/bold]")
        console.print("  • [magenta]Vacuity[/magenta]: Epistemic uncertainty (lack of evidence). Lower = more confident.")
        console.print("  • [cyan]Entropy[/cyan]: Aleatoric uncertainty (prediction entropy). Lower = more decisive.")
        console.print("  • [white]Strength[/white]: Dirichlet strength (total evidence). Higher = more evidence accumulated.")


if __name__ == "__main__":
    app()
