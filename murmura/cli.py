"""Command-line interface for Murmura."""

from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from murmura.config import load_config
from murmura.utils.device import get_device
from murmura.utils.factories import (
    build_aggregator_factory,
    build_attack,
    build_criterion,
    build_dataset_adapter,
    build_model_factory,
)
from murmura.utils.seed import set_seed

app = typer.Typer(
    name="murmura",
    help="Murmura: Decentralized Federated Learning Framework",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# run — simulation or distributed depending on config.backend
# ---------------------------------------------------------------------------

@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to configuration file (YAML/JSON)"),
    device_override: Optional[str] = typer.Option(None, "--device", help="Override device (cpu/cuda/mps)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Enable verbose output"),
):
    """Run a decentralized federated learning experiment from a config file.

    Routes to the ZMQ distributed backend when config sets backend: distributed,
    otherwise runs the existing in-process simulation.

    Examples:
        murmura run experiments/basic_fedavg.yaml
        murmura run experiments/distributed_fedavg.yaml --quiet
    """
    try:
        console.print(f"[bold blue]Loading config:[/bold blue] {config_path}")
        config = load_config(config_path)

        if config.backend == "distributed":
            _run_distributed(config_path, verbose)
        else:
            _run_simulation(config, config_path, device_override, verbose)

    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)


def _run_simulation(config, config_path, device_override, verbose):
    from murmura.core.network import Network

    set_seed(config.experiment.seed)

    device = torch.device(device_override) if device_override else get_device()
    console.print(f"[bold green]Device:[/bold green] {device}")
    console.print(f"\n[bold]Experiment:[/bold] {config.experiment.name}")
    console.print(f"  Backend:    simulation")
    console.print(f"  Rounds:     {config.experiment.rounds}")
    console.print(f"  Topology:   {config.topology.type} ({config.topology.num_nodes} nodes)")
    console.print(f"  Aggregation:{config.aggregation.algorithm}")
    if config.attack.enabled:
        console.print(
            f"  [red]Attack: {config.attack.type} "
            f"({config.attack.percentage * 100:.0f}% nodes)[/red]"
        )

    console.print(f"\n[bold]Loading dataset:[/bold] {config.data.adapter}")
    dataset_adapter = build_dataset_adapter(config)

    console.print(f"[bold]Loading model:[/bold] {config.model.factory}")
    model_factory = build_model_factory(config)

    aggregator_factory = build_aggregator_factory(config, model_factory, device)
    criterion, evidential = build_criterion(config)

    if evidential:
        console.print("[bold cyan]Using evidential deep learning (EDL)[/bold cyan]")

    console.print("\n[bold]Creating network…[/bold]")
    network = Network.from_config(
        config=config,
        model_factory=model_factory,
        dataset_adapter=dataset_adapter,
        aggregator_factory=aggregator_factory,
        device=device,
        criterion=criterion,
        evidential=evidential,
    )

    console.print("\n[bold green]Starting training…[/bold green]\n")
    history = network.train(
        rounds=config.experiment.rounds,
        local_epochs=config.training.local_epochs,
        lr=config.training.lr,
        verbose=verbose or config.experiment.verbose,
    )

    _display_results(history)
    console.print("\n[bold green]✓ Training complete[/bold green]")


def _run_distributed(config_path: Path, verbose: bool):
    from murmura.distributed.runner import DistributedRunner

    config = load_config(config_path)
    console.print(f"\n[bold]Experiment:[/bold] {config.experiment.name}")
    console.print(f"  Backend:    distributed (ZMQ {config.distributed.transport.upper()})")
    console.print(f"  Rounds:     {config.experiment.rounds}")
    console.print(f"  Topology:   {config.topology.type} ({config.topology.num_nodes} nodes)")
    console.print(f"  Aggregation:{config.aggregation.algorithm}")
    if config.attack.enabled:
        console.print(
            f"  [red]Attack: {config.attack.type} "
            f"({config.attack.percentage * 100:.0f}% nodes)[/red]"
        )

    console.print("\n[bold green]Launching distributed processes…[/bold green]\n")
    runner = DistributedRunner(config_path)
    history = runner.run(verbose=verbose or config.experiment.verbose)

    _display_results(history)
    console.print("\n[bold green]✓ Distributed training complete[/bold green]")


# ---------------------------------------------------------------------------
# run-node — launch a single node (multi-machine deployments)
# ---------------------------------------------------------------------------

@app.command(name="run-node")
def run_node(
    config_path: Path = typer.Argument(..., help="Path to the shared config file"),
    node_id: int = typer.Option(..., "--node-id", "-n", help="This node's ID (0-indexed)"),
    t_start: float = typer.Option(
        ...,
        "--t-start",
        help=(
            "Absolute monotonic time (seconds) for round 0 start, "
            "as printed by `murmura run` on the head node. "
            "All nodes in the experiment must use the same value."
        ),
    ),
    run_id: str = typer.Option(
        "default",
        "--run-id",
        help="Run identifier matching the head node (printed by `murmura run`).",
    ),
    host_override: Optional[str] = typer.Option(
        None, "--host", help="Override host for TCP transport (e.g., head-node IP)"
    ),
):
    """Launch a single distributed node (for multi-machine deployments).

    Use this on each worker machine after starting `murmura run` on the head
    node.  The head node prints the t_start and run_id values to pass here.

    Example (3-node cluster):
        Head node:  murmura run config.yaml
                    # prints: run_id=abc123  t_start=1234567890.123
        Worker 1:   murmura run-node config.yaml -n 1 --run-id abc123 --t-start 1234567890.123
        Worker 2:   murmura run-node config.yaml -n 2 --run-id abc123 --t-start 1234567890.123
    """
    try:
        config = load_config(config_path)

        if config.backend != "distributed":
            console.print(
                "[yellow]Warning:[/yellow] config.backend is not 'distributed'. "
                "Proceeding anyway with distributed node."
            )

        dist_cfg = config.distributed
        if host_override:
            from murmura.config.schema import DistributedConfig
            dist_dict = config.distributed.model_dump()
            dist_dict["host"] = host_override
            dist_cfg = DistributedConfig(**dist_dict)

        from murmura.distributed.endpoints import Endpoints
        from murmura.distributed.node_process import NodeProcess

        endpoints = Endpoints(dist_cfg, config.topology.num_nodes, run_id)

        console.print(f"[bold green]Starting node {node_id}[/bold green]  run_id={run_id}")
        proc = NodeProcess.from_config_path(
            node_id=node_id,
            config_path=str(config_path),
            endpoints=endpoints,
            t_start=t_start,
        )
        proc.run()

    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# list-components
# ---------------------------------------------------------------------------

@app.command()
def list_components(
    component_type: str = typer.Argument(
        ..., help="Component type (topologies/aggregators/attacks/backends)"
    ),
):
    """List available components.

    Examples:
        murmura list-components topologies
        murmura list-components aggregators
        murmura list-components backends
    """
    if component_type == "topologies":
        console.print("[bold]Available Topologies:[/bold]")
        console.print("  • ring      — each node connects to 2 neighbours")
        console.print("  • fully     — all-to-all")
        console.print("  • erdos     — Erdős-Rényi random graph (requires p)")
        console.print("  • k-regular — k-regular ring lattice (requires k)")

    elif component_type == "aggregators":
        console.print("[bold]Available Aggregators:[/bold]")
        console.print("  • fedavg           — simple decentralised averaging")
        console.print("  • krum             — Multi-Krum Byzantine-resilient")
        console.print("  • balance          — distance-based with adaptive thresholds")
        console.print("  • sketchguard      — Count-Sketch compression filtering")
        console.print("  • ubar             — two-stage (distance + loss) Byzantine-resilient")
        console.print("  • evidential_trust — uncertainty-aware trust aggregation (EDL)")

    elif component_type == "attacks":
        console.print("[bold]Available Attacks:[/bold]")
        console.print("  • gaussian           — Gaussian noise injection")
        console.print("  • directed_deviation — directional parameter scaling")

    elif component_type == "backends":
        console.print("[bold]Available Backends:[/bold]")
        console.print("  • simulation  — single-process in-memory (default)")
        console.print("  • distributed — multi-process ZMQ (set backend: distributed in config)")
        console.print("\n[bold]Distributed transport options:[/bold]")
        console.print("  • ipc — IPC sockets, single machine (default)")
        console.print("  • tcp — TCP sockets, multi-machine")

    else:
        console.print(f"[red]Unknown component type: {component_type}[/red]")
        console.print("Available: topologies, aggregators, attacks, backends")


# ---------------------------------------------------------------------------
# Results display (shared by simulation and distributed paths)
# ---------------------------------------------------------------------------

def _display_results(history: dict) -> None:
    has_uncertainty = len(history.get("mean_vacuity", [])) > 0

    table = Table(title="Training Results")
    table.add_column("Round", style="cyan")
    table.add_column("Mean Acc", style="green")
    table.add_column("Std Acc", style="yellow")
    table.add_column("Honest Acc", style="blue")
    table.add_column("Comp. Acc", style="red")
    if has_uncertainty:
        table.add_column("Vacuity", style="magenta")
        table.add_column("Entropy", style="cyan")
        table.add_column("Strength", style="white")

    for i in range(len(history["round"])):
        honest = history["honest_accuracy"][i] if i < len(history["honest_accuracy"]) else "-"
        comp = history["compromised_accuracy"][i] if i < len(history["compromised_accuracy"]) else "-"
        row = [
            str(history["round"][i]),
            f"{history['mean_accuracy'][i]:.4f}",
            f"{history['std_accuracy'][i]:.4f}",
            f"{honest:.4f}" if isinstance(honest, float) else honest,
            f"{comp:.4f}" if isinstance(comp, float) else comp,
        ]
        if has_uncertainty:
            row += [
                f"{history['mean_vacuity'][i]:.4f}",
                f"{history['mean_entropy'][i]:.4f}",
                f"{history['mean_strength'][i]:.2f}",
            ]
        table.add_row(*row)

    console.print(table)

    if has_uncertainty:
        console.print("\n[bold]Uncertainty Metrics:[/bold]")
        console.print("  • [magenta]Vacuity[/magenta]: epistemic uncertainty — lower is more confident")
        console.print("  • [cyan]Entropy[/cyan]: aleatoric uncertainty — lower is more decisive")
        console.print("  • [white]Strength[/white]: Dirichlet strength — higher means more evidence")


if __name__ == "__main__":
    app()
