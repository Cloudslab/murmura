"""Network orchestrator for decentralized federated learning."""

from typing import List, Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from murmura.core.node import Node
from murmura.core.types import ModelState
from murmura.topology.base import Topology
from murmura.aggregation.base import Aggregator
from murmura.attacks.base import Attack


class Network:
    """Orchestrates decentralized federated learning across nodes.

    The Network manages a collection of nodes connected via a topology,
    coordinating local training and decentralized aggregation rounds.
    """

    def __init__(
        self,
        nodes: List[Node],
        topology: Topology,
        attack: Optional[Attack] = None
    ):
        """Initialize network.

        Args:
            nodes: List of Node instances
            topology: Network topology defining neighbor relationships
            attack: Optional Byzantine attack mechanism
        """
        if len(nodes) != topology.num_nodes:
            raise ValueError(
                f"Number of nodes ({len(nodes)}) must match topology "
                f"({topology.num_nodes})"
            )

        self.nodes = nodes
        self.topology = topology
        self.attack = attack

        # Training history
        self.history: Dict[str, List[Any]] = {
            "round": [],
            "mean_accuracy": [],
            "std_accuracy": [],
            "mean_loss": [],
            "honest_accuracy": [],
            "compromised_accuracy": [],
            # Evidential uncertainty metrics (if using EDL)
            "mean_vacuity": [],
            "mean_entropy": [],
            "mean_strength": [],
        }

    def train(
        self,
        rounds: int,
        local_epochs: int = 1,
        lr: float = 0.01,
        verbose: bool = False,
        eval_every: int = 1
    ) -> Dict[str, List[Any]]:
        """Run decentralized federated learning.

        Args:
            rounds: Number of training rounds
            local_epochs: Local training epochs per round
            lr: Learning rate
            verbose: Enable verbose logging
            eval_every: Evaluate every N rounds

        Returns:
            Training history dictionary
        """
        for round_num in range(rounds):
            if verbose:
                print(f"\n=== Round {round_num + 1}/{rounds} ===")

            # Step 1: Local training on all nodes
            self._local_training_step(local_epochs, lr, round_num, verbose)

            # Step 2: Exchange models and aggregate
            self._aggregation_step(round_num, verbose)

            # Step 3: Evaluate
            if (round_num + 1) % eval_every == 0:
                self._evaluation_step(round_num + 1, verbose)

        return self.history

    def _local_training_step(self, epochs: int, lr: float, round_num: int, verbose: bool) -> None:
        """Perform local training on all nodes."""
        for node in self.nodes:
            # Skip training for compromised nodes (they keep frozen models)
            if self.attack and self.attack.is_compromised(node.node_id):
                continue

            node.local_train(epochs=epochs, lr=lr, round_num=round_num)

    def _aggregation_step(self, round_num: int, verbose: bool) -> None:
        """Perform decentralized aggregation across the network."""
        # Collect all current states (before aggregation)
        current_states = [node.get_state() for node in self.nodes]

        # Apply attacks if enabled
        if self.attack:
            for node_id, node in enumerate(self.nodes):
                if self.attack.is_compromised(node_id):
                    # Compromised node broadcasts attacked state
                    current_states[node_id] = self.attack.apply_attack(
                        node_id=node_id,
                        model_state=current_states[node_id],
                        round_num=round_num
                    )

        # Each node aggregates with its neighbors
        aggregated_states = []
        for node_id, node in enumerate(self.nodes):
            # Get neighbor states according to topology
            neighbor_ids = self.topology.neighbors[node_id]
            neighbor_states = {
                nid: current_states[nid] for nid in neighbor_ids
            }

            # Aggregate
            aggregated_state = node.aggregate_with_neighbors(
                neighbor_states=neighbor_states,
                round_num=round_num
            )
            aggregated_states.append(aggregated_state)

        # Apply aggregated states to all nodes
        for node, aggregated_state in zip(self.nodes, aggregated_states):
            node.apply_aggregated_state(aggregated_state)

    def _evaluation_step(self, round_num: int, verbose: bool) -> None:
        """Evaluate all nodes and record metrics."""
        accuracies = []
        losses = []
        honest_accuracies = []
        compromised_accuracies = []

        # Evidential metrics
        vacuities = []
        entropies = []
        strengths = []

        for node in self.nodes:
            eval_results = node.evaluate()
            acc = eval_results.get("accuracy", 0.0)
            loss = eval_results.get("loss", 0.0)

            accuracies.append(acc)
            losses.append(loss)

            # Collect evidential metrics if available
            if "vacuity" in eval_results:
                vacuities.append(eval_results["vacuity"])
                entropies.append(eval_results["entropy"])
                strengths.append(eval_results["strength"])

            # Separate honest vs compromised
            if self.attack:
                if self.attack.is_compromised(node.node_id):
                    compromised_accuracies.append(acc)
                else:
                    honest_accuracies.append(acc)

        # Record statistics
        self.history["round"].append(round_num)
        self.history["mean_accuracy"].append(np.mean(accuracies))
        self.history["std_accuracy"].append(np.std(accuracies))
        self.history["mean_loss"].append(np.mean(losses))

        if honest_accuracies:
            self.history["honest_accuracy"].append(np.mean(honest_accuracies))
        if compromised_accuracies:
            self.history["compromised_accuracy"].append(np.mean(compromised_accuracies))

        # Record evidential metrics if available
        if vacuities:
            self.history["mean_vacuity"].append(np.mean(vacuities))
            self.history["mean_entropy"].append(np.mean(entropies))
            self.history["mean_strength"].append(np.mean(strengths))

        if verbose:
            print(f"Round {round_num}: Mean Accuracy = {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            if honest_accuracies and compromised_accuracies:
                print(f"  Honest: {np.mean(honest_accuracies):.4f}, "
                      f"Compromised: {np.mean(compromised_accuracies):.4f}")
            # Print evidential metrics
            if vacuities:
                print(f"  Uncertainty: Vacuity={np.mean(vacuities):.4f}, "
                      f"Entropy={np.mean(entropies):.4f}, Strength={np.mean(strengths):.2f}")

    def get_node_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics from all nodes' aggregators.

        Returns:
            Dictionary mapping node IDs to their statistics
        """
        stats = {}
        for node in self.nodes:
            stats[node.node_id] = node.get_aggregator_statistics()
        return stats

    @classmethod
    def from_config(
        cls,
        config: Any,
        model_factory: Callable[[], nn.Module],
        dataset_adapter: Any,
        aggregator_factory: Callable[[int], Aggregator],
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        evidential: bool = False,
    ) -> "Network":
        """Create network from configuration.

        Args:
            config: Configuration object
            model_factory: Factory function that creates model instances
            dataset_adapter: Dataset adapter with federated partitions
            aggregator_factory: Factory function that creates aggregator instances
            device: Device for computation
            criterion: Optional custom loss function (e.g., EvidentialLoss)
            evidential: Whether models are evidential (output Dirichlet params)

        Returns:
            Configured Network instance
        """
        from murmura.topology import create_topology
        from murmura.attacks.gaussian import GaussianAttack
        from murmura.attacks.directed import DirectedDeviationAttack

        # Create topology
        topology = create_topology(
            topology_type=config.topology.type,
            num_nodes=config.topology.num_nodes,
            p=config.topology.p,
            k=config.topology.k,
            seed=config.topology.seed
        )

        # Create attack if enabled
        attack = None
        if config.attack.enabled:
            if config.attack.type == "gaussian":
                attack = GaussianAttack(
                    num_nodes=config.topology.num_nodes,
                    attack_percentage=config.attack.percentage,
                    noise_std=config.attack.params.get("noise_std", 10.0),
                    seed=config.experiment.seed
                )
            elif config.attack.type == "directed_deviation":
                attack = DirectedDeviationAttack(
                    num_nodes=config.topology.num_nodes,
                    attack_percentage=config.attack.percentage,
                    lambda_param=config.attack.params.get("lambda_param", -5.0),
                    seed=config.experiment.seed
                )

        # Create nodes
        nodes = []
        for node_id in range(config.topology.num_nodes):
            # Create model
            model = model_factory()

            # Get node's data
            train_dataset = dataset_adapter.get_client_data(node_id)

            # Ensure batch size is valid for this node's data
            # BatchNorm requires at least 2 samples; also ensure we have at least 1 batch
            num_samples = len(train_dataset)
            effective_batch_size = min(config.training.batch_size, max(2, num_samples))

            train_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                drop_last=num_samples > effective_batch_size,  # Only drop if we have extra samples
            )

            # For simplicity, use same data for test (in practice, separate test set)
            test_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=False
            )

            # Create aggregator
            aggregator = aggregator_factory(node_id)

            # Create node
            node = Node(
                node_id=node_id,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                aggregator=aggregator,
                device=device,
                criterion=criterion,
                evidential=evidential,
            )
            nodes.append(node)

        return cls(nodes=nodes, topology=topology, attack=attack)
