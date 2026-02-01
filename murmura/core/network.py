"""Network orchestrator for decentralized federated learning."""

from typing import List, Optional, Dict, Any, Callable, Literal
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
        attack: Optional[Attack] = None,
        mode: Literal["decentralized", "centralized"] = "decentralized"
    ):
        """Initialize network.

        Args:
            nodes: List of Node instances
            topology: Network topology defining neighbor relationships
            attack: Optional Byzantine attack mechanism
            mode: Network mode - decentralized (peer-to-peer) or centralized (server-based)
        """
        if len(nodes) != topology.num_nodes:
            raise ValueError(
                f"Number of nodes ({len(nodes)}) must match topology "
                f"({topology.num_nodes})"
            )

        self.nodes = nodes
        self.topology = topology
        self.attack = attack
        self.mode = mode

        # Training history
        self.history: Dict[str, List[Any]] = {
            "round": [],
            "mean_accuracy": [],
            "std_accuracy": [],
            "mean_loss": [],
            "honest_accuracy": [],
            "compromised_accuracy": [],
            # Comprehensive metrics
            "mean_precision": [],
            "mean_recall": [],
            "mean_f1": [],
            "mean_auc": [],
            # Evidential uncertainty metrics (if using EDL)
            "mean_vacuity": [],
            "mean_entropy": [],
            "mean_strength": [],
            # Communication metrics (for FedRansel)
            "communication_cost": [],
        }

        # Detailed per-round per-node metrics storage
        self.detailed_metrics: List[Dict[str, Any]] = []

        # Parameter tracking (can be large - optional)
        self.track_parameters: bool = False
        self.parameter_history: List[Dict[str, Any]] = []

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

            # Step 2: Exchange models and aggregate (based on mode)
            if self.mode == "centralized":
                self._centralized_aggregation_step(round_num, verbose)
            else:
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

    def _centralized_aggregation_step(self, round_num: int, verbose: bool) -> None:
        """Perform centralized aggregation (all nodes -> server -> all nodes).

        In centralized mode, all node states are collected and sent to a single
        aggregator (server), which computes the global aggregate and broadcasts
        the result to all nodes. This is used for algorithms like FedRansel.
        """
        # Collect all current states
        current_states = {
            node.node_id: node.get_state()
            for node in self.nodes
        }

        # Apply attacks if enabled
        if self.attack:
            for node_id in current_states:
                if self.attack.is_compromised(node_id):
                    current_states[node_id] = self.attack.apply_attack(
                        node_id=node_id,
                        model_state=current_states[node_id],
                        round_num=round_num
                    )

        # Use first node's aggregator as the "server" aggregator
        # In centralized mode, all nodes share the same aggregator type
        server_aggregator = self.nodes[0].aggregator

        # Reference state for structure (any node's state)
        reference_state = current_states[0]

        # Server aggregates ALL node states
        # Pass node_id=-1 to indicate server, all states as neighbor_states
        aggregated_state = server_aggregator.aggregate(
            node_id=-1,  # Server ID
            own_state=reference_state,
            neighbor_states=current_states,
            round_num=round_num,
            train_loader=self.nodes[0].train_loader,
            model_template=self.nodes[0].model,
            device=self.nodes[0].device,
        )

        # Track communication cost if aggregator provides it
        if hasattr(server_aggregator, 'communication_costs') and server_aggregator.communication_costs:
            self.history["communication_cost"].append(
                server_aggregator.communication_costs[-1]
            )

        # Broadcast aggregated state to ALL nodes (same state for all)
        for node in self.nodes:
            node.apply_aggregated_state(aggregated_state)

    def _evaluation_step(self, round_num: int, verbose: bool) -> None:
        """Evaluate all nodes and record metrics."""
        accuracies = []
        losses = []
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []
        honest_accuracies = []
        compromised_accuracies = []

        # Evidential metrics
        vacuities = []
        entropies = []
        strengths = []

        # Per-node detailed metrics for this round
        per_node_metrics = {}

        for node in self.nodes:
            eval_results = node.evaluate(comprehensive=True)
            acc = eval_results.get("accuracy", 0.0)
            loss = eval_results.get("loss", 0.0)

            accuracies.append(acc)
            losses.append(loss)

            # Collect comprehensive metrics
            if "precision" in eval_results:
                precisions.append(eval_results["precision"])
                recalls.append(eval_results["recall"])
                f1_scores.append(eval_results["f1"])
                auc_scores.append(eval_results.get("auc", 0.5))

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

            # Store per-node metrics
            per_node_metrics[node.node_id] = eval_results

        # Record statistics
        self.history["round"].append(round_num)
        self.history["mean_accuracy"].append(np.mean(accuracies))
        self.history["std_accuracy"].append(np.std(accuracies))
        self.history["mean_loss"].append(np.mean(losses))

        # Record comprehensive metrics
        if precisions:
            self.history["mean_precision"].append(np.mean(precisions))
            self.history["mean_recall"].append(np.mean(recalls))
            self.history["mean_f1"].append(np.mean(f1_scores))
            self.history["mean_auc"].append(np.mean(auc_scores))

        if honest_accuracies:
            self.history["honest_accuracy"].append(np.mean(honest_accuracies))
        if compromised_accuracies:
            self.history["compromised_accuracy"].append(np.mean(compromised_accuracies))

        # Record evidential metrics if available
        if vacuities:
            self.history["mean_vacuity"].append(np.mean(vacuities))
            self.history["mean_entropy"].append(np.mean(entropies))
            self.history["mean_strength"].append(np.mean(strengths))

        # Store detailed per-node metrics
        self.detailed_metrics.append({
            "round": round_num,
            "per_node": per_node_metrics,
        })

        if verbose:
            print(f"Round {round_num}: Acc={np.mean(accuracies):.4f}±{np.std(accuracies):.4f}", end="")
            if precisions:
                print(f", P={np.mean(precisions):.4f}, R={np.mean(recalls):.4f}, F1={np.mean(f1_scores):.4f}, AUC={np.mean(auc_scores):.4f}", end="")
            print()
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

    def enable_parameter_tracking(self, enabled: bool = True) -> None:
        """Enable or disable full parameter tracking.

        Warning: This can consume significant memory for large models.

        Args:
            enabled: Whether to track full parameters each round
        """
        self.track_parameters = enabled

    def get_detailed_metrics(self) -> List[Dict[str, Any]]:
        """Get detailed per-round per-node metrics.

        Returns:
            List of dicts with per-round metrics for each node
        """
        return self.detailed_metrics

    def get_aggregator_detailed_history(self) -> Dict[str, Any]:
        """Get detailed history from the aggregator (for FedRansel, DP-FedAvg, etc.).

        Returns:
            Dictionary with aggregator-specific detailed history
        """
        if not self.nodes:
            return {}

        aggregator = self.nodes[0].aggregator
        if aggregator is None:
            return {}

        result = {"algorithm": type(aggregator).__name__}

        # Get FedRansel detailed history
        if hasattr(aggregator, 'get_detailed_history'):
            result["detailed_history"] = aggregator.get_detailed_history()

        # Get DP-FedAvg budget history
        if hasattr(aggregator, 'get_budget_history'):
            result["budget_history"] = aggregator.get_budget_history()

        # Get general statistics
        result["statistics"] = aggregator.get_statistics()

        return result

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

        # Get network mode (default to decentralized for backward compatibility)
        mode = "decentralized"
        if hasattr(config, 'network') and hasattr(config.network, 'mode'):
            mode = config.network.mode

        return cls(nodes=nodes, topology=topology, attack=attack, mode=mode)
