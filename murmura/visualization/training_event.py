import time
from typing import List, Dict, Optional, Any
import numpy as np


class TrainingEvent:
    """Base class for training process events"""

    def __init__(self, round_num: int, step_name: str):
        """
        Args:
            round_num (int): The current round number.
            step_name (str): The name of the step in the training process.
        """
        self.round_num = round_num
        self.step_name = step_name
        self.timestamp = time.time()


class LocalTrainingEvent(TrainingEvent):
    """Event for local training process"""

    def __init__(
        self,
        round_num: int,
        active_nodes: List[int],
        metrics: Dict,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ):
        """
        Args:
            round_num (int): The current round number.
            active_nodes (List[int]): List of active nodes.
            metrics (Dict): Dictionary containing training metrics.
            current_epoch (Optional[int]): Current epoch in training.
            total_epochs (Optional[int]): Total number of epochs per round.
        """
        super().__init__(round_num, "local_training")
        self.active_nodes = active_nodes
        self.metrics = metrics
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs


class ParameterTransferEvent(TrainingEvent):
    """Event for parameter transfer process"""

    def __init__(
        self,
        round_num: int,
        source_nodes: List[int],
        target_nodes: List[int],
        param_summary: Dict,
    ):
        """
        Args:
            round_num (int): The current round number.
            source_nodes (List[int]): List of source nodes.
            target_nodes (List[int]): List of target nodes.
            param_summary (Dict): Dictionary containing parameter summary.
        """
        super().__init__(round_num, "parameter_transfer")
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.param_summary = param_summary


class AggregationEvent(TrainingEvent):
    """Event for aggregation process"""

    def __init__(
        self,
        round_num: int,
        participating_nodes: List[int],
        aggregator_node: Optional[int],
        strategy_name: str,
    ):
        """
        Args:
            round_num (int): The current round number.
            participating_nodes (List[int]): List of participating nodes.
            aggregator_node (Optional[int]): The node performing the aggregation.
            strategy_name (str): The name of the aggregation strategy.
        """
        super().__init__(round_num, "aggregation")
        self.participating_nodes = participating_nodes
        self.aggregator_node = aggregator_node
        self.strategy_name = strategy_name


class ModelUpdateEvent(TrainingEvent):
    """Event for model update process"""

    def __init__(
        self, round_num: int, updated_nodes: List[int], param_convergence: float
    ):
        """
        Args:
            round_num (int): The current round number.
            updated_nodes (List[int]): List of nodes that received the updated model.
            param_convergence (float): The convergence metric of the parameters.
        """
        super().__init__(round_num, "model_update")
        self.updated_nodes = updated_nodes
        self.param_convergence = param_convergence


class EvaluationEvent(TrainingEvent):
    """Event for evaluation process"""

    def __init__(self, round_num: int, metrics: Dict):
        """
        Args:
            round_num (int): The current round number.
            metrics (Dict): Dictionary containing evaluation metrics.
        """
        super().__init__(round_num, "evaluation")
        self.metrics = metrics


class NetworkStructureEvent(TrainingEvent):
    """Event containing comprehensive static network structure information"""

    def __init__(
        self,
        topology_type: str,
        num_nodes: int,
        adjacency_matrix: Optional[List[List[int]]] = None,
        node_identifiers: Optional[List[str]] = None,
        edge_weights: Optional[Dict[str, float]] = None,
        node_attributes: Optional[Dict[int, Dict[str, Any]]] = None,
        geographic_info: Optional[Dict[int, Dict[str, float]]] = None,
        organizational_hierarchy: Optional[Dict[int, Dict[str, str]]] = None,
    ):
        """
        :param topology_type: The type of network topology.
        :param num_nodes: The number of nodes in the network.
        :param adjacency_matrix: Complete graph representation showing node connections.
        :param node_identifiers: Unique IDs for each participating device/institution.
        :param edge_weights: Connection strengths, bandwidth capacities, or link qualities.
        :param node_attributes: Hardware specs, computational capabilities, memory constraints.
        :param geographic_info: Physical locations (latitude, longitude, region).
        :param organizational_hierarchy: Institution types, department affiliations, admin levels.
        """
        super().__init__(round_num=0, step_name="network_structure")
        self.topology_type = topology_type
        self.num_nodes = num_nodes
        self.adjacency_matrix = (
            adjacency_matrix or self._create_default_adjacency_matrix(num_nodes)
        )
        self.node_identifiers = node_identifiers or [
            f"node_{i}" for i in range(num_nodes)
        ]
        self.edge_weights = edge_weights or {}
        self.node_attributes = node_attributes or self._create_default_node_attributes(
            num_nodes
        )
        self.geographic_info = geographic_info or {}
        self.organizational_hierarchy = organizational_hierarchy or {}

    def _create_default_adjacency_matrix(self, num_nodes: int) -> List[List[int]]:
        """Create a default adjacency matrix based on topology type"""
        matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

        if self.topology_type == "complete":
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        matrix[i][j] = 1
        elif self.topology_type == "ring":
            for i in range(num_nodes):
                matrix[i][(i + 1) % num_nodes] = 1
                matrix[(i + 1) % num_nodes][i] = 1
        elif self.topology_type == "line":
            for i in range(num_nodes - 1):
                matrix[i][i + 1] = 1
                matrix[i + 1][i] = 1
        # For star and custom, matrix remains mostly zeros (handled elsewhere)

        return matrix

    def _create_default_node_attributes(
        self, num_nodes: int
    ) -> Dict[int, Dict[str, Any]]:
        """Create default node attributes with realistic hardware specifications"""
        attributes = {}

        # Generate diverse but realistic hardware configurations
        cpu_options = [4, 8, 16, 32, 64]  # CPU cores
        memory_options = [8, 16, 32, 64, 128]  # GB RAM
        gpu_options = [0, 1, 2, 4, 8]  # Number of GPUs
        bandwidth_options = [100, 500, 1000, 2000, 5000]  # Mbps

        for i in range(num_nodes):
            # Use modulo to create consistent but diverse configurations
            attributes[i] = {
                "cpu_cores": cpu_options[i % len(cpu_options)],
                "memory_gb": memory_options[i % len(memory_options)],
                "gpu_count": gpu_options[i % len(gpu_options)],
                "bandwidth_mbps": bandwidth_options[i % len(bandwidth_options)],
                "storage_gb": 1000 * (1 + i % 5),  # 1TB to 5TB
                "node_type": "edge"
                if i % 3 == 0
                else ("cloud" if i % 3 == 1 else "mobile"),
                "reliability_score": 0.7 + (i % 4) * 0.075,  # 0.7 to 0.925
                "compute_capability": f"tier_{1 + i % 3}",  # tier_1, tier_2, tier_3
            }

        return attributes

    def get_network_summary(self) -> Dict[str, Any]:
        """Get comprehensive network structure summary"""
        # Calculate network metrics
        total_edges = sum(sum(row) for row in self.adjacency_matrix) // 2
        node_degrees = [sum(row) for row in self.adjacency_matrix]
        avg_degree = np.mean(node_degrees) if node_degrees else 0
        max_degree = max(node_degrees) if node_degrees else 0
        min_degree = min(node_degrees) if node_degrees else 0

        # Calculate connectivity metrics
        density = (
            total_edges / (self.num_nodes * (self.num_nodes - 1) / 2)
            if self.num_nodes > 1
            else 0
        )

        # Aggregate node capabilities
        total_cpu_cores = sum(
            attr.get("cpu_cores", 0) for attr in self.node_attributes.values()
        )
        total_memory = sum(
            attr.get("memory_gb", 0) for attr in self.node_attributes.values()
        )
        total_gpus = sum(
            attr.get("gpu_count", 0) for attr in self.node_attributes.values()
        )
        avg_bandwidth = np.mean(
            [attr.get("bandwidth_mbps", 0) for attr in self.node_attributes.values()]
        )

        # Node type distribution
        node_types: Dict[str, int] = {}
        for attr in self.node_attributes.values():
            node_type = attr.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        return {
            "topology_type": self.topology_type,
            "num_nodes": self.num_nodes,
            "total_edges": total_edges,
            "network_density": density,
            "node_degrees": {
                "average": avg_degree,
                "maximum": max_degree,
                "minimum": min_degree,
                "distribution": node_degrees,
            },
            "resource_totals": {
                "cpu_cores": total_cpu_cores,
                "memory_gb": total_memory,
                "gpu_count": total_gpus,
                "avg_bandwidth_mbps": avg_bandwidth,
            },
            "node_type_distribution": node_types,
            "geographic_coverage": len(self.geographic_info),
            "organizational_entities": len(
                set(
                    attr.get("institution_type", "")
                    for attr in self.organizational_hierarchy.values()
                )
            )
            if self.organizational_hierarchy
            else 0,
        }


class FingerprintEvent(TrainingEvent):
    """Event for gradient fingerprint data collection"""

    def __init__(
        self, 
        round_num: int, 
        node_id: str, 
        fingerprint_data: Dict[str, float]
    ):
        """
        Args:
            round_num (int): The current round number.
            node_id (str): The ID of the node generating the fingerprint.
            fingerprint_data (Dict[str, float]): Dictionary containing fingerprint metrics.
        """
        super().__init__(round_num, "gradient_fingerprint")
        self.node_id = node_id
        self.fingerprint_data = fingerprint_data


class TrustSignalsEvent(TrainingEvent):
    """Event for trust signal data collection"""

    def __init__(
        self,
        round_num: int,
        observer_node: str,
        target_node: str,
        trust_signals: Dict[str, float],
        fingerprint_comparison: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            round_num (int): The current round number.
            observer_node (str): The ID of the node computing trust signals.
            target_node (str): The ID of the node being evaluated.
            trust_signals (Dict[str, float]): Dictionary containing all trust signals.
            fingerprint_comparison (Optional[Dict[str, Any]]): Optional fingerprint comparison data.
        """
        super().__init__(round_num, "trust_signals")
        self.observer_node = observer_node
        self.target_node = target_node
        self.trust_signals = trust_signals
        self.fingerprint_comparison = fingerprint_comparison


class InitialStateEvent(TrainingEvent):
    """Event for initial state process - kept for backward compatibility"""

    def __init__(self, topology_type: str, num_nodes: int):
        """
        :param topology_type: The type of network topology.
        :param num_nodes: The number of nodes in the network.
        """
        super().__init__(round_num=0, step_name="initial_state")
        self.topology_type = topology_type
        self.num_nodes = num_nodes
