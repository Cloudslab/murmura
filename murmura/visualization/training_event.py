import time
from typing import List, Dict, Optional


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

    def __init__(self, round_num: int, active_nodes: List[int], metrics: Dict):
        """
        Args:
            round_num (int): The current round number.
            active_nodes (List[int]): List of active nodes.
            metrics (Dict): Dictionary containing training metrics.
        """
        super().__init__(round_num, "local_training")
        self.active_nodes = active_nodes
        self.metrics = metrics


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


class InitialStateEvent(TrainingEvent):
    """Event for initial state process"""

    def __init__(self, topology_type: str, num_nodes: int):
        """
        :param topology_type: The type of network topology.
        :param num_nodes: The number of nodes in the network.
        """
        super().__init__(round_num=0, step_name="initial_state")
        self.topology_type = topology_type
        self.num_nodes = num_nodes
