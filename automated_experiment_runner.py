#!/usr/bin/env python3
"""
Enhanced Automated Murmura Experiment Runner with Observer Pattern
Uses the built-in observer system to collect metrics directly from training
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from murmura.visualization.training_observer import TrainingObserver
from murmura.visualization.training_event import TrainingEvent, EvaluationEvent


class ExperimentMetricsCollector(TrainingObserver):
    """
    Enhanced observer that collects comprehensive metrics during training experiments.
    Captures all data used by the visualization system for detailed analysis.
    """

    def __init__(self):
        # Core metrics tracking
        self.round_metrics = []
        self.initial_metrics = None
        self.final_metrics = None
        self.training_start_time = None
        self.training_end_time = None

        # Detailed event tracking (same as NetworkVisualizer)
        self.accuracy_history = {}  # round -> accuracy
        self.loss_history = {}  # round -> loss
        self.parameter_history = {}  # node -> [parameter_norms]
        self.convergence_history = []  # [convergence_values]

        # Training process tracking
        self.local_training_events = []
        self.parameter_transfer_events = []
        self.aggregation_events = []
        self.model_update_events = []

        # Network topology info
        self.topology_type = None
        self.num_nodes = 0
        self.aggregation_strategy = None

        # Communication and efficiency metrics
        self.total_parameter_transfers = 0
        self.communication_rounds = 0
        self.active_nodes_per_round = {}

        # Training dynamics
        self.epochs_per_round = {}
        self.batch_metrics = {}

    def reset(self):
        """Reset collector for a new experiment"""
        self.__init__()

    def on_event(self, event: TrainingEvent) -> None:
        """Handle training events to collect comprehensive metrics"""
        from murmura.visualization.training_event import (
            EvaluationEvent, LocalTrainingEvent, ParameterTransferEvent,
            AggregationEvent, ModelUpdateEvent, InitialStateEvent
        )

        if isinstance(event, InitialStateEvent):
            self.topology_type = event.topology_type
            self.num_nodes = event.num_nodes

        elif isinstance(event, LocalTrainingEvent):
            self.local_training_events.append({
                'round': event.round_num,
                'timestamp': event.timestamp,
                'active_nodes': len(event.active_nodes),
                'total_nodes': self.num_nodes,
                'node_list': event.active_nodes.copy(),
                'current_epoch': getattr(event, 'current_epoch', None),
                'total_epochs': getattr(event, 'total_epochs', None),
                'metrics': event.metrics.copy() if event.metrics else {}
            })

            # Track active nodes per round
            self.active_nodes_per_round[event.round_num] = len(event.active_nodes)

            # Track epochs
            if hasattr(event, 'total_epochs') and event.total_epochs:
                self.epochs_per_round[event.round_num] = event.total_epochs

        elif isinstance(event, ParameterTransferEvent):
            transfer_data = {
                'round': event.round_num,
                'timestamp': event.timestamp,
                'source_nodes': event.source_nodes.copy(),
                'target_nodes': event.target_nodes.copy(),
                'num_transfers': len(event.source_nodes) * len(event.target_nodes),
                'param_summary': event.param_summary.copy() if event.param_summary else {}
            }
            self.parameter_transfer_events.append(transfer_data)

            # Track total transfers for communication overhead
            self.total_parameter_transfers += transfer_data['num_transfers']

            # Store parameter norms for convergence analysis
            for node, summary in event.param_summary.items():
                if node not in self.parameter_history:
                    self.parameter_history[node] = []

                if isinstance(summary, dict) and 'norm' in summary:
                    self.parameter_history[node].append(summary['norm'])
                elif isinstance(summary, dict) and 'mean' in summary:
                    # Use mean as fallback for parameter tracking
                    self.parameter_history[node].append(abs(summary['mean']))
                else:
                    # Fallback for raw parameter data
                    self.parameter_history[node].append(1.0)

        elif isinstance(event, AggregationEvent):
            self.aggregation_events.append({
                'round': event.round_num,
                'timestamp': event.timestamp,
                'participating_nodes': len(event.participating_nodes),
                'node_list': event.participating_nodes.copy(),
                'aggregator_node': event.aggregator_node,
                'strategy_name': event.strategy_name
            })

            # Track aggregation strategy
            if not self.aggregation_strategy:
                self.aggregation_strategy = event.strategy_name

        elif isinstance(event, ModelUpdateEvent):
            update_data = {
                'round': event.round_num,
                'timestamp': event.timestamp,
                'updated_nodes': len(event.updated_nodes),
                'node_list': event.updated_nodes.copy(),
                'param_convergence': event.param_convergence
            }
            self.model_update_events.append(update_data)

            # Track convergence over time
            self.convergence_history.append(event.param_convergence)

        elif isinstance(event, EvaluationEvent):
            # Store detailed round metrics
            metrics_data = {
                'round': event.round_num,
                'timestamp': event.timestamp,
                **event.metrics
            }
            self.round_metrics.append(metrics_data)

            # Track accuracy and loss history (same as NetworkVisualizer)
            if 'accuracy' in event.metrics:
                self.accuracy_history[event.round_num] = event.metrics['accuracy']
            if 'loss' in event.metrics:
                self.loss_history[event.round_num] = event.metrics['loss']

            # Track initial and final metrics
            if event.round_num == 0:
                self.initial_metrics = event.metrics.copy()
                self.training_start_time = event.timestamp
            else:
                self.final_metrics = event.metrics.copy()
                self.training_end_time = event.timestamp

    def get_experiment_results(self) -> Dict[str, Any]:
        """Extract comprehensive experiment results matching visualization system"""
        # Add numpy import for statistical calculations
        import numpy as np

        if not self.initial_metrics or not self.final_metrics:
            return {
                'initial_accuracy': 0.0,
                'final_accuracy': 0.0,
                'accuracy_improvement': 0.0,
                'convergence_rounds': 0,
                'training_time_minutes': 0.0,
                'error': 'No metrics collected'
            }

        # Basic performance metrics
        initial_acc = self.initial_metrics.get('accuracy', 0.0)
        final_acc = self.final_metrics.get('accuracy', 0.0)
        initial_loss = self.initial_metrics.get('loss', 0.0)
        final_loss = self.final_metrics.get('loss', 0.0)
        improvement = final_acc - initial_acc

        # Training dynamics
        convergence_rounds = len([m for m in self.round_metrics if m['round'] > 0])
        actual_rounds = max(self.accuracy_history.keys()) if self.accuracy_history else 0

        # Calculate training time
        training_time = 0.0
        if self.training_start_time and self.training_end_time:
            training_time = (self.training_end_time - self.training_start_time) / 60.0

        # Communication and efficiency metrics
        avg_active_nodes = np.mean(list(self.active_nodes_per_round.values())) if self.active_nodes_per_round else 0
        communication_efficiency = self.total_parameter_transfers / max(1, actual_rounds) if actual_rounds > 0 else 0

        # Parameter convergence analysis
        final_convergence = self.convergence_history[-1] if self.convergence_history else 0.0
        avg_convergence = np.mean(self.convergence_history) if self.convergence_history else 0.0
        convergence_trend = 'improving' if len(self.convergence_history) > 1 and self.convergence_history[-1] < self.convergence_history[0] else 'stable'

        # Accuracy and loss trends
        accuracy_trend = 'improving' if final_acc > initial_acc else 'degrading'
        loss_trend = 'improving' if final_loss < initial_loss else 'degrading'

        # Calculate accuracy/loss statistics over rounds
        accuracy_values = list(self.accuracy_history.values())
        loss_values = list(self.loss_history.values())

        accuracy_std = np.std(accuracy_values) if len(accuracy_values) > 1 else 0.0
        loss_std = np.std(loss_values) if len(loss_values) > 1 else 0.0

        accuracy_max = max(accuracy_values) if accuracy_values else final_acc
        accuracy_min = min(accuracy_values) if accuracy_values else initial_acc
        loss_max = max(loss_values) if loss_values else initial_loss
        loss_min = min(loss_values) if loss_values else final_loss

        # Parameter diversity (measure of how different nodes' parameters are)
        parameter_diversity = 0.0
        if len(self.parameter_history) > 1:
            # Calculate standard deviation across nodes for final round
            final_params = []
            for node_params in self.parameter_history.values():
                if node_params:
                    final_params.append(node_params[-1])
            parameter_diversity = np.std(final_params) if len(final_params) > 1 else 0.0

        # Training stability (consistency of improvements)
        accuracy_improvements = []
        if len(accuracy_values) > 1:
            for i in range(1, len(accuracy_values)):
                accuracy_improvements.append(accuracy_values[i] - accuracy_values[i-1])
        training_stability = 1.0 / (1.0 + np.std(accuracy_improvements)) if accuracy_improvements else 1.0

        return {
            # Basic performance metrics
            'initial_accuracy': initial_acc,
            'final_accuracy': final_acc,
            'accuracy_improvement': improvement,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction': initial_loss - final_loss,

            # Training process metrics
            'convergence_rounds': convergence_rounds,
            'actual_rounds': actual_rounds,
            'training_time_minutes': training_time,
            'time_per_round': training_time / max(1, actual_rounds),

            # Accuracy statistics
            'accuracy_max': accuracy_max,
            'accuracy_min': accuracy_min,
            'accuracy_std': accuracy_std,
            'accuracy_trend': accuracy_trend,

            # Loss statistics
            'loss_max': loss_max,
            'loss_min': loss_min,
            'loss_std': loss_std,
            'loss_trend': loss_trend,

            # Network and topology metrics
            'topology_type': self.topology_type or 'unknown',
            'num_nodes': self.num_nodes,
            'aggregation_strategy': self.aggregation_strategy or 'unknown',
            'avg_active_nodes_per_round': avg_active_nodes,
            'node_participation_rate': avg_active_nodes / max(1, self.num_nodes),

            # Communication metrics
            'total_parameter_transfers': self.total_parameter_transfers,
            'parameter_transfers_per_round': communication_efficiency,
            'total_local_training_events': len(self.local_training_events),
            'total_aggregation_events': len(self.aggregation_events),
            'total_model_updates': len(self.model_update_events),

            # Convergence metrics
            'final_parameter_convergence': final_convergence,
            'avg_parameter_convergence': avg_convergence,
            'parameter_diversity': parameter_diversity,
            'convergence_trend': convergence_trend,

            # Training dynamics
            'training_stability': training_stability,
            'rounds_with_improvement': len([i for i in accuracy_improvements if i > 0]) if accuracy_improvements else 0,
            'max_single_round_improvement': max(accuracy_improvements) if accuracy_improvements else 0.0,

            # Detailed data for further analysis
            'round_metrics': self.round_metrics.copy(),
            'accuracy_history': self.accuracy_history.copy(),
            'loss_history': self.loss_history.copy(),
            'convergence_history': self.convergence_history.copy(),
            'parameter_history_summary': {
                node: {'count': len(params), 'final': params[-1] if params else 0.0, 'trend': 'stable' if len(params) < 2 else ('increasing' if params[-1] > params[0] else 'decreasing')}
                for node, params in self.parameter_history.items()
            }
        }


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment - same as before"""
    paradigm: str  # centralized, federated, decentralized
    dataset: str  # mnist, ham10000
    topology: str  # star, ring, complete, random
    heterogeneity: str  # iid, moderate_noniid, extreme_noniid
    privacy: str  # none, moderate_dp, strong_dp
    scale: int  # 5, 10, 20

    # Derived parameters
    aggregation_strategy: str = ""
    alpha: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-5

    # Execution parameters
    rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    actors_per_node: int = 2

    def __post_init__(self):
        """Set derived parameters based on configuration"""
        # Set aggregation strategy based on paradigm
        if self.paradigm == "centralized":
            self.aggregation_strategy = "fedavg"
        elif self.paradigm == "federated":
            self.aggregation_strategy = "fedavg"
        elif self.paradigm == "decentralized":
            self.aggregation_strategy = "gossip_avg"

        # Set alpha based on heterogeneity
        if self.heterogeneity == "iid":
            self.alpha = 1000.0
        elif self.heterogeneity == "moderate_noniid":
            self.alpha = 0.5
        elif self.heterogeneity == "extreme_noniid":
            self.alpha = 0.1

        # Set privacy parameters
        if self.privacy == "moderate_dp":
            self.epsilon = 1.0
            self.delta = 1e-5
        elif self.privacy == "strong_dp":
            self.epsilon = 0.1
            self.delta = 1e-6

        # Adjust parameters for decentralized learning
        if self.paradigm == "decentralized":
            if self.privacy == "moderate_dp":
                self.epsilon = 2.0
            elif self.privacy == "strong_dp":
                self.epsilon = 0.5

        # Different training paradigms need different configurations
        if self.paradigm == "centralized":
            self.scale = 1
            self.rounds = 1
            if self.dataset == "ham10000":
                self.local_epochs = 100
                self.learning_rate = 0.0005
            elif self.dataset == "mnist":
                self.local_epochs = 60
                self.learning_rate = 0.001
            if self.privacy != "none":
                self.local_epochs = int(self.local_epochs * 1.2)
                self.learning_rate *= 0.9
        else:
            # Federated/Decentralized learning
            if self.dataset == "ham10000":
                self.rounds = 75
                self.local_epochs = 4
                self.learning_rate = 0.0005
            elif self.dataset == "mnist":
                self.rounds = 40
                self.local_epochs = 3
                self.learning_rate = 0.001
            if self.privacy != "none":
                self.rounds = int(self.rounds * 1.5)
                self.learning_rate *= 0.8
            if self.scale >= 20:
                self.local_epochs += 1

        # Resource allocation
        if self.scale <= 1:
            self.actors_per_node = 1
        elif self.scale <= 10:
            self.actors_per_node = 2
        elif self.scale <= 20:
            self.actors_per_node = 4
        else:
            self.actors_per_node = 6


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    success: bool
    start_time: str
    end_time: str
    duration_minutes: float

    # Performance metrics - collected via observer
    initial_accuracy: float = 0.0
    final_accuracy: float = 0.0
    accuracy_improvement: float = 0.0
    convergence_rounds: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0

    # Efficiency metrics
    training_time_minutes: float = 0.0
    total_communication_mb: float = 0.0
    communication_per_round: float = 0.0

    # Privacy metrics (if applicable)
    privacy_epsilon_spent: float = 0.0
    privacy_delta_spent: float = 0.0
    privacy_budget_utilization: float = 0.0

    # Error information
    error_message: str = ""
    error_type: str = ""

    # Detailed round metrics
    round_metrics: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.round_metrics is None:
            self.round_metrics = []


class EnhancedExperimentRunner:
    """
    Enhanced experiment runner using observer pattern for metrics collection
    """

    def __init__(self, base_dir: str, ray_address: str = None, max_parallel: int = 2):
        self.base_dir = Path(base_dir)
        self.ray_address = ray_address
        self.max_parallel = max_parallel
        self.results_dir = self.base_dir / "experiment_results"
        self.logs_dir = self.base_dir / "experiment_logs"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Track experiment state
        self.completed_experiments = set()
        self.failed_experiments = set()

    def setup_logging(self):
        """Set up centralized logging"""
        log_file = self.logs_dir / f"experiment_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)

    def generate_experiment_configs(self, experiment_set: str) -> List[ExperimentConfig]:
        """Generate all experiment configurations for a given set"""
        configs = []

        if experiment_set == "core":
            # Core Three-Way Comparison
            paradigms = ["centralized", "federated", "decentralized"]
            datasets = ["mnist", "ham10000"]

            valid_combinations = {
                "centralized": {
                    "topologies": ["star", "complete"],
                    "heterogeneity": ["iid"],
                    "privacy": ["none", "moderate_dp", "strong_dp"],
                    "scales": [1]
                },
                "federated": {
                    "topologies": ["star", "complete"],
                    "heterogeneity": ["iid", "moderate_noniid", "extreme_noniid"],
                    "privacy": ["none", "moderate_dp", "strong_dp"],
                    "scales": [5, 10, 20]
                },
                "decentralized": {
                    "topologies": ["ring", "complete", "line"],
                    "heterogeneity": ["iid", "moderate_noniid", "extreme_noniid"],
                    "privacy": ["none", "moderate_dp", "strong_dp"],
                    "scales": [5, 10, 20]
                }
            }

            for paradigm in paradigms:
                for dataset in datasets:
                    paradigm_config = valid_combinations[paradigm]
                    for topology in paradigm_config["topologies"]:
                        for heterogeneity in paradigm_config["heterogeneity"]:
                            for privacy in paradigm_config["privacy"]:
                                for scale in paradigm_config["scales"]:
                                    configs.append(ExperimentConfig(
                                        paradigm=paradigm,
                                        dataset=dataset,
                                        topology=topology,
                                        heterogeneity=heterogeneity,
                                        privacy=privacy,
                                        scale=int(scale)
                                    ))

        elif experiment_set == "topology_privacy":
            # Topology-Privacy Interaction Analysis
            paradigm_topology_map = {
                "federated": ["star", "complete"],
                "decentralized": ["ring", "complete", "line"]
            }
            datasets = ["mnist", "ham10000"]
            privacy_levels = ["none", "moderate_dp", "strong_dp"]

            for paradigm, valid_topologies in paradigm_topology_map.items():
                for dataset in datasets:
                    for topology in valid_topologies:
                        for privacy in privacy_levels:
                            configs.append(ExperimentConfig(
                                paradigm=paradigm,
                                dataset=dataset,
                                topology=topology,
                                heterogeneity="moderate_noniid",
                                privacy=privacy,
                                scale=10
                            ))

        elif experiment_set == "scalability":
            # Scalability Analysis
            paradigms = ["centralized", "federated", "decentralized"]
            datasets = ["mnist", "ham10000"]
            scales = [5, 10, 15, 20]

            for paradigm in paradigms:
                for dataset in datasets:
                    for scale in scales:
                        if paradigm == "centralized":
                            topology = "star"
                            actual_scale = 1
                        elif paradigm == "federated":
                            topology = "star"
                            actual_scale = scale
                        else:  # decentralized
                            topology = "ring"
                            actual_scale = scale

                        configs.append(ExperimentConfig(
                            paradigm=paradigm,
                            dataset=dataset,
                            topology=topology,
                            heterogeneity="moderate_noniid",
                            privacy="none",
                            scale=actual_scale
                        ))

        self.logger.info(f"Generated {len(configs)} configurations for {experiment_set} experiment set")
        return configs

    def run_single_experiment_with_observer(self, config: ExperimentConfig, experiment_id: str) -> ExperimentResult:
        """
        Run a single experiment using the observer pattern to collect metrics
        """
        start_time = datetime.now()
        self.logger.info(f"Starting experiment {experiment_id}: {config.paradigm}-{config.dataset}-{config.topology}")

        # Create metrics collector
        metrics_collector = ExperimentMetricsCollector()

        try:
            # Import the appropriate learning process
            if config.paradigm == "decentralized":
                from murmura.orchestration.learning_process.decentralized_learning_process import DecentralizedLearningProcess
                from murmura.data_processing.dataset import MDataset, DatasetSource
                from murmura.data_processing.partitioner_factory import PartitionerFactory
                from murmura.orchestration.orchestration_config import OrchestrationConfig
                from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
                from murmura.network_management.topology import TopologyConfig, TopologyType
                from murmura.node.resource_config import RayClusterConfig, ResourceConfig

                # Create model based on dataset
                if config.dataset == "mnist":
                    from murmura.examples.decentralized_mnist_example import MNISTModel, create_mnist_preprocessor
                    from murmura.model.pytorch_model import TorchModelWrapper
                    import torch.nn as nn
                    import torch

                    model = MNISTModel()
                    input_shape = (1, 28, 28)
                    mnist_preprocessor = create_mnist_preprocessor()
                    global_model = TorchModelWrapper(
                        model=model,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer_class=torch.optim.Adam,
                        optimizer_kwargs={"lr": config.learning_rate},
                        input_shape=input_shape,
                        data_preprocessor=mnist_preprocessor,
                    )
                    dataset_name = "mnist"
                    feature_columns = ["image"]
                    label_column = "label"

                elif config.dataset == "ham10000":
                    from murmura.examples.decentralized_skin_lesion_example import (
                        WideResNet, create_skin_lesion_preprocessor, add_integer_labels_to_dataset
                    )
                    from murmura.model.pytorch_model import TorchModelWrapper
                    import torch.nn as nn
                    import torch

                    # Load and process skin lesion dataset
                    train_dataset = MDataset.load_dataset_with_multinode_support(
                        DatasetSource.HUGGING_FACE,
                        dataset_name="marmal88/skin_cancer",
                        split="train",
                    )

                    # Convert string labels to integers
                    dx_categories, num_classes, dx_to_label = add_integer_labels_to_dataset(
                        train_dataset, self.logger
                    )

                    model = WideResNet(num_classes=num_classes)
                    input_shape = (3, 128, 128)
                    skin_lesion_preprocessor = create_skin_lesion_preprocessor(128)
                    global_model = TorchModelWrapper(
                        model=model,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer_class=torch.optim.Adam,
                        optimizer_kwargs={"lr": config.learning_rate, "weight_decay": 1e-4},
                        input_shape=input_shape,
                        device="auto",
                        data_preprocessor=skin_lesion_preprocessor,
                    )
                    dataset_name = "marmal88/skin_cancer"
                    feature_columns = ["image"]
                    label_column = "label"
                else:
                    raise ValueError(f"Unsupported dataset for decentralized learning: {config.dataset}")

                # Create configuration
                ray_cluster_config = RayClusterConfig(
                    address=self.ray_address,
                    namespace=f"murmura_exp_{experiment_id}",
                    auto_detect_cluster=True
                )

                resource_config = ResourceConfig(
                    actors_per_node=config.actors_per_node,
                    placement_strategy="spread",
                )

                orchestration_config = OrchestrationConfig(
                    num_actors=config.scale,
                    partition_strategy="dirichlet" if config.paradigm != "centralized" else "iid",
                    alpha=config.alpha,
                    min_partition_size=50,
                    split="train",
                    topology=TopologyConfig(
                        topology_type=TopologyType(config.topology),
                        hub_index=0,
                    ),
                    aggregation=AggregationConfig(
                        strategy_type=AggregationStrategyType(config.aggregation_strategy),
                        params={"mixing_parameter": 0.5} if config.aggregation_strategy == "gossip_avg" else None,
                    ),
                    dataset_name=dataset_name,
                    ray_cluster=ray_cluster_config,
                    resources=resource_config,
                    feature_columns=feature_columns,
                    label_column=label_column,
                    rounds=config.rounds,
                    epochs=config.local_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    test_split="test",
                )

                # Load dataset if not already loaded
                if config.dataset == "mnist":
                    train_dataset = MDataset.load_dataset_with_multinode_support(
                        DatasetSource.HUGGING_FACE,
                        dataset_name="mnist",
                        split="train",
                    )

                # Create learning process
                learning_process = DecentralizedLearningProcess(
                    config=orchestration_config,
                    dataset=train_dataset,
                    model=global_model,
                )

                # Register metrics collector as observer
                learning_process.register_observer(metrics_collector)

                # Create partitioner
                partitioner = PartitionerFactory.create(orchestration_config)

                # Initialize and execute
                learning_process.initialize(
                    num_actors=orchestration_config.num_actors,
                    topology_config=orchestration_config.topology,
                    aggregation_config=orchestration_config.aggregation,
                    partitioner=partitioner,
                )

                # Execute training
                results = learning_process.execute()

                # Shutdown
                learning_process.shutdown()

            else:
                # Handle federated and centralized learning similarly
                # This is a simplified version - you can extend this
                raise NotImplementedError(f"Observer-based execution not yet implemented for {config.paradigm}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            # Get metrics from collector
            collected_metrics = metrics_collector.get_experiment_results()

            if 'error' not in collected_metrics:
                result = ExperimentResult(
                    config=config,
                    success=True,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_minutes=duration,
                    initial_accuracy=collected_metrics['initial_accuracy'],
                    final_accuracy=collected_metrics['final_accuracy'],
                    accuracy_improvement=collected_metrics['accuracy_improvement'],
                    convergence_rounds=collected_metrics['convergence_rounds'],
                    initial_loss=collected_metrics['initial_loss'],
                    final_loss=collected_metrics['final_loss'],
                    training_time_minutes=collected_metrics['training_time_minutes'],
                    round_metrics=[collected_metrics]  # Store all enhanced metrics in round_metrics
                )
                self.logger.info(f"Experiment {experiment_id} completed successfully in {duration:.1f} minutes")
            else:
                result = ExperimentResult(
                    config=config,
                    success=False,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_minutes=duration,
                    error_message=collected_metrics['error'],
                    error_type="metrics_collection_error"
                )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            result = ExperimentResult(
                config=config,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_minutes=duration,
                error_message=str(e),
                error_type=type(e).__name__
            )

            self.logger.error(f"Experiment {experiment_id} failed with exception: {e}")
            self.logger.error(traceback.format_exc())

        return result

    def run_single_experiment_subprocess(self, config: ExperimentConfig, experiment_id: str) -> ExperimentResult:
        """
        Fallback method using subprocess for compatibility with existing scripts
        """
        start_time = datetime.now()
        self.logger.info(f"Starting experiment {experiment_id} via subprocess: {config.paradigm}-{config.dataset}-{config.topology}")

        # Build command (same as original)
        script_path = self.get_example_script_path(config)
        cmd = self.build_command(config, experiment_id)

        # Setup logging for this experiment
        log_file = self.logs_dir / f"{experiment_id}.log"

        try:
            # Run the experiment
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                    cwd=self.base_dir
                )

                output = process.stdout
                f.write(output)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            if process.returncode == 0:
                # Success - parse metrics (fallback method)
                metrics = self.parse_experiment_output(output, config)

                estimated_setup_time = 1.0
                training_time = max(0.0, duration - estimated_setup_time)
                metrics['training_time_minutes'] = training_time

                if 'convergence_rounds' not in metrics:
                    metrics['convergence_rounds'] = config.rounds

                result = ExperimentResult(
                    config=config,
                    success=True,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_minutes=duration,
                    **metrics
                )

                self.logger.info(f"Experiment {experiment_id} completed successfully in {duration:.1f} minutes")

            else:
                # Failure
                error_msg = f"Process exited with code {process.returncode}"
                result = ExperimentResult(
                    config=config,
                    success=False,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_minutes=duration,
                    error_message=error_msg,
                    error_type="subprocess_error"
                )

                self.logger.error(f"Experiment {experiment_id} failed: {error_msg}")

        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            result = ExperimentResult(
                config=config,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_minutes=duration,
                error_message="Experiment timed out after 1 hour",
                error_type="timeout"
            )

            self.logger.error(f"Experiment {experiment_id} timed out after 1 hour")

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60.0

            result = ExperimentResult(
                config=config,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_minutes=duration,
                error_message=str(e),
                error_type=type(e).__name__
            )

            self.logger.error(f"Experiment {experiment_id} failed with exception: {e}")
            self.logger.error(traceback.format_exc())

        return result

    def run_single_experiment(self, config: ExperimentConfig, experiment_id: str) -> ExperimentResult:
        """
        Main method to run a single experiment.
        Tries observer pattern first, falls back to subprocess if needed.
        """
        try:
            # Try observer-based approach for supported configurations
            if config.paradigm == "decentralized" and config.dataset in ["mnist", "ham10000"]:
                return self.run_single_experiment_with_observer(config, experiment_id)
            else:
                # Fall back to subprocess approach for other configurations
                self.logger.info(f"Using subprocess approach for {config.paradigm} paradigm")
                return self.run_single_experiment_subprocess(config, experiment_id)
        except Exception as e:
            self.logger.warning(f"Observer approach failed for {experiment_id}: {e}")
            self.logger.info("Falling back to subprocess approach")
            return self.run_single_experiment_subprocess(config, experiment_id)

    @staticmethod
    def get_example_script_path(config: ExperimentConfig, experiment_id: str = None) -> str:
        """Get the appropriate example script path"""
        base_path = "murmura/examples"
        dataset_script_map = {
            "mnist": "mnist",
            "ham10000": "skin_lesion"
        }

        script_dataset = dataset_script_map.get(config.dataset, config.dataset)

        if config.paradigm == "centralized":
            if config.privacy != "none":
                script = f"dp_{script_dataset}_example.py"
            else:
                script = f"{script_dataset}_example.py"
        elif config.paradigm == "federated":
            if config.privacy != "none":
                script = f"dp_{script_dataset}_example.py"
            else:
                script = f"{script_dataset}_example.py"
        elif config.paradigm == "decentralized":
            if config.privacy != "none":
                script = f"dp_decentralized_{script_dataset}_example.py"
            else:
                script = f"decentralized_{script_dataset}_example.py"

        return os.path.join(base_path, script)

    def build_command(self, config: ExperimentConfig, experiment_id: str) -> List[str]:
        """Build the command to run an experiment via subprocess"""
        script_path = self.get_example_script_path(config)

        # Base command
        cmd = [
            sys.executable, script_path,
            "--num_actors", str(config.scale),
            "--partition_strategy", "dirichlet" if config.paradigm != "centralized" else "iid",
            "--alpha", str(config.alpha),
            "--min_partition_size", "50",
            "--rounds", str(config.rounds),
            "--epochs", str(config.local_epochs),
            "--batch_size", str(config.batch_size),
            "--lr", str(config.learning_rate),
            "--log_level", "INFO",
            "--actors_per_node", str(config.actors_per_node),
            "--placement_strategy", "spread",
        ]

        # Add topology and aggregation
        cmd.extend([
            "--topology", config.topology,
            "--aggregation_strategy", config.aggregation_strategy,
        ])

        # Add Ray cluster configuration
        if self.ray_address:
            cmd.extend(["--ray_address", self.ray_address])

        cmd.extend([
            "--ray_namespace", f"murmura_exp_{experiment_id}",
            "--auto_detect_cluster"
        ])

        # Privacy parameters
        if config.privacy != "none":
            cmd.extend([
                "--epsilon", str(config.epsilon),
                "--delta", str(config.delta),
                "--clipping_norm", "1.0",
                "--client_sampling_rate", "0.8" if config.paradigm != "decentralized" else "1.0"
            ])

        # Dataset-specific parameters
        if config.dataset == "ham10000":
            cmd.extend([
                "--dataset_name", "marmal88/skin_cancer",
                "--image_size", "128",
                "--widen_factor", "8",
                "--depth", "16",
                "--dropout", "0.3",
                "--weight_decay", "1e-4"
            ])

        # Paradigm-specific parameters
        if config.paradigm == "decentralized":
            cmd.extend(["--mixing_parameter", "0.5"])

        # Save path
        save_path = self.results_dir / f"{experiment_id}_model.pt"
        cmd.extend(["--save_path", str(save_path)])

        return cmd

    def parse_experiment_output(self, output: str, config: ExperimentConfig) -> Dict[str, Any]:
        """Parse experiment output to extract metrics (fallback method)"""
        metrics = {}

        lines = output.split('\n')
        for line in lines:
            line = line.strip()

            # Extract accuracy metrics
            if "Initial Test Accuracy:" in line:
                try:
                    accuracy_str = line.split(':')[1].strip().replace('%', '')
                    metrics['initial_accuracy'] = float(accuracy_str) / 100.0
                except Exception as e:
                    print(f"Failed to parse initial accuracy: {line}, error: {e}")

            elif "Final Test Accuracy:" in line:
                try:
                    accuracy_str = line.split(':')[1].strip().replace('%', '')
                    metrics['final_accuracy'] = float(accuracy_str) / 100.0
                except Exception as e:
                    print(f"Failed to parse final accuracy: {line}, error: {e}")

            elif "Accuracy Improvement:" in line:
                try:
                    improvement_str = line.split(':')[1].strip().replace('%', '')
                    metrics['accuracy_improvement'] = float(improvement_str) / 100.0
                except Exception as e:
                    print(f"Failed to parse accuracy improvement: {line}, error: {e}")

            # Extract other metrics as needed...

        return metrics

    def result_to_dict(self, result: ExperimentResult) -> Dict[str, Any]:
        """Convert ExperimentResult to dictionary for CSV/Excel with comprehensive metrics"""
        base_dict = {
            # Experiment identification
            'experiment_id': f"{result.config.paradigm}_{result.config.dataset}_{result.config.topology}_{result.config.heterogeneity}_{result.config.privacy}_{result.config.scale}",
            'paradigm': result.config.paradigm,
            'dataset': result.config.dataset,
            'topology': result.config.topology,
            'heterogeneity': result.config.heterogeneity,
            'privacy': result.config.privacy,
            'scale': result.config.scale,

            # Configuration parameters
            'aggregation_strategy': result.config.aggregation_strategy,
            'alpha': result.config.alpha,
            'epsilon': result.config.epsilon,
            'delta': result.config.delta,
            'rounds_configured': result.config.rounds,
            'epochs_configured': result.config.local_epochs,
            'batch_size_configured': result.config.batch_size,
            'learning_rate_configured': result.config.learning_rate,

            # Execution metadata
            'success': result.success,
            'start_time': result.start_time,
            'end_time': result.end_time,
            'duration_minutes': result.duration_minutes,

            # Basic performance metrics
            'initial_accuracy': result.initial_accuracy,
            'final_accuracy': result.final_accuracy,
            'accuracy_improvement': result.accuracy_improvement,
            'initial_loss': result.initial_loss,
            'final_loss': result.final_loss,
            'convergence_rounds': result.convergence_rounds,
            'training_time_minutes': result.training_time_minutes,

            # Standard efficiency metrics
            'total_communication_mb': result.total_communication_mb,
            'communication_per_round': result.communication_per_round,

            # Standard privacy metrics
            'privacy_epsilon_spent': result.privacy_epsilon_spent,
            'privacy_delta_spent': result.privacy_delta_spent,
            'privacy_budget_utilization': result.privacy_budget_utilization,

            # Error information
            'error_message': result.error_message,
            'error_type': result.error_type,
        }

        # Add all the enhanced metrics if they exist in the round_metrics
        if result.round_metrics and len(result.round_metrics) > 0:
            # Try to extract enhanced metrics from the first round_metrics entry
            first_metrics = result.round_metrics[0] if isinstance(result.round_metrics, list) else result.round_metrics

            # If round_metrics contains the enhanced data, extract it
            if isinstance(first_metrics, dict):
                enhanced_metrics = [
                    'loss_reduction', 'actual_rounds', 'time_per_round',
                    'accuracy_max', 'accuracy_min', 'accuracy_std', 'accuracy_trend',
                    'loss_max', 'loss_min', 'loss_std', 'loss_trend',
                    'topology_type', 'num_nodes', 'avg_active_nodes_per_round', 'node_participation_rate',
                    'total_parameter_transfers', 'parameter_transfers_per_round',
                    'total_local_training_events', 'total_aggregation_events', 'total_model_updates',
                    'final_parameter_convergence', 'avg_parameter_convergence', 'parameter_diversity', 'convergence_trend',
                    'training_stability', 'rounds_with_improvement', 'max_single_round_improvement'
                ]

                for metric in enhanced_metrics:
                    if metric in first_metrics:
                        base_dict[metric] = first_metrics[metric]
                    else:
                        # Set reasonable defaults for missing enhanced metrics
                        if 'accuracy' in metric or 'loss' in metric:
                            base_dict[metric] = 0.0
                        elif 'trend' in metric:
                            base_dict[metric] = 'unknown'
                        elif 'total' in metric or 'count' in metric:
                            base_dict[metric] = 0
                        elif 'rate' in metric or 'time' in metric or 'parameter' in metric:
                            base_dict[metric] = 0.0
                        else:
                            base_dict[metric] = 0.0
            else:
                # Set defaults for all enhanced metrics if not available
                enhanced_defaults = {
                    'loss_reduction': result.initial_loss - result.final_loss if result.initial_loss and result.final_loss else 0.0,
                    'actual_rounds': result.convergence_rounds,
                    'time_per_round': result.training_time_minutes / max(1, result.convergence_rounds) if result.training_time_minutes and result.convergence_rounds else 0.0,
                    'accuracy_max': result.final_accuracy,
                    'accuracy_min': result.initial_accuracy,
                    'accuracy_std': 0.0,
                    'accuracy_trend': 'improving' if result.accuracy_improvement > 0 else 'stable',
                    'loss_max': result.initial_loss,
                    'loss_min': result.final_loss,
                    'loss_std': 0.0,
                    'loss_trend': 'improving' if (result.initial_loss - result.final_loss) > 0 else 'stable',
                    'topology_type': result.config.topology,
                    'num_nodes': result.config.scale,
                    'avg_active_nodes_per_round': result.config.scale,
                    'node_participation_rate': 1.0,
                    'total_parameter_transfers': 0,
                    'parameter_transfers_per_round': 0.0,
                    'total_local_training_events': result.convergence_rounds,
                    'total_aggregation_events': result.convergence_rounds,
                    'total_model_updates': result.convergence_rounds,
                    'final_parameter_convergence': 0.0,
                    'avg_parameter_convergence': 0.0,
                    'parameter_diversity': 0.0,
                    'convergence_trend': 'stable',
                    'training_stability': 1.0,
                    'rounds_with_improvement': result.convergence_rounds if result.accuracy_improvement > 0 else 0,
                    'max_single_round_improvement': result.accuracy_improvement / max(1, result.convergence_rounds) if result.accuracy_improvement > 0 else 0.0,
                }
                base_dict.update(enhanced_defaults)
        else:
            # If no round_metrics at all, use basic defaults
            enhanced_defaults = {
                'loss_reduction': result.initial_loss - result.final_loss if result.initial_loss and result.final_loss else 0.0,
                'actual_rounds': result.convergence_rounds,
                'time_per_round': result.training_time_minutes / max(1, result.convergence_rounds) if result.training_time_minutes and result.convergence_rounds else 0.0,
                'accuracy_max': result.final_accuracy,
                'accuracy_min': result.initial_accuracy,
                'accuracy_std': 0.0,
                'accuracy_trend': 'improving' if result.accuracy_improvement > 0 else 'stable',
                'loss_max': result.initial_loss,
                'loss_min': result.final_loss,
                'loss_std': 0.0,
                'loss_trend': 'improving' if (result.initial_loss - result.final_loss) > 0 else 'stable',
                'topology_type': result.config.topology,
                'num_nodes': result.config.scale,
                'avg_active_nodes_per_round': result.config.scale,
                'node_participation_rate': 1.0,
                'total_parameter_transfers': 0,
                'parameter_transfers_per_round': 0.0,
                'total_local_training_events': result.convergence_rounds,
                'total_aggregation_events': result.convergence_rounds,
                'total_model_updates': result.convergence_rounds,
                'final_parameter_convergence': 0.0,
                'avg_parameter_convergence': 0.0,
                'parameter_diversity': 0.0,
                'convergence_trend': 'stable',
                'training_stability': 1.0,
                'rounds_with_improvement': result.convergence_rounds if result.accuracy_improvement > 0 else 0,
                'max_single_round_improvement': result.accuracy_improvement / max(1, result.convergence_rounds) if result.accuracy_improvement > 0 else 0.0,
            }
            base_dict.update(enhanced_defaults)

        return base_dict

    def append_result_to_file(self, result: ExperimentResult, output_file: str):
        """Append a single result to the output file immediately"""
        row_data = self.result_to_dict(result)

        try:
            file_exists = os.path.exists(output_file)

            if output_file.endswith('.xlsx'):
                if file_exists:
                    try:
                        df_existing = pd.read_excel(output_file)
                        df_new = pd.concat([df_existing, pd.DataFrame([row_data])], ignore_index=True)
                    except Exception as e:
                        self.logger.warning(f"Error reading existing Excel file: {e}. Creating new file.")
                        df_new = pd.DataFrame([row_data])
                else:
                    df_new = pd.DataFrame([row_data])

                df_new.to_excel(output_file, index=False)

            else:  # CSV format
                df_row = pd.DataFrame([row_data])

                if file_exists:
                    df_row.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    df_row.to_csv(output_file, mode='w', header=True, index=False)

            self.logger.info(f"Result appended to {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to append result to {output_file}: {e}")

    def run_experiments(self, experiment_set: str, output_file: str, resume: bool = False):
        """Run all experiments in a set"""
        configs = self.generate_experiment_configs(experiment_set)

        if resume and os.path.exists(output_file):
            try:
                if output_file.endswith('.xlsx'):
                    existing_df = pd.read_excel(output_file)
                else:
                    existing_df = pd.read_csv(output_file)

                completed_experiment_ids = set(existing_df['experiment_id'].tolist())
                configs = [c for c in configs if self.get_experiment_id(c) not in completed_experiment_ids]

                self.logger.info(f"Resuming: {len(completed_experiment_ids)} experiments already completed")

            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")

        self.logger.info(f"Running {len(configs)} experiments")

        # Group configs by execution priority
        priority_configs = []
        regular_configs = []

        for config in configs:
            if (config.dataset == "mnist" and
                    config.scale <= 10 and
                    config.privacy == "none" and
                    config.heterogeneity == "iid"):
                priority_configs.append(config)
            else:
                regular_configs.append(config)

        all_configs = priority_configs + regular_configs
        results = []

        # Run experiments
        for i, config in enumerate(all_configs):
            experiment_id = self.get_experiment_id(config)

            self.logger.info(f"Progress: {i+1}/{len(all_configs)} - Running {experiment_id}")

            result = self.run_single_experiment(config, experiment_id)
            results.append(result)

            # Append result immediately to file
            self.append_result_to_file(result, output_file)

            # Fail fast: if first 3 experiments fail, stop
            if i < 3 and not result.success:
                self.logger.error(f"Early failure detected. Stopping experiment suite.")
                self.logger.error(f"Fix the issue with {experiment_id} before continuing.")
                break

        # Final summary
        self.print_final_summary(results)
        return results

    def print_final_summary(self, results: List[ExperimentResult]):
        """Print final experiment summary"""
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results if r.success)
        failed_experiments = total_experiments - successful_experiments

        self.logger.info(f"Final Experiment Summary:")
        self.logger.info(f"  Total experiments: {total_experiments}")
        self.logger.info(f"  Successful: {successful_experiments}")
        self.logger.info(f"  Failed: {failed_experiments}")
        self.logger.info(f"  Success rate: {successful_experiments/total_experiments*100:.1f}%")

        if failed_experiments > 0:
            self.logger.info("Failed experiments:")
            for result in results:
                if not result.success:
                    exp_id = f"{result.config.paradigm}_{result.config.dataset}_{result.config.topology}_{result.config.privacy}_{result.config.scale}"
                    self.logger.info(f"  {exp_id}: {result.error_type} - {result.error_message}")

    def get_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID"""
        return f"{config.paradigm}_{config.dataset}_{config.topology}_{config.heterogeneity}_{config.privacy}_{config.scale}"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Murmura Experiment Runner with Observer Pattern")

    parser.add_argument("--experiment_set",
                        choices=["core", "topology_privacy", "scalability", "all"],
                        default="core",
                        help="Which experiment set to run")

    parser.add_argument("--base_dir",
                        type=str,
                        default="/home/ubuntu/murmura",
                        help="Base directory containing Murmura framework")

    parser.add_argument("--output_file",
                        type=str,
                        default="experiment_results.csv",
                        help="Output file for results (CSV or Excel)")

    parser.add_argument("--ray_address",
                        type=str,
                        default=None,
                        help="Ray cluster head node address")

    parser.add_argument("--max_parallel",
                        type=int,
                        default=1,
                        help="Maximum parallel experiments")

    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume from existing results file")

    parser.add_argument("--use_observer",
                        action="store_true",
                        default=True,
                        help="Use observer pattern for metrics collection (default: True)")

    args = parser.parse_args()

    # Create runner
    runner = EnhancedExperimentRunner(
        base_dir=args.base_dir,
        ray_address=args.ray_address,
        max_parallel=args.max_parallel
    )

    # Determine which experiment sets to run
    if args.experiment_set == "all":
        experiment_sets = ["core", "topology_privacy", "scalability"]
    else:
        experiment_sets = [args.experiment_set]

    # Run experiments
    all_results = []
    for exp_set in experiment_sets:
        output_file = args.output_file.replace('.', f'_{exp_set}.')
        runner.logger.info(f"Starting {exp_set} experiment set")

        results = runner.run_experiments(
            experiment_set=exp_set,
            output_file=output_file,
            resume=args.resume
        )
        all_results.extend(results)

    runner.logger.info("Experiment suite completed!")


if __name__ == "__main__":
    main()
