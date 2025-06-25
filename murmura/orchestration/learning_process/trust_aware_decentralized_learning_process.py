"""
Trust-aware decentralized learning process with HSIC-based trust monitoring.

This module extends the decentralized learning process to include trust
monitoring and trust-aware aggregation.
"""

from statistics import mean
from typing import Dict, Any, List, Optional
import logging

import numpy as np
import ray

from murmura.orchestration.learning_process.decentralized_learning_process import (
    DecentralizedLearningProcess
)
from murmura.trust.trust_monitor import TrustMonitor
from murmura.trust.trust_config import TrustMonitoringConfig
from murmura.aggregation.strategies.trust_aware_gossip import TrustAwareGossipAvg
from murmura.visualization.training_event import (
    EvaluationEvent,
    ModelUpdateEvent,
    AggregationEvent,
    ParameterTransferEvent,
    LocalTrainingEvent,
)


class TrustAwareDecentralizedLearningProcess(DecentralizedLearningProcess):
    """
    Extension of decentralized learning process with trust monitoring.
    
    This class adds HSIC-based trust monitoring to detect and mitigate
    malicious behavior in decentralized federated learning.
    """
    
    def __init__(
        self,
        config,
        dataset,
        model,
    ):
        """
        Initialize trust-aware decentralized learning process.
        
        Args:
            config: Learning process configuration
            dataset: Dataset instance
            model: Model instance
        """
        super().__init__(config, dataset, model)
        
        # Trust monitoring components
        self.trust_config: Optional[TrustMonitoringConfig] = None
        self.trust_monitors: Dict[str, TrustMonitor] = {}
        self.trust_enabled = False
        
        # Logger
        self.trust_logger = logging.getLogger(
            "murmura.trust.TrustAwareDecentralizedLearningProcess"
        )
        
        # Trust will be initialized in execute() when cluster_manager is available
        self._trust_initialized = False
    
    def _initialize_trust_config(self) -> None:
        """Initialize trust monitoring configuration."""
        # Initialize trust monitoring if enabled
        if hasattr(self.config, "trust_monitoring") and self.config.trust_monitoring:
            if isinstance(self.config.trust_monitoring, dict):
                self.trust_config = TrustMonitoringConfig.from_dict(
                    self.config.trust_monitoring
                )
            elif isinstance(self.config.trust_monitoring, TrustMonitoringConfig):
                self.trust_config = self.config.trust_monitoring
            else:
                self.trust_logger.warning(
                    "Invalid trust_monitoring config type, using default"
                )
                self.trust_config = TrustMonitoringConfig()
            
            self.trust_enabled = self.trust_config.enabled
        else:
            self.trust_config = TrustMonitoringConfig()
            self.trust_enabled = False
        
        if self.trust_enabled:
            self._initialize_trust_monitors()
            self._setup_trust_aware_aggregation()
            self.trust_logger.info("Trust monitoring enabled")
        else:
            self.trust_logger.info("Trust monitoring disabled")
    
    def _initialize_trust_monitors(self) -> None:
        """Initialize trust monitors for each actor."""
        if not self.cluster_manager or not self.cluster_manager.actors:
            self.trust_logger.warning("No actors available for trust monitoring")
            return
        
        self.trust_logger.info(
            f"Initializing trust monitors for {len(self.cluster_manager.actors)} actors"
        )
        
        # Create trust monitor for each actor
        for i, actor in enumerate(self.cluster_manager.actors):
            node_id = f"node_{i}"
            
            # Create trust monitor actor
            trust_monitor = TrustMonitor.remote(
                node_id=node_id,
                hsic_config=self.trust_config.hsic_config.to_dict(),
                trust_config=self.trust_config.trust_policy_config.to_dict(),
            )
            
            self.trust_monitors[node_id] = trust_monitor
        
        self.trust_logger.info(f"Created {len(self.trust_monitors)} trust monitors")
    
    def _setup_trust_aware_aggregation(self) -> None:
        """Setup trust-aware aggregation strategy."""
        # Get current strategy from cluster manager
        current_strategy = None
        
        # Try different ways to access the aggregation strategy
        if hasattr(self.cluster_manager, 'topology_coordinator'):
            topology_coordinator = self.cluster_manager.topology_coordinator
            if hasattr(topology_coordinator, 'aggregation_strategy'):
                current_strategy = topology_coordinator.aggregation_strategy
            elif hasattr(topology_coordinator, '_aggregation_strategy'):
                current_strategy = topology_coordinator._aggregation_strategy
        
        if current_strategy is None:
            current_strategy = getattr(self.cluster_manager, 'aggregation_strategy', None)
        
        self.trust_logger.info(f"Current aggregation strategy: {current_strategy.__class__.__name__ if current_strategy else 'None'}")
        
        # For ring topology, we need to ensure we're using a gossip-based strategy
        if current_strategy and hasattr(current_strategy, "__class__"):
            strategy_name = current_strategy.__class__.__name__.lower()
            
            if "gossip" in strategy_name or "decentralized" in strategy_name:
                # Replace with trust-aware version
                self.trust_logger.info(
                    f"Replacing {current_strategy.__class__.__name__} "
                    "with TrustAwareGossipAvg"
                )
                
                # Get mixing parameter from current strategy if available
                mixing_param = getattr(current_strategy, "mixing_parameter", 0.5)
                
                # Create trust-aware strategy
                trust_aware_strategy = TrustAwareGossipAvg(
                    mixing_parameter=mixing_param,
                    trust_monitors=self.trust_monitors,
                    use_trust_weights=True,
                )
                
                # Replace strategy in topology coordinator
                if hasattr(self.cluster_manager, 'topology_coordinator'):
                    self.cluster_manager.topology_coordinator.aggregation_strategy = trust_aware_strategy
                else:
                    self.cluster_manager.aggregation_strategy = trust_aware_strategy
                
                self.trust_logger.info("Successfully set up trust-aware aggregation")
            else:
                self.trust_logger.warning(
                    f"Current strategy {current_strategy.__class__.__name__} "
                    "is not gossip-based. Cannot apply trust-aware aggregation."
                )
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the trust-aware decentralized learning process.
        
        Returns:
            Results of the learning process with trust metrics
        """
        if not self.cluster_manager:
            raise ValueError("Learning process not initialized. Call initialize first.")
        
        # Initialize trust monitoring if not done yet
        if not self._trust_initialized:
            self._initialize_trust_config()
            self._trust_initialized = True
        
        # Get configuration parameters
        rounds = self.config.rounds
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        test_split = self.config.test_split
        monitor_resources = self.config.monitor_resources
        health_check_interval = self.config.health_check_interval
        
        # Enhanced logging
        self.log_training_progress(
            0, {
                "status": "starting_trust_aware_decentralized",
                "rounds": rounds,
                "trust_enabled": self.trust_enabled,
            }
        )
        
        # Prepare test data
        test_dataset = self.dataset.get_split(test_split)
        feature_columns = self.config.feature_columns
        label_column = self.config.label_column
        
        self.logger.info("Preparing test data for evaluation...")
        test_features, test_labels = self._prepare_test_data(
            test_dataset, feature_columns, label_column
        )
        
        # Evaluate initial model
        initial_metrics = self.model.evaluate(test_features, test_labels)
        self.logger.info(
            f"Initial Test Accuracy: {initial_metrics['accuracy'] * 100:.2f}%"
        )
        
        # Emit evaluation event
        self.training_monitor.emit_event(
            EvaluationEvent(round_num=0, metrics=initial_metrics)
        )
        
        # Get topology information
        topology_info = self.cluster_manager.get_topology_information()
        topology_type = topology_info.get("type", "unknown")
        adjacency_list = topology_info.get("adjacency_list", {})
        
        round_metrics = []
        trust_metrics = []
        
        # Training rounds
        for round_num in range(1, rounds + 1):
            self.logger.info(f"--- Round {round_num}/{rounds} ---")
            
            # Monitor resources if enabled
            if monitor_resources:
                resource_usage = self.monitor_resource_usage()
                self.logger.debug(
                    f"Round {round_num} resource usage: "
                    f"{resource_usage.get('resource_utilization', {})}"
                )
            
            # Health checks
            if round_num % health_check_interval == 0:
                health_status = self.get_actor_health_status()
                if "error" not in health_status:
                    self.logger.info(
                        f"Round {round_num} health check: "
                        f"{health_status['healthy']}/{health_status['sampled_actors']} healthy"
                    )
            
            # 1. Local Training
            self.logger.info(f"Training on clients for {epochs} epochs...")
            
            # Emit local training event
            self.training_monitor.emit_event(
                LocalTrainingEvent(
                    round_num=round_num,
                    active_nodes=list(range(len(self.cluster_manager.actors))),
                    metrics={},
                    total_epochs=epochs,
                )
            )
            
            # Training
            train_metrics = self.cluster_manager.train_models(
                client_sampling_rate=self.config.client_sampling_rate,
                data_sampling_rate=self.config.data_sampling_rate,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True,
            )
            
            # Calculate average training metrics
            avg_train_loss = mean([m["loss"] for m in train_metrics])
            avg_train_acc = mean([m["accuracy"] for m in train_metrics])
            
            self.log_training_progress(
                round_num,
                {
                    "avg_train_loss": avg_train_loss,
                    "avg_train_accuracy": avg_train_acc,
                    "active_clients": len(train_metrics),
                    "topology": topology_type,
                    "trust_enabled": self.trust_enabled,
                },
            )
            
            # 2. Trust-Aware Parameter Exchange
            if self.trust_enabled:
                self._perform_trust_aware_aggregation(
                    round_num, adjacency_list, test_features, test_labels
                )
            else:
                # Standard aggregation (parent class method)
                self._perform_standard_aggregation(
                    round_num, adjacency_list, test_features, test_labels
                )
            
            # 3. Collect trust metrics if enabled
            if self.trust_enabled and round_num % self.trust_config.trust_report_interval == 0:
                round_trust_metrics = self._collect_trust_metrics()
                trust_metrics.append({
                    "round": round_num,
                    **round_trust_metrics
                })
                
                # Log trust summary
                self._log_trust_summary(round_trust_metrics)
            
            # 4. Evaluation
            test_metrics = self.model.evaluate(test_features, test_labels)
            self.logger.info(f"Global Model Test Loss: {test_metrics['loss']:.4f}")
            self.logger.info(
                f"Global Model Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%"
            )
            
            # Emit evaluation event
            self.training_monitor.emit_event(
                EvaluationEvent(round_num=round_num, metrics=test_metrics)
            )
            
            # Store metrics
            round_metrics.append({
                "round": round_num,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            })
        
        # Final evaluation
        final_metrics = self.model.evaluate(test_features, test_labels)
        improvement = final_metrics["accuracy"] - initial_metrics["accuracy"]
        
        # Log final results
        self.logger.info("=== Final Model Evaluation ===")
        self.logger.info(f"Final Test Accuracy: {final_metrics['accuracy'] * 100:.2f}%")
        self.logger.info(f"Accuracy Improvement: {improvement * 100:.2f}%")
        
        if self.trust_enabled:
            final_trust_metrics = self._collect_trust_metrics()
            self._log_trust_summary(final_trust_metrics)
        
        # Return results
        results = {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "accuracy_improvement": improvement,
            "round_metrics": round_metrics,
            "topology": topology_info,
            "trust_enabled": self.trust_enabled,
        }
        
        if self.trust_enabled:
            results["trust_metrics"] = trust_metrics
            results["final_trust_report"] = final_trust_metrics
        
        return results
    
    def _perform_trust_aware_aggregation(
        self,
        round_num: int,
        adjacency_list: Dict[int, List[int]],
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> None:
        """
        Perform trust-aware parameter aggregation.
        
        Args:
            round_num: Current round number
            adjacency_list: Topology adjacency list
            test_features: Test features for evaluation
            test_labels: Test labels for evaluation
        """
        topology_type = "decentralized"  # Default type
        self.logger.info(
            f"Performing trust-aware decentralized aggregation using {topology_type} topology..."
        )
        
        # First, update trust monitors with current parameters
        for i, actor in enumerate(self.cluster_manager.actors):
            node_id = f"node_{i}"
            if node_id in self.trust_monitors:
                current_params = ray.get(actor.get_model_parameters.remote())
                ray.get(
                    self.trust_monitors[node_id].set_current_parameters.remote(
                        current_params
                    )
                )
        
        # Collect parameters for visualization
        node_params = {}
        for i, actor in enumerate(self.cluster_manager.actors):
            params = ray.get(actor.get_model_parameters.remote(), timeout=1800)
            node_params[i] = params
        
        # Create parameter summaries
        param_summaries = self._create_parameter_summaries(node_params)
        
        # Process trust assessments for each edge in topology
        trust_assessments = {}
        for node, neighbors in adjacency_list.items():
            node_id = f"node_{node}"
            
            if node_id in self.trust_monitors and neighbors:
                # Assess each neighbor
                for neighbor in neighbors:
                    neighbor_id = f"node_{neighbor}"
                    neighbor_params = node_params[neighbor]
                    
                    # Get trust assessment
                    action, trust_score, stats = ray.get(
                        self.trust_monitors[node_id].assess_update.remote(
                            neighbor_id, neighbor_params
                        )
                    )
                    
                    trust_assessments[f"{node_id}->{neighbor_id}"] = {
                        "action": action.value,
                        "trust_score": trust_score,
                        "stats": stats,
                    }
                
                # Emit parameter transfer event with trust info
                self.training_monitor.emit_event(
                    ParameterTransferEvent(
                        round_num=round_num,
                        source_nodes=[node],
                        target_nodes=neighbors,
                        param_summary={node: param_summaries[node]}
                        if node in param_summaries else {},
                    )
                )
        
        # Emit aggregation events
        for node in range(len(self.cluster_manager.actors)):
            neighbors = adjacency_list.get(node, [])
            if neighbors:
                self.training_monitor.emit_event(
                    AggregationEvent(
                        round_num=round_num,
                        participating_nodes=[node] + neighbors,
                        aggregator_node=node,
                        strategy_name="TrustAwareGossipAvg",
                    )
                )
        
        # Get data sizes for weighted aggregation
        split = self.config.split
        partitions = list(self.dataset.get_partitions(split).values())
        client_data_sizes = [len(partition) for partition in partitions]
        weights = [float(size) for size in client_data_sizes]
        
        # Perform trust-aware aggregation
        aggregated_params = self.cluster_manager.aggregate_model_parameters(
            weights=weights
        )
        
        # Update global model
        self.model.set_parameters(aggregated_params)
        
        # Distribute to clients
        self.cluster_manager.update_models(aggregated_params)
        
        # Calculate convergence
        param_convergence = self._calculate_parameter_convergence(
            node_params, aggregated_params
        )
        
        # Emit model update event
        self.training_monitor.emit_event(
            ModelUpdateEvent(
                round_num=round_num,
                updated_nodes=list(range(len(self.cluster_manager.actors))),
                param_convergence=param_convergence,
            )
        )
    
    def _perform_standard_aggregation(
        self,
        round_num: int,
        adjacency_list: Dict[int, List[int]],
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> None:
        """
        Perform standard aggregation (fallback to parent implementation).
        """
        # This is essentially the same as parent class but extracted for clarity
        self.logger.info("Performing standard decentralized aggregation...")
        
        # Collect parameters
        node_params = {}
        for i, actor in enumerate(self.cluster_manager.actors):
            params = ray.get(actor.get_model_parameters.remote(), timeout=1800)
            node_params[i] = params
        
        # Create summaries
        param_summaries = self._create_parameter_summaries(node_params)
        
        # Emit events
        for node, neighbors in adjacency_list.items():
            if neighbors:
                self.training_monitor.emit_event(
                    ParameterTransferEvent(
                        round_num=round_num,
                        source_nodes=[node],
                        target_nodes=neighbors,
                        param_summary={node: param_summaries[node]}
                        if node in param_summaries else {},
                    )
                )
        
        # Get weights
        split = self.config.split
        partitions = list(self.dataset.get_partitions(split).values())
        client_data_sizes = [len(partition) for partition in partitions]
        weights = [float(size) for size in client_data_sizes]
        
        # Aggregate
        aggregated_params = self.cluster_manager.aggregate_model_parameters(
            weights=weights
        )
        
        # Update models
        self.model.set_parameters(aggregated_params)
        self.cluster_manager.update_models(aggregated_params)
        
        # Calculate convergence
        param_convergence = self._calculate_parameter_convergence(
            node_params, aggregated_params
        )
        
        # Emit update event
        self.training_monitor.emit_event(
            ModelUpdateEvent(
                round_num=round_num,
                updated_nodes=list(range(len(self.cluster_manager.actors))),
                param_convergence=param_convergence,
            )
        )
    
    def _collect_trust_metrics(self) -> Dict[str, Any]:
        """
        Collect trust metrics from all trust monitors.
        
        Returns:
            Aggregated trust metrics
        """
        metrics = {
            "total_monitors": len(self.trust_monitors),
            "node_reports": {},
            "global_stats": {
                "total_excluded": 0,
                "total_downgraded": 0,
                "total_warned": 0,
                "avg_trust_score": 0.0,
            },
        }
        
        all_trust_scores = []
        
        for node_id, monitor in self.trust_monitors.items():
            try:
                report = ray.get(monitor.get_trust_report.remote())
                metrics["node_reports"][node_id] = report
                
                # Aggregate statistics
                for neighbor_id, neighbor_info in report["neighbors"].items():
                    trust_score = neighbor_info["trust_score"]
                    all_trust_scores.append(trust_score)
                    
                    # Count actions based on trust level
                    if neighbor_info["trust_level"] == "untrusted":
                        metrics["global_stats"]["total_excluded"] += 1
                    elif neighbor_info["trust_level"] == "suspicious":
                        metrics["global_stats"]["total_downgraded"] += 1
                    elif neighbor_info["drift_rate"] > 0.1:  # High drift rate
                        metrics["global_stats"]["total_warned"] += 1
                        
            except Exception as e:
                self.trust_logger.error(f"Failed to get report from {node_id}: {e}")
        
        # Calculate average trust score
        if all_trust_scores:
            metrics["global_stats"]["avg_trust_score"] = np.mean(all_trust_scores)
        
        return metrics
    
    def _log_trust_summary(self, trust_metrics: Dict[str, Any]) -> None:
        """
        Log a summary of trust metrics.
        
        Args:
            trust_metrics: Trust metrics to summarize
        """
        stats = trust_metrics["global_stats"]
        
        self.trust_logger.info("=== Trust Monitoring Summary ===")
        self.trust_logger.info(
            f"Average Trust Score: {stats['avg_trust_score']:.3f}"
        )
        self.trust_logger.info(
            f"Excluded Nodes: {stats['total_excluded']}"
        )
        self.trust_logger.info(
            f"Downgraded Nodes: {stats['total_downgraded']}"
        )
        self.trust_logger.info(
            f"Warned Nodes: {stats['total_warned']}"
        )
        
        # Log per-node summary if not too many nodes
        if len(self.trust_monitors) <= 20:
            for node_id, report in trust_metrics["node_reports"].items():
                if report["neighbors"]:
                    neighbor_summary = []
                    for n_id, n_info in report["neighbors"].items():
                        neighbor_summary.append(
                            f"{n_id}:{n_info['trust_level']}({n_info['trust_score']:.2f})"
                        )
                    self.trust_logger.debug(
                        f"{node_id} neighbors: {', '.join(neighbor_summary)}"
                    )