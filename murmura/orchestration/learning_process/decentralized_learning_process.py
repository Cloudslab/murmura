from statistics import mean
from typing import Dict, Any, Optional, List
import logging

import numpy as np
import ray

from murmura.orchestration.learning_process.learning_process import LearningProcess
from murmura.visualization.training_event import (
    EvaluationEvent,
    ModelUpdateEvent,
    AggregationEvent,
    ParameterTransferEvent,
    LocalTrainingEvent,
    FingerprintEvent,
)
from murmura.trust_monitoring import TrustMonitor, TrustMonitorConfig


class DecentralizedLearningProcess(LearningProcess):
    """
    Implementation of a decentralized learning process with generic data handling and trust monitoring.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize trust monitors for each node
        self.trust_monitors: Dict[int, TrustMonitor] = {}
        self.trust_config: Optional[TrustMonitorConfig] = None

    def _prepare_test_data(self, test_dataset, feature_columns, label_column):
        """
        Prepare test data with proper preprocessing - same as FederatedLearningProcess.
        """
        # Extract raw data
        if len(feature_columns) == 1:
            feature_data = test_dataset[feature_columns[0]]
        else:
            feature_data = [test_dataset[col] for col in feature_columns]

        label_data = test_dataset[label_column]

        # Let the model wrapper handle preprocessing
        if len(feature_columns) == 1:
            if hasattr(self.model, "data_preprocessor"):
                if self.model.data_preprocessor is not None:
                    try:
                        data_list = (
                            list(feature_data)
                            if not isinstance(feature_data, list)
                            else feature_data
                        )
                        features = self.model.data_preprocessor.preprocess_features(
                            data_list
                        )
                    except Exception as e:
                        self.logger.warning(f"Preprocessor failed, using fallback: {e}")
                        features = np.array(feature_data, dtype=np.float32)
                else:
                    features = np.array(feature_data, dtype=np.float32)
            else:
                features = np.array(feature_data, dtype=np.float32)
        else:
            processed_columns = []
            for col_data in feature_data:
                if hasattr(self.model, "data_preprocessor"):
                    if self.model.data_preprocessor is not None:
                        try:
                            col_features = (
                                self.model.data_preprocessor.preprocess_features(
                                    list(col_data)
                                )
                            )
                            processed_columns.append(col_features)
                        except Exception:
                            processed_columns.append(
                                np.array(col_data, dtype=np.float32)
                            )
                    else:
                        processed_columns.append(np.array(col_data, dtype=np.float32))
                else:
                    processed_columns.append(np.array(col_data, dtype=np.float32))

            features = np.column_stack(processed_columns)

        labels = np.array(label_data, dtype=np.int64)
        self.logger.info(
            f"Prepared test data - Features shape: {features.shape}, Labels shape: {labels.shape}"
        )

        return features, labels

    def _initialize_trust_monitoring(self) -> None:
        """Initialize trust monitors for honest nodes based on topology."""
        if not hasattr(self.config, 'trust_monitoring') or not self.config.trust_monitoring:
            self.logger.info("Trust monitoring not enabled")
            return
            
        self.trust_config = self.config.trust_monitoring
        
        if not self.trust_config.enable_trust_monitoring:
            self.logger.info("Trust monitoring disabled in configuration")
            return
            
        # Get topology information
        topology_info = self.cluster_manager.get_topology_information()
        adjacency_list = topology_info.get("adjacency_list", {})
        
        # Initialize trust monitors only for honest nodes
        for node_idx in range(len(self.cluster_manager.actors)):
            # Skip malicious nodes - they don't need trust monitors
            if node_idx in self.cluster_manager.malicious_client_indices:
                continue
                
            # Create trust monitor for this honest node
            node_id = f"node_{node_idx}"
            # Use model_seed + node_idx for trust monitor seed to ensure deterministic behavior
            trust_seed = self.config.model_seed + node_idx if self.config.model_seed is not None else None
            trust_monitor = TrustMonitor(node_id, self.trust_config, seed=trust_seed)
            
            # Register event callback to forward trust events to training monitor
            trust_monitor.add_event_callback(lambda event: self.training_monitor.emit_event(event))
            
            self.trust_monitors[node_idx] = trust_monitor
            
            neighbors = adjacency_list.get(node_idx, [])
            self.logger.info(f"Initialized trust monitor for honest node {node_idx} with {len(neighbors)} neighbors")
        
        self.logger.info(f"Trust monitoring initialized for {len(self.trust_monitors)} honest nodes")


    def _process_trust_monitoring(self, round_num: int, node_params: Dict[int, Dict[str, Any]], 
                                 train_metrics: List[Dict[str, float]], 
                                 node_malicious_detections: Dict[int, set]) -> Dict[int, Dict[str, float]]:
        """Process trust monitoring for all honest nodes."""
        if not self.trust_monitors:
            return {}
            
        # Get topology information
        topology_info = self.cluster_manager.get_topology_information()
        adjacency_list = topology_info.get("adjacency_list", {})
        
        # Collect trust scores for each honest node
        all_trust_scores = {}
        
        for node_idx, trust_monitor in self.trust_monitors.items():
            neighbors = adjacency_list.get(node_idx, [])
            
            # Collect neighbor parameters and losses
            neighbor_updates = {}
            neighbor_losses = {}
            
            for neighbor_idx in neighbors:
                if neighbor_idx in node_params:
                    neighbor_updates[f"node_{neighbor_idx}"] = node_params[neighbor_idx]
                    
                    # Find corresponding training metrics
                    if neighbor_idx < len(train_metrics):
                        neighbor_losses[f"node_{neighbor_idx}"] = train_metrics[neighbor_idx].get("loss", 0.0)
                        
                    # Log neighbor characteristics for debugging
                    is_malicious = neighbor_idx in self.cluster_manager.malicious_client_indices
                    self.logger.debug(f"Node {node_idx}: Neighbor {neighbor_idx} is {'MALICIOUS' if is_malicious else 'honest'}")
            
            if neighbor_updates:
                self.logger.info(f"Node {node_idx}: Processing {len(neighbor_updates)} neighbor updates: {list(neighbor_updates.keys())}")
                
                # Enable debug logging for this specific trust monitor
                trust_monitor.logger.setLevel(logging.DEBUG)
                
                # Process trust monitoring for this node
                with trust_monitor._measure_trust_resource_usage("process_parameter_updates"):
                    # Pass own parameters for self-referenced baseline detection
                    own_params = node_params.get(node_idx)
                    trust_scores = trust_monitor.process_parameter_updates(
                        round_num=round_num,
                        neighbor_updates=neighbor_updates,
                        neighbor_losses=neighbor_losses,
                        own_parameters=own_params  # NEW: Enable self-referenced detection
                    )
                all_trust_scores[node_idx] = trust_scores
                self.logger.info(f"Node {node_idx}: Trust scores: {trust_scores}")
                
                # Collect fingerprint data for honest nodes  
                if hasattr(trust_monitor, 'compute_gradient_fingerprint') and own_params:
                    try:
                        fingerprint_data = trust_monitor.compute_gradient_fingerprint(own_params)
                        fingerprint_event = FingerprintEvent(
                            round_num=round_num,
                            node_id=f"node_{node_idx}",
                            fingerprint_data=fingerprint_data
                        )
                        self.training_monitor.emit_event(fingerprint_event)
                        self.logger.debug(f"Node {node_idx}: Emitted fingerprint event with data: {fingerprint_data}")
                    except Exception as e:
                        self.logger.warning(f"Node {node_idx}: Failed to collect fingerprint data: {e}")
                
                # Get the actual detections from the trust monitor itself
                with trust_monitor._measure_trust_resource_usage("get_trust_summary"):
                    trust_summary = trust_monitor.get_trust_summary()
                suspicious_neighbors = trust_summary.get("low_trust_neighbors", [])
                
                if suspicious_neighbors:
                    self.logger.info(f"Node {node_idx} has low-trust neighbors: {suspicious_neighbors}")
                    # Track nodes with reduced trust for performance analysis
                    if node_idx not in node_malicious_detections:
                        node_malicious_detections[node_idx] = set()
                    node_malicious_detections[node_idx].update(suspicious_neighbors)
            else:
                self.logger.debug(f"No neighbor updates available for node {node_idx}")
        
        return all_trust_scores

    def execute(self) -> Dict[str, Any]:
        """
        Execute the decentralized learning process with generic data handling.

        :return: Results of the decentralized learning process
        """
        if not self.cluster_manager:
            raise ValueError("Learning process not initialized. Call initialize first.")

        # Get configuration parameters
        rounds = self.config.rounds
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        test_split = self.config.test_split
        monitor_resources = self.config.monitor_resources
        health_check_interval = self.config.health_check_interval

        # Enhanced logging with cluster context
        self.log_training_progress(
            0, {"status": "starting_decentralized", "rounds": rounds}
        )

        # Prepare test data for evaluation
        test_dataset = self.dataset.get_split(test_split)

        # Get feature and label columns from config - these should be set during initialization
        feature_columns = self.config.feature_columns
        label_column = self.config.label_column

        # These should never be None if config validation worked properly
        if feature_columns is None or label_column is None:
            raise ValueError(
                f"feature_columns ({feature_columns}) and label_column ({label_column}) "
                "must be specified in the configuration"
            )

        self.logger.info(
            f"Using feature columns: {feature_columns}, label column: {label_column}"
        )

        self.logger.info("Preparing test data for evaluation...")
        test_features, test_labels = self._prepare_test_data(
            test_dataset, feature_columns, label_column
        )

        # Evaluate initial model
        initial_metrics = self.model.evaluate(test_features, test_labels)
        self.logger.info(
            f"Initial Test Accuracy: {initial_metrics['accuracy'] * 100:.2f}%"
        )

        # Emit evaluation event for visualization
        self.training_monitor.emit_event(
            EvaluationEvent(round_num=0, metrics=initial_metrics)
        )

        # Get topology information
        topology_info = self.cluster_manager.get_topology_information()
        topology_type = topology_info.get("type", "unknown")
        adjacency_list = topology_info.get("adjacency_list", {})

        # Initialize trust monitoring for decentralized learning
        self._initialize_trust_monitoring()

        round_metrics = []
        trust_monitoring_results = []
        # Track actual malicious detections per node
        node_malicious_detections = {}

        # Initialize cumulative privacy tracking
        cumulative_privacy_spent = {"epsilon": 0.0, "delta": 0.0}
        per_round_privacy_metrics = []

        # Training rounds
        for round_num in range(1, rounds + 1):
            self.logger.info(f"--- Round {round_num}/{rounds} ---")

            # Monitor resource usage if enabled
            if monitor_resources:
                resource_usage = self.monitor_resource_usage()
                self.logger.debug(
                    f"Round {round_num} resource usage: {resource_usage.get('resource_utilization', {})}"
                )

            # UPDATED: Periodic health checks
            if round_num % health_check_interval == 0:
                health_status = self.get_actor_health_status()
                if "error" not in health_status:
                    self.logger.info(
                        f"Round {round_num} health check: {health_status['healthy']}/{health_status['sampled_actors']} healthy"
                    )
                    if health_status.get("degraded", 0) > 0:
                        self.logger.warning(
                            f"Degraded actors: {health_status['degraded']}"
                        )
                    if health_status.get("error", 0) > 0:
                        self.logger.error(f"Error actors: {health_status['error']}")

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

            # Training with subsampling support
            train_metrics = self.cluster_manager.train_models(
                current_round=round_num,
                total_rounds=self.config.rounds,
                client_sampling_rate=self.config.client_sampling_rate,
                data_sampling_rate=self.config.data_sampling_rate,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True,
            )

            # Calculate average training metrics
            avg_train_loss = mean([m["loss"] for m in train_metrics])
            avg_train_acc = mean([m["accuracy"] for m in train_metrics])

            # Enhanced logging
            self.log_training_progress(
                round_num,
                {
                    "avg_train_loss": avg_train_loss,
                    "avg_train_accuracy": avg_train_acc,
                    "active_clients": len(train_metrics),
                    "topology": topology_type,
                },
            )

            # 2. Parameter Exchange and Aggregation (Decentralized)
            self.logger.info(
                f"Performing decentralized aggregation using {topology_type} topology..."
            )

            # Collect parameters for visualization
            node_params = {}
            for i, actor in enumerate(self.cluster_manager.actors):
                # Check if this actor index is malicious using cluster manager's tracking
                if i in self.cluster_manager.malicious_client_indices:
                    params = ray.get(actor.get_model_parameters.remote(current_round=round_num, total_rounds=rounds), timeout=1800)
                    self.logger.debug(f"Round {round_num}: Collected parameters from MALICIOUS node {i}")
                else:
                    params = ray.get(actor.get_model_parameters.remote(), timeout=1800)
                    self.logger.debug(f"Round {round_num}: Collected parameters from honest node {i}")
                node_params[i] = params
                
                # Log parameter statistics for debugging
                if params:
                    param_count = sum(p.numel() if hasattr(p, 'numel') else np.prod(np.array(p).shape) 
                                     for p in params.values())
                    self.logger.debug(f"  Node {i}: {len(params)} layers, {param_count} total parameters")

            # Create parameter summaries for visualization
            param_summaries = self._create_parameter_summaries(node_params)

            # Process trust monitoring for malicious behavior detection
            self.logger.info(f"Round {round_num}: Processing trust monitoring for {len(self.trust_monitors)} monitors")
            
            # Log malicious clients for debugging
            if self.cluster_manager.malicious_client_indices:
                self.logger.info(f"Round {round_num}: Known malicious clients: {list(self.cluster_manager.malicious_client_indices)}")
            
            trust_scores = self._process_trust_monitoring(round_num, node_params, train_metrics, node_malicious_detections)
            self.logger.info(f"Round {round_num}: Trust monitoring returned {len(trust_scores)} results")
            if trust_scores:
                trust_monitoring_results.append({
                    "round": round_num,
                    "trust_scores": trust_scores
                })
                
                # Log trust scores summary for this round
                self.logger.info(f"Round {round_num}: Trust scores summary:")
                for node_idx, scores in trust_scores.items():
                    self.logger.info(f"  Node {node_idx}: {scores}")

            # For each node, emit parameter transfer events
            for node, neighbors in adjacency_list.items():
                if neighbors:
                    self.training_monitor.emit_event(
                        ParameterTransferEvent(
                            round_num=round_num,
                            source_nodes=[node],
                            target_nodes=neighbors,
                            param_summary={node: param_summaries[node]}
                            if node in param_summaries
                            else {},
                        )
                    )

            # Emit aggregation events for each node
            for node in range(len(self.cluster_manager.actors)):
                neighbors = adjacency_list.get(node, [])
                if neighbors:
                    self.training_monitor.emit_event(
                        AggregationEvent(
                            round_num=round_num,
                            participating_nodes=[node] + neighbors,
                            aggregator_node=node,
                            strategy_name=self.cluster_manager.aggregation_strategy.__class__.__name__,
                        )
                    )

            # Get client data sizes for weighted aggregation
            split = self.config.split
            partitions = list(self.dataset.get_partitions(split).values())
            client_data_sizes = [len(partition) for partition in partitions]
            weights = [float(size) for size in client_data_sizes]

            # Perform topology-aware decentralized aggregation with optional trust weighting
            # In true decentralized learning, each node performs local aggregation with neighbors
            # No global model is maintained or distributed
            # Trust scores are applied as additional weights to reduce malicious influence
            trust_scores_for_aggregation = None
            if (self.trust_config and 
                self.trust_config.enable_trust_weighted_aggregation and 
                trust_scores):
                trust_scores_for_aggregation = trust_scores
                self.logger.info(f"Round {round_num}: Enabling trust-weighted aggregation")
            else:
                self.logger.info(f"Round {round_num}: Using standard aggregation (trust weighting disabled)")
                
            self.cluster_manager.perform_decentralized_aggregation(
                current_round=round_num,
                total_rounds=rounds,
                weights=weights,
                trust_scores=trust_scores_for_aggregation
            )

            # Calculate parameter convergence across the network
            # Use average of all node parameters as reference for convergence calculation
            avg_params = self._calculate_average_parameters(node_params)
            param_convergence = self._calculate_parameter_convergence(
                node_params, avg_params
            )

            # Emit model update event
            self.training_monitor.emit_event(
                ModelUpdateEvent(
                    round_num=round_num,
                    updated_nodes=list(range(len(self.cluster_manager.actors))),
                    param_convergence=param_convergence,
                )
            )

            # 4. Evaluation
            # In decentralized learning, evaluate by averaging across all honest nodes
            # This gives a better representation of network performance than a single node
            honest_node_indices = [
                idx for idx in range(len(self.cluster_manager.actors))
                if idx not in self.cluster_manager.malicious_client_indices
            ]
            
            if not honest_node_indices:
                self.logger.warning("No honest nodes available for evaluation!")
                # Fallback to first node if somehow no honest nodes exist
                honest_node_indices = [0]
            
            honest_accuracies = []
            honest_losses = []
            honest_precisions = []
            honest_recalls = []
            honest_f1_scores = []
            
            self.logger.info(
                f"Evaluating {len(honest_node_indices)} honest nodes "
                f"(excluding malicious nodes: {self.cluster_manager.malicious_client_indices})"
            )
            
            # Evaluate each honest node's model
            for node_idx in honest_node_indices:
                try:
                    node_actor = self.cluster_manager.actors[node_idx]
                    node_params = ray.get(node_actor.get_model_parameters.remote())
                    
                    # Set model parameters and evaluate
                    self.model.set_parameters(node_params)
                    node_metrics = self.model.evaluate(test_features, test_labels)
                    
                    honest_accuracies.append(node_metrics['accuracy'])
                    honest_losses.append(node_metrics['loss'])
                    honest_precisions.append(node_metrics['precision'])
                    honest_recalls.append(node_metrics['recall'])
                    honest_f1_scores.append(node_metrics['f1_score'])
                    
                    self.logger.debug(
                        f"Node {node_idx} - Loss: {node_metrics['loss']:.4f}, "
                        f"Accuracy: {node_metrics['accuracy'] * 100:.2f}%, "
                        f"Precision: {node_metrics['precision']:.4f}, "
                        f"Recall: {node_metrics['recall']:.4f}, "
                        f"F1: {node_metrics['f1_score']:.4f}"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate node {node_idx}: {e}")
            
            # Calculate average metrics across honest nodes
            if honest_accuracies:
                avg_accuracy = sum(honest_accuracies) / len(honest_accuracies)
                avg_loss = sum(honest_losses) / len(honest_losses)
                avg_precision = sum(honest_precisions) / len(honest_precisions)
                avg_recall = sum(honest_recalls) / len(honest_recalls)
                avg_f1_score = sum(honest_f1_scores) / len(honest_f1_scores)
                
                # Create aggregated test metrics
                test_metrics = {
                    'accuracy': avg_accuracy,
                    'loss': avg_loss,
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1_score': avg_f1_score
                }
                
                self.logger.info(
                    f"Average Test Loss (Honest Nodes): {avg_loss:.4f}"
                )
                self.logger.info(
                    f"Average Test Accuracy (Honest Nodes): {avg_accuracy * 100:.2f}%"
                )
                self.logger.info(
                    f"Average Test Precision (Honest Nodes): {avg_precision:.4f}"
                )
                self.logger.info(
                    f"Average Test Recall (Honest Nodes): {avg_recall:.4f}"
                )
                self.logger.info(
                    f"Average Test F1 Score (Honest Nodes): {avg_f1_score:.4f}"
                )
                self.logger.info(
                    f"Accuracy Range: {min(honest_accuracies) * 100:.2f}% - {max(honest_accuracies) * 100:.2f}%"
                )
            else:
                # Fallback if no honest nodes could be evaluated
                self.logger.error("No honest nodes could be evaluated!")
                test_metrics = {
                    'accuracy': 0.0, 
                    'loss': float('inf'),
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }

            # Emit evaluation event
            self.training_monitor.emit_event(
                EvaluationEvent(round_num=round_num, metrics=test_metrics)
            )

            # Collect privacy metrics after each round
            round_privacy_metrics = self.cluster_manager.collect_privacy_metrics()
            if round_privacy_metrics["dp_enabled"]:
                # Calculate per-round privacy increment
                # Since clients report cumulative epsilon, we need to compute the increment
                round_epsilon_increment = (
                    round_privacy_metrics["epsilon"]
                    - cumulative_privacy_spent["epsilon"]
                )

                # Update cumulative privacy spent (epsilon is additive in DP)
                cumulative_privacy_spent["epsilon"] = round_privacy_metrics["epsilon"]
                cumulative_privacy_spent["delta"] = max(
                    cumulative_privacy_spent["delta"], round_privacy_metrics["delta"]
                )

                # Store per-round privacy metrics (using increment for this round)
                per_round_privacy_metrics.append(
                    {
                        "round": round_num,
                        "epsilon": round_epsilon_increment,
                        "delta": round_privacy_metrics["delta"],
                        "client_count": round_privacy_metrics["client_count"],
                    }
                )

            # Store metrics for this round
            round_metrics.append(
                {
                    "round": round_num,
                    "train_loss": avg_train_loss,
                    "train_accuracy": avg_train_acc,
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                }
            )

        # Final evaluation - average across honest nodes only
        self.logger.info("=== Final Evaluation Across Honest Nodes ===")
        honest_node_indices = [
            idx for idx in range(len(self.cluster_manager.actors))
            if idx not in self.cluster_manager.malicious_client_indices
        ]
        
        if not honest_node_indices:
            self.logger.warning("No honest nodes available for final evaluation!")
            honest_node_indices = [0]
        
        final_honest_accuracies = []
        final_honest_losses = []
        
        # Evaluate each honest node's final model
        for node_idx in honest_node_indices:
            try:
                node_actor = self.cluster_manager.actors[node_idx]
                node_params = ray.get(node_actor.get_model_parameters.remote())
                
                # Set model parameters and evaluate
                self.model.set_parameters(node_params)
                node_metrics = self.model.evaluate(test_features, test_labels)
                
                final_honest_accuracies.append(node_metrics['accuracy'])
                final_honest_losses.append(node_metrics['loss'])
                
                self.logger.debug(
                    f"Final Node {node_idx} - Loss: {node_metrics['loss']:.4f}, "
                    f"Accuracy: {node_metrics['accuracy'] * 100:.2f}%"
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate final node {node_idx}: {e}")
        
        # Calculate final average metrics across honest nodes
        if final_honest_accuracies:
            final_avg_accuracy = sum(final_honest_accuracies) / len(final_honest_accuracies)
            final_avg_loss = sum(final_honest_losses) / len(final_honest_losses)
            
            final_metrics = {
                'accuracy': final_avg_accuracy,
                'loss': final_avg_loss
            }
            
            improvement = final_metrics["accuracy"] - initial_metrics["accuracy"]
            
            self.logger.info(
                f"Final evaluation across {len(final_honest_accuracies)} honest nodes "
                f"(excluding malicious: {self.cluster_manager.malicious_client_indices})"
            )
        else:
            # Fallback if no honest nodes could be evaluated
            self.logger.error("No honest nodes could be evaluated for final metrics!")
            final_metrics = {'accuracy': 0.0, 'loss': float('inf')}
            improvement = final_metrics["accuracy"] - initial_metrics["accuracy"]

        # Use cumulative privacy metrics from round tracking
        if per_round_privacy_metrics:
            privacy_metrics = {
                "dp_enabled": True,
                "epsilon": cumulative_privacy_spent["epsilon"],
                "delta": cumulative_privacy_spent["delta"],
                "client_count": per_round_privacy_metrics[-1]["client_count"],
                "per_round_privacy": per_round_privacy_metrics,
            }
        else:
            privacy_metrics = self.cluster_manager.collect_privacy_metrics()

        # Update representative model with aggregated privacy metrics for compatibility
        if privacy_metrics["dp_enabled"] and hasattr(self.model, "privacy_spent"):
            # Update privacy spent if model has this attribute (e.g., DPTorchModelWrapper)
            self.model.privacy_spent = {
                "epsilon": privacy_metrics["epsilon"],
                "delta": privacy_metrics["delta"],
            }
            # Update privacy accounting if method exists
            if hasattr(self.model, "_update_privacy_spent"):
                self.model._update_privacy_spent()

        # Enhanced final logging
        cluster_summary = self.get_cluster_summary()
        self.logger.info("=== Final Model Evaluation ===")
        self.logger.info(
            f"Cluster type: {cluster_summary.get('cluster_type', 'unknown')}"
        )
        self.logger.info(f"Final Test Accuracy (Honest Nodes Average): {final_metrics['accuracy'] * 100:.2f}%")
        self.logger.info(f"Accuracy Improvement (Honest Nodes): {improvement * 100:.2f}%")
        
        # Also log accuracy variance if we have multiple honest nodes
        if len(final_honest_accuracies) > 1:
            accuracy_std = (sum((acc - final_metrics['accuracy'])**2 for acc in final_honest_accuracies) / len(final_honest_accuracies))**0.5
            self.logger.info(
                f"Accuracy Range (Honest Nodes): {min(final_honest_accuracies) * 100:.2f}% - {max(final_honest_accuracies) * 100:.2f}% "
                f"(std: {accuracy_std * 100:.2f}%)"
            )

        # Collect final trust monitoring summary based on actual detections
        trust_summary = {}
        all_detected_suspicious = set()
        
        if self.trust_monitors:
            for node_idx, trust_monitor in self.trust_monitors.items():
                with trust_monitor._measure_trust_resource_usage("get_trust_summary_final"):
                    summary = trust_monitor.get_trust_summary()
                
                # Use actual detections from runtime instead of recalculating thresholds
                if node_idx in node_malicious_detections:
                    actual_detections = list(node_malicious_detections[node_idx])
                    summary["low_trust_neighbors"] = actual_detections
                    # Collect all suspicious neighbors globally
                    all_detected_suspicious.update(actual_detections)
                
                trust_summary[node_idx] = summary
        
        # Convert sets to lists for JSON serialization
        node_detections_for_logging = {
            node_idx: list(detections) for node_idx, detections in node_malicious_detections.items()
        }
        
        # Update global detection status
        if all_detected_suspicious:
            self.logger.info(f"Trust monitoring detected {len(all_detected_suspicious)} suspicious neighbors across all nodes: {list(all_detected_suspicious)}")
            self.logger.info(f"Node-wise detections: {node_detections_for_logging}")
        else:
            self.logger.info("Trust monitoring: No malicious behavior detected by any node")

        # Collect trust resource monitoring results
        trust_resource_summary = {}
        if self.trust_monitors and self.trust_config and self.trust_config.enable_trust_resource_monitoring:
            self.logger.info("=== Trust Monitor Resource Usage Summary ===")
            
            # Collect individual node data and calculate aggregates
            all_cpu_avg = []
            all_memory_avg = []
            all_processing_total = []
            all_operations = []
            
            for node_idx, trust_monitor in self.trust_monitors.items():
                summary = trust_monitor.get_trust_resource_summary()
                trust_resource_summary[f"node_{node_idx}"] = summary
                
                if summary.get("status") != "no_resource_data":
                    overall = summary.get("overall", {})
                    cpu_stats = overall.get("cpu_stats", {})
                    memory_stats = overall.get("memory_stats", {})
                    timing_stats = overall.get("timing_stats", {})
                    
                    # Collect for aggregation
                    all_cpu_avg.append(cpu_stats.get('avg_percent', 0))
                    all_memory_avg.append(memory_stats.get('avg_mb', 0))
                    all_processing_total.append(timing_stats.get('total_time_ms', 0))
                    all_operations.append(summary.get('total_measurements', 0))
                    
                    self.logger.info(f"Node {node_idx} trust monitor resource usage:")
                    self.logger.info(f"  CPU: {cpu_stats.get('avg_percent', 0):.2f}% avg, {cpu_stats.get('max_percent', 0):.2f}% max")
                    self.logger.info(f"  Memory: {memory_stats.get('avg_mb', 0):.2f}MB avg, {memory_stats.get('peak_memory_mb', 0):.2f}MB peak")
                    self.logger.info(f"  Processing: {timing_stats.get('total_time_ms', 0):.2f}ms total, {timing_stats.get('avg_time_ms', 0):.2f}ms avg")
                    self.logger.info(f"  Measurements: {summary.get('total_measurements', 0)} operations tracked")
            
            # Log aggregate statistics across all honest nodes
            if all_cpu_avg:
                avg_cpu = sum(all_cpu_avg) / len(all_cpu_avg)
                avg_memory = sum(all_memory_avg) / len(all_memory_avg)
                total_processing = sum(all_processing_total)
                total_operations = sum(all_operations)
                
                self.logger.info("=== Aggregate Trust Monitor Resource Usage (All Honest Nodes) ===")
                self.logger.info(f"Average CPU usage across {len(all_cpu_avg)} honest nodes: {avg_cpu:.2f}%")
                self.logger.info(f"Average memory usage across {len(all_memory_avg)} honest nodes: {avg_memory:.2f}MB")
                self.logger.info(f"Total processing time across all honest nodes: {total_processing:.2f}ms")
                self.logger.info(f"Total trust operations across all honest nodes: {total_operations} operations")
                
                # Store aggregate data for CSV extraction
                trust_resource_summary["aggregate"] = {
                    "avg_cpu_percent": avg_cpu,
                    "avg_memory_mb": avg_memory,
                    "total_processing_ms": total_processing,
                    "total_operations": total_operations,
                    "node_count": len(all_cpu_avg)
                }

        # Return results_phase1
        results = {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "accuracy_improvement": improvement,
            "round_metrics": round_metrics,
            "topology": topology_info,
            "privacy_metrics": privacy_metrics,
            "trust_monitoring": {
                "enabled": bool(self.trust_monitors),
                "results": trust_monitoring_results,
                "final_summary": trust_summary,
                "global_suspicious_detected": list(all_detected_suspicious) if self.trust_monitors else [],
                "node_detections": node_detections_for_logging if self.trust_monitors else {},
                "resource_monitoring": trust_resource_summary if trust_resource_summary else {}
            }
        }

        if cluster_summary:
            results["cluster_info"] = cluster_summary

        return results

    def _calculate_average_parameters(
        self, node_params: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate the average parameters across all nodes for convergence analysis.

        :param node_params: Dictionary mapping node indices to their parameters
        :return: Averaged parameters dictionary
        """
        if not node_params:
            return {}

        # Get the first node's parameters to determine structure
        first_node_params = next(iter(node_params.values()))
        avg_params = {}

        for param_name, param_value in first_node_params.items():
            # Calculate average for this parameter across all nodes
            param_values = [node_params[node_id][param_name] for node_id in node_params]
            avg_params[param_name] = np.mean(param_values, axis=0)

        return avg_params
