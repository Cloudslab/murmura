"""
Trust-aware true decentralized learning process with performance-based trust monitoring.

This module extends the true decentralized learning process to include:
- Performance-based trust monitoring (not just HSIC)
- Local validation for detecting malicious behavior
- Trust-aware gossip aggregation
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import ray

from murmura.orchestration.learning_process.true_decentralized_learning_process import (
    TrueDecentralizedLearningProcess
)
from murmura.trust.trust_monitor import TrustMonitor
from murmura.trust.trust_config import TrustMonitoringConfig
from murmura.visualization.training_event import (
    ParameterTransferEvent,
)


class TrustAwareTrueDecentralizedLearningProcess(TrueDecentralizedLearningProcess):
    """
    Trust-aware extension of true decentralized learning.
    
    This class adds:
    - Performance-based trust monitoring for each node
    - Validation-based malicious node detection
    - Trust-weighted gossip aggregation
    """
    
    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)
        
        # Trust monitoring components
        self.trust_config: Optional[TrustMonitoringConfig] = None
        self.trust_monitors: Dict[str, TrustMonitor] = {}
        self.trust_enabled = False
        
        # Performance tracking
        self.node_performance_history: Dict[int, List[float]] = {}
        self.validation_split = 0.1  # Use 10% of data for validation
        
        # Attack tracking
        self.attack_statistics: Dict[str, Any] = {
            "total_attacks": 0,
            "detected_attacks": 0,
            "per_attacker": {},
            "detection_rate": 0.0,
        }
        
        # Logger
        self.trust_logger = logging.getLogger(
            "murmura.trust.TrustAwareTrueDecentralizedLearningProcess"
        )
        
        self._trust_initialized = False
    
    def initialize(
        self,
        num_actors: int,
        topology_config,
        aggregation_config,
        partitioner,
        attack_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize with optional attack configuration."""
        # Store attack config
        self.attack_config = attack_config
        
        # Call parent initialize
        super().initialize(num_actors, topology_config, aggregation_config, partitioner)
        
        # Create mixed actors if attack config is provided
        if attack_config:
            self._create_mixed_actors(num_actors, attack_config)
            self._redistribute_models_and_data()
    
    def _create_mixed_actors(self, num_actors: int, attack_config: Dict[str, Any]) -> None:
        """Replace default actors with mixed honest/malicious actors."""
        import numpy as np
        from murmura.node.client_actor import VirtualClientActor
        from murmura.attacks.gradual_label_flipping import create_gradual_attack_config
        
        if attack_config.get("attack_type") == "gradual_label_flipping":
            dataset_name = getattr(self.config, 'dataset_name', 'mnist')
            
            gradual_config = create_gradual_attack_config(
                dataset_name=dataset_name,
                attack_intensity=attack_config.get("attack_intensity", "moderate"),
                stealth_level=attack_config.get("stealth_level", "medium")
            )
            
            # Determine malicious actors
            malicious_fraction = attack_config.get("malicious_fraction", 0.25)
            num_malicious = int(num_actors * malicious_fraction)
            
            if num_malicious > 0:
                np.random.seed(42)
                malicious_indices = np.random.choice(num_actors, num_malicious, replace=False)
                
                self.trust_logger.info(f"Creating {num_malicious} malicious actors out of {num_actors}")
                
                # Create actors with attack config
                mixed_actors = []
                for i in range(num_actors):
                    actor_attack_config = gradual_config if i in malicious_indices else None
                    
                    actor = VirtualClientActor.remote(
                        client_id=f"client_{i}",
                        attack_config=actor_attack_config
                    )
                    mixed_actors.append(actor)
                
                # Replace actors
                if hasattr(self.cluster_manager, 'actors'):
                    self.cluster_manager.actors = mixed_actors
                    self.trust_logger.info(
                        f"Replaced actors with mixed population: "
                        f"{num_malicious} malicious, {num_actors - num_malicious} honest"
                    )
    
    def _redistribute_models_and_data(self) -> None:
        """Re-distribute models and data to the new mixed actors."""
        self.trust_logger.info("Re-distributing models and data to mixed actors...")
        
        # CRITICAL: Re-apply topology to set actor references
        # Without this, trust_weighted_gossip_aggregate fails with "Actor references not set"
        try:
            self.cluster_manager._apply_topology()
            self.trust_logger.info("Re-applied topology to mixed actors")
        except Exception as e:
            self.trust_logger.error(f"Failed to re-apply topology: {e}")
        
        # Re-distribute data partitions
        partitions = self.dataset.get_partitions("train")
        if partitions:
            partition_list = [partitions[i] for i in range(len(partitions))]
            self.cluster_manager.distribute_data(
                partition_list,
                metadata={
                    "split": "train",
                    "dataset": self.config.dataset_name,
                    "redistribution": True,
                }
            )
        
        # Re-distribute dataset
        if hasattr(self.cluster_manager, 'distribute_dataset'):
            self.cluster_manager.distribute_dataset(
                self.dataset,
                feature_columns=self.config.feature_columns,
                label_column=self.config.label_column
            )
        
        # Re-distribute model
        if hasattr(self.cluster_manager, 'distribute_model'):
            self.cluster_manager.distribute_model(self.model)
    
    def _initialize_trust_config(self) -> None:
        """Initialize trust monitoring configuration."""
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
            self.trust_logger.info("Trust monitoring enabled with performance tracking")
        else:
            self.trust_logger.info("Trust monitoring disabled")
    
    def _initialize_trust_monitors(self) -> None:
        """Initialize trust monitors with performance monitoring enabled."""
        if not self.cluster_manager or not self.cluster_manager.actors:
            self.trust_logger.warning("No actors available for trust monitoring")
            return
        
        self.trust_logger.info(
            f"Initializing trust monitors for {len(self.cluster_manager.actors)} actors"
        )
        
        # Create trust monitor for each actor
        for i, actor in enumerate(self.cluster_manager.actors):
            node_id = f"node_{i}"
            
            # Create trust monitor with performance monitoring ALWAYS enabled
            trust_monitor = TrustMonitor.remote(
                node_id=node_id,
                hsic_config=self.trust_config.hsic_config.to_dict(),
                trust_config=self.trust_config.trust_policy_config.to_dict(),
                model_template=None,
                enable_performance_monitoring=True,  # Always enable for proper detection
            )
            
            self.trust_monitors[node_id] = trust_monitor
            self.node_performance_history[i] = []
        
        self.trust_logger.info(f"Created {len(self.trust_monitors)} trust monitors")
    
    def _perform_trust_aware_gossip_exchange(
        self, 
        round_num: int, 
        adjacency_list: Dict[int, List[int]],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Perform trust-aware gossip exchange with performance validation.
        
        Each node:
        1. Validates neighbor models on local validation data
        2. Computes trust scores based on performance
        3. Performs trust-weighted aggregation
        """
        self.trust_logger.info("Performing trust-aware decentralized gossip exchange...")
        
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
        
        # Prepare trust-aware gossip tasks
        gossip_tasks = []
        
        for node_idx, neighbors in adjacency_list.items():
            if neighbors:
                # Create trust-aware gossip task
                task = self._perform_node_trust_gossip(
                    node_idx, neighbors, round_num, validation_data
                )
                gossip_tasks.append((node_idx, task))
        
        # Wait for all gossip exchanges
        if gossip_tasks:
            try:
                results = ray.get([task for _, task in gossip_tasks], timeout=600)
                
                # Log results
                successful = sum(1 for r in results if r.get('success', False))
                self.trust_logger.info(
                    f"Completed {successful}/{len(gossip_tasks)} trust-aware exchanges"
                )
                
            except Exception as e:
                self.trust_logger.error(f"Trust-aware gossip failed: {e}")
                raise
    
    def _perform_node_trust_gossip(
        self,
        node_idx: int,
        neighbors: List[int],
        round_num: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Perform trust-aware gossip for a single node.
        
        This method runs on the orchestrator and coordinates trust evaluation.
        """
        node_id = f"node_{node_idx}"
        actor = self.cluster_manager.actors[node_idx]
        
        # Get validation data for this node if not provided
        if validation_data is None:
            # Use a portion of node's training data for validation
            try:
                node_data = ray.get(actor.get_partition_data.remote(sample_rate=self.validation_split))
                if node_data and len(node_data) == 2:
                    validation_data = node_data
            except Exception as e:
                self.trust_logger.warning(f"Failed to get validation data for node {node_idx}: {e}")
        
        # Evaluate each neighbor's trustworthiness
        trust_weights = {}
        neighbor_params = {}
        
        for neighbor_idx in neighbors:
            neighbor_id = f"node_{neighbor_idx}"
            
            try:
                # Get neighbor's parameters
                params = ray.get(
                    self.cluster_manager.actors[neighbor_idx].get_model_parameters.remote(),
                    timeout=30
                )
                neighbor_params[neighbor_idx] = params
                
                # Get trust assessment from trust monitor
                if node_id in self.trust_monitors:
                    action, trust_score, stats = ray.get(
                        self.trust_monitors[node_id].assess_update.remote(
                            neighbor_id, params
                        )
                    )
                    
                    # Validate performance if we have validation data
                    performance_score = 1.0
                    if validation_data is not None:
                        performance_score = self._validate_neighbor_performance(
                            params, validation_data, neighbor_idx
                        )
                    
                    # Combine trust and performance scores
                    combined_trust = 0.5 * trust_score + 0.5 * performance_score
                    trust_weights[neighbor_idx] = combined_trust
                    
                    self.trust_logger.debug(
                        f"Node {node_idx} -> Neighbor {neighbor_idx}: "
                        f"HSIC trust={trust_score:.3f}, performance={performance_score:.3f}, "
                        f"combined={combined_trust:.3f}"
                    )
                else:
                    # No trust monitor, use equal weights
                    trust_weights[neighbor_idx] = 1.0
                    
            except Exception as e:
                self.trust_logger.warning(
                    f"Failed to assess neighbor {neighbor_idx} for node {node_idx}: {e}"
                )
                trust_weights[neighbor_idx] = 0.0
        
        # Filter out untrusted neighbors (trust < threshold)
        trust_threshold = 0.3  # Nodes with trust below this are excluded
        trusted_neighbors = [
            n for n in neighbors 
            if trust_weights.get(n, 0) >= trust_threshold
        ]
        
        if not trusted_neighbors:
            self.trust_logger.warning(
                f"Node {node_idx} has no trusted neighbors! Using all neighbors with low weights."
            )
            trusted_neighbors = neighbors
            # Give very low weight to untrusted neighbors
            for n in neighbors:
                if trust_weights.get(n, 0) < trust_threshold:
                    trust_weights[n] = 0.1
        
        # Perform trust-weighted gossip aggregation
        return actor.trust_weighted_gossip_aggregate.remote(
            neighbor_indices=trusted_neighbors,
            trust_weights={str(n): trust_weights.get(n, 1.0) for n in trusted_neighbors},
            mixing_parameter=0.5
        )
    
    def _validate_neighbor_performance(
        self,
        neighbor_params: Dict[str, Any],
        validation_data: Tuple[np.ndarray, np.ndarray],
        neighbor_idx: int
    ) -> float:
        """
        Validate neighbor's model performance on local validation data.
        
        Returns:
            Performance score between 0 and 1
        """
        try:
            features, labels = validation_data
            
            # Temporarily set model parameters
            original_params = self.model.get_parameters()
            self.model.set_parameters(neighbor_params)
            
            # Evaluate on validation data
            metrics = self.model.evaluate(features, labels)
            accuracy = metrics.get('accuracy', 0.0)
            
            # Restore original parameters
            self.model.set_parameters(original_params)
            
            # Track performance history
            if neighbor_idx in self.node_performance_history:
                self.node_performance_history[neighbor_idx].append(accuracy)
                
                # Detect performance anomalies
                if len(self.node_performance_history[neighbor_idx]) > 3:
                    recent_perf = self.node_performance_history[neighbor_idx][-3:]
                    avg_perf = np.mean(recent_perf)
                    
                    # If performance suddenly drops, it might be malicious
                    if accuracy < avg_perf * 0.8:  # 20% drop
                        self.trust_logger.warning(
                            f"Performance anomaly detected for node {neighbor_idx}: "
                            f"current={accuracy:.3f}, average={avg_perf:.3f}"
                        )
                        return max(0.0, accuracy / avg_perf)  # Penalize based on drop
            
            return accuracy
            
        except Exception as e:
            self.trust_logger.error(f"Failed to validate neighbor {neighbor_idx}: {e}")
            return 0.0
    
    def execute(self) -> Dict[str, Any]:
        """Execute trust-aware true decentralized learning."""
        if not self.cluster_manager:
            raise ValueError("Learning process not initialized. Call initialize first.")
        
        # Initialize trust monitoring
        if not self._trust_initialized:
            self._initialize_trust_config()
            self._trust_initialized = True
        
        # Get configuration
        rounds = self.config.rounds
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        test_split = self.config.test_split
        
        self.logger.info("=== Starting Trust-Aware True Decentralized Learning ===")
        self.logger.info(f"Trust monitoring: {'Enabled' if self.trust_enabled else 'Disabled'}")
        
        # Prepare test data
        test_dataset = self.dataset.get_split(test_split)
        test_features, test_labels = self._prepare_test_data(
            test_dataset, self.config.feature_columns, self.config.label_column
        )
        
        # Initial evaluation
        initial_consensus = self._evaluate_network_consensus(test_features, test_labels)
        
        self.logger.info(
            f"Initial Network Consensus: {initial_consensus['consensus_accuracy'] * 100:.2f}%"
        )
        
        # Get topology
        topology_info = self.cluster_manager.get_topology_information()
        adjacency_list = topology_info.get("adjacency_list", {})
        
        round_metrics = []
        trust_metrics = []
        
        # Training rounds
        for round_num in range(1, rounds + 1):
            self.logger.info(f"\n--- Round {round_num}/{rounds} ---")
            
            # Update round numbers for attack progression
            for i, actor in enumerate(self.cluster_manager.actors):
                try:
                    if hasattr(actor, 'set_round_number'):
                        ray.get(actor.set_round_number.remote(round_num), timeout=5)
                except Exception:
                    pass
            
            # Update round for trust monitors
            if self.trust_enabled:
                for node_id, trust_monitor in self.trust_monitors.items():
                    try:
                        ray.get(trust_monitor.set_round_number.remote(round_num), timeout=5)
                    except Exception:
                        pass
            
            # 1. Local Training
            self.logger.info(f"Local training for {epochs} epochs...")
            
            train_metrics = self.cluster_manager.train_models(
                client_sampling_rate=self.config.client_sampling_rate,
                data_sampling_rate=self.config.data_sampling_rate,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True,
            )
            
            avg_train_loss = np.mean([m["loss"] for m in train_metrics])
            avg_train_acc = np.mean([m["accuracy"] for m in train_metrics])
            
            # 2. Trust-Aware Gossip Exchange
            if self.trust_enabled:
                self._perform_trust_aware_gossip_exchange(round_num, adjacency_list)
            else:
                # Standard gossip without trust
                self._perform_gossip_exchange(round_num, adjacency_list)
            
            # 3. Collect attack statistics
            self._collect_attack_statistics(round_num)
            
            # 4. Evaluate network
            consensus_metrics = self._evaluate_network_consensus(test_features, test_labels)
            
            self.logger.info(f"\nRound {round_num} Results:")
            self.logger.info(
                f"  Training: Loss={avg_train_loss:.4f}, Accuracy={avg_train_acc:.4f}"
            )
            self.logger.info(
                f"  Consensus: {consensus_metrics['consensus_accuracy'] * 100:.2f}%"
            )
            self.logger.info(
                f"  Node Agreement: {consensus_metrics['node_agreement'] * 100:.2f}%"
            )
            
            # Collect trust metrics
            if self.trust_enabled and round_num % self.trust_config.trust_report_interval == 0:
                round_trust_metrics = self._collect_trust_metrics()
                trust_metrics.append({
                    "round": round_num,
                    **round_trust_metrics
                })
                self._log_trust_summary(round_trust_metrics)
            
            # Store metrics
            round_metrics.append({
                "round": round_num,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                **consensus_metrics
            })
        
        # Final evaluation
        final_consensus = self._evaluate_network_consensus(test_features, test_labels)
        
        self.logger.info("\n=== Final Evaluation ===")
        self.logger.info(
            f"Final Consensus: {final_consensus['consensus_accuracy'] * 100:.2f}%"
        )
        self.logger.info(
            f"Improvement: "
            f"{(final_consensus['consensus_accuracy'] - initial_consensus['consensus_accuracy']) * 100:.2f}%"
        )
        
        # Log suspected malicious nodes
        if self.trust_enabled:
            self._log_suspected_malicious_nodes()
        
        # Calculate final trust report with global stats
        final_trust_report = self._generate_final_trust_report() if self.trust_enabled else {}
        
        results = {
            "initial_metrics": initial_consensus,
            "final_metrics": final_consensus,
            "accuracy_improvement": final_consensus['consensus_accuracy'] - initial_consensus['consensus_accuracy'],
            "consensus_improvement": final_consensus['consensus_accuracy'] - initial_consensus['consensus_accuracy'],
            "round_metrics": round_metrics,
            "topology": topology_info,
            "trust_enabled": self.trust_enabled,
            "learning_type": "trust_aware_true_decentralized",
            "final_trust_report": final_trust_report
        }
        
        if self.trust_enabled:
            results["trust_metrics"] = trust_metrics
            
        results["attack_statistics"] = self._finalize_attack_statistics()
        
        return results
    
    def _collect_trust_metrics(self) -> Dict[str, Any]:
        """Collect trust metrics from all monitors."""
        metrics = {
            "total_monitors": len(self.trust_monitors),
            "node_reports": {},
            "performance_anomalies": 0,
            "suspected_malicious": []
        }
        
        for node_id, monitor in self.trust_monitors.items():
            try:
                report = ray.get(monitor.get_trust_report.remote())
                metrics["node_reports"][node_id] = report
                
                # Check for performance anomalies
                node_idx = int(node_id.split('_')[1])
                if node_idx in self.node_performance_history:
                    history = self.node_performance_history[node_idx]
                    if len(history) > 3:
                        recent_avg = np.mean(history[-3:])
                        overall_avg = np.mean(history)
                        if recent_avg < overall_avg * 0.8:
                            metrics["performance_anomalies"] += 1
                            metrics["suspected_malicious"].append(node_id)
                            
            except Exception as e:
                self.trust_logger.error(f"Failed to get report from {node_id}: {e}")
        
        return metrics
    
    def _log_trust_summary(self, trust_metrics: Dict[str, Any]) -> None:
        """Log trust monitoring summary."""
        self.trust_logger.info("=== Trust Monitoring Summary ===")
        self.trust_logger.info(
            f"Performance Anomalies Detected: {trust_metrics['performance_anomalies']}"
        )
        if trust_metrics["suspected_malicious"]:
            self.trust_logger.warning(
                f"Suspected Malicious Nodes: {trust_metrics['suspected_malicious']}"
            )
    
    def _collect_attack_statistics(self, round_num: int) -> None:
        """Collect attack statistics from malicious clients."""
        malicious_nodes = []
        
        for i, actor in enumerate(self.cluster_manager.actors):
            try:
                node_id = f"node_{i}"
                
                # Check if this actor has attack configuration (is malicious)
                is_malicious = ray.get(actor.is_malicious.remote(), timeout=5)
                
                if is_malicious:
                    malicious_nodes.append(node_id)
                    self.trust_logger.info(f"Found malicious actor: {node_id}")
                    
                    # Try to get attack report
                    try:
                        attack_report = ray.get(actor.get_attack_report.remote(), timeout=5)
                        if node_id not in self.attack_statistics["per_attacker"]:
                            self.attack_statistics["per_attacker"][node_id] = {}
                        self.attack_statistics["per_attacker"][node_id] = attack_report
                    except AttributeError:
                        # Actor doesn't have get_attack_report method
                        self.attack_statistics["per_attacker"][node_id] = {"rounds_active": round_num}
                    
                    # Check if detected
                    if self.trust_enabled:
                        excluded_nodes = self._get_excluded_nodes()
                        detected = node_id in excluded_nodes
                        
                        # Also check if marked as suspicious by neighbors
                        suspicious_count = 0
                        total_monitors = 0
                        for monitor_id, monitor in self.trust_monitors.items():
                            if monitor_id != node_id:  # Don't count self-evaluation
                                try:
                                    trust_report = ray.get(monitor.get_trust_report.remote(), timeout=5)
                                    neighbors = trust_report.get('neighbors', {})
                                    if node_id in neighbors:
                                        total_monitors += 1
                                        neighbor_data = neighbors[node_id]
                                        trust_level = neighbor_data.get('trust_level', 'trusted')
                                        if trust_level in ['suspicious', 'untrusted']:
                                            suspicious_count += 1
                                except Exception:
                                    pass
                        
                        # Consider detected if majority finds it suspicious
                        if total_monitors > 0:
                            suspicious_ratio = suspicious_count / total_monitors
                            if suspicious_ratio >= 0.5:  # Majority suspicious
                                detected = True
                                self.trust_logger.info(
                                    f"Malicious {node_id} detected by trust levels: "
                                    f"{suspicious_count}/{total_monitors} neighbors find it suspicious"
                                )
                        
                        if detected:
                            try:
                                ray.get(actor.mark_as_detected.remote(round_num, 1), timeout=5)
                            except AttributeError:
                                pass  # Actor doesn't have mark_as_detected method
                            self.trust_logger.warning(
                                f"ATTACK DETECTED: {node_id} at round {round_num}"
                            )
                        else:
                            self.trust_logger.debug(
                                f"Malicious {node_id} not yet detected (excluded: {excluded_nodes}, "
                                f"suspicious: {suspicious_count}/{total_monitors})"
                            )
                
            except AttributeError:
                # Actor doesn't have is_malicious method, assume honest
                continue
            except Exception as e:
                self.trust_logger.debug(f"Error checking actor {i}: {e}")
        
        # Log trust monitor values for all nodes
        if self.trust_enabled and round_num % 2 == 0:  # Log every 2 rounds
            self._log_trust_monitor_values(round_num, malicious_nodes)
    
    def _get_excluded_nodes(self) -> List[str]:
        """Get list of nodes excluded by trust monitoring using consensus."""
        # Track how many nodes exclude each neighbor
        exclusion_votes = {}
        flagging_counts = {}  # Track how many neighbors each node flags
        
        for node_id, monitor in self.trust_monitors.items():
            try:
                excluded_neighbors = ray.get(monitor.get_excluded_neighbors.remote(), timeout=5)
                flagging_counts[node_id] = len(excluded_neighbors)
                
                for neighbor_id in excluded_neighbors:
                    if neighbor_id not in exclusion_votes:
                        exclusion_votes[neighbor_id] = []
                    exclusion_votes[neighbor_id].append(node_id)
            except Exception:
                pass
        
        # Find nodes that flag too many others (likely malicious themselves)
        total_nodes = len(self.trust_monitors)
        suspicious_flaggers = []
        for node_id, flag_count in flagging_counts.items():
            # If a node flags more than half of the network, it's suspicious
            if flag_count > total_nodes // 2:
                suspicious_flaggers.append(node_id)
                self.trust_logger.warning(f"Node {node_id} flags {flag_count}/{total_nodes} nodes - suspicious")
        
        # Determine truly excluded nodes using consensus, excluding votes from suspicious flaggers
        truly_excluded = []
        min_consensus = max(2, total_nodes // 3)  # Need at least 2 votes or 1/3 of network
        
        for neighbor_id, voters in exclusion_votes.items():
            # Filter out votes from suspicious flaggers
            clean_voters = [v for v in voters if v not in suspicious_flaggers]
            
            if len(clean_voters) >= min_consensus:
                truly_excluded.append(neighbor_id)
                self.trust_logger.warning(
                    f"Node {neighbor_id} excluded by consensus: {len(clean_voters)} clean votes "
                    f"(flagged by: {clean_voters})"
                )
        
        return truly_excluded
    
    def _log_trust_monitor_values(self, round_num: int, malicious_nodes: List[str]) -> None:
        """Log trust monitor values for debugging attack detection."""
        self.trust_logger.info(f"\n=== TRUST MONITOR VALUES - ROUND {round_num} ===")
        self.trust_logger.info(f"Malicious nodes: {malicious_nodes}")
        
        for node_id, monitor in self.trust_monitors.items():
            try:
                # Get trust report from monitor
                trust_report = ray.get(monitor.get_trust_report.remote(), timeout=5)
                
                node_type = "MALICIOUS" if node_id in malicious_nodes else "HONEST"
                self.trust_logger.info(f"\n{node_type} NODE {node_id}:")
                
                # Get neighbors this node is monitoring
                neighbors = trust_report.get('neighbors', {})
                if neighbors:
                    for neighbor_id, neighbor_data in neighbors.items():
                        neighbor_type = "MALICIOUS" if neighbor_id in malicious_nodes else "HONEST"
                        trust_score = neighbor_data.get('trust_score', 'N/A')
                        trust_level = neighbor_data.get('trust_level', 'N/A')
                        drift_rate = neighbor_data.get('drift_rate', 'N/A')
                        
                        # Get HSIC stats if available
                        hsic_stats = neighbor_data.get('hsic_stats', {})
                        current_hsic = hsic_stats.get('current_hsic', 'N/A')
                        adaptive_threshold = hsic_stats.get('adaptive_threshold', 'N/A')
                        
                        self.trust_logger.info(
                            f"  -> {neighbor_type} {neighbor_id}: "
                            f"trust={trust_score:.3f}, level={trust_level}, "
                            f"hsic={current_hsic:.3f}, threshold={adaptive_threshold:.3f}, "
                            f"drift_rate={drift_rate:.3f}"
                        )
                else:
                    self.trust_logger.info(f"  No neighbors being monitored")
                    
            except Exception as e:
                self.trust_logger.warning(f"Failed to get trust report from {node_id}: {e}")
    
    def _log_suspected_malicious_nodes(self) -> None:
        """Log final assessment of suspected malicious nodes."""
        suspected = set()
        
        # Check performance histories
        for node_idx, history in self.node_performance_history.items():
            if len(history) > 5:
                # Check for consistent performance degradation
                first_half = np.mean(history[:len(history)//2])
                second_half = np.mean(history[len(history)//2:])
                
                if second_half < first_half * 0.7:  # 30% degradation
                    suspected.add(f"node_{node_idx}")
        
        # Check trust exclusions
        excluded = self._get_excluded_nodes()
        suspected.update(excluded)
        
        if suspected:
            self.trust_logger.warning(f"\n=== SUSPECTED MALICIOUS NODES ===")
            for node_id in sorted(suspected):
                self.trust_logger.warning(f"  - {node_id}")
    
    def _finalize_attack_statistics(self) -> Dict[str, Any]:
        """Finalize attack statistics."""
        # Count malicious actors by checking is_malicious
        total_attackers = 0
        malicious_actors = []
        
        for i, actor in enumerate(self.cluster_manager.actors):
            try:
                node_id = f"node_{i}"
                is_malicious = ray.get(actor.is_malicious.remote(), timeout=5)
                if is_malicious:
                    total_attackers += 1
                    malicious_actors.append(node_id)
                    # Ensure attacker is in per_attacker dict
                    if node_id not in self.attack_statistics["per_attacker"]:
                        self.attack_statistics["per_attacker"][node_id] = {"detected": False}
            except Exception:
                continue
        
        # Count detected attackers (those with majority suspicious ratings)
        detected_attackers = 0
        for node_id in malicious_actors:
            suspicious_count = 0
            total_monitors = 0
            for monitor_id, monitor in self.trust_monitors.items():
                if monitor_id != node_id:  # Don't count self-evaluation
                    try:
                        trust_report = ray.get(monitor.get_trust_report.remote(), timeout=5)
                        neighbors = trust_report.get('neighbors', {})
                        if node_id in neighbors:
                            total_monitors += 1
                            neighbor_data = neighbors[node_id]
                            trust_level = neighbor_data.get('trust_level', 'trusted')
                            if trust_level in ['suspicious', 'untrusted']:
                                suspicious_count += 1
                    except Exception:
                        pass
            
            # Consider detected if majority finds it suspicious
            if total_monitors > 0 and suspicious_count / total_monitors >= 0.5:
                detected_attackers += 1
                self.attack_statistics["per_attacker"][node_id]["detected"] = True
                self.trust_logger.info(f"Attacker {node_id} marked as detected: {suspicious_count}/{total_monitors} suspicious")
        
        self.attack_statistics.update({
            "total_attackers": total_attackers,
            "detected_attacks": detected_attackers,
            "detection_rate": detected_attackers / total_attackers if total_attackers > 0 else 0.0,
            "malicious_actors": malicious_actors,
        })
        
        self.trust_logger.info(f"Final attack statistics: {total_attackers} attackers, {detected_attackers} detected, {detected_attackers/total_attackers*100 if total_attackers > 0 else 0:.1f}% detection rate")
        
        return self.attack_statistics.copy()
    
    def _generate_final_trust_report(self) -> Dict[str, Any]:
        """Generate final trust report with global statistics."""
        if not self.trust_monitors:
            return {}
        
        trust_levels = {"trusted": 0, "suspicious": 0, "untrusted": 0}
        trust_scores = []
        excluded_nodes = set()
        downgraded_nodes = set()
        
        # Collect trust data from all monitors
        for node_id, monitor in self.trust_monitors.items():
            try:
                trust_report = ray.get(monitor.get_trust_report.remote(), timeout=5)
                neighbors = trust_report.get('neighbors', {})
                
                for neighbor_id, neighbor_data in neighbors.items():
                    trust_level = neighbor_data.get('trust_level', 'trusted')
                    trust_score = neighbor_data.get('trust_score', 1.0)
                    
                    # Count trust levels
                    if trust_level in trust_levels:
                        trust_levels[trust_level] += 1
                    
                    # Collect trust scores
                    trust_scores.append(trust_score)
                    
                    # Track excluded and downgraded nodes
                    if trust_level == 'untrusted':
                        excluded_nodes.add(neighbor_id)
                    elif trust_level == 'suspicious':
                        downgraded_nodes.add(neighbor_id)
                        
            except Exception as e:
                self.trust_logger.warning(f"Failed to get final trust report from {node_id}: {e}")
        
        # Calculate global statistics
        total_excluded = len(excluded_nodes)
        total_downgraded = len(downgraded_nodes)
        avg_trust_score = np.mean(trust_scores) if trust_scores else 1.0
        
        # Create global stats
        global_stats = {
            "total_excluded": total_excluded,
            "total_downgraded": total_downgraded, 
            "avg_trust_score": avg_trust_score,
            "trust_level_counts": trust_levels,
            "excluded_nodes": list(excluded_nodes),
            "downgraded_nodes": list(downgraded_nodes),
            "total_trust_assessments": len(trust_scores)
        }
        
        final_report = {
            "global_stats": global_stats,
            "timestamp": time.time(),
            "total_monitors": len(self.trust_monitors)
        }
        
        self.trust_logger.info(f"Final trust report: {total_excluded} excluded, {total_downgraded} downgraded, avg_trust={avg_trust_score:.3f}")
        
        return final_report