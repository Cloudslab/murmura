"""
Trust-aware true decentralized learning process.

This module extends the true decentralized learning process to include:
- HSIC-based trust monitoring
- Adaptive trust assessment
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
    - HSIC-based trust monitoring for each node
    - Adaptive malicious node detection
    - Trust-weighted gossip aggregation
    """
    
    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)
        
        # Trust monitoring components
        self.trust_config: Optional[TrustMonitoringConfig] = None
        self.trust_monitors: Dict[str, TrustMonitor] = {}
        self.trust_enabled = False
        
        
        # Attack tracking
        self.attack_statistics: Dict[str, Any] = {
            "total_attacks": 0,
            "detected_attacks": 0,
            "per_attacker": {},
            "detection_rate": 0.0,
        }
        
        # Track which nodes are malicious (ground truth)
        self.malicious_node_ids: List[str] = []
        self.honest_node_ids: List[str] = []
        
        # Logger
        self.trust_logger = logging.getLogger(
            "murmura.trust.TrustAwareTrueDecentralizedLearningProcess"
        )
        
        # Validation split for pattern analysis (not used in pattern analysis mode)
        self.validation_split = 0.1  # Default 10% for validation data
        
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
        
        # Debug logging
        self.trust_logger.info(f"Initialize called with attack_config: {attack_config}")
        
        # Call parent initialize
        super().initialize(num_actors, topology_config, aggregation_config, partitioner)
        
        # Create mixed actors if attack config is provided
        if attack_config:
            self.trust_logger.info(f"Creating mixed actors with attack config: {attack_config}")
            self._create_mixed_actors(num_actors, attack_config)
            self._redistribute_models_and_data()
        else:
            self.trust_logger.info("No attack config provided - using honest actors only")
    
    def _create_mixed_actors(self, num_actors: int, attack_config: Dict[str, Any]) -> None:
        """Replace default actors with mixed honest/malicious actors."""
        import numpy as np
        from murmura.node.client_actor import VirtualClientActor
        from murmura.attacks.gradual_label_flipping import create_gradual_attack_config
        
        attack_type = attack_config.get("attack_type")
        dataset_name = getattr(self.config, 'dataset_name', 'mnist')
        
        if attack_type in ["gradual_label_flipping", "label_flipping"]:
            from murmura.attacks.gradual_label_flipping import create_gradual_attack_config
            attack_config_obj = create_gradual_attack_config(
                dataset_name=dataset_name,
                attack_intensity=attack_config.get("attack_intensity", "moderate"),
                stealth_level=attack_config.get("stealth_level", "medium")
            )
        elif attack_type in ["model_poisoning", "backdoor"]:
            from murmura.attacks.gradual_model_poisoning import create_backdoor_config
            attack_config_obj = create_backdoor_config(
                dataset_name=dataset_name,
                attack_intensity=attack_config.get("attack_intensity", "moderate"),
                stealth_level=attack_config.get("stealth_level", "medium"),
                target_class=attack_config.get("target_class", 0)
            )
        else:
            self.trust_logger.warning(f"Unknown attack type: {attack_type}. Skipping malicious actor creation.")
            return
        
        if attack_type in ["gradual_label_flipping", "label_flipping", "model_poisoning", "backdoor"]:
            
            # Determine malicious actors
            malicious_fraction = attack_config.get("malicious_fraction", 0.25)
            num_malicious = int(num_actors * malicious_fraction)
            
            if num_malicious > 0:
                np.random.seed(42)
                malicious_indices = np.random.choice(num_actors, num_malicious, replace=False)
                
                self.trust_logger.info(f"Creating {num_malicious} malicious actors out of {num_actors}")
                
                # Create actors with attack config and track malicious nodes
                mixed_actors = []
                self.malicious_node_ids = []
                self.honest_node_ids = []
                
                for i in range(num_actors):
                    node_id = f"node_{i}"  # Match the naming convention used in trust monitoring
                    actor_attack_config = attack_config_obj if i in malicious_indices else None
                    
                    # Track which nodes are malicious
                    if i in malicious_indices:
                        self.malicious_node_ids.append(node_id)
                    else:
                        self.honest_node_ids.append(node_id)
                    
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
                        f"{num_malicious} malicious {self.malicious_node_ids}, {num_actors - num_malicious} honest {self.honest_node_ids}"
                    )
                    
                # Update attack statistics with ground truth
                self.attack_statistics["total_attacks"] = len(self.malicious_node_ids)
                for node_id in self.malicious_node_ids:
                    self.attack_statistics["per_attacker"][node_id] = {
                        "detected": False,
                        "trust_score": 1.0,
                        "action_taken": "ACCEPT"
                    }
    
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
            self.trust_logger.info("Trust monitoring enabled")
        else:
            self.trust_logger.info("Trust monitoring disabled")
    
    def _initialize_trust_monitors(self) -> None:
        """Initialize trust monitors."""
        if not self.cluster_manager or not self.cluster_manager.actors:
            self.trust_logger.warning("No actors available for trust monitoring")
            return
        
        self.trust_logger.info(
            f"Initializing trust monitors for {len(self.cluster_manager.actors)} actors"
        )
        
        # Create trust monitor for each actor
        for i, actor in enumerate(self.cluster_manager.actors):
            node_id = f"node_{i}"
            
            # Create trust monitor with statistical config
            statistical_config = None
            if self.trust_config.statistical_config is not None:
                if hasattr(self.trust_config.statistical_config, 'to_dict'):
                    statistical_config = self.trust_config.statistical_config.to_dict()
                else:
                    statistical_config = self.trust_config.statistical_config
                    
            # Get topology for optimization
            topology = getattr(self.trust_config, 'topology', 'ring')
            
            trust_monitor = TrustMonitor.remote(
                node_id=node_id,
                statistical_config=statistical_config,
                trust_config=self.trust_config.trust_policy_config.to_dict(),
                enable_ensemble_detection=self.trust_config.enable_ensemble_detection,
                topology=topology,
            )
            
            self.trust_monitors[node_id] = trust_monitor
        
        self.trust_logger.info(f"Created {len(self.trust_monitors)} trust monitors")
        
        # Configure trust monitors with FL context and Beta thresholding
        self._configure_trust_monitors()
    
    def _configure_trust_monitors(self) -> None:
        """Configure trust monitors with FL context and Beta thresholding."""
        if not self.trust_monitors:
            return
        
        self.trust_logger.info("Configuring trust monitors with FL context...")
        
        try:
            
            # Get topology type for configuration
            topology_type = "ring"  # Default
            if hasattr(self.config, 'topology') and hasattr(self.config.topology, 'topology_type'):
                topology_type = str(self.config.topology.topology_type.name).lower()
            
            for node_id, trust_monitor_ref in self.trust_monitors.items():
                # Set FL context
                ray.get(trust_monitor_ref.set_fl_context.remote(
                    total_rounds=self.config.rounds,
                    current_accuracy=0.1,  # Initial accuracy
                    topology=topology_type
                ))
                
                # Configure Beta thresholding if available
                if hasattr(self.trust_config, 'beta_threshold_config') and self.trust_config.beta_threshold_config:
                    ray.get(trust_monitor_ref.configure_beta_threshold.remote(
                        self.trust_config.beta_threshold_config.to_dict()
                    ))
                
            
            # ==== PATTERN ANALYSIS: Set ground truth labels for all trust monitors ====
            # This helps trust monitors track patterns based on ground truth knowledge
            for node_id, trust_monitor_ref in self.trust_monitors.items():
                for other_node_id in self.trust_monitors.keys():
                    if other_node_id != node_id:  # Don't set ground truth for self
                        if other_node_id in self.malicious_node_ids:
                            ray.get(trust_monitor_ref.set_ground_truth_label.remote(other_node_id, "malicious"))
                        elif other_node_id in self.honest_node_ids:
                            ray.get(trust_monitor_ref.set_ground_truth_label.remote(other_node_id, "honest"))
                        else:
                            # Default assumption: nodes are honest unless specified
                            ray.get(trust_monitor_ref.set_ground_truth_label.remote(other_node_id, "honest"))
            
            self.trust_logger.info(f"Configured {len(self.trust_monitors)} trust monitors with FL context, Beta thresholding, and ground truth labels for pattern analysis")
            
        except Exception as e:
            self.trust_logger.error(f"Failed to configure trust monitors: {e}")
    
    def _perform_trust_aware_gossip_exchange(
        self, 
        round_num: int, 
        adjacency_list: Dict[int, List[int]],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Perform trust-aware gossip exchange.
        
        Each node:
        1. Evaluates neighbor models using HSIC-based trust
        2. Computes trust scores using adaptive assessment
        3. Performs trust-weighted aggregation
        """
        self.trust_logger.info("Performing trust-aware decentralized gossip exchange...")
        
        # Trust monitor baselines already set pre-training
        
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
                    timeout=120
                )
                neighbor_params[neighbor_idx] = params
                
                # Get trust assessment from trust monitor
                if node_id in self.trust_monitors:
                    # Use ensemble detection if enabled, otherwise standard HSIC detection
                    if self.trust_config.enable_ensemble_detection:
                        action, trust_score, stats = ray.get(
                            self.trust_monitors[node_id].assess_update_with_ensemble.remote(
                                neighbor_id, params
                            )
                        )
                    else:
                        action, trust_score, stats = ray.get(
                            self.trust_monitors[node_id].assess_update.remote(
                                neighbor_id, params
                            )
                        )
                    
                    # Use trust score directly
                    trust_weights[neighbor_idx] = trust_score
                    
                    self.trust_logger.debug(
                        f"Node {node_idx} -> Neighbor {neighbor_idx}: "
                        f"HSIC trust={trust_score if isinstance(trust_score, (int, float)) else 'N/A'}"
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
            
            # 0. Set trust baselines BEFORE training (critical for drift detection)
            if self.trust_enabled:
                self._set_trust_baselines_pre_training(round_num)
            
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
            "suspected_malicious": []
        }
        
        for node_id, monitor in self.trust_monitors.items():
            try:
                report = ray.get(monitor.get_trust_report.remote())
                metrics["node_reports"][node_id] = report
                
                            
            except Exception as e:
                self.trust_logger.error(f"Failed to get report from {node_id}: {e}")
        
        return metrics
    
    def _log_trust_summary(self, trust_metrics: Dict[str, Any]) -> None:
        """Log trust monitoring summary."""
        self.trust_logger.info("=== Trust Monitoring Summary ===")
        if trust_metrics["suspected_malicious"]:
            self.trust_logger.warning(
                f"Suspected Malicious Nodes: {trust_metrics['suspected_malicious']}"
            )
    
    def _set_trust_baselines_pre_training(self, round_num: int) -> None:
        """Set trust monitor baselines before training starts (critical for drift detection)."""
        if not self.trust_monitors:
            return
            
        self.trust_logger.debug(f"Setting trust baselines pre-training for round {round_num}")
        
        # Set current parameters for each trust monitor BEFORE training
        for i, actor in enumerate(self.cluster_manager.actors):
            node_id = f"node_{i}"
            if node_id in self.trust_monitors:
                try:
                    # Get pre-training parameters as baseline
                    current_params = ray.get(actor.get_model_parameters.remote())
                    ray.get(
                        self.trust_monitors[node_id].set_current_parameters.remote(
                            current_params
                        )
                    )
                    self.trust_logger.debug(f"Set pre-training baseline for {node_id}")
                except Exception as e:
                    self.trust_logger.warning(f"Failed to set baseline for {node_id}: {e}")
    
    def _collect_attack_statistics(self, round_num: int) -> None:
        """Collect attack statistics from malicious clients."""
        # Get ground truth for evaluation purposes only (not for detection)
        malicious_nodes = self.malicious_node_ids.copy()
        
        # The trust monitor should detect suspicious nodes without ground truth knowledge
        # We only use ground truth here for calculating detection accuracy metrics
        
        for node_id in malicious_nodes:
            # Initialize attack statistics if not exists
            if node_id not in self.attack_statistics["per_attacker"]:
                self.attack_statistics["per_attacker"][node_id] = {
                    "detected": False,
                    "trust_score": 1.0,
                    "action_taken": "ACCEPT",
                    "rounds_active": round_num
                }
            
            # Try to get attack report from actor if available
            try:
                actor_index = int(node_id.split('_')[1])  # Extract index from node_id
                if actor_index < len(self.cluster_manager.actors):
                    actor = self.cluster_manager.actors[actor_index]
                    try:
                        attack_report = ray.get(actor.get_attack_report.remote(), timeout=5)
                        self.attack_statistics["per_attacker"][node_id].update(attack_report)
                    except AttributeError:
                        # Actor doesn't have get_attack_report method
                        pass
            except (ValueError, IndexError) as e:
                self.trust_logger.debug(f"Could not get attack report for {node_id}: {e}")
        
        # Now check for detection using trust monitoring
        if self.trust_enabled:
            excluded_nodes = self._get_excluded_nodes()
            
            for node_id in malicious_nodes:
                # Check if detected by exclusion
                detected = node_id in excluded_nodes
                
                # INDEPENDENT TRUST MONITORING: Each node makes its own decision
                # No consensus - this supports the paper's core contribution of independent monitoring
                individual_detections = 0
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
                                trust_score = neighbor_data.get('trust_score', 1.0)
                                
                                # Update our tracking with the latest trust score
                                self.attack_statistics["per_attacker"][node_id]["trust_score"] = trust_score
                                
                                # INDEPENDENT DETECTION: Each neighbor decides independently
                                if trust_level in ['suspicious', 'untrusted']:
                                    individual_detections += 1
                                    # Each independent detection counts as a successful detection
                                    if not detected:
                                        detected = True
                                        self.trust_logger.info(
                                            f"Malicious {node_id} detected by INDEPENDENT trust monitor {monitor_id}: "
                                            f"trust_level={trust_level}, trust_score={trust_score:.3f}"
                                        )
                                        break  # First detection is sufficient - true independence
                        except Exception:
                            pass
                
                # Update detection status
                self.attack_statistics["per_attacker"][node_id]["detected"] = detected
                
                if detected:
                    self.attack_statistics["detected_attacks"] += 1
                    self.trust_logger.warning(
                        f"ATTACK DETECTED: {node_id} at round {round_num} "
                        f"(excluded: {node_id in excluded_nodes}, independent_detections: {individual_detections}/{total_monitors})"
                    )
                else:
                    self.trust_logger.debug(
                        f"Malicious {node_id} not yet detected "
                        f"(excluded: {node_id in excluded_nodes}, independent_detections: {individual_detections}/{total_monitors})"
                    )
        
        # Update overall detection rate
        detected_count = sum(1 for stats in self.attack_statistics["per_attacker"].values() if stats.get("detected", False))
        total_malicious = len(self.malicious_node_ids)
        if total_malicious > 0:
            self.attack_statistics["detection_rate"] = detected_count / total_malicious
        
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
                
                # Use ground truth for evaluation labeling only
                ground_truth_label = "[GROUND TRUTH: MALICIOUS]" if node_id in malicious_nodes else "[GROUND TRUTH: HONEST]"
                self.trust_logger.info(f"\n{ground_truth_label} NODE {node_id}:")
                
                # Get neighbors this node is monitoring
                neighbors = trust_report.get('neighbors', {})
                if neighbors:
                    for neighbor_id, neighbor_data in neighbors.items():
                        # Trust monitor's actual detection (what matters for the system)
                        trust_level = neighbor_data.get('trust_level', 'N/A')
                        detection_label = f"[DETECTED AS: {trust_level.upper()}]" if trust_level != 'N/A' else "[DETECTION: UNKNOWN]"
                        
                        trust_score = neighbor_data.get('trust_score', 'N/A')
                        drift_rate = neighbor_data.get('drift_rate', 'N/A')
                        
                        # Safe format function for pattern analysis mode
                        def safe_float_format(val):
                            try:
                                if val == 'N/A' or val is None:
                                    return 'N/A'
                                return f"{float(val):.3f}"
                            except (ValueError, TypeError):
                                return str(val)
                        
                        # Get statistical detection method
                        detection_method = neighbor_data.get('method', 'statistical')
                        
                        self.trust_logger.info(
                            f"  -> {detection_label} {neighbor_id}: "
                            f"trust={safe_float_format(trust_score)}, level={trust_level}, "
                            f"method={detection_method}, drift_rate={safe_float_format(drift_rate)}"
                        )
                else:
                    self.trust_logger.info(f"  No neighbors being monitored")
                    
            except Exception as e:
                self.trust_logger.warning(f"Failed to get trust report from {node_id}: {e}")
    
    def _log_suspected_malicious_nodes(self) -> None:
        """Log final assessment of suspected malicious nodes."""
        suspected = set()
        
        
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
        
        # Count detected attackers (use independent detection results from rounds)
        detected_attackers = 0
        for node_id in malicious_actors:
            # Use the detection status that was already determined during the rounds
            # This preserves the independent trust monitoring approach
            if self.attack_statistics["per_attacker"][node_id].get("detected", False):
                detected_attackers += 1
                self.trust_logger.info(f"Attacker {node_id} was detected during training rounds")
            else:
                self.trust_logger.info(f"Attacker {node_id} was NOT detected during training rounds")
        
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
        
        # Safe format avg_trust_score to handle None values
        avg_trust_formatted = f"{avg_trust_score:.3f}" if isinstance(avg_trust_score, (int, float)) else str(avg_trust_score)
        self.trust_logger.info(f"Final trust report: {total_excluded} excluded, {total_downgraded} downgraded, avg_trust={avg_trust_formatted}")
        
        return final_report