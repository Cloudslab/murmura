import os
import csv
from typing import Dict, List, Optional, Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.artist import Artist

from murmura.network_management.topology_manager import TopologyManager
from murmura.visualization.training_event import (
    TrainingEvent,
    LocalTrainingEvent,
    ParameterTransferEvent,
    AggregationEvent,
    ModelUpdateEvent,
    EvaluationEvent,
    InitialStateEvent,
    NetworkStructureEvent,
    FingerprintEvent,
    TrustSignalsEvent,
    AggregationWeightsEvent,
)
from murmura.attacks.attack_event import (
    AttackEvent,
    LabelFlippingEvent,
    GradientManipulationEvent,
    AttackSummaryEvent,
    AttackDetectionEvent,
)
from murmura.trust_monitoring.trust_events import (
    TrustAnomalyEvent, 
    TrustScoreEvent,
)
from murmura.visualization.training_observer import TrainingObserver


class NetworkVisualizer(TrainingObserver):
    """
    Creates visualizations_phase1 of the training process based on events and stores data in CSV format.

    This visualizer creates both animations and frame sequences showing how
    model parameters flow through the network during training, and exports
    all collected data to CSV files for further analysis.
    """

    def __init__(self, output_dir: str = "./visualizations_phase1"):
        """
        Initialize the network visualizer.

        Args:
            output_dir: Directory to save visualization outputs and CSV files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.topology: Optional[Dict[int, List[int]]] = None
        self.network_type: Optional[str] = None
        self.frames: List[Dict[str, Any]] = []
        # Use dictionaries instead of lists to store metric history by round number
        self.accuracy_history: Dict[int, float] = {}
        self.loss_history: Dict[int, float] = {}
        self.parameter_history: Dict[int, List[float]] = {}
        self.frame_descriptions: List[str] = []
        self.round_metrics: Dict[int, Dict[str, float]] = {}  # Store metrics by round
        # Track at which frame each parameter update occurs
        self.parameter_update_frames: Dict[
            int, Dict[int, int]
        ] = {}  # {node_id: {parameter_idx: frame_idx}}

        # New data structures for comprehensive CSV export
        self.event_log: List[Dict[str, Any]] = []  # Complete event log
        self.node_activities: Dict[
            int, List[Dict[str, Any]]
        ] = {}  # Per-node activity tracking
        self.communication_log: List[Dict[str, Any]] = []  # Communication events
        self.parameter_updates: List[Dict[str, Any]] = []  # Parameter update tracking

        # Enhanced network structure data
        self.network_structure: Optional[Dict[str, Any]] = None
        self.adjacency_matrix: Optional[List[List[int]]] = None
        self.node_identifiers: List[str] = []
        self.edge_weights: Dict[str, float] = {}
        self.node_attributes: Dict[int, Dict[str, Any]] = {}
        self.geographic_info: Dict[int, Dict[str, float]] = {}
        self.organizational_hierarchy: Dict[int, Dict[str, str]] = {}

        # Attack tracking data structures
        self.attack_events: List[Dict[str, Any]] = []  # All attack events
        self.malicious_nodes: set[int] = set()  # Track which nodes are malicious
        self.attack_history: Dict[int, List[Dict[str, Any]]] = {}  # Per-node attack history
        self.attack_intensity_history: Dict[int, List[float]] = {}  # Track intensity over time
        self.attack_detection_events: List[Dict[str, Any]] = []  # Detection events
        
        # Trust monitoring data structures  
        self.trust_events: List[Dict[str, Any]] = []  # All trust events
        self.trust_scores_history: Dict[str, Dict[int, Dict[str, float]]] = {}  # {observer_node: {round: {neighbor: trust_score}}}
        self.influence_weights_history: Dict[str, Dict[int, Dict[str, float]]] = {}  # {observer_node: {round: {neighbor: influence_weight}}}
        self.trust_anomalies: List[Dict[str, Any]] = []  # Trust anomaly events
        self.trust_score_changes: Dict[str, Dict[int, Dict[str, float]]] = {}  # {observer_node: {round: {neighbor: change}}}

    def set_topology(self, topology_manager: TopologyManager) -> None:
        """
        Set the network topology information.

        Args:
            topology_manager: Topology manager containing network structure
        """
        self.topology = topology_manager.adjacency_list
        self.network_type = topology_manager.config.topology_type.value

    def on_event(self, event: TrainingEvent) -> None:
        """
        Process training events to build visualization data and collect CSV data.

        Args:
            event: The training event to process
        """
        if self.topology is None and not isinstance(
            event, (InitialStateEvent, NetworkStructureEvent, AggregationWeightsEvent)
        ):
            return  # Can't visualize without topology information

        # Log the event for CSV export
        event_data = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "step_name": event.step_name,
            "event_type": type(event).__name__,
        }

        frame: Dict[str, Any] = {
            "round": event.round_num,
            "step": event.step_name,
            "timestamp": event.timestamp,
            "topology_type": self.network_type,
            "active_nodes": [],
            "active_edges": [],
            "node_params": {},
            "metrics": {},
        }

        description = f"Round {event.round_num}: {event.step_name}"

        if isinstance(event, NetworkStructureEvent):
            self.network_type = event.topology_type
            frame["topology_type"] = event.topology_type
            frame["num_nodes"] = event.num_nodes
            description = "Network structure initialization"

            # Store comprehensive network structure data
            self.network_structure = event.get_network_summary()
            self.adjacency_matrix = event.adjacency_matrix
            self.node_identifiers = event.node_identifiers
            self.edge_weights = event.edge_weights
            self.node_attributes = event.node_attributes
            self.geographic_info = event.geographic_info
            self.organizational_hierarchy = event.organizational_hierarchy

            # Add comprehensive data to frame
            frame["network_structure"] = self.network_structure
            frame["adjacency_matrix"] = event.adjacency_matrix
            frame["node_identifiers"] = event.node_identifiers
            frame["edge_weights"] = event.edge_weights
            frame["node_attributes"] = event.node_attributes
            frame["geographic_info"] = event.geographic_info
            frame["organizational_hierarchy"] = event.organizational_hierarchy

            # Add to event log with detailed network structure
            event_data.update(
                {
                    "topology_type": event.topology_type,
                    "num_nodes": event.num_nodes,
                    "network_density": self.network_structure.get("network_density", 0),
                    "total_edges": self.network_structure.get("total_edges", 0),
                    "total_cpu_cores": self.network_structure.get(
                        "resource_totals", {}
                    ).get("cpu_cores", 0),
                    "total_memory_gb": self.network_structure.get(
                        "resource_totals", {}
                    ).get("memory_gb", 0),
                    "total_gpus": self.network_structure.get("resource_totals", {}).get(
                        "gpu_count", 0
                    ),
                    "avg_bandwidth_mbps": self.network_structure.get(
                        "resource_totals", {}
                    ).get("avg_bandwidth_mbps", 0),
                    "node_type_distribution": str(
                        self.network_structure.get("node_type_distribution", {})
                    ),
                    "organizational_entities": self.network_structure.get(
                        "organizational_entities", 0
                    ),
                }
            )

        elif isinstance(event, InitialStateEvent):
            self.network_type = event.topology_type
            frame["topology_type"] = event.topology_type
            frame["num_nodes"] = event.num_nodes
            description = "Initial network state"

            # Add to event log
            event_data.update(
                {"topology_type": event.topology_type, "num_nodes": event.num_nodes}
            )

        elif isinstance(event, LocalTrainingEvent):
            frame["active_nodes"] = event.active_nodes
            frame["metrics"] = event.metrics
            description = f"Round {event.round_num}: Local training on {len(event.active_nodes)} nodes"

            # Log node activities
            for node_id in event.active_nodes:
                if node_id not in self.node_activities:
                    self.node_activities[node_id] = []

                activity = {
                    "timestamp": event.timestamp,
                    "round_num": event.round_num,
                    "activity_type": "local_training",
                    "metrics": event.metrics,
                    "current_epoch": event.current_epoch,
                    "total_epochs": event.total_epochs,
                }
                self.node_activities[node_id].append(activity)

            # Add to event log
            event_data.update(
                {
                    "active_nodes": ",".join(map(str, event.active_nodes)),
                    "num_active_nodes": len(event.active_nodes),
                    "metrics": str(event.metrics),
                    "current_epoch": event.current_epoch,
                    "total_epochs": event.total_epochs,
                }
            )

        elif isinstance(event, ParameterTransferEvent):
            for source in event.source_nodes:
                for target in event.target_nodes:
                    frame["active_edges"].append((source, target))

                    # Log communication
                    comm_data = {
                        "timestamp": event.timestamp,
                        "round_num": event.round_num,
                        "source_node": source,
                        "target_node": target,
                        "transfer_type": "parameter_transfer",
                        "param_summary": str(event.param_summary.get(source, {})),
                    }
                    self.communication_log.append(comm_data)

            # Track parameter summaries for visualization
            for node, summary in event.param_summary.items():
                if node not in self.parameter_history:
                    self.parameter_history[node] = []
                    self.parameter_update_frames[node] = {}

                if isinstance(summary, dict) and "norm" in summary:
                    # Store parameter update for this round
                    current_round = event.round_num
                    norm_value = summary["norm"]

                    # Only add if this is a new round or first update
                    if not self.parameter_history[node] or current_round > len(
                        self.parameter_history[node]
                    ):
                        self.parameter_history[node].append(norm_value)
                        # Store at which frame this parameter update happened
                        self.parameter_update_frames[node][
                            len(self.parameter_history[node]) - 1
                        ] = len(self.frames)

                        # Log parameter update
                        param_update = {
                            "timestamp": event.timestamp,
                            "round_num": event.round_num,
                            "node_id": node,
                            "parameter_norm": norm_value,
                            "parameter_summary": str(summary),
                        }
                        self.parameter_updates.append(param_update)
                else:
                    # If we just have raw parameters, use a simple norm
                    if not self.parameter_history[node] or event.round_num > len(
                        self.parameter_history[node]
                    ):
                        self.parameter_history[node].append(1.0)  # Placeholder
                        self.parameter_update_frames[node][
                            len(self.parameter_history[node]) - 1
                        ] = len(self.frames)

                        # Log parameter update
                        param_update = {
                            "timestamp": event.timestamp,
                            "round_num": event.round_num,
                            "node_id": node,
                            "parameter_norm": 1.0,
                            "parameter_summary": str(summary),
                        }
                        self.parameter_updates.append(param_update)

            frame["node_params"] = event.param_summary
            description = (
                f"Round {event.round_num}: Parameter transfer from {len(event.source_nodes)} "
                f"nodes to {len(event.target_nodes)} nodes"
            )

            # Add to event log
            event_data.update(
                {
                    "source_nodes": ",".join(map(str, event.source_nodes)),
                    "target_nodes": ",".join(map(str, event.target_nodes)),
                    "num_source_nodes": len(event.source_nodes),
                    "num_target_nodes": len(event.target_nodes),
                    "param_summary": str(event.param_summary),
                }
            )

        elif isinstance(event, AggregationEvent):
            frame["active_nodes"] = event.participating_nodes
            frame["strategy_name"] = event.strategy_name

            if event.aggregator_node is not None:
                # For star topology, show edges to aggregator
                for node in event.participating_nodes:
                    if node != event.aggregator_node:
                        frame["active_edges"].append((node, event.aggregator_node))

                        # Log communication to aggregator
                        comm_data = {
                            "timestamp": event.timestamp,
                            "round_num": event.round_num,
                            "source_node": node,
                            "target_node": event.aggregator_node,
                            "transfer_type": "aggregation",
                            "strategy_name": event.strategy_name,
                        }
                        self.communication_log.append(comm_data)

                description = (
                    f"Round {event.round_num}: Aggregation at node {event.aggregator_node} "
                    f"using {event.strategy_name}"
                )
            else:
                description = f"Round {event.round_num}: Decentralized aggregation using {event.strategy_name}"

            # Log node activities for aggregation
            for node_id in event.participating_nodes:
                if node_id not in self.node_activities:
                    self.node_activities[node_id] = []

                activity = {
                    "timestamp": event.timestamp,
                    "round_num": event.round_num,
                    "activity_type": "aggregation",
                    "strategy_name": event.strategy_name,
                    "is_aggregator": node_id == event.aggregator_node,
                }
                self.node_activities[node_id].append(activity)

            # Add to event log
            event_data.update(
                {
                    "participating_nodes": ",".join(
                        map(str, event.participating_nodes)
                    ),
                    "aggregator_node": event.aggregator_node,
                    "strategy_name": event.strategy_name,
                    "num_participating_nodes": len(event.participating_nodes),
                }
            )

        elif isinstance(event, ModelUpdateEvent):
            frame["active_nodes"] = event.updated_nodes
            frame["param_convergence"] = event.param_convergence
            description = f"Round {event.round_num}: Model update on {len(event.updated_nodes)} nodes"

            # Log node activities for model update
            for node_id in event.updated_nodes:
                if node_id not in self.node_activities:
                    self.node_activities[node_id] = []

                activity = {
                    "timestamp": event.timestamp,
                    "round_num": event.round_num,
                    "activity_type": "model_update",
                    "param_convergence": event.param_convergence,
                }
                self.node_activities[node_id].append(activity)

            # Add to event log
            event_data.update(
                {
                    "updated_nodes": ",".join(map(str, event.updated_nodes)),
                    "num_updated_nodes": len(event.updated_nodes),
                    "param_convergence": event.param_convergence,
                }
            )

        elif isinstance(event, EvaluationEvent):
            frame["metrics"] = event.metrics

            # Store metrics by round number
            self.round_metrics[event.round_num] = event.metrics

            # Only update global lists if we have valid metrics
            if "accuracy" in event.metrics:
                # Store accuracy value by round number
                accuracy_value = event.metrics["accuracy"]
                self.accuracy_history[event.round_num] = accuracy_value

            if "loss" in event.metrics:
                # Store loss value by round number
                loss_value = event.metrics["loss"]
                self.loss_history[event.round_num] = loss_value

            description = f"Round {event.round_num}: Evaluation - "
            if "accuracy" in event.metrics:
                description += f"Accuracy: {event.metrics['accuracy']:.4f}"
            if "loss" in event.metrics:
                description += f", Loss: {event.metrics['loss']:.4f}"

            # Add to event log
            event_data.update(
                {
                    "metrics": str(event.metrics),
                    "accuracy": event.metrics.get("accuracy"),
                    "loss": event.metrics.get("loss"),
                }
            )

        # Handle attack events
        elif isinstance(event, (LabelFlippingEvent, GradientManipulationEvent)):
            self._handle_attack_event(event, frame, event_data, description)
        
        elif isinstance(event, AttackSummaryEvent):
            self._handle_attack_summary(event, frame, event_data, description)
        
        elif isinstance(event, AttackDetectionEvent):
            self._handle_attack_detection(event, frame, event_data, description)
            
        # Handle trust monitoring events
        elif isinstance(event, TrustAnomalyEvent):
            self._handle_trust_anomaly(event, frame, event_data, description)
        
        elif isinstance(event, TrustScoreEvent):
            self._handle_trust_score_update(event, frame, event_data, description)
            
        elif isinstance(event, FingerprintEvent):
            self._handle_fingerprint_event(event, frame, event_data, description)
            
        elif isinstance(event, TrustSignalsEvent):
            self._handle_trust_signals_event(event, frame, event_data, description)
            
        elif isinstance(event, AggregationWeightsEvent):
            self._handle_aggregation_weights_event(event, frame, event_data, description)

        # Ensure all metrics data is available for all frames
        frame["all_metrics"] = self.round_metrics.copy()
        # Add current metrics history to each frame for consistent animation
        frame["accuracy_history"] = self.accuracy_history.copy()
        frame["loss_history"] = self.loss_history.copy()
        frame["parameter_history"] = {
            node: history.copy() for node, history in self.parameter_history.items()
        }
        
        # Add attack data to all frames
        frame["attack_events"] = self.attack_events.copy()
        frame["malicious_nodes"] = list(self.malicious_nodes)
        frame["attack_intensity_history"] = {
            node: history.copy() for node, history in self.attack_intensity_history.items()
        }

        self.frames.append(frame)
        self.frame_descriptions.append(description)
        self.event_log.append(event_data)

    def _handle_attack_event(self, event: AttackEvent, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle label flipping and gradient manipulation attack events."""
        # Track malicious nodes
        self.malicious_nodes.update(event.malicious_clients)
        
        # Record attack intensity for each malicious client
        for client_id in event.malicious_clients:
            if client_id not in self.attack_intensity_history:
                self.attack_intensity_history[client_id] = []
            self.attack_intensity_history[client_id].append(event.attack_intensity)
        
        # Create attack event record
        attack_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "attack_type": event.attack_type,
            "malicious_clients": event.malicious_clients,
            "attack_intensity": event.attack_intensity,
            "num_malicious_clients": event.num_malicious_clients
        }
        
        # Add specific attack details
        if isinstance(event, LabelFlippingEvent):
            attack_record.update({
                "samples_poisoned": event.samples_poisoned,
                "total_samples_poisoned": event.total_samples_poisoned,
                "target_label": event.target_label,
                "source_label": event.source_label
            })
            
        elif isinstance(event, GradientManipulationEvent):
            attack_record.update({
                "parameters_modified": event.parameters_modified,
                "total_parameters_modified": event.total_parameters_modified,
                "noise_scale": event.noise_scale,
                "sign_flip_prob": event.sign_flip_prob
            })
        
        self.attack_events.append(attack_record)
        
        # Update frame data
        frame["attack_active"] = True
        frame["current_attack"] = attack_record
        frame["malicious_nodes_current"] = event.malicious_clients
        
        # Update event data for CSV
        event_data.update({
            "attack_type": event.attack_type,
            "malicious_clients": ",".join(map(str, event.malicious_clients)),
            "attack_intensity": event.attack_intensity,
            "num_malicious_clients": event.num_malicious_clients
        })
        
        if isinstance(event, LabelFlippingEvent):
            event_data.update({
                "total_samples_poisoned": event.total_samples_poisoned,
                "target_label": event.target_label,
                "source_label": event.source_label
            })
        elif isinstance(event, GradientManipulationEvent):
            event_data.update({
                "total_parameters_modified": event.total_parameters_modified,
                "noise_scale": event.noise_scale,
                "sign_flip_prob": event.sign_flip_prob
            })

    def _handle_attack_summary(self, event: AttackSummaryEvent, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle attack summary events."""
        attack_summary_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "attack_statistics": event.attack_statistics,
            "global_attack_metrics": event.global_attack_metrics
        }
        
        self.attack_events.append(attack_summary_record)
        
        frame["attack_summary"] = attack_summary_record
        
        event_data.update({
            "attack_statistics": str(event.attack_statistics),
            "global_attack_metrics": str(event.global_attack_metrics)
        })

    def _handle_attack_detection(self, event: AttackDetectionEvent, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle attack detection events."""
        detection_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "suspected_malicious_clients": event.suspected_malicious_clients,
            "detection_metrics": event.detection_metrics,
            "detection_method": event.detection_method,
            "confidence_scores": event.confidence_scores
        }
        
        self.attack_detection_events.append(detection_record)
        
        frame["attack_detection"] = detection_record
        
        event_data.update({
            "suspected_malicious_clients": ",".join(map(str, event.suspected_malicious_clients)),
            "detection_method": event.detection_method,
            "detection_metrics": str(event.detection_metrics),
            "confidence_scores": str(event.confidence_scores)
        })

    def export_to_csv(self, prefix: str = "training_data") -> None:
        """
        Export all collected training data to CSV files.

        Args:
            prefix: Prefix for CSV filenames
        """
        # Ensure output directory exists before writing files
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Export main event log
        if self.event_log:
            event_csv_path = os.path.join(self.output_dir, f"{prefix}_events.csv")
            with open(event_csv_path, "w", newline="", encoding="utf-8") as f:
                if self.event_log:
                    # Collect all possible field names from all events
                    all_fieldnames: set[str] = set()
                    for event in self.event_log:
                        all_fieldnames.update(event.keys())

                    # Sort fieldnames for consistent column order
                    fieldnames = sorted(all_fieldnames)

                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.event_log)
            print(f"Event log exported to {event_csv_path}")

        # 2. Export metrics history
        if self.round_metrics:
            metrics_csv_path = os.path.join(self.output_dir, f"{prefix}_metrics.csv")
            with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                # Get all unique metric keys
                all_metric_keys: set[str] = set()
                for metrics in self.round_metrics.values():
                    all_metric_keys.update(metrics.keys())

                fieldnames = ["round_num"] + sorted(all_metric_keys)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for round_num in sorted(self.round_metrics.keys()):
                    row: Dict[str, Any] = {"round_num": round_num}
                    row.update(self.round_metrics[round_num])
                    writer.writerow(row)
            print(f"Metrics history exported to {metrics_csv_path}")

        # 3. Export parameter history
        if self.parameter_history:
            param_csv_path = os.path.join(self.output_dir, f"{prefix}_parameters.csv")
            with open(param_csv_path, "w", newline="", encoding="utf-8") as f:
                # Find maximum number of rounds across all nodes
                max_rounds = max(
                    len(history) for history in self.parameter_history.values()
                )

                fieldnames = ["round_num"] + [
                    f"node_{node_id}"
                    for node_id in sorted(self.parameter_history.keys())
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for round_idx in range(max_rounds):
                    row = {"round_num": round_idx + 1}
                    for node_id in sorted(self.parameter_history.keys()):
                        if round_idx < len(self.parameter_history[node_id]):
                            row[f"node_{node_id}"] = self.parameter_history[node_id][
                                round_idx
                            ]
                        else:
                            row[f"node_{node_id}"] = None
                    writer.writerow(row)
            print(f"Parameter history exported to {param_csv_path}")

        # 4. Export communication log
        if self.communication_log:
            comm_csv_path = os.path.join(
                self.output_dir, f"{prefix}_communications.csv"
            )
            with open(comm_csv_path, "w", newline="", encoding="utf-8") as f:
                if self.communication_log:
                    # Collect all unique fieldnames from all entries
                    all_fieldnames = set()
                    for entry in self.communication_log:
                        all_fieldnames.update(entry.keys())

                    writer = csv.DictWriter(f, fieldnames=sorted(list(all_fieldnames)))
                    writer.writeheader()
                    writer.writerows(self.communication_log)
            print(f"Communication log exported to {comm_csv_path}")

        # 5. Export node activities
        if self.node_activities:
            for node_id, activities in self.node_activities.items():
                if activities:
                    activity_csv_path = os.path.join(
                        self.output_dir, f"{prefix}_node_{node_id}_activities.csv"
                    )
                    with open(
                        activity_csv_path, "w", newline="", encoding="utf-8"
                    ) as f:
                        # Get all possible fieldnames from all activities
                        all_fieldnames = set()
                        for activity in activities:
                            all_fieldnames.update(activity.keys())

                        writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                        writer.writeheader()
                        writer.writerows(activities)
            print(f"Node activities exported for {len(self.node_activities)} nodes")

        # 6. Export parameter updates
        if self.parameter_updates:
            param_update_csv_path = os.path.join(
                self.output_dir, f"{prefix}_parameter_updates.csv"
            )
            with open(param_update_csv_path, "w", newline="", encoding="utf-8") as f:
                if self.parameter_updates:
                    writer = csv.DictWriter(
                        f, fieldnames=self.parameter_updates[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(self.parameter_updates)
            print(f"Parameter updates exported to {param_update_csv_path}")

        # 7. Export enhanced network structure information
        if self.network_structure or self.topology:
            # Basic topology CSV
            if self.topology:
                topology_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_topology.csv"
                )
                with open(topology_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["node_id", "connected_nodes", "degree"]
                    )
                    writer.writeheader()

                    for node_id, neighbors in self.topology.items():
                        writer.writerow(
                            {
                                "node_id": node_id,
                                "connected_nodes": ",".join(map(str, neighbors)),
                                "degree": len(neighbors),
                            }
                        )
                print(f"Topology exported to {topology_csv_path}")

            # Enhanced network structure CSV
            if self.network_structure:
                network_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_network_structure.csv"
                )
                with open(network_csv_path, "w", newline="", encoding="utf-8") as f:
                    # Create a single-row CSV with network-wide statistics
                    fieldnames = [
                        "topology_type",
                        "num_nodes",
                        "total_edges",
                        "network_density",
                        "avg_degree",
                        "max_degree",
                        "min_degree",
                        "total_cpu_cores",
                        "total_memory_gb",
                        "total_gpus",
                        "avg_bandwidth_mbps",
                        "organizational_entities",
                        "geographic_coverage",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    row = {
                        "topology_type": self.network_structure["topology_type"],
                        "num_nodes": self.network_structure["num_nodes"],
                        "total_edges": self.network_structure["total_edges"],
                        "network_density": self.network_structure["network_density"],
                        "avg_degree": self.network_structure["node_degrees"]["average"],
                        "max_degree": self.network_structure["node_degrees"]["maximum"],
                        "min_degree": self.network_structure["node_degrees"]["minimum"],
                        "total_cpu_cores": self.network_structure["resource_totals"][
                            "cpu_cores"
                        ],
                        "total_memory_gb": self.network_structure["resource_totals"][
                            "memory_gb"
                        ],
                        "total_gpus": self.network_structure["resource_totals"][
                            "gpu_count"
                        ],
                        "avg_bandwidth_mbps": self.network_structure["resource_totals"][
                            "avg_bandwidth_mbps"
                        ],
                        "organizational_entities": self.network_structure[
                            "organizational_entities"
                        ],
                        "geographic_coverage": self.network_structure[
                            "geographic_coverage"
                        ],
                    }
                    writer.writerow(row)
                print(f"Network structure exported to {network_csv_path}")

            # Adjacency matrix CSV
            if self.adjacency_matrix:
                adj_matrix_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_adjacency_matrix.csv"
                )
                with open(adj_matrix_csv_path, "w", newline="", encoding="utf-8") as f:
                    fieldnames = ["node_id"] + [
                        f"connects_to_{i}" for i in range(len(self.adjacency_matrix))
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for i, adj_row in enumerate(self.adjacency_matrix):
                        csv_row: Dict[str, Any] = {"node_id": i}
                        for j, connection in enumerate(adj_row):
                            csv_row[f"connects_to_{j}"] = connection
                        writer.writerow(csv_row)
                print(f"Adjacency matrix exported to {adj_matrix_csv_path}")

            # Node attributes CSV
            if self.node_attributes:
                node_attrs_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_node_attributes.csv"
                )
                with open(node_attrs_csv_path, "w", newline="", encoding="utf-8") as f:
                    # Get all possible attribute keys
                    all_attr_keys: set[str] = set()
                    for attrs in self.node_attributes.values():
                        all_attr_keys.update(attrs.keys())

                    fieldnames = ["node_id", "node_identifier"] + sorted(all_attr_keys)
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for node_id, attrs in self.node_attributes.items():
                        row = {
                            "node_id": node_id,
                            "node_identifier": self.node_identifiers[node_id]
                            if node_id < len(self.node_identifiers)
                            else f"node_{node_id}",
                        }
                        row.update(attrs)
                        writer.writerow(row)
                print(f"Node attributes exported to {node_attrs_csv_path}")

            # Edge weights CSV
            if self.edge_weights:
                edge_weights_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_edge_weights.csv"
                )
                with open(
                    edge_weights_csv_path, "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=["edge", "weight"])
                    writer.writeheader()

                    for edge, weight in self.edge_weights.items():
                        writer.writerow({"edge": edge, "weight": weight})
                print(f"Edge weights exported to {edge_weights_csv_path}")

            # Geographic information CSV
            if self.geographic_info:
                geo_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_geographic_info.csv"
                )
                with open(geo_csv_path, "w", newline="", encoding="utf-8") as f:
                    # Get all possible geographic keys
                    all_geo_keys: set[str] = set()
                    for geo_data in self.geographic_info.values():
                        all_geo_keys.update(geo_data.keys())

                    fieldnames = ["node_id"] + sorted(all_geo_keys)
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for node_id, geo_data in self.geographic_info.items():
                        row = {"node_id": node_id}
                        row.update(geo_data)
                        writer.writerow(row)
                print(f"Geographic information exported to {geo_csv_path}")

            # Organizational hierarchy CSV
            if self.organizational_hierarchy:
                org_csv_path = os.path.join(
                    self.output_dir, f"{prefix}_organizational_hierarchy.csv"
                )
                with open(org_csv_path, "w", newline="", encoding="utf-8") as f:
                    # Get all possible organizational keys
                    all_org_keys: set[str] = set()
                    for org_data in self.organizational_hierarchy.values():
                        all_org_keys.update(org_data.keys())

                    fieldnames = ["node_id"] + sorted(all_org_keys)
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for node_id, org_data in self.organizational_hierarchy.items():
                        row = {"node_id": node_id}
                        row.update(org_data)
                        writer.writerow(row)
                print(f"Organizational hierarchy exported to {org_csv_path}")

        # 8. Export attack events
        if self.attack_events:
            attack_csv_path = os.path.join(self.output_dir, f"{prefix}_attack_events.csv")
            with open(attack_csv_path, "w", newline="", encoding="utf-8") as f:
                # Get all possible field names from all attack events
                all_fieldnames: set[str] = set()
                for event in self.attack_events:
                    all_fieldnames.update(event.keys())
                
                fieldnames = sorted(all_fieldnames)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.attack_events)
            print(f"Attack events exported to {attack_csv_path}")

        # 9. Export attack intensity history
        if self.attack_intensity_history:
            intensity_csv_path = os.path.join(self.output_dir, f"{prefix}_attack_intensity.csv")
            with open(intensity_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["node_id", "round_num", "attack_intensity"])
                writer.writeheader()
                
                for node_id, intensities in self.attack_intensity_history.items():
                    for round_num, intensity in enumerate(intensities, 1):
                        writer.writerow({
                            "node_id": node_id,
                            "round_num": round_num,
                            "attack_intensity": intensity
                        })
            print(f"Attack intensity history exported to {intensity_csv_path}")

        # 10. Export attack detection events
        if self.attack_detection_events:
            detection_csv_path = os.path.join(self.output_dir, f"{prefix}_attack_detection.csv")
            with open(detection_csv_path, "w", newline="", encoding="utf-8") as f:
                # Get all possible field names from all detection events
                all_fieldnames: set[str] = set()
                for event in self.attack_detection_events:
                    all_fieldnames.update(event.keys())
                
                fieldnames = sorted(all_fieldnames)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.attack_detection_events)
            print(f"Attack detection events exported to {detection_csv_path}")

        # 11. Export malicious nodes summary
        if self.malicious_nodes:
            malicious_csv_path = os.path.join(self.output_dir, f"{prefix}_malicious_nodes.csv")
            with open(malicious_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["node_id", "is_malicious"])
                writer.writeheader()
                
                # Write all nodes with their malicious status
                all_nodes = set(range(len(self.node_identifiers))) if self.node_identifiers else self.malicious_nodes
                for node_id in all_nodes:
                    writer.writerow({
                        "node_id": node_id,
                        "is_malicious": node_id in self.malicious_nodes
                    })
            print(f"Malicious nodes summary exported to {malicious_csv_path}")

        # 12. Export trust signals events to separate CSV
        if hasattr(self, 'trust_signals_events') and self.trust_signals_events:
            trust_signals_csv_path = os.path.join(self.output_dir, "trust_signals.csv")
            with open(trust_signals_csv_path, "w", newline="", encoding="utf-8") as f:
                # Get all possible field names from all trust signals events
                all_fieldnames: set[str] = set()
                for event in self.trust_signals_events:
                    all_fieldnames.update(event.keys())
                
                fieldnames = sorted(all_fieldnames)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.trust_signals_events)
            print(f"Trust signals exported to {trust_signals_csv_path}")

        # 13. Export aggregation weights data  
        if hasattr(self, 'aggregation_weights_events') and self.aggregation_weights_events:
            weights_csv_path = os.path.join(self.output_dir, "aggregation_weights.csv")
            
            with open(weights_csv_path, "w", newline="", encoding="utf-8") as f:
                # Get all possible field names
                all_fieldnames: set[str] = set()
                for event in self.aggregation_weights_events:
                    all_fieldnames.update(event.keys())
                
                fieldnames = sorted(all_fieldnames)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.aggregation_weights_events)
            print(f"Aggregation weights exported to {weights_csv_path}")

        # 14. Export all trust-related events to consolidated trust_events.csv
        if self.trust_events or (hasattr(self, 'trust_signals_events') and self.trust_signals_events):
            trust_events_csv_path = os.path.join(self.output_dir, "trust_events.csv")
            all_trust_events = []
            
            # Add trust score events
            if self.trust_events:
                all_trust_events.extend(self.trust_events)
            
            # Add trust signals events 
            if hasattr(self, 'trust_signals_events') and self.trust_signals_events:
                all_trust_events.extend(self.trust_signals_events)
            
            # Add fingerprint events
            if hasattr(self, 'fingerprint_events') and self.fingerprint_events:
                all_trust_events.extend(self.fingerprint_events)
            
            # Add aggregation weights events
            if hasattr(self, 'aggregation_weights_events') and self.aggregation_weights_events:
                all_trust_events.extend(self.aggregation_weights_events)
            
            with open(trust_events_csv_path, "w", newline="", encoding="utf-8") as f:
                # Get all possible field names from all events
                all_fieldnames: set[str] = set()
                for event in all_trust_events:
                    all_fieldnames.update(event.keys())
                
                fieldnames = sorted(all_fieldnames)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_trust_events)
            print(f"All trust events exported to {trust_events_csv_path}")

        print(f"All training data (including attack and trust data) exported to {self.output_dir}")

    def render_training_animation(
        self, filename: str = "training_animation.mp4", fps: int = 1
    ) -> None:
        """
        Creates an animation of the training process and exports data to CSV.

        Args:
            filename: Output filename for the animation
            fps: Frames per second
        """
        # Export CSV data first
        self.export_to_csv()

        if not self.frames:
            print("No frames captured. Run training first.")
            return

        # Create figure for the animation
        fig = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[2, 1])
        ax_network = fig.add_subplot(
            gs[:, 0]
        )  # Network visualization (left side, full height)
        ax_params = fig.add_subplot(gs[0, 1])  # Parameter convergence (top right)
        ax_metrics = fig.add_subplot(gs[1, 1])  # Metrics (bottom right)

        # Create metrics twin axis once
        ax_metrics_twin = ax_metrics.twinx()
        ax_metrics_twin.set_ylabel("Loss", color="r")

        # Base graph structure from the topology
        G = nx.Graph()

        if self.topology:
            for node in self.topology.keys():
                G.add_node(node)

            for node, neighbors in self.topology.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
        else:
            # If no topology was set, create a default graph from the first frame
            num_nodes = self.frames[0].get("num_nodes", 5)
            for i in range(num_nodes):
                G.add_node(i)

        # Calculate layout once for consistency
        pos = nx.spring_layout(G, seed=42)

        # Create text objects for titles and descriptions
        ax_network.set_title("Network State", fontsize=14)
        ax_params.set_title("Parameter Convergence", fontsize=12)
        ax_metrics.set_title("Training Metrics", fontsize=12)
        description_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12)

        def update_frame(frame_idx: int) -> List[Artist]:
            """Update animation frame.

            Args:
                frame_idx: Index of the current frame

            Returns:
                List of artists that were updated
            """
            if frame_idx >= len(self.frames):
                return []

            frame = self.frames[frame_idx]
            current_round = frame["round"]

            # Clear previous plots
            ax_network.clear()
            ax_params.clear()
            ax_metrics.clear()
            ax_metrics_twin.clear()

            # Update titles
            ax_network.set_title("Network State", fontsize=14)
            ax_params.set_title("Parameter Convergence", fontsize=12)
            ax_metrics.set_title("Training Metrics", fontsize=12)
            description_text.set_text(self.frame_descriptions[frame_idx])

            # Draw network state
            # Regular edges
            nx.draw_networkx_edges(
                G, pos, ax=ax_network, edge_color="gray", alpha=0.3, width=1.0
            )

            # Active edges (data transfer)
            if frame.get("active_edges"):
                edge_list = frame["active_edges"]
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax_network,
                    edgelist=edge_list,
                    edge_color="red",
                    width=2.0,
                )

            # Node colors based on activity and malicious status
            node_colors = []
            node_sizes = []

            for node in G.nodes():
                # Check if node is malicious
                if node in frame.get("malicious_nodes", []):
                    if node in frame.get("active_nodes", []):
                        node_colors.append("darkred")  # Active malicious node
                        node_sizes.append(1400)
                    else:
                        node_colors.append("red")  # Inactive malicious node
                        node_sizes.append(1000)
                elif node in frame.get("active_nodes", []):
                    node_colors.append("orange")  # Active benign node
                    node_sizes.append(1200)
                else:
                    node_colors.append("skyblue")  # Inactive benign node
                    node_sizes.append(800)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, ax=ax_network, node_color=node_colors, node_size=node_sizes
            )

            # Node labels
            nx.draw_networkx_labels(G, pos, ax=ax_network, font_size=10)

            # Add topology type text
            topology_text = f"Topology: {frame.get('topology_type', 'Unknown')}"
            if "strategy_name" in frame:
                topology_text += f"\nStrategy: {frame['strategy_name']}"
            ax_network.text(
                0.02,
                0.02,
                topology_text,
                transform=ax_network.transAxes,
                fontsize=10,
                verticalalignment="bottom",
            )

            # Plot parameter convergence - show all visible parameters up to current round
            added_legend_entries = False
            for node, history in self.parameter_history.items():
                if len(history) > 0:  # Show all nodes with parameter history
                    # Only show data up to current round
                    # Use the round number rather than frame index for x-axis
                    visible_data = []
                    for i in range(min(current_round, len(history))):
                        visible_data.append(history[i])

                    if visible_data:
                        # Create a simple list of integers for the x-axis
                        rounds = []
                        for r in range(1, len(visible_data) + 1):
                            rounds.append(r)
                        ax_params.plot(rounds, visible_data, label=f"Node {node}")
                        added_legend_entries = True

            ax_params.set_xlabel("Round")
            ax_params.set_ylabel("Parameter Norm")

            # Set x-axis ticks to be integers
            if current_round > 0:
                ax_params.set_xticks(range(1, current_round + 1))

            # Only add legend if we have data
            if added_legend_entries:
                ax_params.legend(loc="upper right", fontsize=8)

            # Plot metrics - show metrics up to current round
            rounds = []
            accuracy_data = []
            loss_data = []

            # Find all available round numbers up to the current round
            available_rounds = sorted([r for r in range(1, current_round + 1)])

            # For each round number, get the latest metrics value
            last_acc = 0.0
            last_loss = 0.0

            for round_num in available_rounds:
                rounds.append(round_num)

                # Get accuracy for this round or use the last known value
                if round_num in self.accuracy_history:
                    last_acc = self.accuracy_history[round_num]
                accuracy_data.append(last_acc)

                # Get loss for this round or use the last known value
                if round_num in self.loss_history:
                    last_loss = self.loss_history[round_num]
                loss_data.append(last_loss)

            # Plot metrics if we have data
            legend_handles = []
            legend_labels = []

            if rounds and accuracy_data:
                # Plot accuracy on left axis
                (acc_line,) = ax_metrics.plot(
                    rounds, accuracy_data, "g-o", label="Accuracy"
                )
                ax_metrics.set_ylabel("Accuracy", color="g")
                legend_handles.append(acc_line)
                legend_labels.append("Accuracy")

            if rounds and loss_data:
                # Plot loss on right axis
                (loss_line,) = ax_metrics_twin.plot(
                    rounds, loss_data, "r-o", label="Loss"
                )
                ax_metrics_twin.set_ylabel("Loss", color="r")
                legend_handles.append(loss_line)
                legend_labels.append("Loss")

            ax_metrics.set_xlabel("Round")

            # Set x-axis ticks to be integers
            if rounds:
                ax_metrics.set_xticks(rounds)

            # Add legend if we have data
            if legend_handles:
                ax_metrics.legend(legend_handles, legend_labels, loc="upper left")

            # Set proper limits
            ax_network.set_xlim((-1.2, 1.2))
            ax_network.set_ylim((-1.2, 1.2))
            ax_network.axis("off")

            # Return updated artists (required for animation)
            return [
                ax_network,
                ax_params,
                ax_metrics,
                ax_metrics_twin,
                description_text,
            ]

        # Create animation with proper typing
        ani = animation.FuncAnimation(
            fig=fig,
            func=update_frame,
            frames=len(self.frames),
            interval=1000 / fps,
            blit=False,
        )

        # Save animation
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Murmura Framework"), bitrate=5000
        )
        output_path = os.path.join(self.output_dir, filename)
        ani.save(output_path, writer=writer)

        plt.close(fig)
        print(f"Animation saved to {output_path}")

    def render_frame_sequence(self, prefix: str = "training_step") -> None:
        """
        Renders each frame as a separate image file and exports CSV data.

        Args:
            prefix: Prefix for output filenames
        """
        # Export CSV data first
        self.export_to_csv()

        for i, frame in enumerate(self.frames):
            # Create a figure with network and metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Create graph
            G = nx.Graph()

            if self.topology:
                for node in self.topology.keys():
                    G.add_node(node)

                for node, neighbors in self.topology.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
            else:
                # If no topology was set, create a default graph
                num_nodes = frame.get("num_nodes", 5)
                for j in range(num_nodes):
                    G.add_node(j)

            # Calculate layout
            pos = nx.spring_layout(G, seed=42)

            # Regular edges
            nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="gray", alpha=0.3)

            # Active edges
            if frame.get("active_edges"):
                edge_list = frame["active_edges"]
                nx.draw_networkx_edges(
                    G, pos, ax=ax1, edgelist=edge_list, edge_color="red", width=2.0
                )

            # Nodes
            node_colors = []
            node_sizes = []

            for node in G.nodes():
                # Check if node is malicious
                if node in frame.get("malicious_nodes", []):
                    if node in frame.get("active_nodes", []):
                        node_colors.append("darkred")  # Active malicious node
                        node_sizes.append(1400)
                    else:
                        node_colors.append("red")  # Inactive malicious node
                        node_sizes.append(1000)
                elif node in frame.get("active_nodes", []):
                    node_colors.append("orange")  # Active benign node
                    node_sizes.append(1200)
                else:
                    node_colors.append("skyblue")  # Inactive benign node
                    node_sizes.append(800)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, ax=ax1, node_color=node_colors, node_size=node_sizes
            )

            # Labels
            nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10)

            # Title and description
            description = (
                self.frame_descriptions[i] if i < len(self.frame_descriptions) else ""
            )
            ax1.set_title(description)

            # Add topology type
            topology_text = f"Topology: {frame.get('topology_type', 'Unknown')}"
            if "strategy_name" in frame:
                topology_text += f"\nStrategy: {frame['strategy_name']}"
            ax1.text(
                0.02,
                0.02,
                topology_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="bottom",
            )

            # Plot metrics if available - accessing round_metrics for consistent data
            has_metrics = False

            if frame["all_metrics"]:
                acc_data = []
                loss_data = []
                rounds = []

                for round_num in sorted(frame["all_metrics"].keys()):
                    if round_num <= frame["round"]:
                        metrics = frame["all_metrics"][round_num]
                        rounds.append(round_num)

                        if "accuracy" in metrics:
                            acc_data.append(metrics["accuracy"])
                        if "loss" in metrics:
                            loss_data.append(metrics["loss"])

                if rounds:
                    has_metrics = True
                    # Add accuracy plot if available
                    if acc_data:
                        ax2.plot(rounds, acc_data, "g-o", label="Accuracy")
                        ax2.set_ylabel("Accuracy", color="g")

                    # Add loss plot with twin axis if available
                    if loss_data:
                        ax2_twin = ax2.twinx()
                        ax2_twin.plot(rounds, loss_data, "r-o", label="Loss")
                        ax2_twin.set_ylabel("Loss", color="r")

                    # Add legend
                    handles, labels = [], []
                    if acc_data:
                        h1, l1 = ax2.get_legend_handles_labels()
                        handles.extend(h1)
                        labels.extend(l1)
                    if loss_data and "ax2_twin" in locals():
                        h2, l2 = locals()["ax2_twin"].get_legend_handles_labels()
                        handles.extend(h2)
                        labels.extend(l2)

                    if handles:
                        ax2.legend(handles, labels, loc="upper left")

            if has_metrics:
                ax2.set_title("Training Metrics")
                ax2.set_xlabel("Round")
            else:
                ax2.set_title("No metrics available")
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Value")

            # Adjust layout
            ax1.set_xlim([-1.2, 1.2])
            ax1.set_ylim([-1.2, 1.2])
            ax1.axis("off")

            plt.tight_layout()

            # Save the figure
            filename = f"{prefix}_{i:03d}.png"
            plt.savefig(
                os.path.join(self.output_dir, filename), dpi=100, bbox_inches="tight"
            )
            plt.close(fig)

        print(f"Frame sequence saved to {self.output_dir} ({len(self.frames)} frames)")

    def render_summary_plot(self, filename: str = "training_summary.png") -> None:
        """
        Creates a summary plot showing key metrics and parameter convergence, and exports CSV data.

        Args:
            filename: Output filename for the summary plot
        """
        # Export CSV data first
        self.export_to_csv()

        if not self.frames:
            print("No frames captured. Run training first.")
            return

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Network topology (top left)
        ax_network = axes[0, 0]
        G = nx.Graph()

        if self.topology:
            for node in self.topology.keys():
                G.add_node(node)

            for node, neighbors in self.topology.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)

            # Draw network
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(
                G, pos, ax=ax_network, node_color="skyblue", node_size=800, font_size=10
            )

            ax_network.set_title(f"Network Topology: {self.network_type}")
        else:
            ax_network.text(
                0.5, 0.5, "No topology information", ha="center", va="center"
            )
            ax_network.set_title("Network Topology")

        ax_network.axis("off")

        # 2. Parameter convergence (top right)
        ax_params = axes[0, 1]

        if self.parameter_history:
            for node, history in self.parameter_history.items():
                if history:  # Only show nodes with data
                    rounds = list(range(1, len(history) + 1))
                    ax_params.plot(rounds, history, label=f"Node {node}")

            ax_params.set_xlabel("Round")
            ax_params.set_ylabel("Parameter Norm")
            ax_params.set_title("Parameter Convergence")

            # Only add legend if we have data
            if any(len(h) > 0 for h in self.parameter_history.values()):
                ax_params.legend(loc="upper right", fontsize=8)
        else:
            ax_params.text(0.5, 0.5, "No parameter data", ha="center", va="center")
            ax_params.set_title("Parameter Convergence")

        # 3. Accuracy over time (bottom left)
        ax_acc = axes[1, 0]

        if self.accuracy_history:
            # Convert dictionary to sorted lists for plotting
            rounds = sorted(self.accuracy_history.keys())
            accuracy_values = [self.accuracy_history[r] for r in rounds]

            if rounds and accuracy_values:
                ax_acc.plot(rounds, accuracy_values, "g-o")
                ax_acc.set_xlabel("Round")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_title("Model Accuracy")

                # Add final accuracy text
                final_acc = accuracy_values[-1]
                ax_acc.text(
                    0.5,
                    0.1,
                    f"Final accuracy: {final_acc:.4f}",
                    transform=ax_acc.transAxes,
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.8),
                )
            else:
                ax_acc.text(0.5, 0.5, "No accuracy data", ha="center", va="center")
                ax_acc.set_title("Model Accuracy")
        else:
            ax_acc.text(0.5, 0.5, "No accuracy data", ha="center", va="center")
            ax_acc.set_title("Model Accuracy")

        # 4. Loss over time (bottom right)
        ax_loss = axes[1, 1]

        if self.loss_history:
            # Convert dictionary to sorted lists for plotting
            rounds = sorted(self.loss_history.keys())
            loss_values = [self.loss_history[r] for r in rounds]

            if rounds and loss_values:
                ax_loss.plot(rounds, loss_values, "r-o")
                ax_loss.set_xlabel("Round")
                ax_loss.set_ylabel("Loss")
                ax_loss.set_title("Model Loss")

                # Add final loss text
                final_loss = loss_values[-1]
                ax_loss.text(
                    0.5,
                    0.1,
                    f"Final loss: {final_loss:.4f}",
                    transform=ax_loss.transAxes,
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.8),
                )
            else:
                ax_loss.text(0.5, 0.5, "No loss data", ha="center", va="center")
                ax_loss.set_title("Model Loss")
        else:
            ax_loss.text(0.5, 0.5, "No loss data", ha="center", va="center")
            ax_loss.set_title("Model Loss")

        # Add overall title
        plt.suptitle("Training Summary", fontsize=16)

        # Save figure - fix the rect parameter to be a tuple
        rect = (0, 0, 1, 0.97)  # left, bottom, right, top
        plt.tight_layout(rect=rect)
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=100, bbox_inches="tight"
        )
        plt.close(fig)

        print(f"Summary plot saved to {os.path.join(self.output_dir, filename)}")
        
    def _handle_trust_anomaly(self, event, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle trust anomaly events."""
        trust_anomaly_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "observer_node": event.node_id,
            "suspected_neighbor": event.suspected_neighbor,
            "anomaly_type": event.anomaly_type,
            "anomaly_score": event.anomaly_score,
            "evidence": event.evidence,
            "trust_scores": event.trust_scores.copy()
        }
        
        self.trust_anomalies.append(trust_anomaly_record)
        self.trust_events.append(trust_anomaly_record)
        
        frame["trust_anomaly"] = trust_anomaly_record
        
        event_data.update({
            "observer_node": event.node_id,
            "suspected_neighbor": event.suspected_neighbor,
            "anomaly_type": event.anomaly_type,
            "anomaly_score": event.anomaly_score,
            "evidence": str(event.evidence),
            "trust_scores": str(event.trust_scores)
        })
        
    def _handle_trust_score_update(self, event, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle trust score update events."""
        observer_node = event.node_id
        round_num = event.round_num
        
        # Store trust scores history
        if observer_node not in self.trust_scores_history:
            self.trust_scores_history[observer_node] = {}
        self.trust_scores_history[observer_node][round_num] = event.trust_scores.copy()
        
        # Store trust score changes
        if observer_node not in self.trust_score_changes:
            self.trust_score_changes[observer_node] = {}
        self.trust_score_changes[observer_node][round_num] = event.score_changes.copy()
        
        # Create trust score event record
        trust_score_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "observer_node": event.node_id,
            "trust_scores": event.trust_scores.copy(),
            "score_changes": event.score_changes.copy(),
            "detection_method": event.detection_method,
            "debug_data": getattr(event, 'debug_data', {})
        }
        
        self.trust_events.append(trust_score_record)
        
        frame["trust_score_update"] = trust_score_record
        
        event_data.update({
            "observer_node": event.node_id,
            "trust_scores": str(event.trust_scores),
            "score_changes": str(event.score_changes),
            "detection_method": event.detection_method
        })
        
    def _handle_fingerprint_event(self, event, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle gradient fingerprint events."""
        node_id = event.node_id
        round_num = event.round_num
        
        # Store fingerprint data history
        if not hasattr(self, 'fingerprint_history'):
            self.fingerprint_history = {}
        if node_id not in self.fingerprint_history:
            self.fingerprint_history[node_id] = {}
        self.fingerprint_history[node_id][round_num] = event.fingerprint_data.copy()
        
        # Create fingerprint event record for CSV export
        fingerprint_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "node_id": event.node_id,
            "fingerprint_data": event.fingerprint_data.copy()
        }
        
        # Store in events list (create if needed)
        if not hasattr(self, 'fingerprint_events'):
            self.fingerprint_events = []
        self.fingerprint_events.append(fingerprint_record)
        
        # Add to frame data
        frame["fingerprint_event"] = fingerprint_record
        
        # Update event data for CSV export
        event_data.update({
            "node_id": event.node_id,
            "fingerprint_data": str(event.fingerprint_data)
        })
    
    def _handle_trust_signals_event(self, event, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle trust signals events with detailed fingerprint comparisons."""
        observer_node = event.observer_node
        target_node = event.target_node
        round_num = event.round_num
        
        # Store trust signals history
        if not hasattr(self, 'trust_signals_history'):
            self.trust_signals_history = {}
        if observer_node not in self.trust_signals_history:
            self.trust_signals_history[observer_node] = {}
        if round_num not in self.trust_signals_history[observer_node]:
            self.trust_signals_history[observer_node][round_num] = {}
        self.trust_signals_history[observer_node][round_num][target_node] = event.trust_signals.copy()
        
        # Create trust signals event record for CSV export
        trust_signals_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "observer_node": event.observer_node,
            "target_node": event.target_node,
            **event.trust_signals  # Expand all signal values as columns
        }
        
        # Add fingerprint comparison data if available
        if event.fingerprint_comparison:
            if "target_fingerprint" in event.fingerprint_comparison:
                for key, value in event.fingerprint_comparison["target_fingerprint"].items():
                    trust_signals_record[f"target_{key}"] = value
            if "own_fingerprint" in event.fingerprint_comparison:
                for key, value in event.fingerprint_comparison["own_fingerprint"].items():
                    trust_signals_record[f"own_{key}"] = value
        
        # Store in events list (create if needed)
        if not hasattr(self, 'trust_signals_events'):
            self.trust_signals_events = []
        self.trust_signals_events.append(trust_signals_record)
        
        # Add to frame data
        frame["trust_signals_event"] = trust_signals_record
        
        # Update event data for CSV export
        event_data.update(trust_signals_record)

    def _handle_aggregation_weights_event(self, event, frame: Dict[str, Any], event_data: Dict[str, Any], description: str) -> None:
        """Handle aggregation weights events and store weight data."""
        
        # Create aggregation weights record
        weights_record = {
            "timestamp": event.timestamp,
            "round_num": event.round_num,
            "observer_node": event.observer_node,
            "aggregation_method": event.aggregation_method,
        }
        
        # Add influence weights data
        for node_id, weight in event.influence_weights.items():
            weights_record[f"influence_weight_{node_id}"] = weight
            
        # Add trust scores data  
        for node_id, score in event.trust_scores.items():
            weights_record[f"trust_score_{node_id}"] = score
        
        # Store in influence weights history
        observer_node = event.observer_node
        round_num = event.round_num
        
        if observer_node not in self.influence_weights_history:
            self.influence_weights_history[observer_node] = {}
        
        self.influence_weights_history[observer_node][round_num] = event.influence_weights.copy()
        
        # Store in events list (create if needed)
        if not hasattr(self, 'aggregation_weights_events'):
            self.aggregation_weights_events = []
        self.aggregation_weights_events.append(weights_record)
        
        # Add to frame data
        frame["aggregation_weights_event"] = weights_record
        
        # Update event data for CSV export
        event_data.update(weights_record)
