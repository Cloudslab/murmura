from statistics import mean
from typing import Dict, Any

import numpy as np
import ray

from murmura.orchestration.learning_process.learning_process import LearningProcess
from murmura.visualization.training_event import (
    EvaluationEvent,
    ModelUpdateEvent,
    AggregationEvent,
    ParameterTransferEvent,
    LocalTrainingEvent,
)


class FederatedLearningProcess(LearningProcess):
    """
    Concrete implementation of the LearningProcess for federated learning with multi-node support.
    """

    def execute(self) -> Dict[str, Any]:
        """
        Execute the federated learning process.

        :return: Results of the federated learning process
        """
        if not self.cluster_manager:
            raise ValueError("Learning process not initialized. Call initialize first.")

        if not self.cluster_manager.topology_manager:
            raise ValueError("Topology manager not set. Call initialize first.")

        # Get configuration parameters using the new helper method
        rounds = self.get_config_value("rounds", 5)
        epochs = self.get_config_value("epochs", 1)
        batch_size = self.get_config_value("batch_size", 32)
        test_split = self.get_config_value("test_split", "test")

        # Enhanced logging with cluster context
        self.log_training_progress(0, {"status": "starting", "rounds": rounds})

        # Prepare test data for global evaluation
        test_dataset = self.dataset.get_split(test_split)
        feature_columns = self.get_config_value("feature_columns", ["image"])
        label_column = self.get_config_value("label_column", "label")

        test_features = np.array(test_dataset[feature_columns[0]])
        test_labels = np.array(test_dataset[label_column])

        # Evaluate initial model
        initial_metrics = self.model.evaluate(test_features, test_labels)
        self.logger.info(
            f"Initial Test Accuracy: {initial_metrics['accuracy'] * 100:.2f}%"
        )

        # Emit evaluation event for visualization
        self.training_monitor.emit_event(
            EvaluationEvent(round_num=0, metrics=initial_metrics)
        )

        round_metrics = []

        # Determine hub node for star topology
        hub_index = None
        if self.cluster_manager.topology_manager.config.topology_type.value == "star":
            hub_index = self.cluster_manager.topology_manager.config.hub_index

        # Training rounds with enhanced monitoring
        for round_num in range(1, rounds + 1):
            self.logger.info(f"--- Round {round_num}/{rounds} ---")

            # Monitor resource usage if enabled
            monitor_resources = self.get_config_value("monitor_resources", False)
            if monitor_resources:
                resource_usage = self.monitor_resource_usage()
                self.logger.debug(
                    f"Round {round_num} resource usage: {resource_usage.get('resource_utilization', {})}"
                )

            # 1. Local Training with enhanced logging
            self.logger.info(f"Training on clients for {epochs} epochs...")

            # Emit local training event with epoch info
            self.training_monitor.emit_event(
                LocalTrainingEvent(
                    round_num=round_num,
                    active_nodes=list(range(len(self.cluster_manager.actors))),
                    metrics={},
                    total_epochs=epochs,
                )
            )

            # Training with enhanced progress logging
            self.logger.info(
                f"Local training progress (each client trains for {epochs} epochs):"
            )
            train_metrics = self.cluster_manager.train_models(
                epochs=epochs, batch_size=batch_size, verbose=True
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
                },
            )

            # 2. Parameter Aggregation
            self.logger.info("Aggregating model parameters...")

            # Collect parameters for visualization
            node_params = {}
            for i, actor in enumerate(self.cluster_manager.actors):
                params = ray.get(actor.get_model_parameters.remote())
                node_params[i] = params

            # Create parameter summaries for visualization
            param_summaries = self._create_parameter_summaries(node_params)

            # Emit parameter transfer event
            if hub_index is not None:
                # Star topology: params flow from nodes to hub
                self.training_monitor.emit_event(
                    ParameterTransferEvent(
                        round_num=round_num,
                        source_nodes=[
                            i
                            for i in range(len(self.cluster_manager.actors))
                            if i != hub_index
                        ],
                        target_nodes=[hub_index],
                        param_summary=param_summaries,
                    )
                )
            else:
                # For other topologies
                for (
                    node,
                    neighbors,
                ) in self.cluster_manager.topology_manager.adjacency_list.items():
                    if neighbors:  # Only emit if the node has neighbors
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

            # Emit aggregation event
            self.training_monitor.emit_event(
                AggregationEvent(
                    round_num=round_num,
                    participating_nodes=list(range(len(self.cluster_manager.actors))),
                    aggregator_node=hub_index,
                    strategy_name=self.cluster_manager.aggregation_strategy.__class__.__name__,
                )
            )

            # Get client data sizes for weighted aggregation (if needed)
            split = self.get_config_value("split", "train")
            partitions = list(self.dataset.get_partitions(split).values())
            client_data_sizes = [len(partition) for partition in partitions]

            # Use data size as weights for aggregation
            weights = [float(size) for size in client_data_sizes]

            # Aggregate parameters
            aggregated_params = self.cluster_manager.aggregate_model_parameters(
                weights=weights
            )

            # 3. Model Update
            # Update global model
            self.model.set_parameters(aggregated_params)

            # Distribute updated model to clients
            self.cluster_manager.update_models(aggregated_params)

            # Calculate parameter convergence
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

            # 4. Evaluation
            # Evaluate global model on test set
            test_metrics = self.model.evaluate(test_features, test_labels)
            self.logger.info(f"Global Model Test Loss: {test_metrics['loss']:.4f}")
            self.logger.info(
                f"Global Model Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%"
            )

            # Emit evaluation event
            self.training_monitor.emit_event(
                EvaluationEvent(round_num=round_num, metrics=test_metrics)
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

        # Final evaluation with cluster context
        final_metrics = self.model.evaluate(test_features, test_labels)
        improvement = final_metrics["accuracy"] - initial_metrics["accuracy"]

        # Enhanced final logging
        cluster_summary = self.get_cluster_summary()
        self.logger.info("=== Final Model Evaluation ===")
        self.logger.info(
            f"Cluster type: {cluster_summary.get('cluster_type', 'unknown')}"
        )
        self.logger.info(f"Final Test Accuracy: {final_metrics['accuracy'] * 100:.2f}%")
        self.logger.info(f"Accuracy Improvement: {improvement * 100:.2f}%")

        # Return enhanced results
        results = {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "accuracy_improvement": improvement,
            "round_metrics": round_metrics,
        }

        # Add cluster information if available
        if cluster_summary:
            results["cluster_info"] = cluster_summary

        return results
