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


class DecentralizedLearningProcess(LearningProcess):
    """
    Implementation of a decentralized learning process with generic data handling.
    """

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

        # Prepare test data for global evaluation
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

        round_metrics = []

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
                params = ray.get(actor.get_model_parameters.remote(), timeout=1800)
                node_params[i] = params

            # Create parameter summaries for visualization
            param_summaries = self._create_parameter_summaries(node_params)

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

            # Perform topology-aware aggregation
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

        # Final evaluation
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

        # Return results
        results = {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "accuracy_improvement": improvement,
            "round_metrics": round_metrics,
            "topology": topology_info,
        }

        if cluster_summary:
            results["cluster_info"] = cluster_summary

        return results
