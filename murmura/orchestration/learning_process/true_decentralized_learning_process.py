"""
True decentralized learning process with proper gossip-based updates.

This module implements a genuinely decentralized learning process where:
- Each node only communicates with its neighbors
- No global model or aggregation exists
- Each node maintains its own model
- Trust monitoring works on local exchanges
"""

import logging
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple
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


class TrueDecentralizedLearningProcess(LearningProcess):
    """
    Implementation of a truly decentralized learning process.
    
    In this process:
    - Each node trains locally
    - Nodes exchange parameters only with neighbors
    - No global aggregation occurs
    - Each node maintains its own model
    - Evaluation considers the network consensus
    """
    
    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)
        self.node_models = {}  # Track individual node models
        self.consensus_threshold = 0.7  # For determining network consensus
        
    def _prepare_test_data(self, test_dataset, feature_columns, label_column):
        """Prepare test data - same as parent class."""
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
            # Multiple feature columns - let model handle it
            features = []
            for i in range(len(feature_data[0])):
                feature_vec = [feature_data[j][i] for j in range(len(feature_columns))]
                features.append(feature_vec)
            features = np.array(features, dtype=np.float32)
            
        labels = np.array(label_data, dtype=np.int64)
        return features, labels
    
    def _evaluate_network_consensus(
        self, 
        test_features: np.ndarray, 
        test_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the network by testing individual nodes and computing consensus.
        
        Returns:
            Metrics including per-node accuracy and network consensus
        """
        node_metrics = []
        node_predictions = []
        
        # Evaluate each node's model
        for i, actor in enumerate(self.cluster_manager.actors):
            try:
                # Get node's parameters
                node_params = ray.get(actor.get_model_parameters.remote())
                
                # Set parameters in a temporary model
                self.model.set_parameters(node_params)
                
                # Evaluate
                metrics = self.model.evaluate(test_features, test_labels)
                node_metrics.append({
                    'node_id': i,
                    'accuracy': metrics['accuracy'],
                    'loss': metrics['loss']
                })
                
                # Get predictions for consensus analysis
                predictions = self.model.predict(test_features)
                node_predictions.append(predictions)
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate node {i}: {e}")
                node_metrics.append({
                    'node_id': i,
                    'accuracy': 0.0,
                    'loss': float('inf'),
                    'error': str(e)
                })
        
        # Calculate consensus metrics
        consensus_metrics = self._calculate_consensus(
            node_predictions, test_labels, node_metrics
        )
        
        return consensus_metrics
    
    def _calculate_consensus(
        self,
        node_predictions: List[np.ndarray],
        true_labels: np.ndarray,
        node_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate network consensus metrics.
        
        Returns metrics including:
        - Average accuracy across nodes
        - Variance in accuracy
        - Consensus accuracy (majority vote)
        - Node agreement rate
        """
        valid_metrics = [m for m in node_metrics if 'error' not in m]
        
        if not valid_metrics:
            return {
                'average_accuracy': 0.0,
                'min_accuracy': 0.0,
                'max_accuracy': 0.0,
                'accuracy_variance': 0.0,
                'consensus_accuracy': 0.0,
                'node_agreement': 0.0,
                'num_valid_nodes': 0,
                'individual_metrics': node_metrics
            }
        
        # Basic statistics
        accuracies = [m['accuracy'] for m in valid_metrics]
        avg_accuracy = np.mean(accuracies)
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        accuracy_variance = np.var(accuracies)
        
        # Calculate consensus predictions (majority vote)
        if node_predictions:
            # Stack predictions from all nodes
            all_predictions = np.stack([p for p in node_predictions if p is not None])
            
            # Majority vote for each sample
            consensus_predictions = []
            agreement_scores = []
            
            for i in range(len(true_labels)):
                # Get predictions for sample i from all nodes
                sample_predictions = all_predictions[:, i]
                
                # Find most common prediction
                unique, counts = np.unique(sample_predictions, return_counts=True)
                majority_pred = unique[np.argmax(counts)]
                consensus_predictions.append(majority_pred)
                
                # Calculate agreement (fraction of nodes with majority prediction)
                agreement = np.max(counts) / len(sample_predictions)
                agreement_scores.append(agreement)
            
            consensus_predictions = np.array(consensus_predictions)
            consensus_accuracy = np.mean(consensus_predictions == true_labels)
            avg_agreement = np.mean(agreement_scores)
        else:
            consensus_accuracy = avg_accuracy
            avg_agreement = 1.0
        
        return {
            'average_accuracy': float(avg_accuracy),
            'min_accuracy': float(min_accuracy),
            'max_accuracy': float(max_accuracy),
            'accuracy_variance': float(accuracy_variance),
            'consensus_accuracy': float(consensus_accuracy),
            'node_agreement': float(avg_agreement),
            'num_valid_nodes': len(valid_metrics),
            'individual_metrics': node_metrics
        }
    
    def _perform_gossip_exchange(self, round_num: int, adjacency_list: Dict[int, List[int]]) -> None:
        """
        Perform true gossip-based parameter exchange.
        
        Each node:
        1. Exchanges parameters with neighbors
        2. Aggregates received parameters locally
        3. Updates its own model
        """
        self.logger.info("Performing decentralized gossip exchange...")
        
        # Prepare tasks for parallel gossip updates
        gossip_tasks = []
        
        for node_idx, neighbors in adjacency_list.items():
            if neighbors:  # Only if node has neighbors
                # Create gossip task for this node
                task = self.cluster_manager.actors[node_idx].gossip_aggregate.remote(
                    neighbor_indices=neighbors,
                    mixing_parameter=0.5  # How much to weight neighbor updates
                )
                gossip_tasks.append((node_idx, task))
        
        # Wait for all gossip exchanges to complete
        if gossip_tasks:
            try:
                results = ray.get([task for _, task in gossip_tasks], timeout=300)
                
                # Log exchange statistics
                successful_exchanges = sum(1 for r in results if r.get('success', False))
                self.logger.info(
                    f"Completed {successful_exchanges}/{len(gossip_tasks)} gossip exchanges"
                )
                
                # Emit parameter transfer events
                for (node_idx, _), result in zip(gossip_tasks, results):
                    if result.get('success', False):
                        neighbors = adjacency_list.get(node_idx, [])
                        self.training_monitor.emit_event(
                            ParameterTransferEvent(
                                round_num=round_num,
                                source_nodes=neighbors,
                                target_nodes=[node_idx],
                                param_summary={'exchanges': len(neighbors)}
                            )
                        )
                        
            except Exception as e:
                self.logger.error(f"Gossip exchange failed: {e}")
                raise
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the true decentralized learning process.
        """
        if not self.cluster_manager:
            raise ValueError("Learning process not initialized. Call initialize first.")
        
        # Get configuration
        rounds = self.config.rounds
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        test_split = self.config.test_split
        
        self.logger.info("=== Starting True Decentralized Learning Process ===")
        
        # Prepare test data
        test_dataset = self.dataset.get_split(test_split)
        test_features, test_labels = self._prepare_test_data(
            test_dataset, self.config.feature_columns, self.config.label_column
        )
        
        # Initial network evaluation
        self.logger.info("Evaluating initial network state...")
        initial_consensus = self._evaluate_network_consensus(test_features, test_labels)
        
        self.logger.info(
            f"Initial Network Consensus Accuracy: {initial_consensus['consensus_accuracy'] * 100:.2f}%"
        )
        self.logger.info(
            f"Initial Average Node Accuracy: {initial_consensus['average_accuracy'] * 100:.2f}%"
        )
        self.logger.info(
            f"Initial Accuracy Range: [{initial_consensus['min_accuracy'] * 100:.2f}%, "
            f"{initial_consensus['max_accuracy'] * 100:.2f}%]"
        )
        
        # Get topology information
        topology_info = self.cluster_manager.get_topology_information()
        adjacency_list = topology_info.get("adjacency_list", {})
        
        round_metrics = []
        
        # Training rounds
        for round_num in range(1, rounds + 1):
            self.logger.info(f"\n--- Round {round_num}/{rounds} ---")
            
            # 1. Local Training
            self.logger.info(f"Local training for {epochs} epochs...")
            
            train_metrics = self.cluster_manager.train_models(
                client_sampling_rate=self.config.client_sampling_rate,
                data_sampling_rate=self.config.data_sampling_rate,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True,
            )
            
            # Log training metrics
            avg_train_loss = mean([m["loss"] for m in train_metrics])
            avg_train_acc = mean([m["accuracy"] for m in train_metrics])
            
            self.logger.info(
                f"Average Local Training - Loss: {avg_train_loss:.4f}, "
                f"Accuracy: {avg_train_acc * 100:.2f}%"
            )
            
            # 2. Gossip-based Parameter Exchange (True Decentralization)
            self._perform_gossip_exchange(round_num, adjacency_list)
            
            # 3. Evaluate Network Consensus
            consensus_metrics = self._evaluate_network_consensus(test_features, test_labels)
            
            self.logger.info(f"\nRound {round_num} Network Evaluation:")
            self.logger.info(
                f"  Consensus Accuracy: {consensus_metrics['consensus_accuracy'] * 100:.2f}%"
            )
            self.logger.info(
                f"  Average Node Accuracy: {consensus_metrics['average_accuracy'] * 100:.2f}%"
            )
            self.logger.info(
                f"  Node Agreement Rate: {consensus_metrics['node_agreement'] * 100:.2f}%"
            )
            self.logger.info(
                f"  Accuracy Variance: {consensus_metrics['accuracy_variance']:.6f}"
            )
            
            # Store round metrics
            round_metrics.append({
                "round": round_num,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "consensus_accuracy": consensus_metrics['consensus_accuracy'],
                "average_accuracy": consensus_metrics['average_accuracy'],
                "min_accuracy": consensus_metrics['min_accuracy'],
                "max_accuracy": consensus_metrics['max_accuracy'],
                "accuracy_variance": consensus_metrics['accuracy_variance'],
                "node_agreement": consensus_metrics['node_agreement']
            })
            
            # Emit evaluation event
            self.training_monitor.emit_event(
                EvaluationEvent(
                    round_num=round_num,
                    metrics={
                        'accuracy': consensus_metrics['consensus_accuracy'],
                        'loss': avg_train_loss  # Use average training loss
                    }
                )
            )
        
        # Final evaluation
        final_consensus = self._evaluate_network_consensus(test_features, test_labels)
        
        self.logger.info("\n=== Final Network Evaluation ===")
        self.logger.info(
            f"Final Consensus Accuracy: {final_consensus['consensus_accuracy'] * 100:.2f}%"
        )
        self.logger.info(
            f"Final Average Accuracy: {final_consensus['average_accuracy'] * 100:.2f}%"
        )
        self.logger.info(
            f"Consensus Improvement: "
            f"{(final_consensus['consensus_accuracy'] - initial_consensus['consensus_accuracy']) * 100:.2f}%"
        )
        
        # Log individual node performance
        self.logger.info("\nFinal Per-Node Performance:")
        for node_metric in final_consensus['individual_metrics']:
            if 'error' not in node_metric:
                self.logger.info(
                    f"  Node {node_metric['node_id']}: "
                    f"Accuracy={node_metric['accuracy'] * 100:.2f}%, "
                    f"Loss={node_metric['loss']:.4f}"
                )
            else:
                self.logger.info(
                    f"  Node {node_metric['node_id']}: ERROR - {node_metric['error']}"
                )
        
        return {
            "initial_metrics": initial_consensus,
            "final_metrics": final_consensus,
            "consensus_improvement": final_consensus['consensus_accuracy'] - initial_consensus['consensus_accuracy'],
            "round_metrics": round_metrics,
            "topology": topology_info,
            "learning_type": "true_decentralized"
        }