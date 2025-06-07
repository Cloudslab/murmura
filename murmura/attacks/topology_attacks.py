"""
Topology-based attacks that exploit communication patterns and parameter flows
to infer information about data distributions even with differential privacy.
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import entropy, pearsonr
import ast


class TopologyAttack(ABC):
    """Base class for topology-based attacks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    def execute_attack(self, visualization_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Execute the attack on visualization data."""
        pass


class CommunicationPatternAttack(TopologyAttack):
    """
    Attack that infers data distribution based on communication frequency and timing.
    
    Hypothesis: Nodes with similar data communicate more frequently or with different
    timing patterns, even with DP noise.
    """
    
    def __init__(self):
        super().__init__("Communication Pattern Attack")
    
    def execute_attack(self, visualization_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze communication patterns to infer node similarities."""
        communications_df = visualization_data.get('communications')
        if communications_df is None:
            return {"error": "No communications data available"}
        
        # Extract communication frequency matrix
        comm_matrix = self._build_communication_matrix(communications_df)
        
        # Cluster nodes based on communication patterns
        node_clusters = self._cluster_communication_patterns(comm_matrix)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(communications_df)
        
        return {
            "communication_matrix": comm_matrix,
            "node_clusters": node_clusters,
            "temporal_patterns": temporal_patterns,
            "attack_success_metric": self._calculate_pattern_coherence(node_clusters)
        }
    
    def _build_communication_matrix(self, communications_df: pd.DataFrame) -> np.ndarray:
        """Build matrix of communication frequencies between nodes."""
        nodes = set(communications_df['source_node'].unique()) | set(communications_df['target_node'].unique())
        n_nodes = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        
        comm_matrix = np.zeros((n_nodes, n_nodes))
        
        for _, row in communications_df.iterrows():
            src_idx = node_to_idx[row['source_node']]
            tgt_idx = node_to_idx[row['target_node']]
            comm_matrix[src_idx][tgt_idx] += 1
        
        return comm_matrix
    
    def _cluster_communication_patterns(self, comm_matrix: np.ndarray) -> Dict[int, int]:
        """Cluster nodes based on their communication patterns."""
        if comm_matrix.shape[0] < 2:
            return {0: 0}
        
        # Use both outgoing and incoming communication patterns
        features = np.concatenate([comm_matrix, comm_matrix.T], axis=1)
        
        # Cluster with k=2 to find two groups
        kmeans = KMeans(n_clusters=min(2, comm_matrix.shape[0]), random_state=42)
        clusters = kmeans.fit_predict(features)
        
        return {node_id: int(cluster) for node_id, cluster in enumerate(clusters)}
    
    def _analyze_temporal_patterns(self, communications_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing patterns in communications."""
        # Group by round and analyze communication timing
        round_stats = communications_df.groupby(['round_num', 'source_node']).agg({
            'timestamp': ['min', 'max', 'std'],
            'target_node': 'count'
        }).reset_index()
        
        # Flatten column names
        round_stats.columns = ['round_num', 'source_node', 'timestamp_min', 'timestamp_max', 'timestamp_std', 'comm_count']
        
        # Calculate per-node temporal statistics
        node_temporal_stats = round_stats.groupby('source_node').agg({
            'timestamp_std': 'mean',
            'comm_count': 'mean'
        }).to_dict('index')
        
        return node_temporal_stats
    
    def _calculate_pattern_coherence(self, node_clusters: Dict[int, int]) -> float:
        """Calculate how coherent the discovered patterns are."""
        if len(set(node_clusters.values())) <= 1:
            return 0.0
        
        # Simple metric: ratio of nodes in majority cluster
        cluster_counts = {}
        for cluster in node_clusters.values():
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        max_cluster_size = max(cluster_counts.values())
        total_nodes = len(node_clusters)
        
        return max_cluster_size / total_nodes


class ParameterMagnitudeAttack(TopologyAttack):
    """
    Attack that infers data characteristics from parameter update magnitudes.
    
    Hypothesis: Even with DP noise, nodes with different data distributions
    will have systematically different parameter update magnitudes.
    """
    
    def __init__(self):
        super().__init__("Parameter Magnitude Attack")
    
    def execute_attack(self, visualization_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze parameter magnitudes to infer data distributions."""
        param_updates_df = visualization_data.get('parameter_updates')
        if param_updates_df is None:
            return {"error": "No parameter updates data available"}
        
        # Extract magnitude statistics per node
        node_magnitude_stats = self._extract_magnitude_statistics(param_updates_df)
        
        # Cluster nodes based on magnitude patterns
        magnitude_clusters = self._cluster_by_magnitude(node_magnitude_stats)
        
        # Analyze magnitude evolution over time
        temporal_magnitude = self._analyze_magnitude_evolution(param_updates_df)
        
        return {
            "node_magnitude_stats": node_magnitude_stats,
            "magnitude_clusters": magnitude_clusters,
            "temporal_magnitude": temporal_magnitude,
            "attack_success_metric": self._calculate_magnitude_separability(node_magnitude_stats)
        }
    
    def _extract_magnitude_statistics(self, param_updates_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Extract statistical features from parameter magnitudes."""
        node_stats = {}
        
        for node_id in param_updates_df['node_id'].unique():
            node_data = param_updates_df[param_updates_df['node_id'] == node_id]
            
            # Parse parameter summaries to extract actual statistics
            norms = []
            means = []
            stds = []
            
            for _, row in node_data.iterrows():
                try:
                    summary = ast.literal_eval(row['parameter_summary'])
                    norms.append(summary['norm'])
                    means.append(summary['mean'])
                    stds.append(summary['std'])
                except:
                    # Fallback to parameter_norm column
                    norms.append(row['parameter_norm'])
            
            node_stats[node_id] = {
                'norm_mean': np.mean(norms),
                'norm_std': np.std(norms),
                'norm_trend': self._calculate_trend(norms),
                'mean_of_means': np.mean(means) if means else 0,
                'mean_of_stds': np.mean(stds) if stds else 0
            }
        
        return node_stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values over time."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        correlation, _ = pearsonr(x, values)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _cluster_by_magnitude(self, node_stats: Dict[int, Dict[str, float]]) -> Dict[int, int]:
        """Cluster nodes based on magnitude statistics."""
        if len(node_stats) < 2:
            return {list(node_stats.keys())[0]: 0} if node_stats else {}
        
        # Create feature matrix
        features = []
        node_ids = []
        
        for node_id, stats in node_stats.items():
            features.append([
                stats['norm_mean'],
                stats['norm_std'],
                stats['norm_trend'],
                stats['mean_of_means'],
                stats['mean_of_stds']
            ])
            node_ids.append(node_id)
        
        features = np.array(features)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(2, len(node_ids)), random_state=42)
        clusters = kmeans.fit_predict(features)
        
        return {node_id: int(cluster) for node_id, cluster in zip(node_ids, clusters)}
    
    def _analyze_magnitude_evolution(self, param_updates_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how parameter magnitudes evolve over training rounds."""
        evolution_stats = {}
        
        for node_id in param_updates_df['node_id'].unique():
            node_data = param_updates_df[param_updates_df['node_id'] == node_id].sort_values('round_num')
            
            norms = node_data['parameter_norm'].values
            rounds = node_data['round_num'].values
            
            if len(norms) > 1:
                # Calculate convergence rate (how quickly norms decrease)
                convergence_rate = self._calculate_trend(norms)
                
                # Calculate stability (variance in later rounds)
                if len(norms) > 3:
                    stability = np.std(norms[-3:])  # Std of last 3 rounds
                else:
                    stability = np.std(norms)
                
                evolution_stats[node_id] = {
                    'convergence_rate': convergence_rate,
                    'stability': stability,
                    'final_norm': norms[-1] if len(norms) > 0 else 0
                }
        
        return evolution_stats
    
    def _calculate_magnitude_separability(self, node_stats: Dict[int, Dict[str, float]]) -> float:
        """Calculate how well nodes can be separated based on magnitude features."""
        if len(node_stats) < 2:
            return 0.0
        
        # Extract norm_mean values
        norms = [stats['norm_mean'] for stats in node_stats.values()]
        
        # Calculate coefficient of variation
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if mean_norm == 0:
            return 0.0
        
        return std_norm / mean_norm  # Higher values indicate better separability


class TopologyStructureAttack(TopologyAttack):
    """
    Attack that uses knowledge of topology structure to infer data placement.
    
    Hypothesis: The position in topology (ring position, distance from center)
    correlates with data characteristics even with DP.
    """
    
    def __init__(self):
        super().__init__("Topology Structure Attack")
    
    def execute_attack(self, visualization_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Use topology structure to predict data distributions."""
        topology_df = visualization_data.get('topology')
        param_updates_df = visualization_data.get('parameter_updates')
        
        if topology_df is None or param_updates_df is None:
            return {"error": "Missing topology or parameter data"}
        
        # Analyze topology structure
        topology_features = self._extract_topology_features(topology_df)
        
        # Extract node characteristics
        node_characteristics = self._extract_node_characteristics(param_updates_df)
        
        # Test correlation between topology position and characteristics
        correlations = self._test_topology_correlations(topology_features, node_characteristics)
        
        # Predict node groups based on topology
        topology_predictions = self._predict_from_topology(topology_features)
        
        return {
            "topology_features": topology_features,
            "node_characteristics": node_characteristics,
            "correlations": correlations,
            "topology_predictions": topology_predictions,
            "attack_success_metric": max(abs(corr) for corr in correlations.values()) if correlations else 0.0
        }
    
    def _extract_topology_features(self, topology_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Extract features based on topology position."""
        topology_features = {}
        
        for _, row in topology_df.iterrows():
            node_id = row['node_id']
            connected_nodes = [int(x) for x in str(row['connected_nodes']).split(',') if x.strip().isdigit()]
            degree = row['degree']
            
            topology_features[node_id] = {
                'degree': degree,
                'position': node_id,  # Position in ordering
                'is_central': 1 if degree > 2 else 0,  # For star topology
                'neighbor_sum': sum(connected_nodes),  # Simple structural feature
            }
        
        return topology_features
    
    def _extract_node_characteristics(self, param_updates_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Extract characteristics of each node from parameter updates."""
        node_chars = {}
        
        for node_id in param_updates_df['node_id'].unique():
            node_data = param_updates_df[param_updates_df['node_id'] == node_id]
            
            norms = node_data['parameter_norm'].values
            
            node_chars[node_id] = {
                'avg_norm': np.mean(norms),
                'norm_variability': np.std(norms),
                'initial_norm': norms[0] if len(norms) > 0 else 0,
                'final_norm': norms[-1] if len(norms) > 0 else 0
            }
        
        return node_chars
    
    def _test_topology_correlations(self, 
                                   topology_features: Dict[int, Dict[str, float]],
                                   node_characteristics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Test correlations between topology position and node characteristics."""
        correlations = {}
        
        # Get common nodes
        common_nodes = set(topology_features.keys()) & set(node_characteristics.keys())
        
        if len(common_nodes) < 3:
            return correlations
        
        # Extract arrays for correlation
        positions = [topology_features[node]['position'] for node in common_nodes]
        degrees = [topology_features[node]['degree'] for node in common_nodes]
        avg_norms = [node_characteristics[node]['avg_norm'] for node in common_nodes]
        norm_vars = [node_characteristics[node]['norm_variability'] for node in common_nodes]
        
        # Calculate correlations (suppress warnings for constant arrays)
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                correlations['position_vs_norm'] = pearsonr(positions, avg_norms)[0]
                correlations['degree_vs_norm'] = pearsonr(degrees, avg_norms)[0]
                correlations['position_vs_variability'] = pearsonr(positions, norm_vars)[0]
        except:
            correlations = {'position_vs_norm': 0.0, 'degree_vs_norm': 0.0, 'position_vs_variability': 0.0}
        
        # Replace NaN with 0
        for key, value in correlations.items():
            if np.isnan(value):
                correlations[key] = 0.0
        
        return correlations
    
    def _predict_from_topology(self, topology_features: Dict[int, Dict[str, float]]) -> Dict[int, str]:
        """Predict node groups based on topology structure."""
        predictions = {}
        
        for node_id, features in topology_features.items():
            # Simple rule-based prediction
            if features['degree'] > 2:
                predictions[node_id] = "central_hub"
            elif features['position'] % 2 == 0:
                predictions[node_id] = "even_position"
            else:
                predictions[node_id] = "odd_position"
        
        return predictions


class AttackEvaluator:
    """Evaluates attack success by comparing predictions to ground truth."""
    
    def __init__(self, ground_truth_partitions: Optional[Dict[int, List[int]]] = None):
        self.ground_truth_partitions = ground_truth_partitions
    
    def evaluate_attacks(self, attack_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all attack results."""
        evaluation = {
            "attack_summaries": [],
            "overall_success": False,
            "best_attack": None,
            "attack_indicators": {}
        }
        
        best_score = 0.0
        
        for attack_result in attack_results:
            attack_name = attack_result.get('attack_name', 'Unknown')
            success_metric = attack_result.get('attack_success_metric', 0.0)
            
            summary = {
                "attack_name": attack_name,
                "success_metric": success_metric,
                "key_findings": self._extract_key_findings(attack_result)
            }
            
            evaluation["attack_summaries"].append(summary)
            
            if success_metric > best_score:
                best_score = success_metric
                evaluation["best_attack"] = attack_name
        
        # Overall success if any attack shows significant signal
        evaluation["overall_success"] = best_score > 0.3  # Threshold for "successful" attack
        evaluation["attack_indicators"]["max_signal"] = best_score
        
        return evaluation
    
    def _extract_key_findings(self, attack_result: Dict[str, Any]) -> List[str]:
        """Extract human-readable key findings from attack results."""
        findings = []
        
        # Communication pattern findings
        if "node_clusters" in attack_result:
            clusters = attack_result["node_clusters"]
            if len(set(clusters.values())) > 1:
                findings.append(f"Detected {len(set(clusters.values()))} distinct communication patterns")
        
        # Magnitude findings
        if "magnitude_clusters" in attack_result:
            findings.append("Parameter magnitudes show systematic differences between nodes")
        
        # Topology correlation findings
        if "correlations" in attack_result:
            corrs = attack_result["correlations"]
            strong_corrs = [(k, v) for k, v in corrs.items() if abs(v) > 0.3]
            if strong_corrs:
                findings.append(f"Strong topology correlations detected: {strong_corrs}")
        
        if not findings:
            findings.append("No significant patterns detected")
        
        return findings


def run_topology_attacks(visualization_dir: str) -> Dict[str, Any]:
    """Run all topology attacks on visualization data from a directory."""
    import os
    
    # Load visualization data
    viz_data = {}
    data_files = {
        'communications': 'training_data_communications.csv',
        'parameter_updates': 'training_data_parameter_updates.csv',
        'topology': 'training_data_topology.csv',
        'metrics': 'training_data_metrics.csv'
    }
    
    for data_type, filename in data_files.items():
        filepath = os.path.join(visualization_dir, filename)
        if os.path.exists(filepath):
            try:
                viz_data[data_type] = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Initialize attacks
    attacks = [
        CommunicationPatternAttack(),
        ParameterMagnitudeAttack(),
        TopologyStructureAttack()
    ]
    
    # Execute attacks
    attack_results = []
    for attack in attacks:
        try:
            result = attack.execute_attack(viz_data)
            result['attack_name'] = attack.name
            attack_results.append(result)
        except Exception as e:
            attack_results.append({
                'attack_name': attack.name,
                'error': str(e),
                'attack_success_metric': 0.0
            })
    
    # Evaluate results
    evaluator = AttackEvaluator()
    evaluation = evaluator.evaluate_attacks(attack_results)
    
    return {
        'attack_results': attack_results,
        'evaluation': evaluation,
        'visualization_data_summary': {k: len(v) for k, v in viz_data.items()}
    }