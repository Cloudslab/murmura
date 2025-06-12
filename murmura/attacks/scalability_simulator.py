# type: ignore
"""
Scalability simulation framework for topology-based attacks.

This module provides synthetic data generation and simulation capabilities 
to test topology attacks on networks with 500+ nodes without the computational
overhead of actual federated learning training.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import math
import networkx as nx
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class NetworkConfig:
    """Configuration for simulated network."""
    num_nodes: int
    topology: str  # 'star', 'ring', 'complete', 'line'
    attack_strategy: str  # 'sensitive_groups', 'topology_correlated', 'imbalanced_sensitive'
    dp_enabled: bool = False
    dp_epsilon: Optional[float] = None
    num_rounds: int = 10
    fl_type: Optional[str] = None  # Will be inferred from topology in __post_init__
    
    def __post_init__(self):
        """Validate configuration with SAME constraints as paper_experiments.py."""
        if self.num_nodes < 2:
            raise ValueError("num_nodes must be >= 2")
        if self.topology not in ['star', 'ring', 'complete', 'line']:
            raise ValueError(f"Unknown topology: {self.topology}")
        if self.attack_strategy not in ['sensitive_groups', 'topology_correlated', 'imbalanced_sensitive']:
            raise ValueError(f"Unknown attack strategy: {self.attack_strategy}")
        
        # SAME compatibility validation as paper_experiments.py
        self._validate_topology_compatibility()
    
    def _validate_topology_compatibility(self):
        """Validate topology-FL type compatibility (SAME as paper_experiments.py)."""
        # Define compatibility matrix based on actual framework constraints
        compatibility_matrix = {
            # Centralized FL (FedAvg/TrimmedMean): ONLY star and complete (global parameter collection)
            "federated": {
                "compatible_topologies": ["star", "complete"],
                "strategies": ["fedavg", "trimmed_mean"]
            },
            # Decentralized FL (GossipAvg): ring, complete, line (NO star support)
            "decentralized": {
                "compatible_topologies": ["ring", "complete", "line"],
                "strategies": ["gossip_avg"]
            }
        }
        
        # For simulation, we infer FL type from topology
        if self.topology in ["star"]:
            self.fl_type = "federated"
        elif self.topology in ["ring", "line"]:
            self.fl_type = "decentralized"
        elif self.topology in ["complete"]:
            # Complete topology can work with both - choose based on strategy
            if self.attack_strategy == "topology_correlated":
                self.fl_type = "federated"  # Use federated for better correlation patterns
            else:
                self.fl_type = "decentralized"  # Default to decentralized
        
        # Validate compatibility
        valid_topologies = compatibility_matrix[self.fl_type]["compatible_topologies"]
        if self.topology not in valid_topologies:
            raise ValueError(f"Topology '{self.topology}' is not compatible with FL type '{self.fl_type}'. "
                           f"Valid topologies for {self.fl_type}: {valid_topologies}")


class SyntheticDataGenerator:
    """Generates synthetic training data that mimics real federated learning patterns."""
    
    def __init__(self, config: NetworkConfig, random_seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(random_seed)
        self.network_graph = self._create_network_graph()
        
    def _create_network_graph(self) -> nx.Graph:
        """Create network graph based on topology."""
        G = nx.Graph()
        G.add_nodes_from(range(self.config.num_nodes))
        
        if self.config.topology == 'star':
            # Node 0 is the center
            for i in range(1, self.config.num_nodes):
                G.add_edge(0, i)
                
        elif self.config.topology == 'ring':
            # Connect nodes in a ring
            for i in range(self.config.num_nodes):
                G.add_edge(i, (i + 1) % self.config.num_nodes)
                
        elif self.config.topology == 'complete':
            # Fully connected graph
            for i in range(self.config.num_nodes):
                for j in range(i + 1, self.config.num_nodes):
                    G.add_edge(i, j)
                    
        elif self.config.topology == 'line':
            # Linear chain
            for i in range(self.config.num_nodes - 1):
                G.add_edge(i, i + 1)
                
        return G
    
    def generate_node_characteristics(self) -> Dict[int, Dict[str, float]]:
        """Generate realistic node characteristics based on real-world patterns and attack strategy."""
        characteristics = {}
        
        # Use realistic parameter ranges from analysis
        if self.config.fl_type == "federated":
            base_norm_range = (3.21, 3.46)  # Real federated learning ranges
            param_std_range = (0.19, 0.24)
        else:
            base_norm_range = (4.47, 5.74)  # Real decentralized learning ranges  
            param_std_range = (0.24, 0.34)
        
        if self.config.attack_strategy == 'sensitive_groups':
            # Create distinct groups with realistic parameter differences
            group_size = self.config.num_nodes // 2
            
            for node_id in range(self.config.num_nodes):
                if node_id < group_size:
                    # Group 1: Higher norms (matching real data patterns)
                    base_norm = self.rng.uniform(base_norm_range[1] * 0.95, base_norm_range[1])
                    param_std = self.rng.uniform(param_std_range[0], param_std_range[0] * 1.1)
                    group_id = 0
                else:
                    # Group 2: Lower norms (matching real data patterns)  
                    base_norm = self.rng.uniform(base_norm_range[0], base_norm_range[0] * 1.05)
                    param_std = self.rng.uniform(param_std_range[1] * 0.9, param_std_range[1])
                    group_id = 1
                
                characteristics[node_id] = {
                    'base_norm': base_norm,
                    'param_std': param_std,
                    'group_id': group_id,
                    'data_heterogeneity': self.rng.uniform(0.3, 0.8),
                    'param_mean': self.rng.uniform(-0.02, 0.08)  # Real parameter mean range
                }
                
        elif self.config.attack_strategy == 'topology_correlated':
            # Characteristics correlate with topology position (realistic ranges)
            for node_id in range(self.config.num_nodes):
                degree = int(self.network_graph.degree(node_id))
                
                # Create strong correlations based on real patterns
                if self.config.topology == 'star' and node_id == 0:
                    # Central node: distinctly higher parameters (real star pattern)
                    base_norm = base_norm_range[1] * 1.05  # 5% higher than max
                    param_std = param_std_range[1] * 1.1   # Higher variation
                    correlation_factor = 1.0
                elif self.config.topology == 'star':
                    # Leaf nodes: lower, similar parameters
                    base_norm = self.rng.uniform(base_norm_range[0], base_norm_range[0] * 1.02)
                    param_std = self.rng.uniform(param_std_range[0], param_std_range[0] * 1.05)
                    correlation_factor = 0.2
                else:
                    # Ring/line: parameters correlate with position
                    position_factor = node_id / max(1, self.config.num_nodes - 1)
                    base_norm = base_norm_range[0] + (base_norm_range[1] - base_norm_range[0]) * position_factor
                    param_std = param_std_range[0] + (param_std_range[1] - param_std_range[0]) * position_factor
                    correlation_factor = position_factor
                
                characteristics[node_id] = {
                    'base_norm': base_norm,
                    'param_std': param_std,
                    'position_correlation': correlation_factor,
                    'topology_degree': degree,
                    'param_mean': self.rng.uniform(-0.02, 0.08)
                }
                
        elif self.config.attack_strategy == 'imbalanced_sensitive':
            # Imbalanced data creates systematic magnitude differences (realistic ranges)
            total_samples = 10000  # Simulated total dataset size
            
            # Create imbalanced distribution: few nodes have much more data
            if self.config.num_nodes <= 10:
                # Small networks: 1-2 nodes get most data
                data_ratios = [0.4] + [0.6 / (self.config.num_nodes - 1)] * (self.config.num_nodes - 1)
            else:
                # Large networks: power law distribution
                ranks = np.arange(1, self.config.num_nodes + 1)
                data_ratios_array = 1.0 / ranks
                data_ratios = data_ratios_array / data_ratios_array.sum()
            
            self.rng.shuffle(data_ratios)  # Random assignment
            
            for node_id in range(self.config.num_nodes):
                data_ratio = data_ratios[node_id]
                samples = int(total_samples * data_ratio)
                
                # More data -> higher norms within realistic ranges
                norm_scale = data_ratio * 2.0  # Scale factor based on data amount
                base_norm = base_norm_range[0] + (base_norm_range[1] - base_norm_range[0]) * min(1.0, norm_scale)
                param_std = param_std_range[0] + (param_std_range[1] - param_std_range[0]) * data_ratio
                
                characteristics[node_id] = {
                    'base_norm': base_norm,
                    'param_std': param_std,
                    'data_samples': samples,
                    'data_ratio': data_ratio,
                    'param_mean': self.rng.uniform(-0.02, 0.08)
                }
        
        return characteristics
    
    def generate_parameter_updates(self, node_characteristics: Dict[int, Dict[str, float]]) -> pd.DataFrame:
        """Generate realistic parameter update traces based on real-world patterns."""
        updates = []
        
        for round_num in range(self.config.num_rounds):
            for node_id in range(self.config.num_nodes):
                char = node_characteristics[node_id]
                
                # Realistic decay pattern (slower than before)
                decay_factor = 0.95 ** round_num  # Slower, more realistic decay
                base_norm = char['base_norm'] * decay_factor
                
                # Use realistic parameter std from characteristics
                param_std_base = char['param_std']
                param_norm = self.rng.normal(base_norm, param_std_base * 0.1)
                param_norm = max(0.01, param_norm)  # Ensure positive
                
                # Realistic parameter mean and std
                param_mean = char['param_mean']
                param_std_actual = param_std_base * self.rng.uniform(0.8, 1.2)  # Add some variation
                
                # Add realistic DP noise if enabled
                if self.config.dp_enabled and self.config.dp_epsilon:
                    # More realistic DP noise based on actual DP mechanisms
                    if self.config.dp_epsilon >= 8.0:  # Medium DP
                        noise_factor = 0.05  # 5% noise
                    else:  # Strong DP  
                        noise_factor = 0.15  # 15% noise - more significant impact
                    
                    dp_noise = self.rng.normal(0, param_norm * noise_factor)
                    param_norm += dp_noise
                    param_norm = max(0.01, param_norm)
                    
                    # DP also affects mean and std significantly
                    param_mean += self.rng.normal(0, abs(param_mean) * noise_factor * 2)
                    param_std_actual *= (1 - noise_factor)  # DP reduces variation more
                
                update = {
                    'round_num': round_num,
                    'node_id': node_id,
                    'parameter_norm': param_norm,
                    'parameter_summary': str({
                        'norm': param_norm,
                        'mean': param_mean,
                        'std': param_std_actual
                    }),
                    'timestamp': round_num * 100 + node_id  # Synthetic timestamp
                }
                
                updates.append(update)
        
        return pd.DataFrame(updates)
    
    def generate_communications(self) -> pd.DataFrame:
        """Generate realistic communication patterns based on real-world analysis."""
        communications = []
        
        # Realistic communication volumes based on topology (from real data analysis)
        if self.config.topology == 'star':
            base_comms_per_round = max(6, int(24 * (self.config.num_nodes / 5)))  # Scale from real 5-node data
        elif self.config.topology == 'ring':
            base_comms_per_round = max(12, int(60 * (self.config.num_nodes / 5)))
        elif self.config.topology == 'complete':
            base_comms_per_round = max(24, int(120 * (self.config.num_nodes / 5)))
        else:  # line
            base_comms_per_round = max(8, int(40 * (self.config.num_nodes / 5)))
        
        for round_num in range(self.config.num_rounds):
            # Generate realistic number of communications per round
            round_communications = 0
            target_communications = int(base_comms_per_round * self.rng.uniform(0.8, 1.2))
            
            while round_communications < target_communications:
                # Choose edges based on topology
                edges = list(self.network_graph.edges())
                
                for source, target in edges:
                    if round_communications >= target_communications:
                        break
                        
                    # Communication frequency varies by topology and attack strategy
                    comm_prob = 1.0
                    
                    if self.config.attack_strategy == 'sensitive_groups':
                        # Same-group nodes communicate more frequently
                        source_group = source < (self.config.num_nodes // 2)
                        target_group = target < (self.config.num_nodes // 2)
                        if source_group == target_group:
                            comm_prob = 1.5  # 50% more likely
                    
                    elif self.config.attack_strategy == 'topology_correlated':
                        # Communication frequency correlates with node degrees
                        source_degree = self.network_graph.degree(source)
                        target_degree = self.network_graph.degree(target)
                        comm_prob = 1.0 + (source_degree + target_degree) * 0.1
                    
                    # Generate communication if probability check passes
                    if self.rng.random() < (comm_prob / len(edges)):
                        # Realistic timestamp with some jitter
                        base_time = round_num * 1000 + round_communications * 10
                        timestamp = base_time + self.rng.uniform(0, 50)  # Add realistic jitter
                        
                        comm = {
                            'round_num': round_num,
                            'source_node': source,
                            'target_node': target,
                            'timestamp': timestamp,
                            'message_size': max(100, self.rng.normal(1024, 256))  # Realistic message sizes
                        }
                        communications.append(comm)
                        round_communications += 1
        
        return pd.DataFrame(communications)
    
    def generate_topology_data(self) -> pd.DataFrame:
        """Generate topology structure data."""
        topology_data = []
        
        for node_id in range(self.config.num_nodes):
            neighbors = list(self.network_graph.neighbors(node_id))
            degree = int(self.network_graph.degree(node_id))
            
            topo_entry = {
                'node_id': node_id,
                'connected_nodes': ','.join(map(str, neighbors)) if neighbors else '',
                'degree': degree,
                'topology_type': self.config.topology
            }
            
            topology_data.append(topo_entry)
        
        return pd.DataFrame(topology_data)
    
    def generate_full_simulation_data(self) -> Dict[str, pd.DataFrame]:
        """Generate complete simulation dataset."""
        # Generate node characteristics
        node_characteristics = self.generate_node_characteristics()
        
        # Generate all data types
        simulation_data = {
            'parameter_updates': self.generate_parameter_updates(node_characteristics),
            'communications': self.generate_communications(),
            'topology': self.generate_topology_data(),
            'node_characteristics': pd.DataFrame.from_dict(node_characteristics, orient='index').reset_index().rename(columns={'index': 'node_id'})
        }
        
        return simulation_data


class ScalabilityAnalyzer:
    """Analyzes attack effectiveness scaling with network size."""
    
    def __init__(self):
        self.scaling_results = {}
    
    def analyze_complexity_scaling(self, network_sizes: List[int], 
                                 topology: str, attack_strategy: str) -> Dict[str, Any]:
        """Analyze theoretical complexity scaling."""
        complexity_analysis = {
            'topology': topology,
            'attack_strategy': attack_strategy,
            'network_sizes': network_sizes,
            'theoretical_complexity': {},
            'empirical_scaling': {}
        }
        
        # Theoretical complexity analysis
        for size in network_sizes:
            if topology == 'complete':
                # Complete graph: O(n²) edges
                comm_complexity = size * (size - 1) // 2
                param_complexity = size * size  # All pairwise comparisons
            elif topology == 'star':
                # Star: O(n) edges, but central node dominates
                comm_complexity = size - 1
                param_complexity = size * (size - 1)  # Hub vs all others
            elif topology == 'ring':
                # Ring: O(n) edges, local structure
                comm_complexity = size
                param_complexity = size * 2  # Local comparisons
            elif topology == 'line':
                # Line: O(n) edges, very local
                comm_complexity = size - 1
                param_complexity = size
            
            complexity_analysis['theoretical_complexity'][size] = {
                'communication_complexity': comm_complexity,
                'parameter_complexity': param_complexity,
                'total_complexity': comm_complexity + param_complexity
            }
        
        return complexity_analysis
    
    def predict_attack_effectiveness(self, network_size: int, 
                                   baseline_effectiveness: Dict[int, float],
                                   topology: str) -> float:
        """Predict attack effectiveness for large networks based on scaling laws."""
        
        # Extract baseline data points
        sizes = sorted(baseline_effectiveness.keys())
        effectiveness_values = [baseline_effectiveness[s] for s in sizes]
        
        if len(sizes) < 2:
            # Not enough data for prediction
            return effectiveness_values[0] if effectiveness_values else 0.0
        
        # Fit scaling model based on topology characteristics
        if topology == 'complete':
            # Complete graphs become harder to attack as size increases (more noise)
            # Effectiveness ~ 1/log(n)
            log_sizes = [math.log(s) for s in sizes]
            scaling_factor = np.polyfit(log_sizes, effectiveness_values, 1)[0]
            predicted = max(0.0, effectiveness_values[-1] + scaling_factor * (math.log(network_size) - math.log(sizes[-1])))
            
        elif topology == 'star':
            # Star topology: central node makes pattern more visible
            # Effectiveness ~ constant or slight increase
            predicted = np.mean(effectiveness_values)
            
        elif topology in ['ring', 'line']:
            # Local structure: effectiveness decreases with size
            # Effectiveness ~ 1/sqrt(n)
            sqrt_sizes = [math.sqrt(s) for s in sizes]
            scaling_factor = np.polyfit(sqrt_sizes, effectiveness_values, 1)[0]
            predicted = max(0.0, effectiveness_values[-1] + scaling_factor * (math.sqrt(network_size) - math.sqrt(sizes[-1])))
        
        else:
            # Default: linear extrapolation
            predicted = np.interp(network_size, sizes, effectiveness_values)
        
        return min(1.0, max(0.0, predicted))  # Clamp to [0, 1]


class LargeScaleAttackSimulator:
    """Simulates topology attacks on large networks without full training."""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.data_generator = SyntheticDataGenerator(config)
        self.analyzer = ScalabilityAnalyzer()
    
    def simulate_attack_execution(self) -> Dict[str, Any]:
        """Simulate attack execution on large network using same validation as paper_experiments.py."""
        
        # Generate synthetic data
        simulation_data = self.data_generator.generate_full_simulation_data()
        
        # Import and run actual attacks on synthetic data
        from .topology_attacks import (
            CommunicationPatternAttack, 
            ParameterMagnitudeAttack, 
            TopologyStructureAttack
        )
        
        # Execute attacks
        attacks = [
            CommunicationPatternAttack(),
            ParameterMagnitudeAttack(), 
            TopologyStructureAttack()
        ]
        
        attack_results = []
        for attack in attacks:
            try:
                result = attack.execute_attack(simulation_data)
                result['attack_name'] = attack.name
                attack_results.append(result)
            except Exception as e:
                attack_results.append({
                    'attack_name': attack.name,
                    'error': str(e),
                    'attack_success_metric': 0.0
                })
        
        # Use SAME evaluation methodology as paper_experiments.py
        evaluation = self._evaluate_attack_comprehensive(attack_results, {
            'attack_strategy': self.config.attack_strategy,
            'topology': self.config.topology,
            'node_count': self.config.num_nodes,
            'dp_setting': {
                'enabled': self.config.dp_enabled,
                'epsilon': self.config.dp_epsilon,
                'name': f"dp_eps_{self.config.dp_epsilon}" if self.config.dp_enabled else "no_dp"
            }
        })
        
        return {
            'config': {
                'num_nodes': self.config.num_nodes,
                'topology': self.config.topology,
                'fl_type': self.config.fl_type,
                'attack_strategy': self.config.attack_strategy,
                'dp_enabled': self.config.dp_enabled,
                'dp_epsilon': self.config.dp_epsilon
            },
            'attack_results': attack_results,
            'evaluation': evaluation['evaluation'],
            'simulation_metadata': {
                'data_generation_method': 'synthetic',
                'network_complexity': self._calculate_network_complexity(),
                'theoretical_bounds': self._calculate_theoretical_bounds()
            }
        }
    
    def _evaluate_attack_comprehensive(self, attack_results: List[Dict[str, Any]], 
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive attack evaluation using SAME methodology as paper_experiments.py."""
        
        evaluation = {
            "attack_success": False,
            "confidence_score": 0.0,
            "attack_type_scores": {},
            "best_attack_type": None,
            "topology_specific_metrics": {},
            "dp_effectiveness_metrics": {},
            "scalability_metrics": {}
        }
        
        # Analyze each attack type (SAME as paper_experiments.py)
        attack_types = ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]
        
        for result in attack_results:
            attack_name = result.get('attack_name', '')
            success_metric = result.get('attack_success_metric', 0.0)
            
            if attack_name in attack_types:
                evaluation['attack_type_scores'][attack_name] = success_metric
                
                if success_metric > evaluation['confidence_score']:
                    evaluation['confidence_score'] = success_metric
                    evaluation['best_attack_type'] = attack_name
        
        # Overall success threshold (SAME as paper_experiments.py)
        evaluation['attack_success'] = evaluation['confidence_score'] > 0.3
        
        # Add strategy-specific analysis
        evaluation.update(self._analyze_attack_strategy_scalability(attack_results, config))
        
        # Add topology-specific metrics
        evaluation['topology_specific_metrics'] = self._analyze_topology_effects_scalability(attack_results, config)
        
        # Add DP effectiveness metrics
        evaluation['dp_effectiveness_metrics'] = self._analyze_dp_effectiveness_scalability(attack_results, config)
        
        # Add scalability metrics
        evaluation['scalability_metrics'] = self._analyze_scalability_effects_scalability(attack_results, config)
        
        # Add fields expected by experiment runner (inside evaluation object)
        evaluation['overall_success'] = evaluation['attack_success']
        evaluation['attack_indicators'] = {'max_signal': evaluation['confidence_score']}
        
        return {
            'attack_results': attack_results,
            'evaluation': evaluation,
            'visualization_data_summary': {'simulated': True}
        }
    
    def _analyze_attack_strategy_scalability(self, attack_results: List[Dict[str, Any]], 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attack success specific to the attack strategy (SAME as paper_experiments.py)."""
        
        strategy = config['attack_strategy']
        analysis = {"strategy_specific_findings": []}
        
        if strategy == "sensitive_groups":
            # Analyze group detection success
            for result in attack_results:
                if result.get('attack_name') == 'Communication Pattern Attack':
                    if 'node_clusters' in result:
                        clusters = result['node_clusters']
                        n_clusters = len(set(clusters.values()))
                        analysis['strategy_specific_findings'].append(
                            f"Detected {n_clusters} distinct groups via communication patterns"
                        )
                        analysis['group_detection_success'] = n_clusters > 1
        
        elif strategy == "topology_correlated":
            # Analyze correlation detection
            for result in attack_results:
                if result.get('attack_name') == 'Topology Structure Attack':
                    if 'correlations' in result:
                        correlations = result['correlations']
                        max_correlation = max([abs(v) for v in correlations.values()], default=0.0)
                        analysis['max_correlation'] = max_correlation
                        analysis['strategy_specific_findings'].append(
                            f"Maximum topology correlation: {max_correlation:.3f}"
                        )
        
        elif strategy == "imbalanced_sensitive":
            # Analyze imbalance detection
            for result in attack_results:
                if result.get('attack_name') == 'Parameter Magnitude Attack':
                    success_metric = result.get('attack_success_metric', 0.0)
                    analysis['imbalance_detection_score'] = success_metric
                    analysis['strategy_specific_findings'].append(
                        f"Imbalance detection score: {success_metric:.3f}"
                    )
        
        return analysis
    
    def _analyze_topology_effects_scalability(self, attack_results: List[Dict[str, Any]], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how topology affects attack success (SAME as paper_experiments.py)."""
        
        topology = config['topology']
        node_count = config['node_count']
        
        metrics = {
            "topology": topology,
            "node_count": node_count,
            "topology_vulnerability_score": 0.0,
            "communication_complexity": self._calculate_communication_complexity_scalability(topology, node_count)
        }
        
        # Calculate topology-specific vulnerability
        max_signal = 0.0
        for result in attack_results:
            success_metric = result.get('attack_success_metric', 0.0)
            max_signal = max(max_signal, success_metric)
        
        metrics['topology_vulnerability_score'] = max_signal
        
        return metrics
    
    def _calculate_communication_complexity_scalability(self, topology: str, node_count: int) -> Dict[str, int]:
        """Calculate communication complexity for different topologies (SAME as paper_experiments.py)."""
        
        if topology == "star":
            # Hub communicates with all others
            return {"total_edges": node_count - 1, "max_degree": node_count - 1, "avg_degree": 2.0}
        elif topology == "ring":
            # Each node connects to 2 neighbors
            return {"total_edges": node_count, "max_degree": 2, "avg_degree": 2.0}
        elif topology == "complete":
            # All nodes connect to all others
            total_edges = node_count * (node_count - 1) // 2
            return {"total_edges": total_edges, "max_degree": node_count - 1, "avg_degree": node_count - 1}
        elif topology == "line":
            # Linear chain
            return {"total_edges": node_count - 1, "max_degree": 2, "avg_degree": 2.0}
        else:
            return {"total_edges": 0, "max_degree": 0, "avg_degree": 0.0}
    
    def _analyze_dp_effectiveness_scalability(self, attack_results: List[Dict[str, Any]], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differential privacy effectiveness (SAME as paper_experiments.py)."""
        
        dp_setting = config['dp_setting']
        
        metrics = {
            "dp_enabled": dp_setting['enabled'],
            "epsilon": dp_setting.get('epsilon'),
            "privacy_level": dp_setting['name']
        }
        
        if dp_setting['enabled']:
            # DP is enabled - measure how well it protects
            max_signal = 0.0
            for result in attack_results:
                success_metric = result.get('attack_success_metric', 0.0)
                max_signal = max(max_signal, success_metric)
            
            attack_success = max_signal > 0.3  # Same threshold as paper_experiments.py
            
            metrics.update({
                "dp_protection_effective": not attack_success,
                "signal_strength_with_dp": max_signal,
                "privacy_budget_utilization": self._estimate_privacy_budget_usage_scalability(config)
            })
        
        return metrics
    
    def _estimate_privacy_budget_usage_scalability(self, config: Dict[str, Any]) -> float:
        """Estimate privacy budget utilization (SAME as paper_experiments.py)."""
        # This is a simplified estimation - in real scenarios you'd get this from the DP mechanism
        rounds = self.config.num_rounds
        epsilon = config['dp_setting'].get('epsilon', 0)
        if epsilon:
            # Rough estimation of budget usage
            return min(1.0, rounds * 0.3)  # Assume ~30% budget per round
        return 0.0
    
    def _analyze_scalability_effects_scalability(self, attack_results: List[Dict[str, Any]], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how network size affects attack success (SAME as paper_experiments.py)."""
        
        node_count = config['node_count']
        
        metrics = {
            "network_size": node_count,
            "size_category": self._categorize_network_size_scalability(node_count),
            "attack_scalability_score": 0.0
        }
        
        # Analyze how attack effectiveness scales with network size
        max_signal = 0.0
        for result in attack_results:
            success_metric = result.get('attack_success_metric', 0.0)
            max_signal = max(max_signal, success_metric)
        
        # Normalize by network size (larger networks might naturally have lower signal)
        # This is a heuristic - larger networks should be harder to attack
        size_normalization = 1.0 + (node_count - 5) * 0.02  # Slight penalty for larger networks
        normalized_score = max_signal * size_normalization
        
        metrics['attack_scalability_score'] = normalized_score
        
        return metrics
    
    def _categorize_network_size_scalability(self, node_count: int) -> str:
        """Categorize network size for analysis (SAME as paper_experiments.py)."""
        if node_count <= 5:
            return "small"
        elif node_count <= 10:
            return "medium" 
        elif node_count <= 15:
            return "large"
        elif node_count <= 50:
            return "very_large"
        elif node_count <= 100:
            return "huge"
        else:
            return "massive"
    
    def _calculate_network_complexity(self) -> Dict[str, float]:
        """Calculate network complexity metrics."""
        G = self.data_generator.network_graph
        
        # Basic graph metrics
        num_edges = G.number_of_edges()
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        
        # Clustering coefficient (how clustered the network is)
        try:
            clustering = nx.average_clustering(G)
        except Exception:
            clustering = 0.0
        
        # Path length (how far apart nodes are on average)
        try:
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                avg_path_length = float('inf')
        except Exception:
            avg_path_length = float('inf')
        
        return {
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'clustering_coefficient': clustering,
            'avg_path_length': avg_path_length,
            'edge_density': num_edges / (self.config.num_nodes * (self.config.num_nodes - 1) / 2)
        }
    
    def _calculate_theoretical_bounds(self) -> Dict[str, float]:
        """Calculate theoretical bounds on attack effectiveness."""
        n = self.config.num_nodes
        
        # Information-theoretic bounds
        if self.config.attack_strategy == 'sensitive_groups':
            # Bound based on group distinguishability
            theoretical_upper_bound = 1.0 - 1.0/math.sqrt(n)  # Groups become more distinguishable
            
        elif self.config.attack_strategy == 'topology_correlated':
            # Bound based on topology structure
            if self.config.topology == 'star':
                theoretical_upper_bound = 1.0 - 1.0/n  # Central node correlation
            else:
                theoretical_upper_bound = 1.0 - 1.0/math.log(n)  # Structure correlation
                
        elif self.config.attack_strategy == 'imbalanced_sensitive':
            # Bound based on data imbalance detection
            theoretical_upper_bound = 1.0 - 1.0/(n * 0.5)  # Harder with more nodes
            
        else:
            theoretical_upper_bound = 0.5  # Default
        
        # DP bounds
        if self.config.dp_enabled and self.config.dp_epsilon:
            # DP reduces effectiveness
            dp_degradation = math.exp(-self.config.dp_epsilon)
            theoretical_upper_bound *= (1 - dp_degradation)
        
        return {
            'theoretical_upper_bound': min(1.0, theoretical_upper_bound),
            'information_theoretic_limit': 1.0 - 1.0/math.log(max(2, n)),
            'dp_privacy_bound': dp_degradation if self.config.dp_enabled else 0.0
        }


def run_scalability_experiments(network_sizes: List[int], 
                              topologies: List[str],
                              attack_strategies: List[str],
                              dp_settings: List[Dict[str, Any]],
                              output_dir: str = "./scalability_results",
                              num_workers: int = 1) -> Dict[str, Any]:
    """Run comprehensive scalability experiments with incremental result saving."""
    
    # Use parallel version if num_workers > 1
    if num_workers > 1:
        from .scalability_simulator_parallel import run_scalability_experiments_parallel
        return run_scalability_experiments_parallel(
            network_sizes, topologies, attack_strategies, 
            dp_settings, output_dir, num_workers
        )
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Files for incremental saving
    results_file = output_path / "scalability_results.json"
    checkpoint_file = output_path / "experiment_checkpoint.json"
    
    # Load existing results and checkpoint if available
    all_results = []
    completed_experiments = set()
    experiment_id = 0
    
    if results_file.exists():
        print(f"📂 Loading existing results from {results_file}")
        try:
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            print(f"   Loaded {len(all_results)} existing results")
        except Exception as e:
            print(f"   ⚠️  Error loading results: {e}, starting fresh")
            all_results = []
    
    if checkpoint_file.exists():
        print(f"📂 Loading checkpoint from {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed_experiments = set(checkpoint.get('completed_experiments', []))
                experiment_id = checkpoint.get('last_experiment_id', 0)
            print(f"   Resuming from experiment {experiment_id}")
        except Exception as e:
            print(f"   ⚠️  Error loading checkpoint: {e}, starting fresh")
    
    print("🚀 Starting scalability experiments...")
    print(f"   Network sizes: {network_sizes}")
    print(f"   Topologies: {topologies}")
    print(f"   Attack strategies: {attack_strategies}")
    print(f"   DP settings: {len(dp_settings)}")
    
    # Calculate total experiments for progress tracking
    total_experiments = len(network_sizes) * len(topologies) * len(attack_strategies) * len(dp_settings)
    skipped_count = 0
    
    for size in network_sizes:
        for topology in topologies:
            for attack_strategy in attack_strategies:
                for dp_setting in dp_settings:
                    
                    # Create experiment key for deduplication
                    experiment_key = f"{size}_{topology}_{attack_strategy}_{dp_setting.get('name', 'no_dp')}"
                    
                    # Skip if already completed
                    if experiment_key in completed_experiments:
                        skipped_count += 1
                        continue
                    
                    config = NetworkConfig(
                        num_nodes=size,
                        topology=topology,
                        attack_strategy=attack_strategy,
                        dp_enabled=dp_setting.get('enabled', False),
                        dp_epsilon=dp_setting.get('epsilon'),
                        num_rounds=5  # Reduced for large-scale simulation
                    )
                    
                    current_experiment_num = experiment_id + 1
                    progress_pct = (len(completed_experiments) + skipped_count) / total_experiments * 100
                    print(f"📊 Experiment {current_experiment_num} ({progress_pct:.1f}% complete): {size} nodes, {topology}, {attack_strategy}, DP={dp_setting.get('name', 'off')}")
                    
                    try:
                        simulator = LargeScaleAttackSimulator(config)
                        result = simulator.simulate_attack_execution()
                        result['experiment_id'] = experiment_id
                        result['experiment_key'] = experiment_key
                        result['status'] = 'success'
                        
                        all_results.append(result)
                        completed_experiments.add(experiment_key)
                        
                        # Log brief result
                        eval_result = result['evaluation']
                        attack_success = eval_result.get('overall_success', False)
                        max_signal = eval_result.get('attack_indicators', {}).get('max_signal', 0.0)
                        print(f"   ✅ Attack {'SUCCESS' if attack_success else 'FAILED'} (signal: {max_signal:.3f})")
                        
                    except Exception as e:
                        print(f"   ❌ Experiment failed: {str(e)}")
                        all_results.append({
                            'experiment_id': experiment_id,
                            'experiment_key': experiment_key,
                            'config': config.__dict__,
                            'status': 'failed',
                            'error': str(e)
                        })
                        completed_experiments.add(experiment_key)
                    
                    # Incremental save after each experiment
                    try:
                        # Save results
                        with open(results_file, 'w') as f:
                            json_results = [_convert_for_json(r) for r in all_results]
                            json.dump(json_results, f, indent=2)
                        
                        # Save checkpoint
                        checkpoint_data = {
                            'last_experiment_id': experiment_id,
                            'completed_experiments': list(completed_experiments),
                            'total_experiments': total_experiments,
                            'timestamp': str(pd.Timestamp.now())
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                        
                    except Exception as e:
                        print(f"   ⚠️  Warning: Failed to save incremental results: {e}")
                    
                    experiment_id += 1
    
    # Final save to ensure everything is persisted
    print("\n💾 Finalizing results...")
    
    # Save final results one more time to ensure completeness
    try:
        with open(results_file, 'w') as f:
            json_results = [_convert_for_json(r) for r in all_results]
            json.dump(json_results, f, indent=2)
        print(f"   Results saved to: {results_file}")
    except Exception as e:
        print(f"   ⚠️  Error saving final results: {e}")
    
    # Generate analysis
    print("📊 Generating analysis...")
    analysis = analyze_scalability_results(all_results)
    analysis_file = output_path / "scalability_analysis.json"
    
    try:
        with open(analysis_file, 'w') as f:
            json.dump(_convert_for_json(analysis), f, indent=2)
        print(f"   Analysis saved to: {analysis_file}")
    except Exception as e:
        print(f"   ⚠️  Error saving analysis: {e}")
    
    # Clean up checkpoint file since we're done
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("   Removed checkpoint file")
    except Exception as e:
        print(f"   ⚠️  Could not remove checkpoint: {e}")
    
    # Calculate final statistics
    successful_experiments = len([r for r in all_results if r.get('status') == 'success'])
    actual_total = len(all_results)
    
    print("\n✅ Experiments completed!")
    print(f"   Total experiments run: {actual_total}")
    print(f"   Skipped (already completed): {skipped_count}")
    print(f"   Successful: {successful_experiments}")
    print(f"   Failed: {actual_total - successful_experiments}")
    
    return {
        'total_experiments': actual_total,
        'successful_experiments': successful_experiments,
        'skipped_experiments': skipped_count,
        'results_file': str(results_file),
        'analysis_file': str(analysis_file),
        'results': all_results
    }


def analyze_scalability_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze scalability experiment results."""
    
    successful_results = [r for r in results if r.get('status') == 'success']
    
    if not successful_results:
        return {'error': 'No successful results to analyze'}
    
    analysis = {
        'summary': {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(results)
        },
        'scaling_trends': {},
        'topology_analysis': {},
        'dp_effectiveness': {},
        'complexity_bounds': {}
    }
    
    # Extract data for analysis
    data_points = []
    for result in successful_results:
        config = result['config']
        evaluation = result['evaluation']
        
        data_points.append({
            'num_nodes': config['num_nodes'],
            'topology': config['topology'],
            'attack_strategy': config['attack_strategy'],
            'dp_enabled': config['dp_enabled'],
            'dp_epsilon': config.get('dp_epsilon'),
            'attack_success': evaluation.get('overall_success', False),
            'max_signal': evaluation.get('attack_indicators', {}).get('max_signal', 0.0)
        })
    
    df = pd.DataFrame(data_points)
    
    # Scaling trends analysis
    for topology in df['topology'].unique():
        topo_data = df[df['topology'] == topology]
        
        # Group by network size
        size_analysis = topo_data.groupby('num_nodes').agg({
            'attack_success': ['mean', 'count'],
            'max_signal': ['mean', 'std']
        }).round(3)
        
        analysis['scaling_trends'][topology] = {
            'attack_success_by_size': size_analysis['attack_success']['mean'].to_dict(),
            'signal_strength_by_size': size_analysis['max_signal']['mean'].to_dict(),
            'sample_sizes': size_analysis['attack_success']['count'].to_dict()
        }
    
    # Topology vulnerability ranking
    topo_vuln = df.groupby('topology').agg({
        'attack_success': 'mean',
        'max_signal': 'mean'
    }).sort_values('attack_success', ascending=False)
    
    analysis['topology_analysis'] = {
        'vulnerability_ranking': topo_vuln.to_dict('index'),
        'most_vulnerable': topo_vuln.index[0] if len(topo_vuln) > 0 else None,
        'least_vulnerable': topo_vuln.index[-1] if len(topo_vuln) > 0 else None
    }
    
    # DP effectiveness analysis
    if 'dp_enabled' in df.columns:
        dp_comparison = df.groupby('dp_enabled').agg({
            'attack_success': 'mean',
            'max_signal': 'mean'
        })
        
        analysis['dp_effectiveness'] = {
            'without_dp': dp_comparison.loc[False].to_dict() if False in dp_comparison.index else {},
            'with_dp': dp_comparison.loc[True].to_dict() if True in dp_comparison.index else {},
            'protection_effectiveness': {}
        }
        
        # Calculate protection effectiveness
        if False in dp_comparison.index and True in dp_comparison.index:
            no_dp_success = dp_comparison.loc[False, 'attack_success']
            dp_success = dp_comparison.loc[True, 'attack_success']
            protection_rate = (no_dp_success - dp_success) / max(no_dp_success, 0.001)
            
            analysis['dp_effectiveness']['protection_effectiveness'] = {
                'attack_reduction': protection_rate,
                'relative_protection': 1.0 - (dp_success / max(no_dp_success, 0.001))
            }
    
    # Complexity bounds analysis
    max_size = df['num_nodes'].max()
    min_size = df['num_nodes'].min()
    
    analysis['complexity_bounds'] = {
        'tested_size_range': {'min': int(min_size), 'max': int(max_size)},
        'extrapolation_validity': 'valid' if max_size >= 100 else 'limited',
        'recommended_max_extrapolation': int(max_size * 2),
        'theoretical_limits': {
            'information_theoretic': 1.0 - 1.0/math.log(max(2, max_size)),
            'network_effect_bound': 1.0 - 1.0/math.sqrt(max_size)
        }
    }
    
    return analysis


def _convert_for_json(obj):
    """Convert numpy types and other objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item') and hasattr(obj, 'size'):
        try:
            if obj.size == 1:
                return obj.item()
            else:
                return obj.tolist()
        except (ValueError, AttributeError):
            return str(obj)
    elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            return obj.tolist()
        except (ValueError, AttributeError):
            return str(obj)
    return obj