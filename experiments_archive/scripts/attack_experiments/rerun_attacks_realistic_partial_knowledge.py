#!/usr/bin/env python3
"""
Re-run topology attacks with realistic partial knowledge scenarios.
These scenarios reflect real-world adversary capabilities.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
from murmura.attacks.topology_attacks import (
    CommunicationPatternAttack,
    ParameterMagnitudeAttack,
    TopologyStructureAttack,
    AttackEvaluator
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


class RealisticKnowledgeScenario:
    """Base class for realistic partial knowledge scenarios."""
    
    def __init__(self, scenario_name: str, seed: int = 42):
        self.scenario_name = scenario_name
        self.rng = np.random.RandomState(seed)
    
    def apply_knowledge(self, viz_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply the knowledge scenario to visualization data."""
        raise NotImplementedError


class NeighborhoodKnowledgeScenario(RealisticKnowledgeScenario):
    """
    Scenario 1: Adversary knows local topology within k hops from their position.
    Realistic for insider threats or compromised nodes.
    """
    
    def __init__(self, adversary_position: Optional[int] = None, k_hops: int = 2, seed: int = 42):
        super().__init__(f"Neighborhood-{k_hops}hop", seed)
        self.adversary_position = adversary_position
        self.k_hops = k_hops
    
    def apply_knowledge(self, viz_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply neighborhood knowledge masking."""
        masked_data = viz_data.copy()
        
        # Build network graph from topology
        if "topology" not in viz_data or viz_data["topology"].empty:
            return masked_data
        
        G = self._build_graph(viz_data["topology"])
        
        # Choose adversary position if not specified
        if self.adversary_position is None:
            self.adversary_position = self.rng.choice(list(G.nodes()))
        
        # Find nodes within k hops
        visible_nodes = set()
        visible_nodes.add(self.adversary_position)
        
        # BFS to find k-hop neighbors
        current_level = {self.adversary_position}
        for hop in range(self.k_hops):
            next_level = set()
            for node in current_level:
                next_level.update(G.neighbors(node))
            visible_nodes.update(next_level)
            current_level = next_level
        
        # Mask topology to only show visible nodes and their connections
        if "topology" in masked_data:
            topology_df = masked_data["topology"]
            masked_topology = topology_df[topology_df['node_id'].isin(visible_nodes)].copy()
            
            # Update connections to only show visible edges
            for idx in masked_topology.index:
                node_id = masked_topology.loc[idx, 'node_id']
                connected = str(masked_topology.loc[idx, 'connected_nodes']).split(',')
                visible_connected = [c for c in connected if c.strip().isdigit() and int(c.strip()) in visible_nodes]
                masked_topology.loc[idx, 'connected_nodes'] = ','.join(visible_connected)
                masked_topology.loc[idx, 'degree'] = len(visible_connected)
            
            masked_data["topology"] = masked_topology
        
        # Mask communications to only visible edges
        if "communications" in masked_data:
            comm_df = masked_data["communications"]
            mask = comm_df.apply(
                lambda row: row['source_node'] in visible_nodes and row['target_node'] in visible_nodes,
                axis=1
            )
            masked_data["communications"] = comm_df[mask]
        
        # Mask parameter updates to only visible nodes
        if "parameter_updates" in masked_data:
            param_df = masked_data["parameter_updates"]
            masked_data["parameter_updates"] = param_df[param_df['node_id'].isin(visible_nodes)]
        
        return masked_data
    
    def _build_graph(self, topology_df: pd.DataFrame) -> nx.Graph:
        """Build NetworkX graph from topology dataframe."""
        G = nx.Graph()
        
        for _, row in topology_df.iterrows():
            node_id = row['node_id']
            G.add_node(node_id)
            
            connected = str(row['connected_nodes']).split(',')
            for neighbor in connected:
                if neighbor.strip().isdigit():
                    G.add_edge(node_id, int(neighbor.strip()))
        
        return G


class StatisticalKnowledgeScenario(RealisticKnowledgeScenario):
    """
    Scenario 2: Adversary knows topology type (star/ring/complete) but not exact structure.
    Realistic for external adversaries with domain knowledge.
    """
    
    def __init__(self, inferred_topology_type: Optional[str] = None, seed: int = 42):
        super().__init__("Statistical", seed)
        self.inferred_topology_type = inferred_topology_type
    
    def apply_knowledge(self, viz_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply statistical knowledge - provide aggregate statistics only."""
        masked_data = viz_data.copy()
        
        if "topology" not in viz_data or viz_data["topology"].empty:
            return masked_data
        
        # Infer topology type if not provided
        if self.inferred_topology_type is None:
            self.inferred_topology_type = self._infer_topology_type(viz_data["topology"])
        
        # Create synthetic topology based on type
        n_nodes = len(viz_data["topology"])
        synthetic_topology = self._create_synthetic_topology(n_nodes, self.inferred_topology_type)
        masked_data["topology"] = synthetic_topology
        
        # For communications, preserve temporal patterns but anonymize nodes
        if "communications" in masked_data:
            comm_df = masked_data["communications"].copy()
            
            # Create node mapping to anonymize
            unique_nodes = set(comm_df['source_node'].unique()) | set(comm_df['target_node'].unique())
            node_mapping = {node: i for i, node in enumerate(sorted(unique_nodes))}
            
            # Apply mapping
            comm_df['source_node'] = comm_df['source_node'].map(node_mapping)
            comm_df['target_node'] = comm_df['target_node'].map(node_mapping)
            
            masked_data["communications"] = comm_df
        
        # For parameter updates, preserve statistical properties but shuffle node assignments
        if "parameter_updates" in masked_data:
            param_df = masked_data["parameter_updates"].copy()
            
            # Shuffle node IDs while preserving temporal structure
            for round_num in param_df['round_num'].unique():
                round_mask = param_df['round_num'] == round_num
                round_data = param_df[round_mask]
                
                # Shuffle node assignments for this round
                node_ids = round_data['node_id'].values
                self.rng.shuffle(node_ids)
                param_df.loc[round_mask, 'node_id'] = node_ids
            
            masked_data["parameter_updates"] = param_df
        
        return masked_data
    
    def _infer_topology_type(self, topology_df: pd.DataFrame) -> str:
        """Infer topology type from structure."""
        degrees = topology_df['degree'].values
        n_nodes = len(topology_df)
        
        # Check for star (one hub with high degree)
        if max(degrees) == n_nodes - 1 and min(degrees) == 1:
            return "star"
        
        # Check for complete (all nodes connected)
        elif all(d == n_nodes - 1 for d in degrees):
            return "complete"
        
        # Check for ring (all nodes have degree 2)
        elif all(d == 2 for d in degrees):
            return "ring"
        
        # Check for line (two nodes have degree 1, rest have degree 2)
        elif sum(d == 1 for d in degrees) == 2 and all(d <= 2 for d in degrees):
            return "line"
        
        else:
            return "unknown"
    
    def _create_synthetic_topology(self, n_nodes: int, topology_type: str) -> pd.DataFrame:
        """Create synthetic topology of given type."""
        topology_data = []
        
        if topology_type == "star":
            # Node 0 is hub
            for i in range(n_nodes):
                if i == 0:
                    connected = ','.join(str(j) for j in range(1, n_nodes))
                    degree = n_nodes - 1
                else:
                    connected = '0'
                    degree = 1
                
                topology_data.append({
                    'node_id': i,
                    'connected_nodes': connected,
                    'degree': degree
                })
        
        elif topology_type == "complete":
            for i in range(n_nodes):
                connected = ','.join(str(j) for j in range(n_nodes) if j != i)
                topology_data.append({
                    'node_id': i,
                    'connected_nodes': connected,
                    'degree': n_nodes - 1
                })
        
        elif topology_type == "ring":
            for i in range(n_nodes):
                prev_node = (i - 1) % n_nodes
                next_node = (i + 1) % n_nodes
                topology_data.append({
                    'node_id': i,
                    'connected_nodes': f'{prev_node},{next_node}',
                    'degree': 2
                })
        
        else:  # Default to random
            for i in range(n_nodes):
                topology_data.append({
                    'node_id': i,
                    'connected_nodes': '',
                    'degree': 0
                })
        
        return pd.DataFrame(topology_data)


class OrganizationalKnowledgeScenario(RealisticKnowledgeScenario):
    """
    Scenario 3: Adversary knows organizational structure but not network details.
    Realistic for adversaries with insider information about departments/groups.
    """
    
    def __init__(self, n_organizations: int = 3, seed: int = 42):
        super().__init__(f"Organizational-{n_organizations}groups", seed)
        self.n_organizations = n_organizations
    
    def apply_knowledge(self, viz_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply organizational knowledge - group nodes by inferred organizations."""
        masked_data = viz_data.copy()
        
        if "topology" not in viz_data or viz_data["topology"].empty:
            return masked_data
        
        n_nodes = len(viz_data["topology"])
        
        # Assign nodes to organizations (could be based on communication patterns in reality)
        org_assignments = self._infer_organizations(viz_data)
        
        # Create coarse topology showing only inter-org connections
        if "topology" in masked_data:
            coarse_topology = self._create_organizational_topology(
                viz_data["topology"], org_assignments
            )
            masked_data["topology"] = coarse_topology
        
        # Aggregate communications by organization
        if "communications" in masked_data:
            comm_df = masked_data["communications"].copy()
            
            # Map nodes to organizations
            comm_df['source_org'] = comm_df['source_node'].map(org_assignments)
            comm_df['target_org'] = comm_df['target_node'].map(org_assignments)
            
            # Aggregate by organization pairs
            org_comm = comm_df.groupby(['round_num', 'source_org', 'target_org']).agg({
                'timestamp': 'mean',
                'source_node': 'count'  # Count as communication volume
            }).reset_index()
            
            org_comm.rename(columns={'source_node': 'comm_count'}, inplace=True)
            org_comm['source_node'] = org_comm['source_org']
            org_comm['target_node'] = org_comm['target_org']
            
            masked_data["communications"] = org_comm[['round_num', 'timestamp', 'source_node', 'target_node']]
        
        # Aggregate parameter updates by organization
        if "parameter_updates" in masked_data:
            param_df = masked_data["parameter_updates"].copy()
            
            # Map nodes to organizations
            param_df['organization'] = param_df['node_id'].map(org_assignments)
            
            # Aggregate by organization
            org_params = param_df.groupby(['round_num', 'organization']).agg({
                'parameter_norm': ['mean', 'std'],
                'timestamp': 'mean'
            }).reset_index()
            
            org_params.columns = ['round_num', 'node_id', 'parameter_norm', 'norm_std', 'timestamp']
            masked_data["parameter_updates"] = org_params
        
        return masked_data
    
    def _infer_organizations(self, viz_data: Dict[str, pd.DataFrame]) -> Dict[int, int]:
        """Infer organizational structure from communication patterns."""
        org_assignments = {}
        
        if "communications" in viz_data and not viz_data["communications"].empty:
            # Use communication frequency to cluster nodes
            comm_df = viz_data["communications"]
            nodes = set(comm_df['source_node'].unique()) | set(comm_df['target_node'].unique())
            
            # Simple assignment based on node ID ranges
            nodes_list = sorted(nodes)
            nodes_per_org = len(nodes_list) // self.n_organizations
            
            for i, node in enumerate(nodes_list):
                org_id = min(i // nodes_per_org, self.n_organizations - 1)
                org_assignments[node] = org_id
        
        else:
            # Fallback: assign based on topology
            topology_df = viz_data["topology"]
            n_nodes = len(topology_df)
            nodes_per_org = n_nodes // self.n_organizations
            
            for i in range(n_nodes):
                org_id = min(i // nodes_per_org, self.n_organizations - 1)
                org_assignments[i] = org_id
        
        return org_assignments
    
    def _create_organizational_topology(
        self, 
        topology_df: pd.DataFrame, 
        org_assignments: Dict[int, int]
    ) -> pd.DataFrame:
        """Create coarse topology showing organizational structure."""
        org_topology = []
        
        # Create one super-node per organization
        for org_id in range(self.n_organizations):
            # Find all inter-org connections
            org_nodes = [n for n, o in org_assignments.items() if o == org_id]
            connected_orgs = set()
            
            for node in org_nodes:
                node_row = topology_df[topology_df['node_id'] == node]
                if not node_row.empty:
                    connected = str(node_row.iloc[0]['connected_nodes']).split(',')
                    for neighbor in connected:
                        if neighbor.strip().isdigit():
                            neighbor_id = int(neighbor.strip())
                            if neighbor_id in org_assignments:
                                neighbor_org = org_assignments[neighbor_id]
                                if neighbor_org != org_id:
                                    connected_orgs.add(neighbor_org)
            
            org_topology.append({
                'node_id': org_id,
                'connected_nodes': ','.join(str(o) for o in sorted(connected_orgs)),
                'degree': len(connected_orgs),
                'organization_size': len(org_nodes)
            })
        
        return pd.DataFrame(org_topology)


def run_attacks_with_scenario(
    visualization_dir: str,
    scenario: RealisticKnowledgeScenario
) -> Dict[str, Any]:
    """Run topology attacks with a specific knowledge scenario."""
    
    # Load original visualization data
    viz_data = {}
    data_files = {
        "communications": "training_data_communications.csv",
        "parameter_updates": "training_data_parameter_updates.csv",
        "topology": "training_data_topology.csv",
        "metrics": "training_data_metrics.csv",
    }
    
    for data_type, filename in data_files.items():
        filepath = os.path.join(visualization_dir, filename)
        if os.path.exists(filepath):
            try:
                viz_data[data_type] = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Apply knowledge scenario
    masked_data = scenario.apply_knowledge(viz_data)
    
    # Initialize and run attacks
    attacks = [
        CommunicationPatternAttack(),
        ParameterMagnitudeAttack(),
        TopologyStructureAttack(),
    ]
    
    attack_results = []
    for attack in attacks:
        try:
            result = attack.execute_attack(masked_data)
            result["attack_name"] = attack.name
            result["scenario"] = scenario.scenario_name
            attack_results.append(result)
        except Exception as e:
            attack_results.append({
                "attack_name": attack.name,
                "error": str(e),
                "attack_success_metric": 0.0,
                "scenario": scenario.scenario_name
            })
    
    return {
        "attack_results": attack_results,
        "scenario": scenario.scenario_name,
        "visualization_data_summary": {
            k: len(v) if hasattr(v, '__len__') else 0 
            for k, v in masked_data.items()
        },
    }


def run_experiments_with_realistic_scenarios(
    visualization_base_dir: str = "experiments_archive/results/training_visualizations/visualizations_phase1",
    output_dir: str = "experiments_archive/results/attack_results/realistic_knowledge_analysis",
    experiment_filter: Optional[List[str]] = None,
    min_nodes: int = 10  # Only run on experiments with >= 10 nodes
) -> None:
    """Run attacks with realistic partial knowledge scenarios."""
    
    print(f"Starting realistic partial knowledge analysis...")
    print(f"Minimum nodes required: {min_nodes}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of experiment directories
    viz_base_path = Path(visualization_base_dir)
    if not viz_base_path.exists():
        print(f"Error: Visualization base directory not found: {visualization_base_dir}")
        return
    
    experiment_dirs = [d for d in viz_base_path.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    
    # Filter by node count
    filtered_dirs = []
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        # Extract node count from name (e.g., "10n" -> 10)
        parts = exp_name.split('_')
        for part in parts:
            if part.endswith('n'):
                try:
                    node_count = int(part[:-1])
                    if node_count >= min_nodes:
                        filtered_dirs.append(exp_dir)
                    break
                except ValueError:
                    continue
    
    experiment_dirs = filtered_dirs
    experiment_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    # Apply additional filter if provided
    if experiment_filter:
        experiment_dirs = [d for d in experiment_dirs if d.name in experiment_filter]
    
    print(f"Found {len(experiment_dirs)} experiments with >= {min_nodes} nodes")
    
    # Define scenarios to test
    scenarios_to_test = [
        ("Complete Knowledge (Baseline)", None),
        ("Neighborhood 1-hop", lambda: NeighborhoodKnowledgeScenario(k_hops=1)),
        ("Neighborhood 2-hop", lambda: NeighborhoodKnowledgeScenario(k_hops=2)),
        ("Statistical Knowledge", lambda: StatisticalKnowledgeScenario()),
        ("Organizational 3-groups", lambda: OrganizationalKnowledgeScenario(n_organizations=3)),
        ("Organizational 5-groups", lambda: OrganizationalKnowledgeScenario(n_organizations=5)),
    ]
    
    # Process each experiment
    all_results = []
    
    for exp_idx, exp_dir in enumerate(experiment_dirs):
        exp_name = exp_dir.name
        viz_dir = str(exp_dir)
        
        print(f"\n[{exp_idx+1}/{len(experiment_dirs)}] Processing {exp_name}")
        
        exp_results = {
            "experiment_name": exp_name,
            "scenario_results": {}
        }
        
        for scenario_name, scenario_factory in scenarios_to_test:
            print(f"  - {scenario_name}")
            
            start_time = time.time()
            
            if scenario_factory is None:
                # Baseline: complete knowledge
                from murmura.attacks.topology_attacks import run_topology_attacks
                result = run_topology_attacks(viz_dir)
                result["scenario"] = "Complete Knowledge"
            else:
                scenario = scenario_factory()
                result = run_attacks_with_scenario(viz_dir, scenario)
            
            runtime = time.time() - start_time
            result["runtime_seconds"] = runtime
            
            exp_results["scenario_results"][scenario_name] = result
        
        all_results.append(exp_results)
        
        # Save intermediate results
        if (exp_idx + 1) % 3 == 0:
            save_results(all_results, os.path.join(output_dir, "realistic_knowledge_results.json"))
    
    # Save final results
    save_results(all_results, os.path.join(output_dir, "realistic_knowledge_results.json"))
    
    # Generate analysis summary
    generate_scenario_summary(all_results, output_dir)


def convert_keys_to_str(obj):
    """Recursively convert dictionary keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    else:
        return obj


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to JSON file."""
    results_converted = convert_keys_to_str(results)
    with open(output_file, 'w') as f:
        json.dump(results_converted, f, indent=2, cls=NumpyEncoder)


def generate_scenario_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate summary analysis of realistic scenarios."""
    
    print("\nGenerating realistic scenario analysis summary...")
    
    # Collect metrics by scenario and attack type
    metrics_by_scenario = {}
    
    for exp_result in results:
        exp_name = exp_result["experiment_name"]
        
        for scenario_name, scenario_result in exp_result["scenario_results"].items():
            if scenario_name not in metrics_by_scenario:
                metrics_by_scenario[scenario_name] = {
                    "Communication Pattern Attack": [],
                    "Parameter Magnitude Attack": [],
                    "Topology Structure Attack": []
                }
            
            # Handle both direct attack results and nested structure
            attack_results = scenario_result.get("attack_results", [])
            if not attack_results and "attack_results" in scenario_result:
                # Nested structure from baseline
                attack_results = scenario_result["attack_results"].get("attack_results", [])
            
            for attack_result in attack_results:
                attack_name = attack_result["attack_name"]
                success_metric = attack_result.get("attack_success_metric", 0.0)
                
                if attack_name in metrics_by_scenario[scenario_name]:
                    metrics_by_scenario[scenario_name][attack_name].append({
                        "experiment": exp_name,
                        "metric": success_metric
                    })
    
    # Calculate averages and create summary
    summary = {
        "scenario_effectiveness": {},
        "attack_robustness": {},
        "recommendations": []
    }
    
    # Baseline for comparison
    baseline_metrics = {}
    if "Complete Knowledge (Baseline)" in metrics_by_scenario:
        for attack_name, metrics in metrics_by_scenario["Complete Knowledge (Baseline)"].items():
            if metrics:
                baseline_metrics[attack_name] = np.mean([m["metric"] for m in metrics])
    
    # Analyze each scenario
    for scenario_name in metrics_by_scenario:
        scenario_summary = {}
        
        for attack_name, metrics in metrics_by_scenario[scenario_name].items():
            if metrics:
                avg_metric = np.mean([m["metric"] for m in metrics])
                scenario_summary[attack_name] = {
                    "average_success": avg_metric,
                    "n_experiments": len(metrics),
                    "reduction_from_baseline": (baseline_metrics.get(attack_name, 0) - avg_metric) / baseline_metrics.get(attack_name, 1) * 100 if baseline_metrics.get(attack_name, 0) > 0 else 0
                }
        
        summary["scenario_effectiveness"][scenario_name] = scenario_summary
    
    # Generate recommendations
    for scenario in ["Neighborhood 1-hop", "Neighborhood 2-hop", "Statistical Knowledge", "Organizational 3-groups"]:
        if scenario in summary["scenario_effectiveness"]:
            scenario_data = summary["scenario_effectiveness"][scenario]
            
            # Check which attacks remain effective
            effective_attacks = []
            for attack_name, data in scenario_data.items():
                if data["average_success"] > 0.5:
                    effective_attacks.append(attack_name.replace(" Attack", ""))
            
            if effective_attacks:
                summary["recommendations"].append(
                    f"{scenario}: {', '.join(effective_attacks)} remain effective (>50% success)"
                )
    
    # Save summary
    summary_file = os.path.join(output_dir, "realistic_scenario_summary.json")
    summary_converted = convert_keys_to_str(summary)
    with open(summary_file, 'w') as f:
        json.dump(summary_converted, f, indent=2, cls=NumpyEncoder)
    
    # Print key findings
    print("\nKey Findings:")
    print("-" * 70)
    
    for scenario in ["Neighborhood 1-hop", "Neighborhood 2-hop", "Statistical Knowledge", 
                     "Organizational 3-groups", "Organizational 5-groups"]:
        if scenario in summary["scenario_effectiveness"]:
            print(f"\n{scenario}:")
            scenario_data = summary["scenario_effectiveness"][scenario]
            
            for attack_name in ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]:
                if attack_name in scenario_data:
                    data = scenario_data[attack_name]
                    print(f"  {attack_name.replace(' Attack', '')}: {data['average_success']:.3f} success " +
                          f"(-{data['reduction_from_baseline']:.1f}% from baseline)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test on a few experiments
            print("Running test with realistic scenarios...")
            # First check what experiments we have
            base_dir = "experiments_archive/results/training_visualizations/visualizations_phase1"
            from pathlib import Path
            
            # List available experiments
            if Path(base_dir).exists():
                exps = [d.name for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('exp_')]
                print(f"Available experiments: {exps[:10]}...")  # Show first 10
            
            run_experiments_with_realistic_scenarios(
                min_nodes=5,  # Lower threshold for testing
                experiment_filter=None  # Will process all with >=5 nodes
            )
        else:
            print("Usage:")
            print("  python rerun_attacks_realistic_partial_knowledge.py --test  # Test run")
            print("  python rerun_attacks_realistic_partial_knowledge.py        # Full run (10+ nodes)")
    else:
        # Full run on 10+ node experiments
        run_experiments_with_realistic_scenarios()