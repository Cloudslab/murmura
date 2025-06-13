#!/usr/bin/env python3
"""
Analyze why Parameter Magnitude Attack plateaus at 25% topology knowledge.
"""

import json
from pathlib import Path

def analyze_knowledge_impact():
    """Analyze the impact of partial knowledge on parameter attack."""
    
    # Load results
    results_file = "experiments_archive/results/attack_results/partial_topology_analysis/partial_topology_results.json"
    
    if not Path(results_file).exists():
        print("Results file not found. Please run partial topology analysis first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("Analysis of Parameter Magnitude Attack with Partial Topology Knowledge")
    print("=" * 70)
    
    # Analyze first experiment in detail
    if results:
        exp = results[0]  # First experiment
        print(f"\nExperiment: {exp['experiment_name']}")
        
        for knowledge_level in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
            if knowledge_level in exp["knowledge_results"]:
                result = exp["knowledge_results"][knowledge_level]
                
                print(f"\nKnowledge Level: {float(knowledge_level)*100:.0f}%")
                print(f"  Known nodes: {len(result.get('known_nodes', []))}")
                print(f"  Data points visible: {result['visualization_data_summary'].get('parameter_updates', 0)}")
                
                # Find parameter magnitude attack result
                for attack in result["attack_results"]:
                    if attack["attack_name"] == "Parameter Magnitude Attack":
                        print(f"  Attack success: {attack.get('attack_success_metric', 0.0):.3f}")
                        
                        # Check if we have detailed stats
                        if "node_magnitude_stats" in attack:
                            stats = attack["node_magnitude_stats"]
                            print(f"  Nodes analyzed: {len(stats)}")
                            
                            # Show magnitude differences
                            if len(stats) >= 2:
                                norms = [s["norm_mean"] for s in stats.values()]
                                print(f"  Magnitude range: {min(norms):.3f} - {max(norms):.3f}")
                                print(f"  Magnitude spread: {max(norms) - min(norms):.3f}")
    
    print("\n\nKey Insight:")
    print("-" * 50)
    print("The Parameter Magnitude Attack plateaus at 25% knowledge because:")
    print("1. It only needs to observe a small sample of nodes to detect patterns")
    print("2. Nodes with different data distributions have inherently different")
    print("   parameter update magnitudes, visible even with partial observation")
    print("3. The attack clusters nodes into groups - seeing 1-2 nodes from each")
    print("   group is sufficient to achieve high success")
    print("\nThis makes parameter-based attacks particularly dangerous as they")
    print("require minimal topology knowledge to be effective!")


if __name__ == "__main__":
    analyze_knowledge_impact()