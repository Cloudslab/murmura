"""
Helper functions for integrating trust monitoring with visualization.
"""

from typing import Dict, Any, Optional
from murmura.trust_monitoring.trust_events import TrustScoreEvent
from murmura.visualization.network_visualizer import NetworkVisualizer


def integrate_trust_monitoring_with_visualizer(
    visualizer: NetworkVisualizer,
    trust_monitors: Dict[str, Any],
    round_num: int
) -> None:
    """
    Integrate trust monitoring data with the network visualizer.
    
    Args:
        visualizer: The network visualizer instance
        trust_monitors: Dictionary of trust monitors {node_id: trust_monitor}
        round_num: Current training round
    """
    for node_id, trust_monitor in trust_monitors.items():
        if hasattr(trust_monitor, 'get_trust_summary'):
            summary = trust_monitor.get_trust_summary()
            
            # Create trust score event for visualization
            if summary.get('trust_scores'):
                trust_event = TrustScoreEvent(
                    node_id=node_id,
                    round_num=round_num,
                    trust_scores=summary['trust_scores'],
                    score_changes=summary.get('trust_velocities', {}),
                    detection_method="adaptive_trust_scoring"
                )
                
                # Process the event through the visualizer
                visualizer.on_event(trust_event)
                
            # Add influence weights if available
            if summary.get('influence_weights'):
                if hasattr(visualizer, 'add_influence_weights_data'):
                    visualizer.add_influence_weights_data(
                        observer_node=node_id,
                        round_num=round_num,
                        influence_weights=summary['influence_weights']
                    )


def create_trust_summary_for_round(
    trust_monitors: Dict[str, Any],
    round_num: int
) -> Dict[str, Dict[str, float]]:
    """
    Create a consolidated trust summary for all monitors at a given round.
    
    Args:
        trust_monitors: Dictionary of trust monitors
        round_num: Training round number
        
    Returns:
        Dictionary mapping observer nodes to their trust scores
    """
    trust_summary = {}
    
    for node_id, trust_monitor in trust_monitors.items():
        if hasattr(trust_monitor, 'get_trust_summary'):
            summary = trust_monitor.get_trust_summary()
            if summary.get('trust_scores'):
                trust_summary[node_id] = summary['trust_scores']
    
    return trust_summary


def export_trust_evolution_data(
    trust_monitors: Dict[str, Any],
    output_path: str
) -> None:
    """
    Export detailed trust evolution data for post-processing analysis.
    
    Args:
        trust_monitors: Dictionary of trust monitors
        output_path: Path to export the data
    """
    import json
    
    trust_evolution = {}
    
    for node_id, trust_monitor in trust_monitors.items():
        if hasattr(trust_monitor, 'get_trust_summary'):
            summary = trust_monitor.get_trust_summary()
            
            trust_evolution[node_id] = {
                'final_trust_scores': summary.get('trust_scores', {}),
                'influence_weights': summary.get('influence_weights', {}),
                'trust_statistics': summary.get('trust_statistics', {}),
                'suspicious_neighbors': summary.get('suspicious_neighbors', []),
                'rounds_completed': summary.get('rounds_completed', 0),
                'monitoring_enabled': summary.get('monitoring_enabled', False)
            }
    
    with open(output_path, 'w') as f:
        json.dump(trust_evolution, f, indent=2)
    
    print(f"Trust evolution data exported to {output_path}")


def analyze_trust_network_health(
    trust_monitors: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze the overall health of the trust network.
    
    Args:
        trust_monitors: Dictionary of trust monitors
        
    Returns:
        Network health analysis
    """
    health_metrics = {
        'total_nodes': len(trust_monitors),
        'monitoring_enabled_nodes': 0,
        'average_trust_scores': {},
        'low_trust_pairs': [],
        'network_connectivity': 1.0,
        'suspicious_activity': False
    }
    
    all_trust_scores = []
    low_trust_threshold = 0.3
    
    for node_id, trust_monitor in trust_monitors.items():
        if hasattr(trust_monitor, 'get_trust_summary'):
            summary = trust_monitor.get_trust_summary()
            
            if summary.get('monitoring_enabled', False):
                health_metrics['monitoring_enabled_nodes'] += 1
            
            trust_scores = summary.get('trust_scores', {})
            for neighbor_id, trust_score in trust_scores.items():
                all_trust_scores.append(trust_score)
                
                # Track low trust pairs
                if trust_score < low_trust_threshold:
                    health_metrics['low_trust_pairs'].append({
                        'observer': node_id,
                        'neighbor': neighbor_id,
                        'trust_score': trust_score
                    })
            
            # Check for suspicious neighbors
            if summary.get('suspicious_neighbors'):
                health_metrics['suspicious_activity'] = True
    
    # Calculate network-wide metrics
    if all_trust_scores:
        import numpy as np
        health_metrics['average_trust_scores'] = {
            'mean': float(np.mean(all_trust_scores)),
            'std': float(np.std(all_trust_scores)),
            'min': float(np.min(all_trust_scores)),
            'max': float(np.max(all_trust_scores))
        }
        
        # Network connectivity based on trust scores above threshold
        high_trust_count = sum(1 for score in all_trust_scores if score > 0.7)
        health_metrics['network_connectivity'] = high_trust_count / len(all_trust_scores)
    
    return health_metrics


def generate_trust_monitoring_report(
    trust_monitors: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate a comprehensive trust monitoring report.
    
    Args:
        trust_monitors: Dictionary of trust monitors
        output_path: Path to save the report
    """
    health_analysis = analyze_trust_network_health(trust_monitors)
    
    report = []
    report.append("# Trust Monitoring Report\\n")
    report.append(f"**Total Nodes**: {health_analysis['total_nodes']}")
    report.append(f"**Monitoring Enabled**: {health_analysis['monitoring_enabled_nodes']}")
    report.append(f"**Network Connectivity**: {health_analysis['network_connectivity']:.3f}")
    report.append(f"**Suspicious Activity Detected**: {health_analysis['suspicious_activity']}\\n")
    
    if health_analysis['average_trust_scores']:
        stats = health_analysis['average_trust_scores']
        report.append("## Trust Score Statistics")
        report.append(f"- **Mean**: {stats['mean']:.3f}")
        report.append(f"- **Std Dev**: {stats['std']:.3f}")
        report.append(f"- **Min**: {stats['min']:.3f}")
        report.append(f"- **Max**: {stats['max']:.3f}\\n")
    
    if health_analysis['low_trust_pairs']:
        report.append("## Low Trust Relationships")
        for pair in health_analysis['low_trust_pairs']:
            report.append(f"- Node {pair['observer']} → Node {pair['neighbor']}: {pair['trust_score']:.3f}")
        report.append("")
    
    # Add individual node summaries
    report.append("## Individual Node Status")
    for node_id, trust_monitor in trust_monitors.items():
        if hasattr(trust_monitor, 'get_trust_summary'):
            summary = trust_monitor.get_trust_summary()
            report.append(f"### Node {node_id}")
            report.append(f"- **Monitoring**: {'Enabled' if summary.get('monitoring_enabled') else 'Disabled'}")
            report.append(f"- **Neighbors Monitored**: {summary.get('total_neighbors', 0)}")
            
            if summary.get('suspicious_neighbors'):
                report.append(f"- **Suspicious Neighbors**: {', '.join(summary['suspicious_neighbors'])}")
            
            if summary.get('trust_statistics'):
                stats = summary['trust_statistics']
                report.append(f"- **Mean Trust**: {stats.get('mean_trust', 0):.3f}")
                report.append(f"- **Min Trust**: {stats.get('min_trust', 0):.3f}")
            
            report.append("")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\\n'.join(report))
    
    print(f"Trust monitoring report saved to {output_path}")