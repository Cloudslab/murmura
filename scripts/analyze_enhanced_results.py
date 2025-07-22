#!/usr/bin/env python3
"""
Enhanced Results Analysis Script for EdgeDrift Trust Monitoring Evaluation
Analyzes trust monitoring performance across different node counts and configurations.
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

def parse_experiment_filename(filename: str) -> Dict[str, str]:
    """Parse experiment filename to extract parameters."""
    # Format: dataset_topology_attack_nX_trust/baseline.txt
    basename = Path(filename).stem
    parts = basename.split('_')
    
    if len(parts) >= 5:
        return {
            'dataset': parts[0],
            'topology': parts[1], 
            'attack': parts[2],
            'node_count': int(parts[3].replace('n', '')),
            'trust_enabled': parts[4] == 'trust'
        }
    else:
        print(f"Warning: Could not parse filename {filename}")
        return {}

def extract_metrics_from_file(filepath: str) -> Dict[str, any]:
    """Extract key metrics from experiment output file."""
    metrics = {
        'status': 'FAILED',
        'final_accuracy': None,
        'accuracy_improvement': None,
        'initial_accuracy': None,
        'detection_count': 0,
        'detection_round': None,
        'malicious_nodes': [],
        'detected_nodes': [],
        'total_rounds': 10
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if experiment completed successfully
        if 'Final Test Accuracy' in content:
            metrics['status'] = 'SUCCESS'
        
        # Extract accuracy metrics
        final_acc_match = re.search(r'Final Test Accuracy: ([0-9.]+)%', content)
        if final_acc_match:
            metrics['final_accuracy'] = float(final_acc_match.group(1))
        
        initial_acc_match = re.search(r'Initial Test Accuracy: ([0-9.]+)%', content)
        if initial_acc_match:
            metrics['initial_accuracy'] = float(initial_acc_match.group(1))
        
        acc_improvement_match = re.search(r'Accuracy Improvement: ([-+]?[0-9.]+)%', content)
        if acc_improvement_match:
            metrics['accuracy_improvement'] = float(acc_improvement_match.group(1))
        
        # Extract trust monitoring results
        malicious_match = re.search(r'Known malicious clients: \[([^\]]+)\]', content)
        if malicious_match:
            malicious_str = malicious_match.group(1)
            metrics['malicious_nodes'] = [int(x.strip()) for x in malicious_str.split(',')]
        
        detection_match = re.search(r'Trust monitoring detected ([0-9]+) suspicious neighbors', content)
        if detection_match:
            metrics['detection_count'] = int(detection_match.group(1))
        
        # Find detection round
        detection_rounds = re.findall(r'Round ([0-9]+):.*detected suspicious neighbors', content)
        if detection_rounds:
            metrics['detection_round'] = int(detection_rounds[0])
        
        # Extract detected neighbors from final summary
        detected_match = re.search(r'"global_suspicious_detected": \[([^\]]*)\]', content)
        if detected_match and detected_match.group(1).strip():
            detected_str = detected_match.group(1)
            # Parse the detected nodes (format: "node_X")
            detected_nodes = re.findall(r'"node_([0-9]+)"', detected_str)
            metrics['detected_nodes'] = [int(x) for x in detected_nodes]
    
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    
    return metrics

def calculate_detection_metrics(malicious_nodes: List[int], detected_nodes: List[int]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for detection."""
    if not malicious_nodes:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    
    malicious_set = set(malicious_nodes)
    detected_set = set(detected_nodes)
    
    true_positives = len(malicious_set & detected_set)
    false_positives = len(detected_set - malicious_set)
    false_negatives = len(malicious_set - detected_set)
    
    precision = true_positives / len(detected_set) if detected_set else 0.0
    recall = true_positives / len(malicious_set) if malicious_set else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def analyze_results_directory(results_dir: str) -> pd.DataFrame:
    """Analyze all experiment results in the directory."""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt') and not filename.startswith('experiment_summary'):
            filepath = os.path.join(results_dir, filename)
            
            # Parse experiment parameters
            params = parse_experiment_filename(filename)
            if not params:
                continue
            
            # Extract metrics from file
            metrics = extract_metrics_from_file(filepath)
            
            # Calculate detection performance
            detection_metrics = calculate_detection_metrics(
                metrics['malicious_nodes'], 
                metrics['detected_nodes']
            )
            
            # Combine all data
            result = {
                **params,
                **metrics,
                **detection_metrics,
                'filename': filename
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations of the results."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Trust vs Baseline Accuracy Comparison
    plt.figure(figsize=(15, 10))
    
    for i, (dataset, attack) in enumerate([('mnist', 'gradient'), ('mnist', 'label_flip'), 
                                          ('cifar10', 'gradient'), ('cifar10', 'label_flip')]):
        plt.subplot(2, 2, i+1)
        
        subset = df[(df['dataset'] == dataset) & (df['attack'] == attack) & (df['status'] == 'SUCCESS')]
        
        if not subset.empty:
            trust_data = subset[subset['trust_enabled'] == True]
            baseline_data = subset[subset['trust_enabled'] == False]
            
            if not trust_data.empty and not baseline_data.empty:
                plt.plot(trust_data['node_count'], trust_data['final_accuracy'], 'o-', 
                        label='Trust-weighted', linewidth=2, markersize=8)
                plt.plot(baseline_data['node_count'], baseline_data['final_accuracy'], 's--', 
                        label='Baseline', linewidth=2, markersize=8)
        
        plt.title(f'{dataset.upper()} - {attack.replace("_", " ").title()}')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Final Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'trust_vs_baseline_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detection Performance by Node Count
    plt.figure(figsize=(12, 8))
    
    trust_only = df[(df['trust_enabled'] == True) & (df['status'] == 'SUCCESS')]
    if not trust_only.empty:
        detection_summary = trust_only.groupby(['dataset', 'attack', 'node_count']).agg({
            'recall': 'mean',
            'precision': 'mean',
            'f1_score': 'mean'
        }).reset_index()
        
        for dataset in ['mnist', 'cifar10']:
            for attack in ['gradient', 'label_flip']:
                subset = detection_summary[(detection_summary['dataset'] == dataset) & 
                                         (detection_summary['attack'] == attack)]
                if not subset.empty:
                    label = f'{dataset.upper()} {attack.replace("_", " ")}'
                    plt.plot(subset['node_count'], subset['recall'], 'o-', label=label, linewidth=2)
    
    plt.title('Detection Recall vs Network Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Detection Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'detection_recall_vs_nodes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scalability Analysis - Accuracy Improvement
    plt.figure(figsize=(12, 8))
    
    successful_experiments = df[df['status'] == 'SUCCESS']
    if not successful_experiments.empty:
        # Calculate accuracy improvement: trust - baseline
        improvements = []
        
        for (dataset, topology, attack, node_count), group in successful_experiments.groupby(['dataset', 'topology', 'attack', 'node_count']):
            trust_acc = group[group['trust_enabled'] == True]['final_accuracy'].values
            baseline_acc = group[group['trust_enabled'] == False]['final_accuracy'].values
            
            if len(trust_acc) > 0 and len(baseline_acc) > 0:
                improvement = trust_acc[0] - baseline_acc[0]
                improvements.append({
                    'dataset': dataset,
                    'topology': topology,
                    'attack': attack,
                    'node_count': node_count,
                    'accuracy_improvement': improvement
                })
        
        if improvements:
            improvement_df = pd.DataFrame(improvements)
            
            for dataset in ['mnist', 'cifar10']:
                subset = improvement_df[improvement_df['dataset'] == dataset]
                if not subset.empty:
                    for attack in subset['attack'].unique():
                        attack_data = subset[subset['attack'] == attack]
                        if not attack_data.empty:
                            plt.plot(attack_data['node_count'], attack_data['accuracy_improvement'], 
                                   'o-', label=f'{dataset.upper()} {attack}', linewidth=2)
    
    plt.title('Trust-weighted vs Baseline Accuracy Improvement')
    plt.xlabel('Number of Nodes') 
    plt.ylabel('Accuracy Improvement (%)')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'accuracy_improvement_vs_nodes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of Detection Performance
    trust_results = df[(df['trust_enabled'] == True) & (df['status'] == 'SUCCESS')]
    if not trust_results.empty:
        pivot_data = trust_results.pivot_table(
            values='f1_score', 
            index=['dataset', 'attack'], 
            columns='node_count', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', vmin=0, vmax=1, 
                   fmt='.2f', cbar_kws={'label': 'F1 Score'})
        plt.title('Detection F1 Score by Dataset/Attack and Node Count')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'detection_f1_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Visualizations saved to: {plots_dir}")

def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate comprehensive summary report."""
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("EdgeDrift Trust Monitoring Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Successful experiments: {len(df[df['status'] == 'SUCCESS'])}\n")
        f.write(f"Failed experiments: {len(df[df['status'] == 'FAILED'])}\n")
        f.write(f"Trust-enabled experiments: {len(df[df['trust_enabled'] == True])}\n")
        f.write(f"Baseline experiments: {len(df[df['trust_enabled'] == False])}\n\n")
        
        # Accuracy Analysis
        successful = df[df['status'] == 'SUCCESS']
        if not successful.empty:
            f.write("ACCURACY ANALYSIS\n")
            f.write("-" * 17 + "\n")
            
            # Trust vs Baseline comparison
            trust_results = successful[successful['trust_enabled'] == True]
            baseline_results = successful[successful['trust_enabled'] == False]
            
            if not trust_results.empty and not baseline_results.empty:
                trust_mean_acc = trust_results['final_accuracy'].mean()
                baseline_mean_acc = baseline_results['final_accuracy'].mean()
                
                f.write(f"Average accuracy (Trust-weighted): {trust_mean_acc:.2f}%\n")
                f.write(f"Average accuracy (Baseline): {baseline_mean_acc:.2f}%\n")
                f.write(f"Average improvement: {trust_mean_acc - baseline_mean_acc:+.2f}%\n\n")
            
            # By dataset and attack type
            for dataset in successful['dataset'].unique():
                for attack in successful['attack'].unique():
                    subset = successful[(successful['dataset'] == dataset) & (successful['attack'] == attack)]
                    if not subset.empty:
                        trust_subset = subset[subset['trust_enabled'] == True]
                        baseline_subset = subset[subset['trust_enabled'] == False]
                        
                        if not trust_subset.empty and not baseline_subset.empty:
                            trust_acc = trust_subset['final_accuracy'].mean()
                            baseline_acc = baseline_subset['final_accuracy'].mean()
                            improvement = trust_acc - baseline_acc
                            
                            f.write(f"{dataset.upper()} {attack.replace('_', ' ')}: {improvement:+.2f}% improvement\n")
            f.write("\n")
        
        # Detection Performance Analysis
        trust_only = df[(df['trust_enabled'] == True) & (df['status'] == 'SUCCESS')]
        if not trust_only.empty:
            f.write("DETECTION PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average detection recall: {trust_only['recall'].mean():.3f}\n")
            f.write(f"Average detection precision: {trust_only['precision'].mean():.3f}\n")
            f.write(f"Average F1 score: {trust_only['f1_score'].mean():.3f}\n\n")
            
            # Best performing configurations
            best_f1 = trust_only.loc[trust_only['f1_score'].idxmax()]
            f.write(f"Best F1 score: {best_f1['f1_score']:.3f}\n")
            f.write(f"  Configuration: {best_f1['dataset']} {best_f1['topology']} {best_f1['attack']} (n={best_f1['node_count']})\n\n")
        
        # Scalability Analysis
        f.write("SCALABILITY ANALYSIS\n")
        f.write("-" * 19 + "\n")
        
        for node_count in sorted(successful['node_count'].unique()):
            subset = successful[successful['node_count'] == node_count]
            if not subset.empty:
                success_rate = len(subset[subset['status'] == 'SUCCESS']) / len(df[df['node_count'] == node_count]) * 100
                avg_accuracy = subset['final_accuracy'].mean()
                f.write(f"Node count {node_count}: {success_rate:.1f}% success rate, {avg_accuracy:.2f}% avg accuracy\n")
        
        f.write(f"\nReport generated: {pd.Timestamp.now()}\n")
    
    print(f"📄 Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze enhanced trust monitoring results')
    parser.add_argument('results_dir', help='Directory containing experiment results')
    parser.add_argument('--create-plots', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)
    
    print(f"🔍 Analyzing results in: {args.results_dir}")
    
    # Analyze results
    df = analyze_results_directory(args.results_dir)
    
    if df.empty:
        print("No valid experiment results found!")
        sys.exit(1)
    
    print(f"📊 Found {len(df)} experiments")
    
    # Save detailed CSV
    detailed_csv = os.path.join(args.results_dir, 'detailed_analysis.csv')
    df.to_csv(detailed_csv, index=False)
    print(f"💾 Detailed analysis saved to: {detailed_csv}")
    
    # Generate summary report
    generate_summary_report(df, args.results_dir)
    
    # Create visualizations if requested
    if args.create_plots:
        create_visualizations(df, args.results_dir)
    
    # Print key findings
    print("\n" + "=" * 50)
    print("KEY FINDINGS")
    print("=" * 50)
    
    successful = df[df['status'] == 'SUCCESS']
    if not successful.empty:
        # Trust vs baseline comparison
        trust_results = successful[successful['trust_enabled'] == True]
        baseline_results = successful[successful['trust_enabled'] == False]
        
        if not trust_results.empty and not baseline_results.empty:
            trust_acc = trust_results['final_accuracy'].mean()
            baseline_acc = baseline_results['final_accuracy'].mean()
            improvement = trust_acc - baseline_acc
            
            print(f"🎯 Overall accuracy improvement: {improvement:+.2f}%")
            print(f"   Trust-weighted: {trust_acc:.2f}%")
            print(f"   Baseline: {baseline_acc:.2f}%")
        
        if not trust_results.empty:
            print(f"🔍 Average detection performance:")
            print(f"   Recall: {trust_results['recall'].mean():.3f}")
            print(f"   Precision: {trust_results['precision'].mean():.3f}")
            print(f"   F1 Score: {trust_results['f1_score'].mean():.3f}")
        
        # Node count analysis
        node_performance = successful.groupby('node_count')['final_accuracy'].mean()
        print(f"📈 Performance by node count:")
        for node_count, avg_acc in node_performance.items():
            print(f"   {node_count} nodes: {avg_acc:.2f}% average accuracy")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()