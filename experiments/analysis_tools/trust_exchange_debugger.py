#!/usr/bin/env python3
"""
Trust Exchange Debugger - Quick analysis of why exchanges are failing

This tool specifically focuses on debugging trust exchange issues
and capturing HSIC/performance values in a rapid test.
"""

import json
import os
import sys
import logging
import time
from typing import Dict, Any, List

# Add the murmura path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist


def debug_trust_exchanges():
    """Quick debug of trust exchange issues with minimal rounds."""
    
    print("🔍 Trust Exchange Debugger")
    print("Analyzing why trust exchanges complete as '0/6'...")
    print("=" * 50)
    
    # Test scenarios - very short for rapid debugging
    scenarios = [
        {
            "name": "honest_quick",
            "description": "Honest baseline - check exchange success",
            "attack_config": None
        },
        {
            "name": "attack_quick", 
            "description": "Single attacker - check detection",
            "attack_config": {
                "attack_type": "gradual_label_flipping",
                "malicious_fraction": 0.167,  # 1 out of 6
                "attack_intensity": "moderate",
                "stealth_level": "medium"
            }
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        start_time = time.time()
        
        try:
            # Very short test - just 6 rounds for speed
            result = run_adaptive_trust_mnist(
                num_actors=6,
                num_rounds=6,  # Minimal rounds for debugging
                topology_type="ring",
                trust_profile="strict",
                use_beta_threshold=False,
                attack_config=scenario["attack_config"],
                output_dir=f"debug_{scenario['name']}",
                log_level="INFO"  # Reduce log spam
            )
            
            # Extract key metrics
            fl_results = result.get("fl_results", {})
            trust_metrics = fl_results.get("trust_metrics", [])
            trust_analysis = result.get("trust_analysis", {})
            attack_stats = fl_results.get("attack_statistics", {})
            
            # Analyze exchange patterns
            exchange_analysis = analyze_exchange_patterns(trust_metrics)
            
            # Analyze HSIC and performance values
            metric_analysis = analyze_metric_values(trust_metrics)
            
            scenario_result = {
                "scenario": scenario["name"],
                "attack_config": scenario["attack_config"],
                "time_taken": time.time() - start_time,
                "exchange_analysis": exchange_analysis,
                "metric_analysis": metric_analysis,
                "final_trust": trust_analysis,
                "attack_stats": attack_stats,
                "detection_rate": attack_stats.get("detection_rate", 0.0),
                "total_excluded": trust_analysis.get("total_excluded", 0)
            }
            
            results[scenario["name"]] = scenario_result
            
            # Quick feedback
            print(f"  ⏱️  Time: {scenario_result['time_taken']:.1f}s")
            print(f"  🔄 Exchange Success: {exchange_analysis['overall_success_rate']:.1%}")
            print(f"  🎯 Detection Rate: {scenario_result['detection_rate']:.1%}")
            print(f"  ❌ Excluded Nodes: {scenario_result['total_excluded']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results[scenario["name"]] = {"error": str(e)}
    
    return results


def analyze_exchange_patterns(trust_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trust exchange patterns to identify failure causes."""
    
    total_exchanges = 0
    successful_exchanges = 0
    failed_exchanges = 0
    zero_exchange_rounds = 0
    
    round_details = []
    
    for round_data in trust_metrics:
        round_num = round_data.get("round", 0)
        node_reports = round_data.get("node_trust_reports", {})
        
        round_exchanges = 0
        round_successes = 0
        round_has_zero = False
        
        for node_id, node_report in node_reports.items():
            exchange_summary = node_report.get("exchange_summary", {})
            completed = exchange_summary.get("completed_exchanges", 0)
            total_neighbors = exchange_summary.get("total_neighbors", 0)
            
            round_exchanges += total_neighbors
            round_successes += completed
            
            if completed == 0 and total_neighbors > 0:
                round_has_zero = True
        
        total_exchanges += round_exchanges
        successful_exchanges += round_successes
        
        if round_has_zero:
            zero_exchange_rounds += 1
        
        round_details.append({
            "round": round_num,
            "total_possible": round_exchanges,
            "successful": round_successes,
            "success_rate": round_successes / round_exchanges if round_exchanges > 0 else 0,
            "has_zero_exchanges": round_has_zero
        })
    
    return {
        "total_exchanges": total_exchanges,
        "successful_exchanges": successful_exchanges,
        "failed_exchanges": total_exchanges - successful_exchanges,
        "overall_success_rate": successful_exchanges / total_exchanges if total_exchanges > 0 else 0,
        "zero_exchange_rounds": zero_exchange_rounds,
        "total_rounds": len(trust_metrics),
        "round_details": round_details
    }


def analyze_metric_values(trust_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze HSIC and performance metric values."""
    
    all_hsic_values = []
    all_performance_values = []
    all_trust_scores = []
    suspicious_hsic_count = 0
    suspicious_performance_count = 0
    
    for round_data in trust_metrics:
        node_reports = round_data.get("node_trust_reports", {})
        
        for node_id, node_report in node_reports.items():
            # HSIC metrics
            hsic_data = node_report.get("hsic_metrics", {})
            if hsic_data:
                hsic_score = hsic_data.get("hsic_score", 0.0)
                hsic_threshold = hsic_data.get("threshold", 0.0)
                is_suspicious = hsic_data.get("is_suspicious", False)
                
                all_hsic_values.append(hsic_score)
                if is_suspicious:
                    suspicious_hsic_count += 1
            
            # Performance metrics
            perf_data = node_report.get("performance_metrics", {})
            if perf_data:
                perf_score = perf_data.get("validation_accuracy", 0.0)
                perf_threshold = perf_data.get("threshold", 0.0)
                is_suspicious = perf_data.get("is_suspicious", False)
                
                all_performance_values.append(perf_score)
                if is_suspicious:
                    suspicious_performance_count += 1
            
            # Trust scores
            trust_data = node_report.get("trust_scores", {})
            if trust_data:
                current_trust = trust_data.get("current_trust", 1.0)
                all_trust_scores.append(current_trust)
    
    import numpy as np
    
    return {
        "hsic_analysis": {
            "values": all_hsic_values,
            "mean": np.mean(all_hsic_values) if all_hsic_values else 0,
            "std": np.std(all_hsic_values) if all_hsic_values else 0,
            "min": np.min(all_hsic_values) if all_hsic_values else 0,
            "max": np.max(all_hsic_values) if all_hsic_values else 0,
            "suspicious_count": suspicious_hsic_count,
            "total_evaluations": len(all_hsic_values)
        },
        "performance_analysis": {
            "values": all_performance_values,
            "mean": np.mean(all_performance_values) if all_performance_values else 0,
            "std": np.std(all_performance_values) if all_performance_values else 0,
            "min": np.min(all_performance_values) if all_performance_values else 0,
            "max": np.max(all_performance_values) if all_performance_values else 0,
            "suspicious_count": suspicious_performance_count,
            "total_evaluations": len(all_performance_values)
        },
        "trust_analysis": {
            "values": all_trust_scores,
            "mean": np.mean(all_trust_scores) if all_trust_scores else 1.0,
            "std": np.std(all_trust_scores) if all_trust_scores else 0,
            "min": np.min(all_trust_scores) if all_trust_scores else 1.0,
            "degraded_count": sum(1 for x in all_trust_scores if x < 1.0)
        }
    }


def generate_quick_report(results: Dict[str, Any]) -> str:
    """Generate a quick diagnostic report."""
    
    report = []
    report.append("# Quick Trust Exchange Debug Report")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Exchange Analysis
    report.append("## Trust Exchange Analysis")
    for scenario_name, scenario_data in results.items():
        if "error" in scenario_data:
            report.append(f"### {scenario_name}: FAILED")
            report.append(f"Error: {scenario_data['error']}")
            continue
        
        exchange_analysis = scenario_data["exchange_analysis"]
        report.append(f"### {scenario_name}")
        report.append(f"- Overall Success Rate: {exchange_analysis['overall_success_rate']:.1%}")
        report.append(f"- Zero Exchange Rounds: {exchange_analysis['zero_exchange_rounds']}/{exchange_analysis['total_rounds']}")
        report.append(f"- Total Exchanges: {exchange_analysis['successful_exchanges']}/{exchange_analysis['total_exchanges']}")
        
        if exchange_analysis['overall_success_rate'] < 0.5:
            report.append("  ❌ **CRITICAL**: High exchange failure rate")
        elif exchange_analysis['zero_exchange_rounds'] > 0:
            report.append("  ⚠️ **WARNING**: Some rounds have zero exchanges")
        else:
            report.append("  ✅ **OK**: Exchange success rate acceptable")
        
        report.append("")
    
    # Metric Value Analysis
    report.append("## HSIC and Performance Value Analysis")
    for scenario_name, scenario_data in results.items():
        if "error" in scenario_data:
            continue
        
        metric_analysis = scenario_data["metric_analysis"]
        
        report.append(f"### {scenario_name}")
        
        # HSIC Analysis
        hsic_data = metric_analysis["hsic_analysis"]
        report.append(f"**HSIC Values:**")
        report.append(f"- Mean: {hsic_data['mean']:.6f}")
        report.append(f"- Range: {hsic_data['min']:.6f} to {hsic_data['max']:.6f}")
        report.append(f"- Suspicious: {hsic_data['suspicious_count']}/{hsic_data['total_evaluations']}")
        
        # Performance Analysis
        perf_data = metric_analysis["performance_analysis"]
        report.append(f"**Performance Values:**")
        report.append(f"- Mean: {perf_data['mean']:.4f}")
        report.append(f"- Range: {perf_data['min']:.4f} to {perf_data['max']:.4f}")
        report.append(f"- Suspicious: {perf_data['suspicious_count']}/{perf_data['total_evaluations']}")
        
        # Trust Analysis
        trust_data = metric_analysis["trust_analysis"]
        report.append(f"**Trust Scores:**")
        report.append(f"- Mean: {trust_data['mean']:.4f}")
        report.append(f"- Degraded Nodes: {trust_data['degraded_count']}")
        
        report.append("")
    
    # Key Issues
    report.append("## Key Issues Identified")
    issues = []
    
    for scenario_name, scenario_data in results.items():
        if "error" in scenario_data:
            issues.append(f"Test execution failed in {scenario_name}")
            continue
        
        exchange_analysis = scenario_data["exchange_analysis"]
        metric_analysis = scenario_data["metric_analysis"]
        
        # Exchange issues
        if exchange_analysis['overall_success_rate'] < 0.5:
            issues.append(f"High exchange failure rate in {scenario_name} ({exchange_analysis['overall_success_rate']:.1%})")
        
        # Detection issues
        if "attack" in scenario_name and scenario_data["detection_rate"] < 0.1:
            issues.append(f"Extremely low detection rate in {scenario_name} ({scenario_data['detection_rate']:.1%})")
        
        # HSIC issues
        hsic_data = metric_analysis["hsic_analysis"]
        if hsic_data['suspicious_count'] == 0 and "attack" in scenario_name:
            issues.append(f"HSIC not detecting any suspicious behavior in {scenario_name}")
        
        # Performance issues
        perf_data = metric_analysis["performance_analysis"]
        if perf_data['suspicious_count'] == 0 and "attack" in scenario_name:
            issues.append(f"Performance monitoring not detecting any suspicious behavior in {scenario_name}")
    
    if issues:
        for issue in issues:
            report.append(f"- ❌ {issue}")
    else:
        report.append("- ✅ No major issues detected")
    
    report.append("")
    
    # Recommendations
    report.append("## Immediate Recommendations")
    
    if any("exchange failure" in issue for issue in issues):
        report.append("### Fix Trust Exchange Implementation")
        report.append("1. Check trust-aware gossip aggregation logic")
        report.append("2. Verify validation data is available for performance monitoring")
        report.append("3. Debug neighbor parameter sharing mechanism")
        report.append("")
    
    if any("HSIC not detecting" in issue for issue in issues):
        report.append("### Improve HSIC Sensitivity")
        report.append("1. Reduce HSIC threshold (currently too high)")
        report.append("2. Check HSIC calculation implementation")
        report.append("3. Verify parameter differences between honest and malicious nodes")
        report.append("")
    
    if any("Performance monitoring not detecting" in issue for issue in issues):
        report.append("### Enhance Performance Monitoring")
        report.append("1. Lower performance drop threshold")
        report.append("2. Increase validation dataset size")
        report.append("3. Check performance validation implementation")
        report.append("")
    
    return "\n".join(report)


def main():
    """Run quick trust exchange debugging."""
    
    start_time = time.time()
    
    # Run debugging
    results = debug_trust_exchanges()
    
    # Generate report
    report = generate_quick_report(results)
    
    # Save results
    os.makedirs("trust_debug_results", exist_ok=True)
    
    with open("trust_debug_results/exchange_debug.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open("trust_debug_results/debug_report.md", "w") as f:
        f.write(report)
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 50)
    print("🎯 TRUST EXCHANGE DEBUG COMPLETE")
    print("=" * 50)
    print(f"⏱️  Total time: {total_time:.1f}s")
    print("📁 Results saved to: trust_debug_results/")
    print("📊 Check debug_report.md for findings")
    print("🔧 Review recommendations for fixes")


if __name__ == "__main__":
    main()