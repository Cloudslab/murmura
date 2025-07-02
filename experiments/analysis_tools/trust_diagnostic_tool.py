#!/usr/bin/env python3
"""
Trust Diagnostic Tool - Deep Analysis of Trust Monitor Values

This tool captures and analyzes actual HSIC values, performance metrics, and trust scores
during both honest and attack scenarios to identify why the trust monitor isn't detecting attacks.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# Add the murmura path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist


class TrustDiagnosticAnalyzer:
    """Comprehensive diagnostic analysis of trust monitor behavior."""
    
    def __init__(self, output_dir: str = "trust_diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup detailed diagnostic logging."""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "trust_diagnostics.log")
            ]
        )
        self.logger = logging.getLogger("TrustDiagnostics")
    
    def run_diagnostic_scenario(
        self,
        scenario_name: str,
        attack_config: Dict[str, Any] = None,
        num_rounds: int = 12,
        capture_detailed_logs: bool = True
    ) -> Dict[str, Any]:
        """Run a diagnostic scenario with detailed metric capture."""
        
        self.logger.info(f"=== Running Diagnostic: {scenario_name} ===")
        
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        # Configure for maximum logging detail
        result = run_adaptive_trust_mnist(
            num_actors=6,
            num_rounds=num_rounds,
            topology_type="ring",
            trust_profile="strict",
            use_beta_threshold=False,  # Use manual for consistent analysis
            attack_config=attack_config,
            output_dir=str(scenario_dir),
            log_level="DEBUG" if capture_detailed_logs else "INFO"
        )
        
        # Extract detailed metrics
        detailed_metrics = self._extract_detailed_metrics(result, scenario_dir)
        
        return {
            "scenario_name": scenario_name,
            "attack_config": attack_config,
            "raw_results": result,
            "detailed_metrics": detailed_metrics
        }
    
    def _extract_detailed_metrics(self, result: Dict[str, Any], scenario_dir: Path) -> Dict[str, Any]:
        """Extract detailed trust and performance metrics."""
        
        fl_results = result.get("fl_results", {})
        trust_metrics = fl_results.get("trust_metrics", [])
        round_metrics = fl_results.get("round_metrics", [])
        
        # Process trust metrics by round
        trust_evolution = []
        for round_data in trust_metrics:
            round_num = round_data.get("round", 0)
            
            # Extract HSIC values per node
            hsic_scores = {}
            performance_scores = {}
            trust_scores = {}
            exchange_counts = {}
            
            node_trust_reports = round_data.get("node_trust_reports", {})
            for node_id, node_report in node_trust_reports.items():
                node_id_int = int(node_id)
                
                # HSIC metrics
                hsic_data = node_report.get("hsic_metrics", {})
                hsic_scores[node_id_int] = {
                    "hsic_score": hsic_data.get("hsic_score", 0.0),
                    "threshold": hsic_data.get("threshold", 0.0),
                    "is_suspicious": hsic_data.get("is_suspicious", False)
                }
                
                # Performance metrics
                perf_data = node_report.get("performance_metrics", {})
                performance_scores[node_id_int] = {
                    "validation_accuracy": perf_data.get("validation_accuracy", 0.0),
                    "threshold": perf_data.get("threshold", 0.0),
                    "is_suspicious": perf_data.get("is_suspicious", False)
                }
                
                # Trust scores
                trust_data = node_report.get("trust_scores", {})
                trust_scores[node_id_int] = {
                    "current_trust": trust_data.get("current_trust", 1.0),
                    "trust_change": trust_data.get("trust_change", 0.0),
                    "status": trust_data.get("status", "active")
                }
                
                # Exchange counts
                exchange_data = node_report.get("exchange_summary", {})
                exchange_counts[node_id_int] = {
                    "completed_exchanges": exchange_data.get("completed_exchanges", 0),
                    "total_neighbors": exchange_data.get("total_neighbors", 0),
                    "success_rate": exchange_data.get("success_rate", 0.0)
                }
            
            trust_evolution.append({
                "round": round_num,
                "hsic_scores": hsic_scores,
                "performance_scores": performance_scores,
                "trust_scores": trust_scores,
                "exchange_counts": exchange_counts
            })
        
        # Performance evolution
        performance_evolution = []
        for round_data in round_metrics:
            round_num = round_data.get("round", 0)
            
            performance_evolution.append({
                "round": round_num,
                "consensus_accuracy": round_data.get("consensus_accuracy", 0.0),
                "node_agreement": round_data.get("node_agreement", 0.0),
                "accuracy_variance": round_data.get("accuracy_variance", 0.0),
                "individual_accuracies": [
                    node.get("accuracy", 0.0) 
                    for node in round_data.get("individual_metrics", [])
                ]
            })
        
        return {
            "trust_evolution": trust_evolution,
            "performance_evolution": performance_evolution,
            "final_trust_analysis": result.get("trust_analysis", {}),
            "attack_statistics": fl_results.get("attack_statistics", {})
        }
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """Run comparative analysis between honest and attack scenarios."""
        
        self.logger.info("=== Running Comparative Trust Analysis ===")
        
        scenarios = [
            {
                "name": "honest_baseline",
                "description": "Pure honest scenario - baseline metrics",
                "attack_config": None
            },
            {
                "name": "subtle_attack",
                "description": "Low intensity gradual attack",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.167,  # 1 out of 6
                    "attack_intensity": "low",
                    "stealth_level": "high"
                }
            },
            {
                "name": "moderate_attack", 
                "description": "Moderate intensity gradual attack",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.167,  # 1 out of 6
                    "attack_intensity": "moderate",
                    "stealth_level": "medium"
                }
            },
            {
                "name": "aggressive_attack",
                "description": "High intensity gradual attack",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.167,  # 1 out of 6
                    "attack_intensity": "high",
                    "stealth_level": "low"
                }
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            self.logger.info(f"\nRunning scenario: {scenario['name']}")
            self.logger.info(f"Description: {scenario['description']}")
            
            try:
                result = self.run_diagnostic_scenario(
                    scenario_name=scenario["name"],
                    attack_config=scenario["attack_config"],
                    num_rounds=15  # Longer for better attack progression
                )
                results[scenario["name"]] = result
                
                self.logger.info(f"✅ Completed: {scenario['name']}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed: {scenario['name']} - {str(e)}")
                results[scenario["name"]] = {"error": str(e)}
        
        return results
    
    def analyze_trust_patterns(self, comparative_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in trust metrics to identify discriminative features."""
        
        self.logger.info("=== Analyzing Trust Patterns ===")
        
        analysis = {
            "honest_patterns": {},
            "attack_patterns": {},
            "discriminative_features": {},
            "exchange_analysis": {},
            "threshold_analysis": {}
        }
        
        # Extract patterns for each scenario
        for scenario_name, scenario_data in comparative_results.items():
            if "error" in scenario_data:
                continue
                
            metrics = scenario_data["detailed_metrics"]
            trust_evolution = metrics["trust_evolution"]
            
            # Analyze HSIC patterns
            hsic_values = []
            performance_values = []
            trust_changes = []
            exchange_success_rates = []
            
            for round_data in trust_evolution:
                round_hsic = []
                round_perf = []
                round_trust_changes = []
                round_exchanges = []
                
                for node_id in range(6):  # 6 nodes
                    if node_id in round_data["hsic_scores"]:
                        hsic_score = round_data["hsic_scores"][node_id]["hsic_score"]
                        round_hsic.append(hsic_score)
                    
                    if node_id in round_data["performance_scores"]:
                        perf_score = round_data["performance_scores"][node_id]["validation_accuracy"]
                        round_perf.append(perf_score)
                    
                    if node_id in round_data["trust_scores"]:
                        trust_change = round_data["trust_scores"][node_id]["trust_change"]
                        round_trust_changes.append(trust_change)
                    
                    if node_id in round_data["exchange_counts"]:
                        exchange_rate = round_data["exchange_counts"][node_id]["success_rate"]
                        round_exchanges.append(exchange_rate)
                
                if round_hsic:
                    hsic_values.append({
                        "round": round_data["round"],
                        "values": round_hsic,
                        "mean": np.mean(round_hsic),
                        "std": np.std(round_hsic),
                        "suspicious_count": sum(1 for node_id, data in round_data["hsic_scores"].items() if data["is_suspicious"])
                    })
                
                if round_perf:
                    performance_values.append({
                        "round": round_data["round"],
                        "values": round_perf,
                        "mean": np.mean(round_perf),
                        "std": np.std(round_perf),
                        "suspicious_count": sum(1 for node_id, data in round_data["performance_scores"].items() if data["is_suspicious"])
                    })
                
                if round_trust_changes:
                    trust_changes.append({
                        "round": round_data["round"],
                        "values": round_trust_changes,
                        "mean": np.mean(round_trust_changes),
                        "negative_changes": sum(1 for x in round_trust_changes if x < 0)
                    })
                
                if round_exchanges:
                    exchange_success_rates.append({
                        "round": round_data["round"],
                        "values": round_exchanges,
                        "mean": np.mean(round_exchanges),
                        "zero_exchanges": sum(1 for x in round_exchanges if x == 0.0)
                    })
            
            # Store patterns
            pattern_data = {
                "hsic_patterns": hsic_values,
                "performance_patterns": performance_values,
                "trust_change_patterns": trust_changes,
                "exchange_patterns": exchange_success_rates,
                "final_detection_rate": metrics["attack_statistics"].get("detection_rate", 0.0),
                "total_excluded": metrics["final_trust_analysis"].get("total_excluded", 0)
            }
            
            if scenario_name == "honest_baseline":
                analysis["honest_patterns"] = pattern_data
            else:
                analysis["attack_patterns"][scenario_name] = pattern_data
        
        # Identify discriminative features
        analysis["discriminative_features"] = self._identify_discriminative_features(analysis)
        
        # Analyze exchange issues
        analysis["exchange_analysis"] = self._analyze_exchange_issues(analysis)
        
        # Analyze threshold effectiveness
        analysis["threshold_analysis"] = self._analyze_threshold_effectiveness(analysis)
        
        return analysis
    
    def _identify_discriminative_features(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify features that can discriminate between honest and attack scenarios."""
        
        honest_patterns = analysis["honest_patterns"]
        attack_patterns = analysis["attack_patterns"]
        
        if not honest_patterns or not attack_patterns:
            return {"error": "Insufficient data for discriminative analysis"}
        
        features = {
            "hsic_discrimination": {},
            "performance_discrimination": {},
            "trust_change_discrimination": {},
            "exchange_discrimination": {}
        }
        
        # HSIC discrimination
        honest_hsic_means = [round_data["mean"] for round_data in honest_patterns["hsic_patterns"]]
        attack_hsic_means = []
        for attack_name, attack_data in attack_patterns.items():
            attack_means = [round_data["mean"] for round_data in attack_data["hsic_patterns"]]
            attack_hsic_means.extend(attack_means)
        
        if honest_hsic_means and attack_hsic_means:
            features["hsic_discrimination"] = {
                "honest_mean": np.mean(honest_hsic_means),
                "honest_std": np.std(honest_hsic_means),
                "attack_mean": np.mean(attack_hsic_means),
                "attack_std": np.std(attack_hsic_means),
                "separation": abs(np.mean(honest_hsic_means) - np.mean(attack_hsic_means)),
                "overlap": self._calculate_overlap(honest_hsic_means, attack_hsic_means)
            }
        
        # Performance discrimination
        honest_perf_means = [round_data["mean"] for round_data in honest_patterns["performance_patterns"]]
        attack_perf_means = []
        for attack_name, attack_data in attack_patterns.items():
            attack_means = [round_data["mean"] for round_data in attack_data["performance_patterns"]]
            attack_perf_means.extend(attack_means)
        
        if honest_perf_means and attack_perf_means:
            features["performance_discrimination"] = {
                "honest_mean": np.mean(honest_perf_means),
                "honest_std": np.std(honest_perf_means),
                "attack_mean": np.mean(attack_perf_means),
                "attack_std": np.std(attack_perf_means),
                "separation": abs(np.mean(honest_perf_means) - np.mean(attack_perf_means)),
                "overlap": self._calculate_overlap(honest_perf_means, attack_perf_means)
            }
        
        return features
    
    def _calculate_overlap(self, honest_values: List[float], attack_values: List[float]) -> float:
        """Calculate the overlap between honest and attack value distributions."""
        if not honest_values or not attack_values:
            return 1.0
        
        honest_min, honest_max = min(honest_values), max(honest_values)
        attack_min, attack_max = min(attack_values), max(attack_values)
        
        overlap_start = max(honest_min, attack_min)
        overlap_end = min(honest_max, attack_max)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        total_range = max(honest_max, attack_max) - min(honest_min, attack_min)
        overlap_range = overlap_end - overlap_start
        
        return overlap_range / total_range if total_range > 0 else 1.0
    
    def _analyze_exchange_issues(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze why trust exchanges are failing (completing as 0/6)."""
        
        exchange_issues = {
            "exchange_failure_patterns": {},
            "success_rate_analysis": {},
            "recommendations": []
        }
        
        # Analyze honest scenario exchanges
        honest_patterns = analysis.get("honest_patterns", {})
        if honest_patterns:
            honest_exchanges = honest_patterns.get("exchange_patterns", [])
            if honest_exchanges:
                zero_exchange_rounds = sum(1 for round_data in honest_exchanges if round_data["zero_exchanges"] > 0)
                total_rounds = len(honest_exchanges)
                
                exchange_issues["success_rate_analysis"]["honest"] = {
                    "zero_exchange_rounds": zero_exchange_rounds,
                    "total_rounds": total_rounds,
                    "failure_rate": zero_exchange_rounds / total_rounds if total_rounds > 0 else 0,
                    "average_success_rate": np.mean([round_data["mean"] for round_data in honest_exchanges])
                }
        
        # Analyze attack scenario exchanges
        attack_patterns = analysis.get("attack_patterns", {})
        for attack_name, attack_data in attack_patterns.items():
            attack_exchanges = attack_data.get("exchange_patterns", [])
            if attack_exchanges:
                zero_exchange_rounds = sum(1 for round_data in attack_exchanges if round_data["zero_exchanges"] > 0)
                total_rounds = len(attack_exchanges)
                
                exchange_issues["success_rate_analysis"][attack_name] = {
                    "zero_exchange_rounds": zero_exchange_rounds,
                    "total_rounds": total_rounds,
                    "failure_rate": zero_exchange_rounds / total_rounds if total_rounds > 0 else 0,
                    "average_success_rate": np.mean([round_data["mean"] for round_data in attack_exchanges])
                }
        
        # Generate recommendations
        if any(data.get("failure_rate", 0) > 0.5 for data in exchange_issues["success_rate_analysis"].values()):
            exchange_issues["recommendations"].append("High exchange failure rate detected - check trust-aware gossip implementation")
        
        return exchange_issues
    
    def _analyze_threshold_effectiveness(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze whether current thresholds are effective for detection."""
        
        threshold_analysis = {
            "hsic_threshold_effectiveness": {},
            "performance_threshold_effectiveness": {},
            "recommended_adjustments": []
        }
        
        honest_patterns = analysis.get("honest_patterns", {})
        attack_patterns = analysis.get("attack_patterns", {})
        
        if not honest_patterns or not attack_patterns:
            return threshold_analysis
        
        # Analyze HSIC threshold effectiveness
        honest_hsic = honest_patterns.get("hsic_patterns", [])
        attack_hsic_all = []
        for attack_data in attack_patterns.values():
            attack_hsic_all.extend(attack_data.get("hsic_patterns", []))
        
        if honest_hsic and attack_hsic_all:
            honest_suspicious = sum(round_data["suspicious_count"] for round_data in honest_hsic)
            attack_suspicious = sum(round_data["suspicious_count"] for round_data in attack_hsic_all)
            
            threshold_analysis["hsic_threshold_effectiveness"] = {
                "honest_false_positives": honest_suspicious,
                "attack_true_positives": attack_suspicious,
                "total_honest_evaluations": len(honest_hsic) * 6,  # 6 nodes per round
                "total_attack_evaluations": len(attack_hsic_all) * 6,
                "false_positive_rate": honest_suspicious / (len(honest_hsic) * 6) if honest_hsic else 0,
                "detection_rate": attack_suspicious / (len(attack_hsic_all) * 6) if attack_hsic_all else 0
            }
        
        return threshold_analysis
    
    def generate_diagnostic_report(self, comparative_results: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive diagnostic report."""
        
        report = []
        report.append("# Trust Monitor Diagnostic Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("This report analyzes why the trust monitor is failing to detect attacks")
        report.append("by examining actual HSIC values, performance metrics, and trust scores.")
        report.append("")
        
        # Exchange Analysis
        exchange_analysis = pattern_analysis.get("exchange_analysis", {})
        if exchange_analysis:
            report.append("## Trust Exchange Issues")
            success_rates = exchange_analysis.get("success_rate_analysis", {})
            
            for scenario, data in success_rates.items():
                failure_rate = data.get("failure_rate", 0)
                avg_success = data.get("average_success_rate", 0)
                
                report.append(f"### {scenario.title()}")
                report.append(f"- Exchange failure rate: {failure_rate:.2%}")
                report.append(f"- Average success rate: {avg_success:.2%}")
                report.append(f"- Zero exchange rounds: {data.get('zero_exchange_rounds', 0)}/{data.get('total_rounds', 0)}")
                
                if failure_rate > 0.3:
                    report.append(f"  ⚠️ **HIGH FAILURE RATE** - Trust exchanges are failing frequently")
                report.append("")
        
        # Discriminative Features Analysis
        discriminative = pattern_analysis.get("discriminative_features", {})
        if discriminative and "error" not in discriminative:
            report.append("## Discriminative Feature Analysis")
            
            # HSIC Analysis
            hsic_disc = discriminative.get("hsic_discrimination", {})
            if hsic_disc:
                report.append("### HSIC Discrimination")
                report.append(f"- Honest mean: {hsic_disc.get('honest_mean', 0):.6f}")
                report.append(f"- Attack mean: {hsic_disc.get('attack_mean', 0):.6f}")
                report.append(f"- Separation: {hsic_disc.get('separation', 0):.6f}")
                report.append(f"- Overlap: {hsic_disc.get('overlap', 0):.2%}")
                
                if hsic_disc.get("separation", 0) < 0.01:
                    report.append("  ❌ **POOR SEPARATION** - HSIC values don't discriminate well")
                if hsic_disc.get("overlap", 0) > 0.8:
                    report.append("  ❌ **HIGH OVERLAP** - Honest and attack distributions overlap significantly")
                report.append("")
            
            # Performance Analysis
            perf_disc = discriminative.get("performance_discrimination", {})
            if perf_disc:
                report.append("### Performance Discrimination")
                report.append(f"- Honest mean accuracy: {perf_disc.get('honest_mean', 0):.4f}")
                report.append(f"- Attack mean accuracy: {perf_disc.get('attack_mean', 0):.4f}")
                report.append(f"- Separation: {perf_disc.get('separation', 0):.4f}")
                report.append(f"- Overlap: {perf_disc.get('overlap', 0):.2%}")
                
                if perf_disc.get("separation", 0) < 0.05:
                    report.append("  ❌ **POOR SEPARATION** - Performance metrics don't discriminate well")
                report.append("")
        
        # Threshold Effectiveness
        threshold_analysis = pattern_analysis.get("threshold_analysis", {})
        if threshold_analysis:
            hsic_eff = threshold_analysis.get("hsic_threshold_effectiveness", {})
            if hsic_eff:
                report.append("## Threshold Effectiveness Analysis")
                report.append("### HSIC Threshold")
                report.append(f"- False positive rate: {hsic_eff.get('false_positive_rate', 0):.2%}")
                report.append(f"- Detection rate: {hsic_eff.get('detection_rate', 0):.2%}")
                report.append(f"- Honest false positives: {hsic_eff.get('honest_false_positives', 0)}")
                report.append(f"- Attack true positives: {hsic_eff.get('attack_true_positives', 0)}")
                
                if hsic_eff.get("detection_rate", 0) < 0.1:
                    report.append("  ❌ **EXTREMELY LOW DETECTION** - Current threshold is too permissive")
                report.append("")
        
        # Key Issues Identified
        report.append("## Key Issues Identified")
        issues = []
        
        # Check for exchange issues
        if exchange_analysis:
            for scenario, data in exchange_analysis.get("success_rate_analysis", {}).items():
                if data.get("failure_rate", 0) > 0.3:
                    issues.append(f"High trust exchange failure rate in {scenario} ({data.get('failure_rate', 0):.1%})")
        
        # Check for discrimination issues
        if discriminative and "error" not in discriminative:
            hsic_disc = discriminative.get("hsic_discrimination", {})
            if hsic_disc and hsic_disc.get("separation", 0) < 0.01:
                issues.append("HSIC values show poor separation between honest and attack scenarios")
            
            perf_disc = discriminative.get("performance_discrimination", {})
            if perf_disc and perf_disc.get("separation", 0) < 0.05:
                issues.append("Performance metrics show poor separation between honest and attack scenarios")
        
        # Check for threshold issues
        if threshold_analysis:
            hsic_eff = threshold_analysis.get("hsic_threshold_effectiveness", {})
            if hsic_eff and hsic_eff.get("detection_rate", 0) < 0.1:
                issues.append("HSIC threshold is too permissive - extremely low detection rate")
        
        if issues:
            for issue in issues:
                report.append(f"- ❌ {issue}")
        else:
            report.append("- ✅ No major issues identified in current analysis")
        report.append("")
        
        # Recommendations
        report.append("## Specific Recommendations")
        
        if issues:
            report.append("### Immediate Actions")
            
            if any("exchange failure" in issue for issue in issues):
                report.append("1. **Fix Trust Exchange Implementation**:")
                report.append("   - Review trust-aware gossip aggregation logic")
                report.append("   - Check validation data availability for performance monitoring")
                report.append("   - Ensure proper neighbor parameter sharing")
                report.append("")
            
            if any("HSIC" in issue and "separation" in issue for issue in issues):
                report.append("2. **Improve HSIC Sensitivity**:")
                report.append("   - Reduce HSIC threshold from current value to 0.005")
                report.append("   - Adjust RBF kernel gamma parameter for better discrimination")
                report.append("   - Implement adaptive HSIC windowing")
                report.append("")
            
            if any("Performance" in issue and "separation" in issue for issue in issues):
                report.append("3. **Enhance Performance Monitoring**:")
                report.append("   - Increase validation dataset size")
                report.append("   - Lower performance drop threshold")
                report.append("   - Implement performance trend analysis")
                report.append("")
            
            if any("threshold" in issue and "permissive" in issue for issue in issues):
                report.append("4. **Adjust Trust Thresholds**:")
                report.append("   - Reduce exclude_threshold from 0.5 to 0.2")
                report.append("   - Reduce downgrade_threshold from 0.3 to 0.15")
                report.append("   - Increase trust report frequency")
                report.append("")
        else:
            report.append("### Optimization Opportunities")
            report.append("- Fine-tune detection sensitivity for better early warning")
            report.append("- Optimize exchange frequency for better coverage")
            report.append("")
        
        return "\n".join(report)
    
    def save_diagnostic_results(self, comparative_results: Dict[str, Any], pattern_analysis: Dict[str, Any], report: str):
        """Save all diagnostic results."""
        
        # Save raw comparative results
        with open(self.output_dir / "comparative_results.json", "w") as f:
            json.dump(comparative_results, f, indent=2, default=str)
        
        # Save pattern analysis
        with open(self.output_dir / "pattern_analysis.json", "w") as f:
            json.dump(pattern_analysis, f, indent=2, default=str)
        
        # Save diagnostic report
        with open(self.output_dir / "diagnostic_report.md", "w") as f:
            f.write(report)
        
        # Create summary CSV for easy analysis
        self._create_summary_csv(comparative_results, pattern_analysis)
        
        self.logger.info(f"Diagnostic results saved to {self.output_dir}")
    
    def _create_summary_csv(self, comparative_results: Dict[str, Any], pattern_analysis: Dict[str, Any]):
        """Create CSV summary for easy analysis."""
        
        summary_data = []
        
        for scenario_name, scenario_data in comparative_results.items():
            if "error" in scenario_data:
                continue
            
            metrics = scenario_data["detailed_metrics"]
            attack_config = scenario_data.get("attack_config")
            
            # Get final metrics
            final_trust = metrics["final_trust_analysis"]
            attack_stats = metrics["attack_statistics"]
            
            # Calculate average metrics across rounds
            trust_evolution = metrics["trust_evolution"]
            if trust_evolution:
                avg_hsic = np.mean([
                    np.mean([node_data["hsic_score"] for node_data in round_data["hsic_scores"].values()])
                    for round_data in trust_evolution if round_data["hsic_scores"]
                ])
                
                avg_performance = np.mean([
                    np.mean([node_data["validation_accuracy"] for node_data in round_data["performance_scores"].values()])
                    for round_data in trust_evolution if round_data["performance_scores"]
                ])
                
                avg_exchange_success = np.mean([
                    np.mean([node_data["success_rate"] for node_data in round_data["exchange_counts"].values()])
                    for round_data in trust_evolution if round_data["exchange_counts"]
                ])
            else:
                avg_hsic = avg_performance = avg_exchange_success = 0.0
            
            summary_data.append({
                "scenario": scenario_name,
                "is_attack": attack_config is not None,
                "attack_intensity": attack_config.get("attack_intensity", "none") if attack_config else "none",
                "malicious_fraction": attack_config.get("malicious_fraction", 0.0) if attack_config else 0.0,
                "detection_rate": attack_stats.get("detection_rate", 0.0),
                "total_excluded": final_trust.get("total_excluded", 0),
                "total_downgraded": final_trust.get("total_downgraded", 0),
                "avg_hsic_score": avg_hsic,
                "avg_performance_score": avg_performance,
                "avg_exchange_success_rate": avg_exchange_success,
                "final_consensus_accuracy": metrics["performance_evolution"][-1]["consensus_accuracy"] if metrics["performance_evolution"] else 0.0
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / "diagnostic_summary.csv", index=False)


def main():
    """Run comprehensive trust diagnostic analysis."""
    
    print("🔍 Trust Monitor Diagnostic Analysis")
    print("Analyzing actual HSIC values, performance metrics, and trust scores")
    print("to identify why attacks are not being detected.")
    print("=" * 70)
    
    analyzer = TrustDiagnosticAnalyzer()
    
    try:
        # Run comparative analysis
        print("📊 Running comparative scenarios...")
        comparative_results = analyzer.run_comparative_analysis()
        
        # Analyze patterns
        print("🔍 Analyzing trust patterns...")
        pattern_analysis = analyzer.analyze_trust_patterns(comparative_results)
        
        # Generate diagnostic report
        print("📝 Generating diagnostic report...")
        report = analyzer.generate_diagnostic_report(comparative_results, pattern_analysis)
        
        # Save results
        analyzer.save_diagnostic_results(comparative_results, pattern_analysis, report)
        
        print("\n" + "=" * 70)
        print("🎯 TRUST DIAGNOSTIC ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"📁 Results saved to: {analyzer.output_dir}")
        print("📊 Check diagnostic_report.md for detailed findings")
        print("📈 Review diagnostic_summary.csv for metric comparisons")
        print("🔧 Follow specific recommendations to fix trust monitor")
        
    except Exception as e:
        print(f"❌ Diagnostic analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()