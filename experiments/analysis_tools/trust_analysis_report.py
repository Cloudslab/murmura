#!/usr/bin/env python3
"""
Advanced Trust Monitor Analysis and Debugging Tool

This script provides detailed analysis of trust monitor performance,
investigates why detection rates are low, and provides recommendations
for improvement.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, List

# Add the murmura path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist


class TrustAnalyzer:
    """Advanced trust monitor analysis and debugging."""
    
    def __init__(self, output_dir: str = "trust_analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup detailed logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, "trust_analysis.log"))
            ]
        )
        self.logger = logging.getLogger("TrustAnalyzer")
    
    def run_detailed_attack_scenario(
        self,
        scenario_name: str,
        malicious_fraction: float = 0.25,
        attack_intensity: str = "moderate",
        num_rounds: int = 20,
        trust_profile: str = "strict"
    ) -> Dict[str, Any]:
        """Run a detailed attack scenario with extensive logging."""
        
        self.logger.info(f"=== Running Detailed Attack Analysis: {scenario_name} ===")
        self.logger.info(f"Malicious Fraction: {malicious_fraction}")
        self.logger.info(f"Attack Intensity: {attack_intensity}")
        self.logger.info(f"Trust Profile: {trust_profile}")
        self.logger.info(f"Rounds: {num_rounds}")
        
        # Create attack config
        attack_config = {
            "attack_type": "gradual_label_flipping",
            "malicious_fraction": malicious_fraction,
            "attack_intensity": attack_intensity,
            "stealth_level": "medium"
        }
        
        # Run with Beta thresholding
        self.logger.info("Running with Beta distribution thresholding...")
        results_beta = run_adaptive_trust_mnist(
            num_actors=6,
            num_rounds=num_rounds,
            topology_type="ring",
            trust_profile=trust_profile,
            use_beta_threshold=True,
            attack_config=attack_config,
            output_dir=os.path.join(self.output_dir, f"{scenario_name}_beta"),
            log_level="DEBUG"  # More detailed logging
        )
        
        # Run with Manual thresholding
        self.logger.info("Running with Manual thresholding...")
        results_manual = run_adaptive_trust_mnist(
            num_actors=6,
            num_rounds=num_rounds,
            topology_type="ring",
            trust_profile=trust_profile,
            use_beta_threshold=False,
            attack_config=attack_config,
            output_dir=os.path.join(self.output_dir, f"{scenario_name}_manual"),
            log_level="DEBUG"
        )
        
        return {
            "scenario": scenario_name,
            "attack_config": attack_config,
            "results_beta": results_beta,
            "results_manual": results_manual
        }
    
    def analyze_trust_drift_detection(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trust drift detection capabilities in detail."""
        
        analysis = {
            "scenario": results["scenario"],
            "attack_config": results["attack_config"],
            "comparison": {}
        }
        
        for method, result_data in [("beta", results["results_beta"]), ("manual", results["results_manual"])]:
            self.logger.info(f"Analyzing {method} thresholding results...")
            
            # Extract trust metrics
            trust_analysis = result_data.get("trust_analysis", {})
            fl_results = result_data.get("fl_results", {})
            
            # Performance metrics
            performance = result_data.get("performance_metrics", {})
            final_accuracy = performance.get("final_accuracy", 0)
            training_time = performance.get("training_time", 0)
            
            # Trust detection metrics
            total_excluded = trust_analysis.get("total_excluded", 0)
            total_downgraded = trust_analysis.get("total_downgraded", 0)
            avg_trust_score = trust_analysis.get("avg_trust_score", 1.0)
            
            # Attack statistics
            attack_stats = fl_results.get("attack_statistics", {})
            total_attackers = attack_stats.get("total_attackers", 0)
            detected_attacks = attack_stats.get("detected_attacks", 0)
            detection_rate = attack_stats.get("detection_rate", 0.0)
            
            # Round-by-round analysis
            round_metrics = fl_results.get("round_metrics", [])
            consensus_progression = [r.get("consensus_accuracy", 0) for r in round_metrics]
            
            # Trust metrics over time
            trust_metrics = fl_results.get("trust_metrics", [])
            
            analysis["comparison"][method] = {
                "performance": {
                    "final_accuracy": final_accuracy,
                    "training_time": training_time,
                    "consensus_progression": consensus_progression
                },
                "trust_detection": {
                    "total_excluded": total_excluded,
                    "total_downgraded": total_downgraded,
                    "avg_trust_score": avg_trust_score,
                    "total_attackers": total_attackers,
                    "detected_attacks": detected_attacks,
                    "detection_rate": detection_rate
                },
                "trust_metrics": trust_metrics,
                "raw_attack_stats": attack_stats
            }
            
            self.logger.info(f"{method.upper()} Results:")
            self.logger.info(f"  Final Accuracy: {final_accuracy:.4f}")
            self.logger.info(f"  Total Attackers: {total_attackers}")
            self.logger.info(f"  Detected Attacks: {detected_attacks}")
            self.logger.info(f"  Detection Rate: {detection_rate:.2%}")
            self.logger.info(f"  Excluded Nodes: {total_excluded}")
            self.logger.info(f"  Downgraded Nodes: {total_downgraded}")
        
        return analysis
    
    def run_sensitivity_analysis(self) -> List[Dict[str, Any]]:
        """Run sensitivity analysis with different trust profiles and attack intensities."""
        
        self.logger.info("=== Running Trust Monitor Sensitivity Analysis ===")
        
        scenarios = [
            {
                "name": "low_intensity_strict",
                "malicious_fraction": 0.167,  # 1 out of 6
                "attack_intensity": "low",
                "trust_profile": "strict",
                "description": "Low intensity attack with strict trust monitoring"
            },
            {
                "name": "moderate_intensity_default",
                "malicious_fraction": 0.167,  # 1 out of 6
                "attack_intensity": "moderate",
                "trust_profile": "default",
                "description": "Moderate intensity attack with default trust monitoring"
            },
            {
                "name": "high_intensity_permissive",
                "malicious_fraction": 0.167,  # 1 out of 6
                "attack_intensity": "high",
                "trust_profile": "permissive",
                "description": "High intensity attack with permissive trust monitoring"
            },
            {
                "name": "multiple_attackers_strict",
                "malicious_fraction": 0.333,  # 2 out of 6
                "attack_intensity": "moderate",
                "trust_profile": "strict",
                "description": "Multiple attackers with strict trust monitoring"
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            self.logger.info(f"\\n{'='*60}")
            self.logger.info(f"Running scenario: {scenario['name']}")
            self.logger.info(f"Description: {scenario['description']}")
            self.logger.info(f"{'='*60}")
            
            # Run the scenario
            scenario_results = self.run_detailed_attack_scenario(
                scenario_name=scenario["name"],
                malicious_fraction=scenario["malicious_fraction"],
                attack_intensity=scenario["attack_intensity"],
                trust_profile=scenario["trust_profile"],
                num_rounds=15  # Extended rounds for better drift detection
            )
            
            # Analyze the results
            analysis = self.analyze_trust_drift_detection(scenario_results)
            analysis["scenario_config"] = scenario
            
            results.append(analysis)
        
        return results
    
    def generate_comprehensive_report(self, sensitivity_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive analysis report."""
        
        report = []
        report.append("# Comprehensive Trust Monitor Analysis Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        
        total_scenarios = len(sensitivity_results)
        successful_detections = 0
        total_detection_rate = 0
        total_false_positives = 0
        
        for result in sensitivity_results:
            for method in ["beta", "manual"]:
                detection_data = result["comparison"][method]["trust_detection"]
                detection_rate = detection_data["detection_rate"]
                total_detection_rate += detection_rate
                
                if detection_rate > 0.5:  # Consider 50%+ as successful
                    successful_detections += 1
        
        avg_detection_rate = total_detection_rate / (total_scenarios * 2)  # *2 for beta and manual
        
        report.append(f"- **Total Scenarios Tested**: {total_scenarios}")
        report.append(f"- **Average Detection Rate**: {avg_detection_rate:.2%}")
        report.append(f"- **Successful Detection Scenarios**: {successful_detections}/{total_scenarios * 2}")
        report.append("")
        
        # Detailed Analysis by Scenario
        report.append("## Detailed Scenario Analysis")
        
        for result in sensitivity_results:
            scenario_config = result["scenario_config"]
            report.append(f"### {scenario_config['name'].replace('_', ' ').title()}")
            report.append(f"**Description**: {scenario_config['description']}")
            report.append(f"**Configuration**:")
            report.append(f"- Malicious Fraction: {scenario_config['malicious_fraction']:.1%}")
            report.append(f"- Attack Intensity: {scenario_config['attack_intensity']}")
            report.append(f"- Trust Profile: {scenario_config['trust_profile']}")
            report.append("")
            
            # Beta vs Manual comparison
            beta_data = result["comparison"]["beta"]
            manual_data = result["comparison"]["manual"]
            
            report.append("#### Beta Distribution Thresholding")
            beta_detection = beta_data["trust_detection"]
            report.append(f"- Detection Rate: {beta_detection['detection_rate']:.2%}")
            report.append(f"- Total Attackers: {beta_detection['total_attackers']}")
            report.append(f"- Detected Attacks: {beta_detection['detected_attacks']}")
            report.append(f"- Excluded Nodes: {beta_detection['total_excluded']}")
            report.append(f"- Final Accuracy: {beta_data['performance']['final_accuracy']:.4f}")
            report.append("")
            
            report.append("#### Manual Thresholding")
            manual_detection = manual_data["trust_detection"]
            report.append(f"- Detection Rate: {manual_detection['detection_rate']:.2%}")
            report.append(f"- Total Attackers: {manual_detection['total_attackers']}")
            report.append(f"- Detected Attacks: {manual_detection['detected_attacks']}")
            report.append(f"- Excluded Nodes: {manual_detection['total_excluded']}")
            report.append(f"- Final Accuracy: {manual_data['performance']['final_accuracy']:.4f}")
            report.append("")
            
            # Performance comparison
            better_method = "Beta" if beta_detection['detection_rate'] > manual_detection['detection_rate'] else "Manual"
            report.append(f"**Winner**: {better_method} thresholding performed better")
            report.append("")
        
        # Trust Profile Analysis
        report.append("## Trust Profile Effectiveness")
        
        profile_performance = {}
        for result in sensitivity_results:
            profile = result["scenario_config"]["trust_profile"]
            if profile not in profile_performance:
                profile_performance[profile] = {"total": 0, "successful": 0, "detection_rates": []}
            
            for method in ["beta", "manual"]:
                detection_rate = result["comparison"][method]["trust_detection"]["detection_rate"]
                profile_performance[profile]["total"] += 1
                profile_performance[profile]["detection_rates"].append(detection_rate)
                if detection_rate > 0.5:
                    profile_performance[profile]["successful"] += 1
        
        for profile, stats in profile_performance.items():
            avg_detection = np.mean(stats["detection_rates"])
            success_rate = stats["successful"] / stats["total"]
            
            report.append(f"### {profile.title()} Profile")
            report.append(f"- Average Detection Rate: {avg_detection:.2%}")
            report.append(f"- Success Rate: {success_rate:.2%}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations for Trust Monitor Improvement")
        
        if avg_detection_rate < 0.5:
            report.append("### 🚨 Critical Issues Identified")
            report.append("1. **Low Detection Rate**: Trust monitor is missing most attacks")
            report.append("2. **Possible Causes**:")
            report.append("   - Trust thresholds may be too permissive")
            report.append("   - HSIC parameters may need tuning")
            report.append("   - Performance monitoring may not be sensitive enough")
            report.append("   - Attack progression may be too gradual")
            report.append("")
            
            report.append("### 🔧 Recommended Fixes")
            report.append("1. **Tighten Trust Thresholds**:")
            report.append("   - Reduce exclude_threshold from 0.5 to 0.3")
            report.append("   - Reduce downgrade_threshold from 0.3 to 0.2")
            report.append("   - Increase trust report frequency")
            report.append("")
            
            report.append("2. **Enhance HSIC Sensitivity**:")
            report.append("   - Reduce HSIC threshold from 0.1 to 0.05")
            report.append("   - Decrease window size for faster detection")
            report.append("   - Increase gamma parameter for RBF kernel")
            report.append("")
            
            report.append("3. **Improve Performance Monitoring**:")
            report.append("   - Increase validation data size")
            report.append("   - Lower performance drop threshold")
            report.append("   - Implement rolling average performance tracking")
            report.append("")
        else:
            report.append("### ✅ Trust Monitor Performance Acceptable")
            report.append(f"Detection rate of {avg_detection_rate:.2%} meets minimum requirements")
        
        return "\\n".join(report)
    
    def save_results(self, sensitivity_results: List[Dict[str, Any]], report: str):
        """Save analysis results."""
        
        # Save raw results
        results_file = os.path.join(self.output_dir, "sensitivity_analysis.json")
        with open(results_file, "w") as f:
            json.dump(sensitivity_results, f, indent=2, default=str)
        
        # Save report
        report_file = os.path.join(self.output_dir, "trust_analysis_report.md")
        with open(report_file, "w") as f:
            f.write(report)
        
        # Create summary CSV
        summary_data = []
        for result in sensitivity_results:
            scenario = result["scenario_config"]
            for method in ["beta", "manual"]:
                detection_data = result["comparison"][method]["trust_detection"]
                performance_data = result["comparison"][method]["performance"]
                
                summary_data.append({
                    "scenario": scenario["name"],
                    "method": method,
                    "trust_profile": scenario["trust_profile"],
                    "attack_intensity": scenario["attack_intensity"],
                    "malicious_fraction": scenario["malicious_fraction"],
                    "detection_rate": detection_data["detection_rate"],
                    "total_attackers": detection_data["total_attackers"],
                    "detected_attacks": detection_data["detected_attacks"],
                    "excluded_nodes": detection_data["total_excluded"],
                    "final_accuracy": performance_data["final_accuracy"]
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.output_dir, "sensitivity_summary.csv"), index=False)
        
        self.logger.info(f"Analysis results saved to {self.output_dir}")


def main():
    """Run comprehensive trust monitor analysis."""
    
    print("🔍 Starting Advanced Trust Monitor Analysis...")
    print("This will run multiple scenarios with detailed logging to identify")
    print("why trust drift detection rates are low and provide recommendations.")
    print("=" * 70)
    
    analyzer = TrustAnalyzer()
    
    try:
        # Run sensitivity analysis
        sensitivity_results = analyzer.run_sensitivity_analysis()
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(sensitivity_results)
        
        # Save results
        analyzer.save_results(sensitivity_results, report)
        
        # Print summary
        print("\\n" + "=" * 70)
        print("🎯 TRUST MONITOR ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Scenarios Tested: {len(sensitivity_results)}")
        print(f"Results saved to: trust_analysis_results/")
        print("📊 Check trust_analysis_report.md for detailed findings")
        print("🔧 Review recommendations for trust monitor improvements")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()