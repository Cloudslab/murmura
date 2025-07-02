#!/usr/bin/env python3
"""
Comprehensive Trust Monitor Testing Framework

This script performs extensive end-to-end testing of the trust monitoring system
using both MNIST and CIFAR-10 examples with various attack scenarios to evaluate
trust drift detection accuracy.

Test Scenarios:
1. Honest-only baseline (should have zero false positives)
2. Single malicious node with gradual attack
3. Multiple malicious nodes with different intensities
4. Stealth attacks with various evasion levels
5. Trust monitor sensitivity analysis
6. Performance comparison between Beta and manual thresholding
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Tuple
import subprocess
import pandas as pd
import numpy as np

# Add the murmura path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

# Import our examples
from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist
from murmura.examples.adaptive_trust_cifar10_example import run_adaptive_trust_cifar10


class TrustMonitorTester:
    """Comprehensive testing framework for trust monitoring system."""
    
    def __init__(self, output_dir: str = "trust_test_results"):
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory first
        os.makedirs(output_dir, exist_ok=True)
        self.setup_logging()
        
        # Test configuration
        self.test_configs = {
            "quick": {
                "num_actors": 4,
                "num_rounds": 8,
                "datasets": ["mnist"],
                "trust_profiles": ["default"],
                "attack_scenarios": ["honest", "single_malicious"]
            },
            "standard": {
                "num_actors": 6,
                "num_rounds": 15,
                "datasets": ["mnist", "cifar10"],
                "trust_profiles": ["default", "strict"],
                "attack_scenarios": ["honest", "single_malicious", "multiple_malicious"]
            },
            "comprehensive": {
                "num_actors": 8,
                "num_rounds": 20,
                "datasets": ["mnist", "cifar10"],
                "trust_profiles": ["permissive", "default", "strict"],
                "attack_scenarios": ["honest", "single_malicious", "multiple_malicious", "stealth_attack"]
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, "trust_test.log"))
            ]
        )
        
        self.logger = logging.getLogger("TrustMonitorTester")
    
    def create_attack_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive attack scenarios for testing."""
        return {
            "honest": {
                "name": "Honest-Only Baseline",
                "description": "All nodes are honest - should have zero false positives",
                "attack_config": None,
                "expected_detections": 0,
                "expected_false_positives": 0
            },
            "single_malicious": {
                "name": "Single Malicious Node",
                "description": "One node performs gradual label flipping attack",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.167,  # 1 out of 6 nodes
                    "attack_intensity": "moderate",
                    "stealth_level": "medium"
                },
                "expected_detections": 1,
                "expected_false_positives": 0
            },
            "multiple_malicious": {
                "name": "Multiple Malicious Nodes",
                "description": "Two nodes perform gradual attacks with different intensities",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.333,  # 2 out of 6 nodes
                    "attack_intensity": "moderate",
                    "stealth_level": "medium"
                },
                "expected_detections": 2,
                "expected_false_positives": 0
            },
            "stealth_attack": {
                "name": "Stealth Attack",
                "description": "Low intensity, high stealth attack to test sensitivity",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.167,  # 1 out of 6 nodes
                    "attack_intensity": "low",
                    "stealth_level": "high"
                },
                "expected_detections": 1,
                "expected_false_positives": 0
            },
            "aggressive_attack": {
                "name": "Aggressive Attack",
                "description": "High intensity attack for detection robustness",
                "attack_config": {
                    "attack_type": "gradual_label_flipping",
                    "malicious_fraction": 0.167,  # 1 out of 6 nodes
                    "attack_intensity": "high",
                    "stealth_level": "low"
                },
                "expected_detections": 1,
                "expected_false_positives": 0
            }
        }
    
    def run_single_test(
        self,
        dataset: str,
        scenario_name: str,
        scenario_config: Dict[str, Any],
        trust_profile: str = "default",
        use_beta_threshold: bool = True,
        num_actors: int = 6,
        num_rounds: int = 15
    ) -> Dict[str, Any]:
        """Run a single trust monitoring test."""
        
        self.logger.info(f"=== Running Test: {dataset.upper()} - {scenario_config['name']} ===")
        self.logger.info(f"Trust Profile: {trust_profile}, Beta Threshold: {use_beta_threshold}")
        self.logger.info(f"Actors: {num_actors}, Rounds: {num_rounds}")
        
        test_start_time = time.time()
        test_id = f"{dataset}_{scenario_name}_{trust_profile}_{'beta' if use_beta_threshold else 'manual'}_{int(time.time())}"
        
        # Create test-specific output directory
        test_output_dir = os.path.join(self.output_dir, test_id)
        os.makedirs(test_output_dir, exist_ok=True)
        
        try:
            # Run the appropriate example
            if dataset == "mnist":
                results = run_adaptive_trust_mnist(
                    num_actors=num_actors,
                    num_rounds=num_rounds,
                    topology_type="ring",
                    trust_profile=trust_profile,
                    use_beta_threshold=use_beta_threshold,
                    attack_config=scenario_config.get("attack_config"),
                    output_dir=test_output_dir,
                    log_level="INFO"
                )
            elif dataset == "cifar10":
                results = run_adaptive_trust_cifar10(
                    num_actors=num_actors,
                    num_rounds=num_rounds,
                    topology_type="ring",
                    trust_profile=trust_profile,
                    use_beta_threshold=use_beta_threshold,
                    model_type="standard",
                    attack_config=scenario_config.get("attack_config"),
                    output_dir=test_output_dir,
                    log_level="INFO"
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            
            # Extract trust analysis results
            trust_analysis = results.get("trust_analysis", {})
            fl_results = results.get("fl_results", {})
            
            # Calculate detection metrics
            total_excluded = trust_analysis.get("total_excluded", 0)
            total_downgraded = trust_analysis.get("total_downgraded", 0)
            avg_trust_score = trust_analysis.get("avg_trust_score", 1.0)
            
            # Determine detection accuracy
            expected_detections = scenario_config.get("expected_detections", 0)
            expected_false_positives = scenario_config.get("expected_false_positives", 0)
            
            # Calculate metrics
            detection_rate = min(total_excluded / expected_detections, 1.0) if expected_detections > 0 else 1.0
            false_positive_rate = max(0, total_excluded - expected_detections) / (num_actors - expected_detections) if num_actors > expected_detections else 0.0
            
            # Performance metrics
            final_accuracy = results.get("performance_metrics", {}).get("final_accuracy", 0)
            accuracy_improvement = results.get("performance_metrics", {}).get("accuracy_improvement", 0)
            total_time = results.get("performance_metrics", {}).get("total_time", 0)
            
            # Attack statistics if available
            attack_stats = fl_results.get("attack_statistics", {})
            
            test_result = {
                "test_id": test_id,
                "timestamp": time.time(),
                "dataset": dataset,
                "scenario": scenario_name,
                "scenario_description": scenario_config["description"],
                "trust_profile": trust_profile,
                "use_beta_threshold": use_beta_threshold,
                "num_actors": num_actors,
                "num_rounds": num_rounds,
                
                # Trust Detection Metrics
                "total_excluded": total_excluded,
                "total_downgraded": total_downgraded,
                "avg_trust_score": avg_trust_score,
                "expected_detections": expected_detections,
                "detection_rate": detection_rate,
                "false_positive_rate": false_positive_rate,
                
                # Performance Metrics
                "final_accuracy": final_accuracy,
                "accuracy_improvement": accuracy_improvement,
                "training_time": total_time,
                
                # Attack Statistics
                "attack_statistics": attack_stats,
                
                # Test Success Criteria
                "detection_success": detection_rate >= 0.8,  # 80% detection rate
                "false_positive_success": false_positive_rate <= 0.1,  # Max 10% false positives
                "overall_success": detection_rate >= 0.8 and false_positive_rate <= 0.1,
                
                # Raw Results
                "raw_results": results
            }
            
            self.logger.info(f"Test completed in {time.time() - test_start_time:.2f}s")
            self.logger.info(f"Detection Rate: {detection_rate:.2%}")
            self.logger.info(f"False Positive Rate: {false_positive_rate:.2%}")
            self.logger.info(f"Final Accuracy: {final_accuracy:.4f}")
            self.logger.info(f"Test Success: {test_result['overall_success']}")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Test failed: {str(e)}", exc_info=True)
            return {
                "test_id": test_id,
                "timestamp": time.time(),
                "dataset": dataset,
                "scenario": scenario_name,
                "error": str(e),
                "overall_success": False
            }
    
    def run_comprehensive_tests(self, test_level: str = "standard") -> List[Dict[str, Any]]:
        """Run comprehensive trust monitoring tests."""
        
        if test_level not in self.test_configs:
            raise ValueError(f"Invalid test level: {test_level}. Choose from {list(self.test_configs.keys())}")
        
        config = self.test_configs[test_level]
        scenarios = self.create_attack_scenarios()
        
        self.logger.info(f"=== Starting Comprehensive Trust Monitor Testing ({test_level.upper()}) ===")
        self.logger.info(f"Datasets: {config['datasets']}")
        self.logger.info(f"Trust Profiles: {config['trust_profiles']}")
        self.logger.info(f"Attack Scenarios: {config['attack_scenarios']}")
        self.logger.info(f"Total Tests: {len(config['datasets']) * len(config['trust_profiles']) * len(config['attack_scenarios']) * 2}")  # *2 for beta vs manual
        
        test_results = []
        total_tests = 0
        successful_tests = 0
        
        for dataset in config["datasets"]:
            for trust_profile in config["trust_profiles"]:
                for scenario_name in config["attack_scenarios"]:
                    if scenario_name not in scenarios:
                        self.logger.warning(f"Skipping unknown scenario: {scenario_name}")
                        continue
                    
                    scenario_config = scenarios[scenario_name]
                    
                    # Test with Beta thresholding
                    total_tests += 1
                    result_beta = self.run_single_test(
                        dataset=dataset,
                        scenario_name=scenario_name,
                        scenario_config=scenario_config,
                        trust_profile=trust_profile,
                        use_beta_threshold=True,
                        num_actors=config["num_actors"],
                        num_rounds=config["num_rounds"]
                    )
                    test_results.append(result_beta)
                    if result_beta.get("overall_success", False):
                        successful_tests += 1
                    
                    # Test with Manual thresholding
                    total_tests += 1
                    result_manual = self.run_single_test(
                        dataset=dataset,
                        scenario_name=scenario_name,
                        scenario_config=scenario_config,
                        trust_profile=trust_profile,
                        use_beta_threshold=False,
                        num_actors=config["num_actors"],
                        num_rounds=config["num_rounds"]
                    )
                    test_results.append(result_manual)
                    if result_manual.get("overall_success", False):
                        successful_tests += 1
        
        self.logger.info(f"=== Testing Complete ===")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Successful Tests: {successful_tests}")
        self.logger.info(f"Success Rate: {successful_tests/total_tests:.2%}")
        
        return test_results
    
    def analyze_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and summarize test results."""
        
        self.logger.info("=== Analyzing Test Results ===")
        
        # Filter out failed tests
        valid_results = [r for r in test_results if "error" not in r]
        failed_tests = [r for r in test_results if "error" in r]
        
        if not valid_results:
            return {"error": "No valid test results to analyze"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(valid_results)
        
        # Overall statistics
        total_tests = len(valid_results)
        successful_tests = sum(df["overall_success"])
        success_rate = successful_tests / total_tests
        
        # Detection performance
        avg_detection_rate = df["detection_rate"].mean()
        avg_false_positive_rate = df["false_positive_rate"].mean()
        avg_trust_score = df["avg_trust_score"].mean()
        
        # Performance by dataset
        dataset_analysis = {}
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            dataset_analysis[dataset] = {
                "total_tests": len(dataset_df),
                "success_rate": dataset_df["overall_success"].mean(),
                "avg_detection_rate": dataset_df["detection_rate"].mean(),
                "avg_false_positive_rate": dataset_df["false_positive_rate"].mean(),
                "avg_final_accuracy": dataset_df["final_accuracy"].mean()
            }
        
        # Performance by trust profile
        profile_analysis = {}
        for profile in df["trust_profile"].unique():
            profile_df = df[df["trust_profile"] == profile]
            profile_analysis[profile] = {
                "total_tests": len(profile_df),
                "success_rate": profile_df["overall_success"].mean(),
                "avg_detection_rate": profile_df["detection_rate"].mean(),
                "avg_false_positive_rate": profile_df["false_positive_rate"].mean()
            }
        
        # Beta vs Manual thresholding comparison
        beta_df = df[df["use_beta_threshold"] == True]
        manual_df = df[df["use_beta_threshold"] == False]
        
        threshold_comparison = {
            "beta_threshold": {
                "success_rate": beta_df["overall_success"].mean() if len(beta_df) > 0 else 0,
                "avg_detection_rate": beta_df["detection_rate"].mean() if len(beta_df) > 0 else 0,
                "avg_false_positive_rate": beta_df["false_positive_rate"].mean() if len(beta_df) > 0 else 0
            },
            "manual_threshold": {
                "success_rate": manual_df["overall_success"].mean() if len(manual_df) > 0 else 0,
                "avg_detection_rate": manual_df["detection_rate"].mean() if len(manual_df) > 0 else 0,
                "avg_false_positive_rate": manual_df["false_positive_rate"].mean() if len(manual_df) > 0 else 0
            }
        }
        
        # Scenario analysis
        scenario_analysis = {}
        for scenario in df["scenario"].unique():
            scenario_df = df[df["scenario"] == scenario]
            scenario_analysis[scenario] = {
                "total_tests": len(scenario_df),
                "success_rate": scenario_df["overall_success"].mean(),
                "avg_detection_rate": scenario_df["detection_rate"].mean(),
                "avg_false_positive_rate": scenario_df["false_positive_rate"].mean()
            }
        
        analysis = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "failed_tests": len(failed_tests),
                "avg_detection_rate": avg_detection_rate,
                "avg_false_positive_rate": avg_false_positive_rate,
                "avg_trust_score": avg_trust_score
            },
            "dataset_analysis": dataset_analysis,
            "trust_profile_analysis": profile_analysis,
            "threshold_comparison": threshold_comparison,
            "scenario_analysis": scenario_analysis,
            "failed_tests": failed_tests
        }
        
        return analysis
    
    def generate_report(self, test_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        
        report = []
        report.append("# Comprehensive Trust Monitor Test Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        summary = analysis["summary"]
        report.append("## Executive Summary")
        report.append(f"- **Total Tests**: {summary['total_tests']}")
        report.append(f"- **Success Rate**: {summary['success_rate']:.2%}")
        report.append(f"- **Average Detection Rate**: {summary['avg_detection_rate']:.2%}")
        report.append(f"- **Average False Positive Rate**: {summary['avg_false_positive_rate']:.2%}")
        report.append(f"- **Failed Tests**: {summary['failed_tests']}")
        report.append("")
        
        # Dataset Performance
        report.append("## Dataset Performance")
        for dataset, stats in analysis["dataset_analysis"].items():
            report.append(f"### {dataset.upper()}")
            report.append(f"- Success Rate: {stats['success_rate']:.2%}")
            report.append(f"- Detection Rate: {stats['avg_detection_rate']:.2%}")
            report.append(f"- False Positive Rate: {stats['avg_false_positive_rate']:.2%}")
            report.append(f"- Final Accuracy: {stats['avg_final_accuracy']:.4f}")
            report.append("")
        
        # Trust Profile Analysis
        report.append("## Trust Profile Analysis")
        for profile, stats in analysis["trust_profile_analysis"].items():
            report.append(f"### {profile.title()} Profile")
            report.append(f"- Success Rate: {stats['success_rate']:.2%}")
            report.append(f"- Detection Rate: {stats['avg_detection_rate']:.2%}")
            report.append(f"- False Positive Rate: {stats['avg_false_positive_rate']:.2%}")
            report.append("")
        
        # Threshold Comparison
        report.append("## Threshold Method Comparison")
        threshold_comp = analysis["threshold_comparison"]
        report.append("### Beta Distribution Thresholding")
        beta_stats = threshold_comp["beta_threshold"]
        report.append(f"- Success Rate: {beta_stats['success_rate']:.2%}")
        report.append(f"- Detection Rate: {beta_stats['avg_detection_rate']:.2%}")
        report.append(f"- False Positive Rate: {beta_stats['avg_false_positive_rate']:.2%}")
        report.append("")
        
        report.append("### Manual Thresholding")
        manual_stats = threshold_comp["manual_threshold"]
        report.append(f"- Success Rate: {manual_stats['success_rate']:.2%}")
        report.append(f"- Detection Rate: {manual_stats['avg_detection_rate']:.2%}")
        report.append(f"- False Positive Rate: {manual_stats['avg_false_positive_rate']:.2%}")
        report.append("")
        
        # Scenario Analysis
        report.append("## Attack Scenario Analysis")
        for scenario, stats in analysis["scenario_analysis"].items():
            report.append(f"### {scenario.replace('_', ' ').title()}")
            report.append(f"- Success Rate: {stats['success_rate']:.2%}")
            report.append(f"- Detection Rate: {stats['avg_detection_rate']:.2%}")
            report.append(f"- False Positive Rate: {stats['avg_false_positive_rate']:.2%}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if summary["success_rate"] >= 0.8:
            report.append("✅ **Overall Performance**: EXCELLENT - Trust monitor is performing well")
        elif summary["success_rate"] >= 0.6:
            report.append("⚠️ **Overall Performance**: GOOD - Some improvements needed")
        else:
            report.append("❌ **Overall Performance**: POOR - Significant improvements required")
        
        if summary["avg_detection_rate"] >= 0.8:
            report.append("✅ **Detection Capability**: STRONG - Effectively detecting malicious behavior")
        else:
            report.append("❌ **Detection Capability**: WEAK - Missing too many attacks")
        
        if summary["avg_false_positive_rate"] <= 0.1:
            report.append("✅ **False Positive Rate**: ACCEPTABLE - Low false alarm rate")
        else:
            report.append("❌ **False Positive Rate**: HIGH - Too many false alarms")
        
        # Beta vs Manual recommendation
        beta_better = beta_stats['success_rate'] > manual_stats['success_rate']
        if beta_better:
            report.append("✅ **Threshold Method**: Beta distribution thresholding outperforms manual")
        else:
            report.append("⚠️ **Threshold Method**: Manual thresholding performs better than Beta")
        
        report.append("")
        
        # Failed Tests
        if analysis["failed_tests"]:
            report.append("## Failed Tests")
            for failed_test in analysis["failed_tests"]:
                report.append(f"- {failed_test['test_id']}: {failed_test.get('error', 'Unknown error')}")
            report.append("")
        
        return "\\n".join(report)
    
    def save_results(self, test_results: List[Dict[str, Any]], analysis: Dict[str, Any], report: str):
        """Save test results, analysis, and report."""
        
        # Save raw results
        results_file = os.path.join(self.output_dir, "test_results.json")
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # Save analysis
        analysis_file = os.path.join(self.output_dir, "analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save report
        report_file = os.path.join(self.output_dir, "test_report.md")
        with open(report_file, "w") as f:
            f.write(report)
        
        # Save CSV for easy analysis
        valid_results = [r for r in test_results if "error" not in r]
        if valid_results:
            df = pd.DataFrame(valid_results)
            # Select key columns for CSV
            key_columns = [
                "test_id", "dataset", "scenario", "trust_profile", "use_beta_threshold",
                "detection_rate", "false_positive_rate", "final_accuracy", "overall_success"
            ]
            df[key_columns].to_csv(os.path.join(self.output_dir, "results_summary.csv"), index=False)
        
        self.logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main function for running comprehensive trust monitor tests."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Trust Monitor Testing Framework"
    )
    
    parser.add_argument(
        "--test_level", 
        choices=["quick", "standard", "comprehensive"], 
        default="standard",
        help="Test level (quick=basic, standard=thorough, comprehensive=extensive)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="trust_test_results",
        help="Output directory for test results"
    )
    
    args = parser.parse_args()
    
    print("🔍 Starting Comprehensive Trust Monitor Testing...")
    print(f"Test Level: {args.test_level.upper()}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)
    
    # Create tester
    tester = TrustMonitorTester(output_dir=args.output_dir)
    
    try:
        # Run tests
        test_results = tester.run_comprehensive_tests(test_level=args.test_level)
        
        # Analyze results
        analysis = tester.analyze_results(test_results)
        
        # Generate report
        report = tester.generate_report(test_results, analysis)
        
        # Save everything
        tester.save_results(test_results, analysis, report)
        
        # Print summary
        print("\\n" + "=" * 60)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {analysis['summary']['total_tests']}")
        print(f"Success Rate: {analysis['summary']['success_rate']:.2%}")
        print(f"Detection Rate: {analysis['summary']['avg_detection_rate']:.2%}")
        print(f"False Positive Rate: {analysis['summary']['avg_false_positive_rate']:.2%}")
        print(f"\\nResults saved to: {args.output_dir}")
        print("📊 Check test_report.md for detailed analysis")
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()