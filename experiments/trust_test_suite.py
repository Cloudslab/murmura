#!/usr/bin/env python3
"""
Comprehensive Trust System Test Suite

This unified test suite consolidates all trust system testing functionality:
- Unit tests for components (HSIC, Beta thresholds)
- Integration tests (Trust monitor with adaptive thresholds)
- End-to-end tests (Full FL with attacks)
- Performance benchmarks
- Baseline experiments

Usage:
    # Quick test (3 rounds)
    python trust_test_suite.py quick

    # Unit tests only
    python trust_test_suite.py unit

    # Integration tests
    python trust_test_suite.py integration

    # End-to-end test with specific scenario
    python trust_test_suite.py e2e --scenario attack --intensity moderate

    # Full benchmark suite
    python trust_test_suite.py benchmark

    # Compare all scenarios
    python trust_test_suite.py compare

    # Run everything
    python trust_test_suite.py all
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

# Add murmura to path
sys.path.insert(0, str(Path(__file__).parent))

import ray
from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist
from murmura.examples.adaptive_trust_cifar10_example import run_adaptive_trust_cifar10
from murmura.trust.hsic import ModelUpdateHSIC
from murmura.trust.trust_monitor import TrustMonitor
from murmura.trust.adaptive_trust_agent import DatasetIndependentTrustSystem
from murmura.trust.beta_threshold import BetaThresholdConfig


@dataclass
class TestResult:
    """Result from a single test."""
    test_name: str
    test_type: str  # unit, integration, e2e, benchmark
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type,
            'success': self.success,
            'duration': self.duration,
            'metrics': self._make_json_serializable(self.metrics),
            'error': self.error
        }
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if obj is None:
            return None
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return str(obj)


class TrustTestSuite:
    """Comprehensive trust system test suite."""
    
    def __init__(self, output_dir: str = "trust_test_results"):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.output_dir / f"test_run_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.results_dir / "test_suite.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("TrustTestSuite")
    
    # ==================== UNIT TESTS ====================
    
    def run_unit_tests(self) -> List[TestResult]:
        """Run unit tests for individual components."""
        self.logger.info("=" * 60)
        self.logger.info("Running Unit Tests")
        self.logger.info("=" * 60)
        
        results = []
        results.append(self._test_hsic_threshold_integration())
        results.append(self._test_beta_threshold_config())
        results.append(self._test_adaptive_trust_system())
        
        return results
    
    def _test_hsic_threshold_integration(self) -> TestResult:
        """Test HSIC threshold integration."""
        test_name = "HSIC Threshold Integration"
        self.logger.info(f"Testing {test_name}...")
        
        start_time = time.time()
        try:
            # Create HSIC monitor
            hsic_monitor = ModelUpdateHSIC(
                window_size=20,
                threshold=0.1,
                gamma=0.1
            )
            
            # Test 1: Default behavior
            default_threshold = hsic_monitor.get_effective_threshold()
            assert default_threshold == 0.3, f"Expected 0.3, got {default_threshold}"
            
            # Test 2: Adaptive threshold
            test_threshold = 0.85
            hsic_monitor.set_adaptive_threshold(test_threshold)
            effective_threshold = hsic_monitor.get_effective_threshold()
            assert effective_threshold == test_threshold, f"Expected {test_threshold}, got {effective_threshold}"
            
            # Test 3: Reset to default by creating new monitor
            hsic_monitor_new = ModelUpdateHSIC(
                window_size=20,
                threshold=0.1,
                gamma=0.1
            )
            cleared_threshold = hsic_monitor_new.get_effective_threshold()
            assert cleared_threshold == 0.3, f"Expected 0.3 for new monitor, got {cleared_threshold}"
            
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type="unit",
                success=True,
                duration=duration,
                metrics={
                    "default_threshold": default_threshold,
                    "adaptive_threshold": effective_threshold,
                    "cleared_threshold": cleared_threshold
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="unit",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    def _test_beta_threshold_config(self) -> TestResult:
        """Test Beta threshold configuration."""
        test_name = "Beta Threshold Configuration"
        self.logger.info(f"Testing {test_name}...")
        
        start_time = time.time()
        try:
            # Create beta config with default parameters
            beta_config = BetaThresholdConfig()
            
            # Test serialization
            config_dict = beta_config.to_dict()
            assert config_dict['alpha_prior'] == 1.0
            assert 'learning_rate' in config_dict
            
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type="unit",
                success=True,
                duration=duration,
                metrics={"config": config_dict}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="unit",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    def _test_adaptive_trust_system(self) -> TestResult:
        """Test adaptive trust system."""
        test_name = "Adaptive Trust System"
        self.logger.info(f"Testing {test_name}...")
        
        start_time = time.time()
        try:
            # Create trust system
            trust_system = DatasetIndependentTrustSystem(use_beta_threshold=True)
            
            # Test with sample data
            update_data = {
                'round': 5,
                'total_rounds': 10,
                'accuracy': 0.95,
                'hsic': 0.98,
                'update_norm': 0.01,
                'consistency': 0.9,
                'neighbor_trusts': [0.9, 0.95, 0.92],
                'topology': 'ring'
            }
            
            result = trust_system.assess_trust("node_1", update_data)
            
            assert 'trust_score' in result
            assert 'malicious' in result
            assert 'confidence' in result
            assert 0 <= result['trust_score'] <= 1
            
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type="unit",
                success=True,
                duration=duration,
                metrics=result
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="unit",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    # ==================== INTEGRATION TESTS ====================
    
    def run_integration_tests(self) -> List[TestResult]:
        """Run integration tests."""
        self.logger.info("=" * 60)
        self.logger.info("Running Integration Tests")
        self.logger.info("=" * 60)
        
        results = []
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        results.append(self._test_trust_monitor_integration())
        
        return results
    
    def _test_trust_monitor_integration(self) -> TestResult:
        """Test trust monitor with beta threshold integration."""
        test_name = "Trust Monitor Integration"
        self.logger.info(f"Testing {test_name}...")
        
        start_time = time.time()
        try:
            # Create trust monitor
            trust_monitor = TrustMonitor.remote(
                node_id="test_node",
                hsic_config={
                    "window_size": 30,
                    "gamma": 0.1,
                    "baseline_percentile": 85.0
                },
                trust_config={
                    "warn_threshold": 0.7,
                    "downgrade_threshold": 0.5,
                    "exclude_threshold": 0.3
                }
            )
            
            # Configure beta threshold
            beta_config = BetaThresholdConfig()
            ray.get(trust_monitor.configure_beta_threshold.remote(beta_config.to_dict()))
            
            # Set FL context
            ray.get(trust_monitor.set_fl_context.remote(
                total_rounds=10,
                current_accuracy=0.5,
                topology="ring"
            ))
            
            # Generate dummy parameters
            params1 = {f"layer_{i}": np.random.randn(10, 10) for i in range(3)}
            params2 = {f"layer_{i}": params1[f"layer_{i}"] + np.random.randn(10, 10) * 0.01 for i in range(3)}
            
            # Set current parameters
            ray.get(trust_monitor.set_current_parameters.remote(params1))
            
            # Assess neighbor
            action, trust_score, stats = ray.get(
                trust_monitor.assess_neighbor_trust.remote("neighbor_1", params2)
            )
            
            # Get trust report
            report = ray.get(trust_monitor.get_trust_report.remote())
            
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type="integration",
                success=True,
                duration=duration,
                metrics={
                    "action": str(action),
                    "trust_score": trust_score,
                    "report": report
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="integration",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    # ==================== END-TO-END TESTS ====================
    
    def run_e2e_tests(self, scenario: str = "all", **kwargs) -> List[TestResult]:
        """Run end-to-end tests with full FL simulation."""
        self.logger.info("=" * 60)
        self.logger.info("Running End-to-End Tests")
        self.logger.info("=" * 60)
        
        results = []
        
        scenarios = {
            "baseline": self._test_honest_baseline,
            "attack": self._test_attack_scenario,
            "stealth": self._test_stealth_attack,
            "multiple": self._test_multiple_attackers
        }
        
        if scenario == "all":
            for name, test_func in scenarios.items():
                results.append(test_func(**kwargs))
        elif scenario in scenarios:
            results.append(scenarios[scenario](**kwargs))
        else:
            self.logger.error(f"Unknown scenario: {scenario}")
        
        return results
    
    def _test_honest_baseline(self, rounds: int = 5, **kwargs) -> TestResult:
        """Test honest baseline - should have zero false positives."""
        test_name = "Honest Baseline"
        self.logger.info(f"Testing {test_name} ({rounds} rounds)...")
        
        start_time = time.time()
        try:
            result = run_adaptive_trust_mnist(
                num_rounds=rounds,
                attack_config=None,
                log_level='WARNING'
            )
            
            fl_results = result.get('fl_results', {})
            trust_report = fl_results.get('trust_report', {})
            
            # Check for false positives
            false_positives = trust_report.get('false_positive_rate', 0)
            excluded_nodes = trust_report.get('excluded_nodes', 0)
            
            duration = time.time() - start_time
            success = false_positives == 0 and excluded_nodes == 0
            
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=success,
                duration=duration,
                metrics={
                    "false_positives": false_positives,
                    "excluded_nodes": excluded_nodes,
                    "final_accuracy": fl_results.get('final_consensus', {}).get('consensus_accuracy', 0),
                    "trust_report": trust_report
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    def _test_attack_scenario(self, intensity: str = "moderate", rounds: int = 5, **kwargs) -> TestResult:
        """Test attack scenario - should detect malicious nodes."""
        test_name = f"Attack Scenario ({intensity})"
        self.logger.info(f"Testing {test_name} ({rounds} rounds)...")
        
        start_time = time.time()
        try:
            attack_config = {
                'attack_type': 'gradual_label_flipping',
                'malicious_fraction': 0.167,  # 1 out of 6
                'attack_intensity': intensity,
                'stealth_level': 'medium'
            }
            
            result = run_adaptive_trust_mnist(
                num_rounds=rounds,
                attack_config=attack_config,
                log_level='WARNING'
            )
            
            fl_results = result.get('fl_results', {})
            attack_stats = fl_results.get('attack_statistics', {})
            trust_report = fl_results.get('trust_report', {})
            
            # Check detection
            total_attackers = attack_stats.get('total_attackers', 0)
            detected_attackers = attack_stats.get('detected_attackers', 0)
            detection_rate = attack_stats.get('detection_rate', 0)
            
            duration = time.time() - start_time
            success = detection_rate > 0  # Should detect at least some attackers
            
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=success,
                duration=duration,
                metrics={
                    "total_attackers": total_attackers,
                    "detected_attackers": detected_attackers,
                    "detection_rate": detection_rate,
                    "false_positive_rate": trust_report.get('false_positive_rate', 0),
                    "final_accuracy": fl_results.get('final_consensus', {}).get('consensus_accuracy', 0)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    def _test_stealth_attack(self, rounds: int = 10, **kwargs) -> TestResult:
        """Test stealth attack - harder to detect."""
        test_name = "Stealth Attack"
        self.logger.info(f"Testing {test_name} ({rounds} rounds)...")
        
        start_time = time.time()
        try:
            attack_config = {
                'attack_type': 'gradual_label_flipping',
                'malicious_fraction': 0.167,
                'attack_intensity': 'low',
                'stealth_level': 'high'
            }
            
            result = run_adaptive_trust_mnist(
                num_rounds=rounds,
                attack_config=attack_config,
                log_level='WARNING'
            )
            
            fl_results = result.get('fl_results', {})
            attack_stats = fl_results.get('attack_statistics', {})
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=True,  # Success means test ran without error
                duration=duration,
                metrics={
                    "detection_rate": attack_stats.get('detection_rate', 0),
                    "rounds_to_detection": attack_stats.get('per_attacker', {}).get('node_0', {}).get('detection_round', -1),
                    "final_accuracy": fl_results.get('final_consensus', {}).get('consensus_accuracy', 0)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    def _test_multiple_attackers(self, num_attackers: int = 2, rounds: int = 5, **kwargs) -> TestResult:
        """Test multiple attackers scenario."""
        test_name = f"Multiple Attackers ({num_attackers})"
        self.logger.info(f"Testing {test_name} ({rounds} rounds)...")
        
        start_time = time.time()
        try:
            attack_config = {
                'attack_type': 'gradual_label_flipping',
                'malicious_fraction': num_attackers / 6,  # Out of 6 total nodes
                'attack_intensity': 'moderate',
                'stealth_level': 'medium'
            }
            
            result = run_adaptive_trust_mnist(
                num_rounds=rounds,
                attack_config=attack_config,
                log_level='WARNING'
            )
            
            fl_results = result.get('fl_results', {})
            attack_stats = fl_results.get('attack_statistics', {})
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=True,
                duration=duration,
                metrics={
                    "total_attackers": attack_stats.get('total_attackers', 0),
                    "detected_attackers": attack_stats.get('detected_attackers', 0),
                    "detection_rate": attack_stats.get('detection_rate', 0),
                    "final_accuracy": fl_results.get('final_consensus', {}).get('consensus_accuracy', 0)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="e2e",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    # ==================== QUICK TEST ====================
    
    def run_quick_test(self) -> List[TestResult]:
        """Run quick 3-round test for rapid iteration."""
        self.logger.info("=" * 60)
        self.logger.info("Running Quick Test (3 rounds)")
        self.logger.info("=" * 60)
        
        results = []
        
        # Quick baseline
        results.append(self._test_honest_baseline(rounds=3))
        
        # Quick attack
        results.append(self._test_attack_scenario(intensity="moderate", rounds=3))
        
        return results
    
    # ==================== BENCHMARK TESTS ====================
    
    def run_benchmark_tests(self) -> List[TestResult]:
        """Run comprehensive benchmark tests."""
        self.logger.info("=" * 60)
        self.logger.info("Running Benchmark Tests")
        self.logger.info("=" * 60)
        
        results = []
        
        # Test configurations
        configurations = [
            {"dataset": "mnist", "rounds": 10, "topology": "ring"},
            {"dataset": "mnist", "rounds": 10, "topology": "fully_connected"},
            {"dataset": "cifar10", "rounds": 10, "topology": "ring"},
        ]
        
        attack_intensities = ["low", "moderate", "high"]
        
        for config in configurations:
            # Baseline
            results.append(self._run_benchmark_config(config, attack_config=None))
            
            # With attacks
            for intensity in attack_intensities:
                attack_config = {
                    'attack_type': 'gradual_label_flipping',
                    'malicious_fraction': 0.167,
                    'attack_intensity': intensity,
                    'stealth_level': 'medium'
                }
                results.append(self._run_benchmark_config(config, attack_config=attack_config))
        
        return results
    
    def _run_benchmark_config(self, config: Dict[str, Any], attack_config: Optional[Dict] = None) -> TestResult:
        """Run a single benchmark configuration."""
        test_name = f"Benchmark: {config['dataset']} - {config.get('topology', 'ring')}"
        if attack_config:
            test_name += f" - {attack_config['attack_intensity']} attack"
        else:
            test_name += " - baseline"
        
        self.logger.info(f"Running {test_name}...")
        
        start_time = time.time()
        try:
            if config['dataset'] == 'mnist':
                result = run_adaptive_trust_mnist(
                    num_rounds=config['rounds'],
                    topology=config.get('topology', 'ring'),
                    attack_config=attack_config,
                    log_level='WARNING'
                )
            elif config['dataset'] == 'cifar10':
                result = run_adaptive_trust_cifar10(
                    num_rounds=config['rounds'],
                    topology=config.get('topology', 'ring'),
                    attack_config=attack_config,
                    log_level='WARNING'
                )
            else:
                raise ValueError(f"Unknown dataset: {config['dataset']}")
            
            fl_results = result.get('fl_results', {})
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                test_type="benchmark",
                success=True,
                duration=duration,
                metrics={
                    "config": config,
                    "attack_config": attack_config,
                    "final_accuracy": fl_results.get('final_consensus', {}).get('consensus_accuracy', 0),
                    "training_time": fl_results.get('total_time', 0),
                    "trust_report": fl_results.get('trust_report', {})
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test failed: {str(e)}")
            return TestResult(
                test_name=test_name,
                test_type="benchmark",
                success=False,
                duration=duration,
                metrics={},
                error=str(e)
            )
    
    # ==================== COMPARISON TESTS ====================
    
    def run_comparison_tests(self) -> Dict[str, Any]:
        """Run comparison between different scenarios."""
        self.logger.info("=" * 60)
        self.logger.info("Running Comparison Tests")
        self.logger.info("=" * 60)
        
        scenarios = {
            "Honest Baseline": None,
            "Low Attack": {
                'attack_type': 'gradual_label_flipping',
                'malicious_fraction': 0.167,
                'attack_intensity': 'low',
                'stealth_level': 'high'
            },
            "Moderate Attack": {
                'attack_type': 'gradual_label_flipping',
                'malicious_fraction': 0.167,
                'attack_intensity': 'moderate',
                'stealth_level': 'medium'
            },
            "High Attack": {
                'attack_type': 'gradual_label_flipping',
                'malicious_fraction': 0.167,
                'attack_intensity': 'high',
                'stealth_level': 'low'
            }
        }
        
        comparison_results = {}
        
        for scenario_name, attack_config in scenarios.items():
            self.logger.info(f"Testing scenario: {scenario_name}")
            
            try:
                result = run_adaptive_trust_mnist(
                    num_rounds=5,
                    attack_config=attack_config,
                    log_level='WARNING'
                )
                
                fl_results = result.get('fl_results', {})
                attack_stats = fl_results.get('attack_statistics', {})
                trust_report = fl_results.get('trust_report', {})
                
                comparison_results[scenario_name] = {
                    "final_accuracy": fl_results.get('final_consensus', {}).get('consensus_accuracy', 0),
                    "detection_rate": attack_stats.get('detection_rate', 0),
                    "false_positive_rate": trust_report.get('false_positive_rate', 0),
                    "excluded_nodes": trust_report.get('excluded_nodes', 0),
                    "average_trust": trust_report.get('average_trust_score', 1.0)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to test {scenario_name}: {str(e)}")
                comparison_results[scenario_name] = {"error": str(e)}
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        # Save comparison results
        comparison_file = self.results_dir / "comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump({
                "scenarios": comparison_results,
                "summary": summary
            }, f, indent=2)
        
        return comparison_results
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of comparison results."""
        summary = {
            "best_accuracy": None,
            "best_detection": None,
            "lowest_false_positives": None
        }
        
        # Find best metrics
        best_accuracy = 0
        best_detection = 0
        lowest_fp = float('inf')
        
        for scenario, metrics in results.items():
            if 'error' in metrics:
                continue
                
            if metrics.get('final_accuracy', 0) > best_accuracy:
                best_accuracy = metrics['final_accuracy']
                summary['best_accuracy'] = scenario
                
            if metrics.get('detection_rate', 0) > best_detection:
                best_detection = metrics['detection_rate']
                summary['best_detection'] = scenario
                
            if metrics.get('false_positive_rate', float('inf')) < lowest_fp:
                lowest_fp = metrics['false_positive_rate']
                summary['lowest_false_positives'] = scenario
        
        return summary
    
    # ==================== MAIN EXECUTION ====================
    
    def run_all_tests(self) -> None:
        """Run all test suites."""
        all_results = []
        
        # Unit tests
        all_results.extend(self.run_unit_tests())
        
        # Integration tests
        all_results.extend(self.run_integration_tests())
        
        # End-to-end tests
        all_results.extend(self.run_e2e_tests())
        
        # Save all results
        self._save_results(all_results)
        self._print_summary(all_results)
    
    def _save_results(self, results: List[TestResult]) -> None:
        """Save test results to file."""
        results_data = {
            "timestamp": self.timestamp,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.success),
            "failed_tests": sum(1 for r in results if not r.success),
            "results": [r.to_dict() for r in results]
        }
        
        results_file = self.results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _print_summary(self, results: List[TestResult]) -> None:
        """Print test summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TEST SUMMARY")
        self.logger.info("=" * 60)
        
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        
        self.logger.info(f"Total Tests: {total}")
        self.logger.info(f"Passed: {passed} ✓")
        self.logger.info(f"Failed: {failed} ✗")
        self.logger.info(f"Success Rate: {passed/total*100:.1f}%")
        
        if failed > 0:
            self.logger.info("\nFailed Tests:")
            for r in results:
                if not r.success:
                    self.logger.info(f"  - {r.test_name}: {r.error}")
        
        # Print timing summary
        total_time = sum(r.duration for r in results)
        self.logger.info(f"\nTotal Time: {total_time:.2f}s")
        
        # Group by test type
        by_type = {}
        for r in results:
            if r.test_type not in by_type:
                by_type[r.test_type] = []
            by_type[r.test_type].append(r)
        
        self.logger.info("\nResults by Type:")
        for test_type, type_results in by_type.items():
            type_passed = sum(1 for r in type_results if r.success)
            self.logger.info(f"  {test_type}: {type_passed}/{len(type_results)} passed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Trust System Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'command',
        choices=['quick', 'unit', 'integration', 'e2e', 'benchmark', 'compare', 'all'],
        help='Test command to run'
    )
    
    parser.add_argument(
        '--scenario',
        choices=['baseline', 'attack', 'stealth', 'multiple', 'all'],
        default='all',
        help='E2E test scenario (for e2e command)'
    )
    
    parser.add_argument(
        '--intensity',
        choices=['low', 'moderate', 'high'],
        default='moderate',
        help='Attack intensity (for attack scenarios)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=5,
        help='Number of FL rounds'
    )
    
    parser.add_argument(
        '--output-dir',
        default='trust_test_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create test suite
    suite = TrustTestSuite(args.output_dir)
    
    # Run requested tests
    if args.command == 'quick':
        results = suite.run_quick_test()
        suite._save_results(results)
        suite._print_summary(results)
        
    elif args.command == 'unit':
        results = suite.run_unit_tests()
        suite._save_results(results)
        suite._print_summary(results)
        
    elif args.command == 'integration':
        results = suite.run_integration_tests()
        suite._save_results(results)
        suite._print_summary(results)
        
    elif args.command == 'e2e':
        results = suite.run_e2e_tests(
            scenario=args.scenario,
            intensity=args.intensity,
            rounds=args.rounds
        )
        suite._save_results(results)
        suite._print_summary(results)
        
    elif args.command == 'benchmark':
        results = suite.run_benchmark_tests()
        suite._save_results(results)
        suite._print_summary(results)
        
    elif args.command == 'compare':
        comparison_results = suite.run_comparison_tests()
        suite.logger.info(f"\nComparison results saved to {suite.results_dir}/comparison_results.json")
        
    elif args.command == 'all':
        suite.run_all_tests()
    
    # Cleanup Ray if initialized
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()