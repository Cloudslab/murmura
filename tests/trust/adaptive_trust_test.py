#!/usr/bin/env python3
"""
Test suite for adaptive trust system.

This test verifies that the adaptive trust system produces zero false positives
on honest-only federated learning and can properly integrate with the trust
monitoring framework.
"""

import logging
import json
import sys
import time
from pathlib import Path

import ray
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_adaptive_trust_basic():
    """Basic test of adaptive trust system."""
    
    logger = logging.getLogger("adaptive_trust_test")
    logger.info("Starting basic adaptive trust test...")
    
    # Test the adaptive trust agent directly
    from murmura.trust.adaptive_trust_agent import (
        DatasetIndependentTrustSystem,
        TrustContext
    )
    
    # Initialize the trust system
    trust_system = DatasetIndependentTrustSystem()
    logger.info("Initialized adaptive trust system")
    
    # Test with normal FL parameters (should not flag as malicious)
    normal_context = TrustContext(
        current_round=3,
        total_rounds=10,
        convergence_rate=0.8,
        global_accuracy=0.85,
        topology_type="ring",
        network_stability=0.95,
        communication_latency=0.1,
        recent_attack_rate=0.0,
        false_positive_rate=0.0,
        hsic_value=0.94,  # Normal FL HSIC value
        update_magnitude=0.02,
        update_direction_consistency=0.9,
        neighbor_trust_scores=[0.8, 0.9, 0.85]
    )
    
    # Test normal parameters
    update_data = {
        'round': 3,
        'total_rounds': 10,
        'accuracy': 0.85,
        'hsic': 0.94,
        'update_norm': 0.02,
        'consistency': 0.9,
        'neighbor_trusts': [0.8, 0.9, 0.85],
        'topology': 'ring',
    }
    
    result = trust_system.assess_trust("test_node", update_data)
    
    logger.info(f"Trust assessment result: {result}")
    logger.info(f"Decision: {'MALICIOUS' if result['malicious'] else 'HONEST'}")
    logger.info(f"Confidence: {result['confidence']:.3f}")
    logger.info(f"Reasoning: {result['reasoning']}")
    
    # Test with multiple normal samples to verify no false positives
    false_positives = 0
    total_tests = 20
    
    logger.info(f"Testing {total_tests} normal FL scenarios...")
    
    for i in range(total_tests):
        # Vary HSIC values in normal FL range
        test_hsic = 0.90 + (i % 10) * 0.01  # Range 0.90 to 0.99
        test_update_data = {
            'round': 2 + i % 8,
            'total_rounds': 10,
            'accuracy': 0.7 + (i % 5) * 0.05,
            'hsic': test_hsic,
            'update_norm': 0.01 + (i % 3) * 0.01,
            'consistency': 0.8 + (i % 4) * 0.05,
            'neighbor_trusts': [0.7 + (j % 3) * 0.1 for j in range(3)],
            'topology': 'ring',
        }
        
        test_result = trust_system.assess_trust(f"node_{i}", test_update_data)
        
        if test_result['malicious']:
            false_positives += 1
            logger.warning(f"False positive detected for node_{i}: {test_result['reasoning']}")
    
    false_positive_rate = false_positives / total_tests
    
    logger.info("=" * 60)
    logger.info("ADAPTIVE TRUST TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"False Positives: {false_positives}")
    logger.info(f"False Positive Rate: {false_positive_rate:.1%}")
    logger.info(f"Test Status: {'✅ PASSED' if false_positives == 0 else '❌ FAILED'}")
    logger.info("=" * 60)
    
    return false_positives == 0

def test_trust_monitor_integration():
    """Test trust monitor with adaptive agent integration."""
    
    logger = logging.getLogger("trust_monitor_test")
    logger.info("Testing trust monitor integration...")
    
    from murmura.trust.trust_monitor import TrustMonitor
    import numpy as np
    
    # Create a trust monitor
    monitor = TrustMonitor.remote(
        node_id="test_node",
        hsic_config={},
        trust_config={}
    )
    
    # Set FL context
    ray.get(monitor.set_fl_context.remote(
        total_rounds=10,
        current_accuracy=0.8,
        topology="ring"
    ))
    
    # Create dummy model parameters
    dummy_params = {
        "layer1.weight": np.random.randn(10, 784),
        "layer1.bias": np.random.randn(10),
        "layer2.weight": np.random.randn(10, 10),
        "layer2.bias": np.random.randn(10),
    }
    
    # Set current parameters
    ray.get(monitor.set_current_parameters.remote(dummy_params))
    ray.get(monitor.set_round_number.remote(3))
    
    # Create slightly different neighbor parameters (normal FL update)
    neighbor_params = {}
    for key, value in dummy_params.items():
        # Add small random noise (typical FL update)
        noise = np.random.normal(0, 0.01, value.shape)
        neighbor_params[key] = value + noise
    
    # Assess the update
    action, trust_score, stats = ray.get(monitor.assess_update.remote(
        "neighbor_1", neighbor_params
    ))
    
    logger.info(f"Trust assessment completed:")
    logger.info(f"Action: {action}")
    logger.info(f"Trust Score: {trust_score:.3f}")
    logger.info(f"Adaptive Decision: {stats.get('adaptive_decision', 'N/A')}")
    logger.info(f"Adaptive Confidence: {stats.get('adaptive_confidence', 'N/A'):.3f}")
    logger.info(f"Reasoning: {stats.get('adaptive_reasoning', 'N/A')}")
    
    # Test should pass if we don't exclude honest neighbor
    test_passed = action.value != "exclude"
    
    logger.info(f"Integration test: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    return test_passed

def main():
    """Main test function."""
    
    logger = logging.getLogger("main")
    logger.info("Starting adaptive trust system tests")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Run basic test
        logger.info("Running basic adaptive trust test...")
        basic_test_passed = test_adaptive_trust_basic()
        
        # Run integration test
        logger.info("Running trust monitor integration test...")
        integration_test_passed = test_trust_monitor_integration()
        
        # Summary
        all_tests_passed = basic_test_passed and integration_test_passed
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Basic Test: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
        logger.info(f"Integration Test: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
        logger.info(f"Overall: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
        logger.info("=" * 60)
        
        return 0 if all_tests_passed else 1
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)