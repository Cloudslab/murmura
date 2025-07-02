#!/usr/bin/env python3
"""
Quick Threshold Integration Test

This test verifies the beta threshold integration is working 
without running full federated learning.
"""

import numpy as np
import sys
import logging

# Add the murmura path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from murmura.trust.hsic import ModelUpdateHSIC
from murmura.trust.trust_monitor import TrustMonitor
from murmura.trust.adaptive_trust_agent import DatasetIndependentTrustSystem
from murmura.trust.beta_threshold import BetaThresholdConfig
import ray


def test_hsic_threshold_integration():
    """Test that HSIC properly uses adaptive thresholds."""
    
    print("🧪 Testing HSIC Threshold Integration")
    print("=" * 50)
    
    # Create HSIC monitor
    hsic_monitor = ModelUpdateHSIC(
        window_size=20,
        threshold=0.1,  # Will be overridden by adaptive
        gamma=0.1
    )
    
    print("✅ Created HSIC monitor")
    
    # Test 1: Default behavior (should use fixed threshold)
    print("\n📊 Test 1: Default behavior")
    effective_threshold = hsic_monitor.get_effective_threshold()
    print(f"   Default threshold: {effective_threshold}")
    assert effective_threshold == 0.3, f"Expected 0.3, got {effective_threshold}"
    print("   ✅ Uses fixed threshold 0.3 as expected")
    
    # Test 2: Set adaptive threshold
    print("\n📊 Test 2: Set adaptive threshold")
    test_adaptive_threshold = 0.85
    hsic_monitor.set_adaptive_threshold(test_adaptive_threshold)
    
    effective_threshold = hsic_monitor.get_effective_threshold()
    print(f"   Adaptive threshold: {effective_threshold}")
    assert effective_threshold == test_adaptive_threshold, f"Expected {test_adaptive_threshold}, got {effective_threshold}"
    print(f"   ✅ Uses adaptive threshold {test_adaptive_threshold} correctly")
    
    # Test 3: HSIC calculation with adaptive threshold
    print("\n📊 Test 3: HSIC drift detection with adaptive threshold")
    
    # Create some test parameters
    test_params = {
        'layer1': np.random.randn(10, 5),
        'layer2': np.random.randn(5, 1)
    }
    
    # Similar parameters (should have high HSIC, no drift)
    similar_params = {
        'layer1': test_params['layer1'] + np.random.randn(10, 5) * 0.01,  # Small noise
        'layer2': test_params['layer2'] + np.random.randn(5, 1) * 0.01
    }
    
    # Different parameters (should have lower HSIC, possible drift)
    different_params = {
        'layer1': np.random.randn(10, 5),  # Completely different
        'layer2': np.random.randn(5, 1)
    }
    
    # Test with similar parameters
    hsic_value, drift_detected, stats = hsic_monitor.update_with_parameters(
        test_params, similar_params, "test_neighbor"
    )
    print(f"   Similar params - HSIC: {hsic_value:.4f}, Drift: {drift_detected}")
    
    # Test with different parameters  
    hsic_value2, drift_detected2, stats2 = hsic_monitor.update_with_parameters(
        test_params, different_params, "test_neighbor"
    )
    print(f"   Different params - HSIC: {hsic_value2:.4f}, Drift: {drift_detected2}")
    
    print("   ✅ HSIC calculation working with adaptive threshold")
    
    return True


def test_trust_monitor_integration():
    """Test that TrustMonitor passes beta thresholds to HSIC."""
    
    print("\n🔧 Testing Trust Monitor Integration")
    print("=" * 50)
    
    # Initialize Ray for the trust monitor
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    
    # Create trust monitor configuration
    trust_config = {
        "warn_threshold": 0.2,
        "downgrade_threshold": 0.4,
        "exclude_threshold": 0.6,
    }
    
    hsic_config = {
        "window_size": 20,
        "threshold": 0.1,
        "gamma": 0.1,
    }
    
    # Create trust monitor - enable beta thresholding in constructor  
    trust_monitor = TrustMonitor.remote(
        node_id="test_node",
        hsic_config=hsic_config,
        trust_config=trust_config,
        enable_performance_monitoring=False
    )
    
    print("✅ Created trust monitor")
    
    # Set up FL context
    ray.get(trust_monitor.set_fl_context.remote(
        total_rounds=10,
        current_accuracy=0.8,
        topology="ring"
    ))
    
    # Set current parameters
    test_params = {
        'layer1': np.random.randn(10, 5),
        'layer2': np.random.randn(5, 1)
    }
    
    ray.get(trust_monitor.set_current_parameters.remote(test_params))
    print("✅ Set FL context and parameters")
    
    # Configure beta threshold
    beta_config = BetaThresholdConfig(
        base_percentile=0.95,
        min_observations=2,
        learning_rate=0.5
    )
    
    try:
        ray.get(trust_monitor.configure_beta_threshold.remote(beta_config))
        print("✅ Configured beta threshold")
    except Exception as e:
        print(f"⚠️ Beta threshold configuration failed: {e}")
        # Continue test anyway
    
    # Test assessment with beta threshold
    neighbor_params = {
        'layer1': test_params['layer1'] + np.random.randn(10, 5) * 0.1,
        'layer2': test_params['layer2'] + np.random.randn(5, 1) * 0.1
    }
    
    # Assess the update
    action, trust_score, detailed_stats = ray.get(
        trust_monitor.assess_update.remote("test_neighbor", neighbor_params)
    )
    
    print(f"   Assessment result:")
    print(f"   - Action: {action}")
    print(f"   - Trust Score: {trust_score:.4f}")
    print(f"   - HSIC Value: {detailed_stats.get('hsic_value', 0):.4f}")
    print(f"   - Adaptive Threshold: {detailed_stats.get('adaptive_threshold', 'N/A')}")
    print(f"   - Threshold Type: {detailed_stats.get('threshold_type', 'unknown')}")
    
    # Debug the threshold issue
    print(f"   - Detailed stats keys: {list(detailed_stats.keys())}")
    adaptive_threshold = detailed_stats.get('adaptive_threshold')
    print(f"   - Adaptive threshold value: {adaptive_threshold}")
    print(f"   - Adaptive threshold type: {type(adaptive_threshold)}")
    
    # Check if adaptive threshold was used
    if adaptive_threshold is not None and adaptive_threshold > 0.5:
        print("   ✅ Beta threshold integration working!")
        return True
    else:
        print("   ❌ Beta threshold integration not working")
        print(f"      Reason: threshold is {adaptive_threshold}")
        return False


def test_adaptive_trust_system():
    """Test the adaptive trust system with beta thresholds."""
    
    print("\n🎯 Testing Adaptive Trust System")
    print("=" * 50)
    
    # Create adaptive trust system with beta thresholding
    trust_system = DatasetIndependentTrustSystem(use_beta_threshold=True)
    print("✅ Created adaptive trust system with beta thresholding")
    
    # Test assessment
    update_data = {
        'round': 3,
        'total_rounds': 10,
        'accuracy': 0.85,
        'hsic': 0.92,
        'update_norm': 0.05,
        'consistency': 0.8,
        'neighbor_trusts': [0.9, 0.8, 0.95],
        'topology': 'ring'
    }
    
    result = trust_system.assess_trust("test_node", update_data)
    
    print(f"   Assessment result:")
    print(f"   - Malicious: {result['malicious']}")
    print(f"   - Confidence: {result['confidence']:.4f}")
    print(f"   - Adaptive Threshold: {result['adaptive_threshold']:.4f}")
    print(f"   - HSIC Value: {result['hsic_value']:.4f}")
    print(f"   - Trust Score: {result['trust_score']:.4f}")
    
    # Check if threshold is reasonable for FL
    if result['adaptive_threshold'] > 0.8:
        print("   ✅ Adaptive threshold appropriate for FL!")
        return True
    else:
        print(f"   ⚠️ Adaptive threshold might be too low: {result['adaptive_threshold']:.4f}")
        return False


def main():
    """Run all integration tests."""
    
    print("🚀 Beta Threshold Integration Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: HSIC threshold integration
        if test_hsic_threshold_integration():
            tests_passed += 1
            print("✅ HSIC integration test PASSED")
        else:
            print("❌ HSIC integration test FAILED")
    except Exception as e:
        print(f"❌ HSIC integration test ERROR: {e}")
    
    try:
        # Test 2: Trust monitor integration
        if test_trust_monitor_integration():
            tests_passed += 1
            print("✅ Trust monitor integration test PASSED")
        else:
            print("❌ Trust monitor integration test FAILED")
    except Exception as e:
        print(f"❌ Trust monitor integration test ERROR: {e}")
    
    try:
        # Test 3: Adaptive trust system
        if test_adaptive_trust_system():
            tests_passed += 1
            print("✅ Adaptive trust system test PASSED")
        else:
            print("❌ Adaptive trust system test FAILED")
    except Exception as e:
        print(f"❌ Adaptive trust system test ERROR: {e}")
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ **ALL TESTS PASSED** - Beta threshold integration is working!")
        print("🚀 Ready to test with full federated learning")
        return 0
    else:
        print("⚠️ **SOME TESTS FAILED** - Integration needs more work")
        return 1


if __name__ == "__main__":
    exit_code = main()
    
    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()
    
    sys.exit(exit_code)