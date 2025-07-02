#!/usr/bin/env python3
"""
Test Beta Threshold Integration - Verify the fix works

This test specifically verifies that:
1. Beta thresholds are being passed to HSIC monitors
2. HSIC drift detection uses adaptive thresholds (not fixed 0.3)
3. Gradual attacks are now detected with proper thresholds
"""

import json
import os
import sys
import logging
import time
from typing import Dict, Any

# Add the murmura path
sys.path.append('/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura')

from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist


def test_beta_integration():
    """Test that beta threshold integration is working."""
    
    print("🧪 Testing Beta Threshold Integration")
    print("Verifying that HSIC uses adaptive thresholds from beta system...")
    print("=" * 60)
    
    # Test scenarios to verify integration
    scenarios = [
        {
            "name": "honest_with_beta",
            "description": "Honest baseline with beta thresholding",
            "attack_config": None,
            "use_beta": True,
            "expected_threshold": "> 0.9"  # Beta should set high thresholds
        },
        {
            "name": "honest_without_beta",
            "description": "Honest baseline without beta (fallback to fixed)",
            "attack_config": None,
            "use_beta": False,
            "expected_threshold": "0.3"  # Should fallback to fixed
        },
        {
            "name": "attack_with_beta",
            "description": "Gradual attack with beta thresholding",
            "attack_config": {
                "attack_type": "gradual_label_flipping",
                "malicious_fraction": 0.167,  # 1 out of 6
                "attack_intensity": "moderate",
                "stealth_level": "medium"
            },
            "use_beta": True,
            "expected_detection": True  # Should detect with proper threshold
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Beta threshold: {scenario['use_beta']}")
        
        start_time = time.time()
        
        try:
            # Run short test to verify integration
            result = run_adaptive_trust_mnist(
                num_actors=6,
                num_rounds=8,  # Enough rounds for beta calibration
                topology_type="ring",
                trust_profile="strict",
                use_beta_threshold=scenario["use_beta"],
                attack_config=scenario["attack_config"],
                output_dir=f"beta_test_{scenario['name']}",
                log_level="DEBUG"  # Capture detailed logs
            )
            
            # Extract key metrics
            fl_results = result.get("fl_results", {})
            trust_metrics = fl_results.get("trust_metrics", [])
            trust_analysis = result.get("trust_analysis", {})
            attack_stats = fl_results.get("attack_statistics", {})
            
            # Analyze threshold usage
            threshold_analysis = analyze_threshold_usage(trust_metrics)
            
            test_result = {
                "scenario": scenario["name"],
                "use_beta": scenario["use_beta"],
                "time_taken": time.time() - start_time,
                "threshold_analysis": threshold_analysis,
                "detection_rate": attack_stats.get("detection_rate", 0.0),
                "total_excluded": trust_analysis.get("total_excluded", 0),
                "attack_config": scenario["attack_config"]
            }
            
            results[scenario["name"]] = test_result
            
            # Quick analysis
            print(f"  ⏱️  Time: {test_result['time_taken']:.1f}s")
            print(f"  🎯 Detection Rate: {test_result['detection_rate']:.1%}")
            print(f"  ❌ Excluded Nodes: {test_result['total_excluded']}")
            
            # Threshold analysis
            if threshold_analysis["adaptive_thresholds_used"]:
                avg_threshold = threshold_analysis["average_adaptive_threshold"]
                print(f"  📊 Adaptive Threshold: {avg_threshold:.3f} (✅ Using Beta)")
            else:
                print(f"  📊 Using Fixed Threshold: 0.3 (⚠️ Fallback)")
            
            print(f"  🔄 Threshold Type: {threshold_analysis['threshold_type']}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {str(e)}")
            results[scenario["name"]] = {"error": str(e)}
    
    return results


def analyze_threshold_usage(trust_metrics):
    """Analyze which thresholds are being used in the trust metrics."""
    
    adaptive_thresholds = []
    threshold_types = []
    has_adaptive = False
    
    for round_data in trust_metrics:
        node_reports = round_data.get("node_trust_reports", {})
        
        for node_id, node_report in node_reports.items():
            # Check if adaptive threshold info is available
            trust_stats = node_report.get("trust_scores", {})
            
            # Look for threshold information in the detailed stats
            if "adaptive_threshold" in node_report:
                adaptive_thresholds.append(node_report["adaptive_threshold"])
                has_adaptive = True
            
            # Check threshold type
            if "threshold_type" in node_report:
                threshold_types.append(node_report["threshold_type"])
    
    return {
        "adaptive_thresholds_used": has_adaptive,
        "adaptive_thresholds": adaptive_thresholds,
        "average_adaptive_threshold": sum(adaptive_thresholds) / len(adaptive_thresholds) if adaptive_thresholds else 0.0,
        "threshold_types": threshold_types,
        "threshold_type": threshold_types[0] if threshold_types else "unknown"
    }


def generate_integration_report(results):
    """Generate report on the beta integration test."""
    
    print("\n" + "=" * 60)
    print("🎯 BETA THRESHOLD INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    # Check if beta integration is working
    integration_working = True
    issues_found = []
    
    for scenario_name, result in results.items():
        if "error" in result:
            print(f"\n❌ {scenario_name}: FAILED")
            print(f"   Error: {result['error']}")
            integration_working = False
            issues_found.append(f"Test execution failed in {scenario_name}")
            continue
        
        print(f"\n📊 {scenario_name}:")
        
        use_beta = result["use_beta"]
        threshold_analysis = result["threshold_analysis"]
        detection_rate = result["detection_rate"]
        
        if use_beta:
            if threshold_analysis["adaptive_thresholds_used"]:
                avg_threshold = threshold_analysis["average_adaptive_threshold"]
                print(f"   ✅ Beta Integration: Working (avg threshold: {avg_threshold:.3f})")
                
                if avg_threshold > 0.8:
                    print(f"   ✅ Threshold Level: Appropriate for FL ({avg_threshold:.3f})")
                else:
                    print(f"   ⚠️ Threshold Level: Lower than expected ({avg_threshold:.3f})")
                    issues_found.append(f"Low beta threshold in {scenario_name}")
            else:
                print(f"   ❌ Beta Integration: Not working (fallback to fixed)")
                integration_working = False
                issues_found.append(f"Beta threshold not used in {scenario_name}")
        else:
            if not threshold_analysis["adaptive_thresholds_used"]:
                print(f"   ✅ Fixed Threshold: Correctly using fallback")
            else:
                print(f"   ⚠️ Unexpected: Using adaptive when beta disabled")
        
        # Check detection for attack scenarios
        if result["attack_config"] is not None:
            if detection_rate > 0.1:
                print(f"   ✅ Attack Detection: {detection_rate:.1%} (Improved!)")
            else:
                print(f"   ❌ Attack Detection: {detection_rate:.1%} (Still not working)")
                issues_found.append(f"Still no detection in {scenario_name}")
        
        print(f"   📈 Detection Rate: {detection_rate:.1%}")
        print(f"   ❌ Excluded Nodes: {result['total_excluded']}")
    
    # Overall assessment
    print(f"\n" + "=" * 60)
    print("🔍 INTEGRATION ASSESSMENT")
    print("=" * 60)
    
    if integration_working and not issues_found:
        print("✅ **INTEGRATION SUCCESSFUL**")
        print("   - Beta thresholds are being passed to HSIC")
        print("   - Adaptive thresholds are being used")
        print("   - Fallback to fixed thresholds works")
        print("   - Ready for full testing!")
    else:
        print("⚠️ **INTEGRATION ISSUES FOUND**")
        if issues_found:
            for issue in issues_found:
                print(f"   • {issue}")
    
    # Recommendations
    print(f"\n🔧 NEXT STEPS")
    if integration_working:
        print("1. Run full trust evaluation with beta thresholding enabled")
        print("2. Test with different attack intensities")
        print("3. Verify detection rates improve significantly")
        print("4. Fine-tune beta threshold percentiles if needed")
    else:
        print("1. Debug why beta thresholds aren't being passed to HSIC")
        print("2. Check trust monitor configuration")
        print("3. Verify adaptive trust system is enabled")
        print("4. Review integration code paths")
    
    return {
        "integration_working": integration_working,
        "issues_found": issues_found,
        "results": results
    }


def main():
    """Run beta threshold integration test."""
    
    # Setup logging to capture details
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    start_time = time.time()
    
    # Run integration tests
    results = test_beta_integration()
    
    # Generate report
    assessment = generate_integration_report(results)
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
    
    # Save results
    os.makedirs("beta_integration_results", exist_ok=True)
    
    with open("beta_integration_results/integration_test.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": total_time,
            "assessment": assessment
        }, f, indent=2, default=str)
    
    print("💾 Results saved to beta_integration_results/")
    
    # Return status for scripting
    return 0 if assessment["integration_working"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)