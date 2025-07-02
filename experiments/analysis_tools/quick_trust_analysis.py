#!/usr/bin/env python3
"""
Quick Trust Monitor Analysis - Focused on identifying key issues

This script provides rapid analysis of trust monitor performance with
minimal execution time to quickly identify problems and solutions.
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


def quick_trust_test():
    """Run a quick but comprehensive trust monitor test."""
    
    print("🔍 Quick Trust Monitor Analysis")
    print("Running focused tests to identify key performance issues...")
    print("=" * 60)
    
    # Test scenarios - minimal but comprehensive
    scenarios = [
        {
            "name": "honest_baseline",
            "description": "Honest nodes only - should have 0% false positives",
            "config": None,
            "expected_excluded": 0
        },
        {
            "name": "obvious_attack",
            "description": "High intensity attack - should be easily detected", 
            "config": {
                "attack_type": "gradual_label_flipping",
                "malicious_fraction": 0.33,  # 2 out of 6 nodes
                "attack_intensity": "high",
                "stealth_level": "low"
            },
            "expected_excluded": 2
        },
        {
            "name": "subtle_attack",
            "description": "Low intensity attack - test sensitivity",
            "config": {
                "attack_type": "gradual_label_flipping", 
                "malicious_fraction": 0.17,  # 1 out of 6 nodes
                "attack_intensity": "low",
                "stealth_level": "high"
            },
            "expected_excluded": 1
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📊 Test {i+1}/3: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        start_time = time.time()
        
        try:
            # Run quick test - reduced rounds for speed
            result = run_adaptive_trust_mnist(
                num_actors=6,
                num_rounds=8,  # Reduced from 15 for speed
                topology_type="ring",
                trust_profile="strict",  # Most sensitive profile
                use_beta_threshold=True,
                attack_config=scenario["config"],
                output_dir=f"quick_test_{scenario['name']}",
                log_level="WARNING"  # Reduce logging for speed
            )
            
            # Extract key metrics
            trust_analysis = result.get("trust_analysis", {})
            attack_stats = result.get("fl_results", {}).get("attack_statistics", {})
            
            excluded = trust_analysis.get("total_excluded", 0)
            downgraded = trust_analysis.get("total_downgraded", 0)
            total_attackers = attack_stats.get("total_attackers", 0)
            detected = attack_stats.get("detected_attacks", 0)
            detection_rate = attack_stats.get("detection_rate", 0.0)
            
            test_result = {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "expected_excluded": scenario["expected_excluded"],
                "actual_excluded": excluded,
                "total_attackers": total_attackers,
                "detected_attacks": detected,
                "detection_rate": detection_rate,
                "downgraded": downgraded,
                "time_taken": time.time() - start_time,
                "success": (excluded >= scenario["expected_excluded"] * 0.5)  # 50% threshold
            }
            
            results.append(test_result)
            
            # Quick feedback
            print(f"  ⏱️  Time: {test_result['time_taken']:.1f}s")
            print(f"  🎯 Expected/Actual Exclusions: {scenario['expected_excluded']}/{excluded}")
            print(f"  📈 Detection Rate: {detection_rate:.1%}")
            print(f"  ✅ Success: {'YES' if test_result['success'] else 'NO'}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {str(e)}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e),
                "success": False
            })
    
    return results


def analyze_quick_results(results):
    """Analyze results and provide immediate insights."""
    
    print("\n" + "=" * 60)
    print("🎯 QUICK ANALYSIS RESULTS")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r.get("success", False))
    total_tests = len(results)
    
    print(f"Overall Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})")
    
    # Analyze each scenario
    issues_found = []
    
    for result in results:
        if "error" in result:
            print(f"\n❌ {result['scenario']}: FAILED")
            print(f"   Error: {result['error']}")
            issues_found.append(f"Test execution failure in {result['scenario']}")
            continue
            
        scenario = result["scenario"]
        expected = result["expected_excluded"]
        actual = result["actual_excluded"]
        detection_rate = result.get("detection_rate", 0)
        
        print(f"\n📊 {scenario}:")
        print(f"   Expected Exclusions: {expected}")
        print(f"   Actual Exclusions: {actual}")
        print(f"   Detection Rate: {detection_rate:.1%}")
        print(f"   Status: {'✅ PASS' if result['success'] else '❌ FAIL'}")
        
        # Identify specific issues
        if scenario == "honest_baseline" and actual > 0:
            issues_found.append("FALSE POSITIVES: Honest nodes being flagged as malicious")
        
        elif scenario == "obvious_attack" and actual < expected:
            issues_found.append("LOW SENSITIVITY: Missing obvious high-intensity attacks")
            
        elif scenario == "subtle_attack" and detection_rate < 0.5:
            issues_found.append("POOR DISCRIMINATION: Cannot detect subtle attacks")
    
    # Provide immediate recommendations
    print("\n" + "=" * 60)
    print("🔧 IMMEDIATE RECOMMENDATIONS")
    print("=" * 60)
    
    if not issues_found:
        print("✅ Trust monitor is performing well!")
        print("   - Zero false positives")
        print("   - Good detection of attacks")
        print("   - Appropriate sensitivity levels")
    else:
        print("⚠️  Issues identified:")
        for issue in issues_found:
            print(f"   • {issue}")
        
        print("\n🛠️  Quick fixes to try:")
        
        if any("FALSE POSITIVES" in issue for issue in issues_found):
            print("   1. REDUCE FALSE POSITIVES:")
            print("      - Increase exclude_threshold from 0.5 to 0.7")
            print("      - Increase downgrade_threshold from 0.3 to 0.5")
            print("      - Use 'permissive' trust profile instead of 'strict'")
        
        if any("LOW SENSITIVITY" in issue for issue in issues_found):
            print("   2. INCREASE SENSITIVITY:")
            print("      - Decrease exclude_threshold from 0.5 to 0.3") 
            print("      - Decrease HSIC threshold from 0.1 to 0.05")
            print("      - Increase trust_report_interval frequency")
        
        if any("POOR DISCRIMINATION" in issue for issue in issues_found):
            print("   3. IMPROVE DISCRIMINATION:")
            print("      - Enable performance-based monitoring")
            print("      - Reduce validation data threshold")
            print("      - Combine HSIC + performance metrics")
    
    # Calculate overall trust monitor health
    if successful_tests == total_tests:
        health_score = "EXCELLENT"
        color = "🟢"
    elif successful_tests >= total_tests * 0.7:
        health_score = "GOOD"
        color = "🟡"
    else:
        health_score = "POOR"
        color = "🔴"
    
    print(f"\n{color} TRUST MONITOR HEALTH: {health_score}")
    
    return {
        "success_rate": successful_tests / total_tests,
        "issues_found": issues_found,
        "health_score": health_score,
        "results": results
    }


def main():
    """Run quick trust analysis."""
    
    start_time = time.time()
    
    # Run quick tests
    results = quick_trust_test()
    
    # Analyze results
    analysis = analyze_quick_results(results)
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total analysis time: {total_time:.1f} seconds")
    print("📊 Quick analysis complete!")
    
    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": total_time,
        "analysis": analysis
    }
    
    with open("quick_trust_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("💾 Summary saved to quick_trust_analysis_summary.json")


if __name__ == "__main__":
    main()