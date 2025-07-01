#!/usr/bin/env python3
"""
Quick trust system testing script for rapid iteration.

Usage:
    python scripts/quick_trust_test.py                    # Baseline test
    python scripts/quick_trust_test.py --attack moderate  # Attack test
    python scripts/quick_trust_test.py --all             # All scenarios
"""

import argparse
import sys
import time
from pathlib import Path

# Add murmura to path
sys.path.append(str(Path(__file__).parent.parent))

from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist


def quick_test(attack_intensity=None, rounds=3):
    """Run a quick test scenario."""
    
    if attack_intensity:
        attack_config = {
            'attack_type': 'gradual_label_flipping',
            'malicious_fraction': 0.167,  # 1 out of 6
            'attack_intensity': attack_intensity,
            'stealth_level': 'medium'
        }
        scenario_name = f"{attack_intensity.title()} Attack"
    else:
        attack_config = None
        scenario_name = "Honest Baseline"
    
    print(f"🔍 Testing: {scenario_name} ({rounds} rounds)")
    start_time = time.time()
    
    try:
        result = run_adaptive_trust_mnist(
            num_rounds=rounds,
            attack_config=attack_config,
            log_level='WARNING'
        )
        
        elapsed = time.time() - start_time
        # Attack statistics are nested in fl_results
        fl_results = result.get('fl_results', {})
        attack_stats = fl_results.get('attack_statistics', {})
        
        total_attackers = attack_stats.get('total_attackers', 0)
        detected = attack_stats.get('detected_attacks', 0) 
        detection_rate = attack_stats.get('detection_rate', 0)
        
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Attackers: {total_attackers}")
        print(f"   Detected: {detected}")
        print(f"   Rate: {detection_rate*100:.1f}%")
        
        if attack_config is None:
            # Baseline - should have 0 exclusions
            if detected == 0:
                print("   ✅ No false positives")
                return True
            else:
                print("   ⚠️  False positives detected")
                return False
        else:
            # Attack - should detect
            if detection_rate >= 0.5:
                print("   ✅ Attack detected")
                return True
            else:
                print("   ❌ Attack missed")
                return False
                
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick trust system testing")
    
    parser.add_argument(
        '--attack',
        choices=['low', 'moderate', 'high'],
        help='Test attack scenario with specified intensity'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all scenarios'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of rounds (default: 3 for speed)'
    )
    
    args = parser.parse_args()
    
    print("⚡ QUICK TRUST SYSTEM TEST")
    print("=" * 40)
    
    if args.all:
        scenarios = [None, 'low', 'moderate', 'high']
        results = []
        
        for scenario in scenarios:
            results.append(quick_test(scenario, args.rounds))
            print()
        
        passed = sum(results)
        total = len(results)
        print(f"Results: {passed}/{total} passed")
        
        return 0 if passed == total else 1
        
    else:
        success = quick_test(args.attack, args.rounds)
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())