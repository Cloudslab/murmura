#!/usr/bin/env python3
"""
User-friendly trust system testing script.

Usage examples:
    # Test honest baseline
    python scripts/test_trust_system.py --scenario baseline

    # Test light attack
    python scripts/test_trust_system.py --scenario attack --intensity low

    # Test with custom parameters
    python scripts/test_trust_system.py --scenario attack --intensity moderate --attackers 2 --rounds 10

    # Compare all scenarios
    python scripts/test_trust_system.py --scenario compare
"""

import argparse
import sys
import time
from pathlib import Path

# Add murmura to path
sys.path.append(str(Path(__file__).parent.parent))

from murmura.examples.adaptive_trust_mnist_example import run_adaptive_trust_mnist


def create_attack_config(intensity: str, num_attackers: int = 1, total_nodes: int = 6):
    """Create attack configuration based on parameters."""
    malicious_fraction = num_attackers / total_nodes
    
    intensity_configs = {
        'low': {'attack_intensity': 'low', 'stealth_level': 'high'},
        'moderate': {'attack_intensity': 'moderate', 'stealth_level': 'medium'},
        'high': {'attack_intensity': 'high', 'stealth_level': 'low'}
    }
    
    config = intensity_configs.get(intensity, intensity_configs['moderate'])
    
    return {
        'attack_type': 'gradual_label_flipping',
        'malicious_fraction': malicious_fraction,
        **config
    }


def test_baseline(rounds: int = 5, verbose: bool = False):
    """Test honest baseline scenario."""
    print("🔍 Testing Honest Baseline Scenario...")
    print(f"   Nodes: 6, Rounds: {rounds}, Expected exclusions: 0")
    print()
    
    log_level = 'INFO' if verbose else 'WARNING'
    start_time = time.time()
    
    try:
        result = run_adaptive_trust_mnist(
            num_rounds=rounds,
            attack_config=None,
            log_level=log_level
        )
        
        elapsed = time.time() - start_time
        attack_stats = result.get('attack_statistics', {})
        excluded = attack_stats.get('detected_attacks', 0)
        
        print("📊 BASELINE RESULTS:")
        print(f"   Total nodes: 6")
        print(f"   Excluded nodes: {excluded}")
        print(f"   False positive rate: {excluded/6*100:.1f}%")
        print(f"   Execution time: {elapsed:.1f}s")
        
        if excluded == 0:
            print("   ✅ SUCCESS: No false positives!")
        else:
            print("   ⚠️  WARNING: False positives detected!")
            
        return excluded == 0
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def test_attack(intensity: str, num_attackers: int = 1, rounds: int = 8, verbose: bool = False):
    """Test attack scenario."""
    print(f"⚔️  Testing {intensity.title()} Attack Scenario...")
    print(f"   Attackers: {num_attackers}/6, Rounds: {rounds}")
    print()
    
    attack_config = create_attack_config(intensity, num_attackers)
    log_level = 'INFO' if verbose else 'WARNING'
    start_time = time.time()
    
    try:
        result = run_adaptive_trust_mnist(
            num_rounds=rounds,
            attack_config=attack_config,
            log_level=log_level
        )
        
        elapsed = time.time() - start_time
        attack_stats = result.get('attack_statistics', {})
        total_attackers = attack_stats.get('total_attackers', 0)
        detected = attack_stats.get('detected_attacks', 0)
        detection_rate = attack_stats.get('detection_rate', 0)
        
        print("📊 ATTACK RESULTS:")
        print(f"   Total attackers: {total_attackers}")
        print(f"   Detected attackers: {detected}")
        print(f"   Detection rate: {detection_rate*100:.1f}%")
        print(f"   Execution time: {elapsed:.1f}s")
        
        if detection_rate >= 0.8:
            print("   ✅ EXCELLENT: High detection rate!")
        elif detection_rate >= 0.5:
            print("   ✅ GOOD: Attack detected!")
        elif detection_rate > 0:
            print("   ⚠️  PARTIAL: Some detection but not complete")
        else:
            print("   ❌ FAILED: Attack not detected!")
            
        return detection_rate > 0.5
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False


def compare_scenarios(rounds: int = 5, verbose: bool = False):
    """Compare all scenarios."""
    print("🔄 Comparing All Trust System Scenarios...")
    print("=" * 60)
    
    scenarios = [
        ("Honest Baseline", lambda: test_baseline(rounds, verbose)),
        ("Low Attack", lambda: test_attack('low', 1, rounds, verbose)),
        ("Moderate Attack", lambda: test_attack('moderate', 1, rounds, verbose)),
        ("High Attack", lambda: test_attack('high', 1, rounds, verbose)),
        ("Multiple Attackers", lambda: test_attack('moderate', 2, rounds, verbose))
    ]
    
    results = {}
    
    for name, test_func in scenarios:
        print(f"\n🧪 {name}")
        print("-" * 40)
        results[name] = test_func()
        print()
    
    print("=" * 60)
    print("📋 SUMMARY:")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    return passed_tests == total_tests


def analyze_beta_thresholds(scenario: str = 'baseline', rounds: int = 6):
    """Analyze how beta thresholds adapt."""
    print(f"📈 Analyzing Beta Threshold Adaptation ({scenario})...")
    print()
    
    attack_config = None
    if scenario != 'baseline':
        attack_config = create_attack_config(scenario, 1)
    
    try:
        result = run_adaptive_trust_mnist(
            num_rounds=rounds,
            attack_config=attack_config,
            log_level='WARNING'
        )
        
        trust_metrics = result.get('fl_results', {}).get('trust_metrics', [])
        
        if trust_metrics:
            print("🎯 Threshold Evolution:")
            for round_data in trust_metrics:
                round_num = round_data.get('round', 0)
                node_reports = round_data.get('node_trust_reports', {})
                
                if node_reports:
                    # Get threshold from first node
                    first_node = list(node_reports.values())[0]
                    threshold = first_node.get('adaptive_threshold', 'N/A')
                    print(f"   Round {round_num}: threshold = {threshold}")
        else:
            print("   No trust metrics available")
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="User-friendly trust system testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scenario baseline                    # Test honest scenario
  %(prog)s --scenario attack --intensity low      # Test low-intensity attack  
  %(prog)s --scenario attack --intensity moderate --attackers 2  # Multiple attackers
  %(prog)s --scenario compare                     # Compare all scenarios
  %(prog)s --scenario analyze --analysis-type beta  # Analyze beta thresholds
        """
    )
    
    parser.add_argument(
        '--scenario', 
        choices=['baseline', 'attack', 'compare', 'analyze'],
        default='baseline',
        help='Test scenario to run'
    )
    
    parser.add_argument(
        '--intensity',
        choices=['low', 'moderate', 'high'],
        default='moderate',
        help='Attack intensity (for attack scenario)'
    )
    
    parser.add_argument(
        '--attackers',
        type=int,
        default=1,
        help='Number of malicious nodes (1-5)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=5,
        help='Number of FL rounds to run'
    )
    
    parser.add_argument(
        '--analysis-type',
        choices=['beta', 'performance'],
        default='beta',
        help='Type of analysis to perform (for analyze scenario)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.attackers < 1 or args.attackers > 5:
        print("❌ Error: Number of attackers must be between 1 and 5")
        return 1
    
    if args.rounds < 1 or args.rounds > 50:
        print("❌ Error: Number of rounds must be between 1 and 50")
        return 1
    
    print("🛡️  TRUST SYSTEM TESTING")
    print("=" * 50)
    print()
    
    # Run the requested scenario
    success = True
    
    if args.scenario == 'baseline':
        success = test_baseline(args.rounds, args.verbose)
        
    elif args.scenario == 'attack':
        success = test_attack(args.intensity, args.attackers, args.rounds, args.verbose)
        
    elif args.scenario == 'compare':
        success = compare_scenarios(args.rounds, args.verbose)
        
    elif args.scenario == 'analyze':
        if args.analysis_type == 'beta':
            analyze_beta_thresholds(args.intensity, args.rounds)
        # Could add more analysis types here
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())