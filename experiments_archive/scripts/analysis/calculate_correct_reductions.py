#!/usr/bin/env python3
"""
Calculate correct reduction percentages from actual paper baseline values.
"""
import json

def calculate_correct_reductions():
    """Calculate reduction percentages from actual paper baseline values."""
    
    # Actual paper baseline values (from extract_paper_baseline.py)
    paper_baseline = {
        "Communication Pattern Attack": 0.841,  # 84.1%
        "Parameter Magnitude Attack": 0.650,   # 65.0%
        "Topology Structure Attack": 0.472     # 47.2%
    }
    
    # Load realistic scenario results
    with open("experiments_archive/results/attack_results/realistic_knowledge_full_analysis/realistic_scenario_summary.json", 'r') as f:
        results = json.load(f)
    
    scenarios = ["Neighborhood 1-hop", "Neighborhood 2-hop", "Statistical Knowledge", 
                 "Organizational 3-groups", "Organizational 5-groups"]
    
    print("Realistic Scenario Performance vs Actual Paper Baseline")
    print("=" * 80)
    
    corrected_results = {}
    
    for scenario_name in scenarios:
        if scenario_name in results["scenario_effectiveness"]:
            scenario_results = results["scenario_effectiveness"][scenario_name]
            
            print(f"\n{scenario_name}:")
            print("```")
            print("Attack Vector                Success Rate    Baseline    Reduction    Status")
            
            corrected_results[scenario_name] = {}
            
            for attack_name in ["Communication Pattern Attack", "Parameter Magnitude Attack", "Topology Structure Attack"]:
                if attack_name in scenario_results:
                    success_rate = scenario_results[attack_name]["average_success"]
                    baseline = paper_baseline[attack_name]
                    
                    # Calculate reduction: (baseline - actual) / baseline * 100
                    reduction = ((baseline - success_rate) / baseline) * 100
                    status = "✓" if success_rate > 0.30 else "✗"
                    
                    attack_short = attack_name.replace(" Attack", "")
                    
                    print(f"{attack_short:24} {success_rate:.1%}          {baseline:.1%}      {reduction:+.1f}%        {status}")
                    
                    corrected_results[scenario_name][attack_name] = {
                        "success_rate": success_rate,
                        "baseline": baseline,
                        "reduction_percent": reduction,
                        "effective": success_rate > 0.30
                    }
            
            print("```")
    
    # Summary analysis
    print("\n\nSUMMARY ANALYSIS")
    print("=" * 50)
    
    effective_scenarios = 0
    total_scenarios = len(scenarios)
    
    for scenario_name in scenarios:
        if scenario_name in corrected_results:
            effective_attacks = sum(1 for data in corrected_results[scenario_name].values() if data["effective"])
            total_attacks = len(corrected_results[scenario_name])
            
            if effective_attacks == total_attacks:
                effective_scenarios += 1
                print(f"✓ {scenario_name}: All attacks remain effective")
            else:
                failed_attacks = [attack for attack, data in corrected_results[scenario_name].items() if not data["effective"]]
                print(f"⚠ {scenario_name}: {len(failed_attacks)} attack(s) below threshold: {[a.replace(' Attack', '') for a in failed_attacks]}")
    
    print(f"\nRobustness: {effective_scenarios}/{total_scenarios} scenarios maintain full attack effectiveness")
    
    # Save corrected results
    with open("experiments_archive/results/attack_results/realistic_knowledge_full_analysis/corrected_reductions.json", 'w') as f:
        json.dump(corrected_results, f, indent=2)
    
    return corrected_results

if __name__ == "__main__":
    calculate_correct_reductions()