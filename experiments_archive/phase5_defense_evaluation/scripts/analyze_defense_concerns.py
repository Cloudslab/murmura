#!/usr/bin/env python3
"""
Analysis script to address critical concerns about defense evaluation:

1. Strong DP alone reduced attack success by 18.8%. How are our defense approaches any better?
2. How is combined strategy performing worse than just structural noise?

This script provides honest analysis without data manipulation.
"""

import json
import numpy as np

def load_baseline_attack_results():
    """Load baseline attack results to understand DP effectiveness."""
    baseline_file = "/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase1_baseline_analysis/results_phase1/rerun_attack_results.json"
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    return baseline_data

def load_defense_evaluation_results():
    """Load defense evaluation results."""
    defense_file = "/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/experiments_archive/phase5_defense_evaluation/defense_evaluation/comprehensive_phase1_results/comprehensive_phase1_evaluation_results.json"
    
    with open(defense_file, 'r') as f:
        defense_data = json.load(f)
    
    return defense_data

def analyze_dp_baseline_effectiveness():
    """Analyze the effectiveness of different DP levels from baseline experiments."""
    
    baseline_data = load_baseline_attack_results()
    
    dp_analysis = {
        'no_dp': {'experiments': [], 'attack_success': []},
        'weak_dp': {'experiments': [], 'attack_success': []},
        'medium_dp': {'experiments': [], 'attack_success': []},
        'strong_dp': {'experiments': [], 'attack_success': []},
        'very_strong_dp': {'experiments': [], 'attack_success': []}
    }
    
    for experiment in baseline_data:
        dp_setting = experiment['config']['dp_setting']['name']
        exp_name = experiment['experiment_name']
        
        if dp_setting in dp_analysis:
            dp_analysis[dp_setting]['experiments'].append(exp_name)
            
            # Calculate average attack success across all attacks
            attack_results = experiment['attack_results']['attack_results']
            success_metrics = [attack['attack_success_metric'] for attack in attack_results]
            avg_success = np.mean(success_metrics)
            
            dp_analysis[dp_setting]['attack_success'].append(avg_success)
    
    # Calculate statistics
    dp_stats = {}
    for dp_level in dp_analysis:
        if dp_analysis[dp_level]['attack_success']:
            success_rates = dp_analysis[dp_level]['attack_success']
            dp_stats[dp_level] = {
                'mean_attack_success': np.mean(success_rates),
                'std_attack_success': np.std(success_rates),
                'count': len(success_rates),
                'attack_reduction_vs_no_dp': 0.0  # Will calculate below
            }
    
    # Calculate attack reduction relative to no_dp baseline
    no_dp_baseline = dp_stats['no_dp']['mean_attack_success']
    
    for dp_level in dp_stats:
        if dp_level != 'no_dp':
            current_success = dp_stats[dp_level]['mean_attack_success']
            reduction = ((no_dp_baseline - current_success) / no_dp_baseline) * 100
            dp_stats[dp_level]['attack_reduction_vs_no_dp'] = reduction
    
    return dp_stats

def analyze_defense_vs_dp_comparison():
    """Compare our defense mechanisms against baseline DP effectiveness."""
    
    # Get baseline DP effectiveness
    dp_stats = analyze_dp_baseline_effectiveness()
    
    # Get our defense results
    defense_data = load_defense_evaluation_results()
    defense_effectiveness = defense_data['summary_stats']['defense_effectiveness']
    
    print("="*80)
    print("CRITICAL ANALYSIS: DEFENSE MECHANISMS vs BASELINE DP")
    print("="*80)
    
    print("\n1. BASELINE DP EFFECTIVENESS (from Phase 1 results):")
    print("   Attack Success Rates and Reductions vs No-DP:")
    
    for dp_level, stats in dp_stats.items():
        success_rate = stats['mean_attack_success'] * 100
        reduction = stats.get('attack_reduction_vs_no_dp', 0)
        count = stats['count']
        
        if dp_level == 'no_dp':
            print(f"   {dp_level:15}: {success_rate:5.1f}% attack success (baseline)")
        else:
            print(f"   {dp_level:15}: {success_rate:5.1f}% attack success -> {reduction:5.1f}% reduction (n={count})")
    
    print("\n2. OUR DEFENSE MECHANISMS EFFECTIVENESS:")
    print("   Average Attack Reduction:")
    
    defense_ranking = [
        ('structural_noise_strong', defense_effectiveness['structural_noise_strong']['mean']),
        ('structural_noise_medium', defense_effectiveness['structural_noise_medium']['mean']),
        ('structural_noise_weak', defense_effectiveness['structural_noise_weak']['mean']),
        ('combined_medium', defense_effectiveness['combined_medium']['mean']),
        ('combined_strong', defense_effectiveness['combined_strong']['mean']),
        ('combined_weak', defense_effectiveness['combined_weak']['mean']),
        ('topology_aware_dp_weak', defense_effectiveness['topology_aware_dp_weak']['mean']),
        ('topology_aware_dp_strong', defense_effectiveness['topology_aware_dp_strong']['mean']),
        ('topology_aware_dp_medium', defense_effectiveness['topology_aware_dp_medium']['mean']),
    ]
    
    for defense_name, effectiveness in defense_ranking:
        print(f"   {defense_name:25}: {effectiveness:5.1f}% attack reduction")
    
    print("\n3. CRITICAL COMPARISON:")
    strong_dp_reduction = dp_stats['strong_dp']['attack_reduction_vs_no_dp']
    best_defense_reduction = max([eff for _, eff in defense_ranking])
    
    print(f"   Baseline Strong DP:        {strong_dp_reduction:5.1f}% attack reduction")
    print(f"   Best Defense Mechanism:    {best_defense_reduction:5.1f}% attack reduction")
    print(f"   Difference:                {best_defense_reduction - strong_dp_reduction:+5.1f}% points")
    
    if strong_dp_reduction > best_defense_reduction:
        print(f"\n   ⚠️  CONCERN VALIDATED: Strong DP outperforms our best defense by {strong_dp_reduction - best_defense_reduction:.1f}% points")
    else:
        print(f"\n   ✅ Our defenses outperform strong DP by {best_defense_reduction - strong_dp_reduction:.1f}% points")
    
    return dp_stats, defense_effectiveness

def analyze_combined_vs_structural_concern():
    """Analyze why combined strategies might perform worse than pure structural noise."""
    
    defense_data = load_defense_evaluation_results()
    detailed_results = defense_data.get('detailed_results', [])
    
    print("\n" + "="*80)
    print("ANALYSIS: WHY COMBINED STRATEGIES UNDERPERFORM STRUCTURAL NOISE")
    print("="*80)
    
    # Analyze attack-specific performance
    attack_reductions = defense_data['summary_stats']['attack_reductions']
    
    structural_strong = attack_reductions['structural_noise_strong']
    combined_medium = attack_reductions['combined_medium']
    combined_strong = attack_reductions['combined_strong']
    
    print("\n1. ATTACK-SPECIFIC PERFORMANCE BREAKDOWN:")
    
    attacks = ['Communication Pattern Attack', 'Parameter Magnitude Attack', 'Topology Structure Attack']
    
    for attack in attacks:
        print(f"\n   {attack}:")
        struct_perf = structural_strong[attack]['mean']
        comb_med_perf = combined_medium[attack]['mean']
        comb_str_perf = combined_strong[attack]['mean']
        
        print(f"     Structural Noise (Strong): {struct_perf:5.1f}%")
        print(f"     Combined (Medium):         {comb_med_perf:5.1f}%")
        print(f"     Combined (Strong):         {comb_str_perf:5.1f}%")
        print(f"     Loss vs Structural:        {struct_perf - comb_med_perf:+5.1f}% (medium), {struct_perf - comb_str_perf:+5.1f}% (strong)")
    
    print("\n2. POTENTIAL EXPLANATIONS FOR UNDERPERFORMANCE:")
    
    explanations = [
        "Defense Interference: Multiple mechanisms may interfere with each other",
        "Over-perturbation: Adding DP on top of structural noise may cause excessive noise",
        "Diminishing Returns: Benefits don't stack additively",
        "Implementation Issues: Combined implementation may have bugs or suboptimal integration",
        "Evaluation Methodology: Our comparison may not be fair or comprehensive"
    ]
    
    for i, explanation in enumerate(explanations, 1):
        print(f"   {i}. {explanation}")
    
    return True

def investigate_methodological_issues():
    """Investigate potential methodological problems in our evaluation."""
    
    print("\n" + "="*80)
    print("METHODOLOGICAL INVESTIGATION")
    print("="*80)
    
    print("\n1. POTENTIAL ISSUES WITH OUR EVALUATION:")
    
    issues = [
        {
            "issue": "Different Baseline Comparisons",
            "description": "Our defenses are compared against no-defense baseline, while DP comparison uses no-DP baseline",
            "impact": "May not be directly comparable"
        },
        {
            "issue": "Attack Implementation Differences", 
            "description": "Attacks in defense evaluation vs baseline phase1 may have different implementations",
            "impact": "Could affect success rate measurements"
        },
        {
            "issue": "Experimental Conditions",
            "description": "Defense evaluation uses different FL training conditions than baseline experiments",
            "impact": "May affect attack effectiveness and defense measurements"
        },
        {
            "issue": "Defense Implementation Quality",
            "description": "Our defense implementations may not be optimal or may have bugs",
            "impact": "Could underestimate true defense potential"
        },
        {
            "issue": "Evaluation Metrics",
            "description": "How we calculate 'attack reduction' may not be equivalent to DP effectiveness measurement",
            "impact": "Apples-to-oranges comparison"
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"\n   {i}. {issue['issue']}:")
        print(f"      Description: {issue['description']}")
        print(f"      Impact: {issue['impact']}")
    
    print("\n2. RECOMMENDATIONS FOR HONEST EVALUATION:")
    
    recommendations = [
        "Re-run defense evaluation using exact same attack implementations as baseline",
        "Compare defenses against same experimental conditions as DP baseline",
        "Implement head-to-head comparison: DP vs our defenses on identical experiments",
        "Validate defense implementations for correctness and optimality",
        "Use consistent evaluation metrics across all comparisons",
        "Report both absolute effectiveness and relative comparisons honestly"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

def main():
    """Run comprehensive analysis of defense evaluation concerns."""
    
    print("DEFENSE EVALUATION CONCERNS ANALYSIS")
    print("====================================")
    print("Analyzing two critical concerns without data manipulation:")
    print("1. Strong DP alone reduced attack success by 18.8% - are our defenses better?") 
    print("2. Why is combined strategy performing worse than structural noise?")
    
    # Analyze DP baseline vs our defenses
    dp_stats, defense_effectiveness = analyze_defense_vs_dp_comparison()
    
    # Analyze combined vs structural noise
    analyze_combined_vs_structural_concern()
    
    # Investigate methodological issues
    investigate_methodological_issues()
    
    print("\n" + "="*80)
    print("HONEST CONCLUSIONS")
    print("="*80)
    
    strong_dp_reduction = dp_stats['strong_dp']['attack_reduction_vs_no_dp']
    best_defense = max(defense_effectiveness.values(), key=lambda x: x['mean'])['mean']
    
    print("\n1. BASELINE DP EFFECTIVENESS:")
    print(f"   Strong DP achieves {strong_dp_reduction:.1f}% attack reduction")
    print(f"   Our best defense achieves {best_defense:.1f}% attack reduction")
    
    if strong_dp_reduction > best_defense:
        print("   ⚠️  HONEST ASSESSMENT: Baseline DP outperforms our defenses")
        print("   This suggests our defense implementations need improvement")
    else:
        print("   ✅ Our defenses show improvement over baseline DP")
    
    print("\n2. COMBINED VS STRUCTURAL NOISE:")
    structural_mean = defense_effectiveness['structural_noise_strong']['mean']
    combined_mean = defense_effectiveness['combined_medium']['mean']
    print(f"   Structural Noise (Strong): {structural_mean:.1f}%")
    print(f"   Combined (Medium): {combined_mean:.1f}%")
    print(f"   Difference: {structural_mean - combined_mean:+.1f}% points")
    
    if structural_mean > combined_mean:
        print("   ⚠️  CONFIRMED: Combined approach underperforms pure structural noise")
        print("   This indicates defense interference or implementation issues")
    
    print("\n3. NEXT STEPS:")
    print("   - Validate defense implementations for correctness")
    print("   - Run head-to-head comparison with baseline DP")
    print("   - Investigate defense interference mechanisms")
    print("   - Consider methodology improvements")
    
    # Save analysis results
    output_file = "/Users/MRANGWALA/Documents/Projects/PhD-Projects/murmura/defense_concerns_analysis.json"
    
    analysis_results = {
        'baseline_dp_effectiveness': dp_stats,
        'defense_effectiveness': defense_effectiveness,
        'key_findings': {
            'strong_dp_reduction': strong_dp_reduction,
            'best_defense_reduction': best_defense,
            'dp_outperforms_defenses': strong_dp_reduction > best_defense,
            'combined_underperforms_structural': structural_mean > combined_mean,
            'methodology_concerns': [
                "Different baseline comparisons",
                "Potential attack implementation differences", 
                "Experimental condition variations",
                "Defense implementation quality questions",
                "Evaluation metric consistency issues"
            ]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()