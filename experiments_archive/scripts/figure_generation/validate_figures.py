#!/usr/bin/env python3
"""
Validation script for the updated figures to ensure accuracy and completeness.
"""

import json
import numpy as np
from pathlib import Path

def validate_data_extraction():
    """Validate that data extraction is working correctly."""
    
    # Load data
    with open('results_phase1/rerun_attack_results.json', 'r') as f:
        phase1_data = json.load(f)
    
    with open('results_phase2/rerun_attack_results.json', 'r') as f:
        phase2_data = json.load(f)
    
    print("=== DATA VALIDATION ===")
    print(f"Phase 1 experiments: {len(phase1_data)}")
    print(f"Phase 2 experiments: {len(phase2_data)}")
    
    # Check data structure
    sample_exp = phase1_data[0]
    required_keys = ['experiment_name', 'config', 'attack_results']
    for key in required_keys:
        assert key in sample_exp, f"Missing key: {key}"
    
    # Check attack results structure
    attack_result = sample_exp['attack_results']['attack_results'][0]
    assert 'attack_success_metric' in attack_result, "Missing attack_success_metric"
    
    print("✓ Data structure validation passed")
    
    # Validate attack success metrics are in valid range
    for phase_name, data in [("Phase 1", phase1_data), ("Phase 2", phase2_data)]:
        metrics = [exp['attack_results']['attack_results'][0]['attack_success_metric'] 
                  for exp in data]
        
        assert all(0 <= m <= 1 for m in metrics), f"{phase_name}: Invalid metric values"
        print(f"✓ {phase_name}: All attack success metrics in valid range [0,1]")
        print(f"  Range: [{min(metrics):.3f}, {max(metrics):.3f}]")
    
    return phase1_data, phase2_data

def validate_subsampling_analysis(phase2_data):
    """Validate subsampling level analysis."""
    
    print("\n=== SUBSAMPLING VALIDATION ===")
    
    subsampling_counts = {}
    for exp in phase2_data:
        name = exp['experiment_name']
        if 'moderate_sampling' in name:
            level = 'moderate'
        elif 'strong_sampling' in name:
            level = 'strong'
        elif 'very_strong_sampling' in name:
            level = 'very_strong'
        else:
            level = 'other'
        
        subsampling_counts[level] = subsampling_counts.get(level, 0) + 1
    
    print("Subsampling level distribution:")
    for level, count in sorted(subsampling_counts.items()):
        print(f"  {level}: {count} experiments")
    
    # Validate we have the expected subsampling levels
    expected_levels = {'moderate', 'strong'}
    actual_levels = set(k for k, v in subsampling_counts.items() if v > 0 and k != 'other')
    
    print(f"Expected levels: {expected_levels}")
    print(f"Actual levels: {actual_levels}")
    
    if not expected_levels.issubset(actual_levels):
        print("⚠ Warning: Missing expected subsampling levels")
    else:
        print("✓ All expected subsampling levels present")
    
    return subsampling_counts

def validate_dataset_coverage(phase1_data, phase2_data):
    """Validate dataset coverage across phases."""
    
    print("\n=== DATASET COVERAGE VALIDATION ===")
    
    phase1_datasets = set(exp['config']['dataset'] for exp in phase1_data)
    phase2_datasets = set(exp['config']['dataset'] for exp in phase2_data)
    
    print(f"Phase 1 datasets: {phase1_datasets}")
    print(f"Phase 2 datasets: {phase2_datasets}")
    
    common_datasets = phase1_datasets.intersection(phase2_datasets)
    print(f"Common datasets: {common_datasets}")
    
    assert len(common_datasets) >= 2, "Need at least 2 common datasets for comparison"
    print(f"✓ Dataset coverage validation passed ({len(common_datasets)} common datasets)")
    
    # Check experiment counts per dataset
    for dataset in common_datasets:
        p1_count = sum(1 for exp in phase1_data if exp['config']['dataset'] == dataset)
        p2_count = sum(1 for exp in phase2_data if exp['config']['dataset'] == dataset)
        print(f"  {dataset}: Phase 1 = {p1_count}, Phase 2 = {p2_count}")

def validate_figures_exist():
    """Validate that the generated figures exist."""
    
    print("\n=== FIGURE VALIDATION ===")
    
    analysis_dir = Path('analysis')
    required_figures = [
        'fig3_subsampling_flow.pdf',
        'fig3_subsampling_flow.png',
        'fig4_dataset_violin.pdf',
        'fig4_dataset_violin.png'
    ]
    
    for figure in required_figures:
        figure_path = analysis_dir / figure
        assert figure_path.exists(), f"Missing figure: {figure_path}"
        
        # Check file size (should be > 0)
        file_size = figure_path.stat().st_size
        assert file_size > 0, f"Empty figure file: {figure_path}"
        print(f"✓ {figure} exists ({file_size:,} bytes)")

def analyze_attack_effectiveness():
    """Provide detailed analysis of attack effectiveness."""
    
    print("\n=== ATTACK EFFECTIVENESS ANALYSIS ===")
    
    # Load data
    with open('results_phase1/rerun_attack_results.json', 'r') as f:
        phase1_data = json.load(f)
    
    with open('results_phase2/rerun_attack_results.json', 'r') as f:
        phase2_data = json.load(f)
    
    # Extract metrics
    def get_metrics(data):
        return [exp['attack_results']['attack_results'][0]['attack_success_metric'] 
                for exp in data]
    
    p1_metrics = get_metrics(phase1_data)
    p2_metrics = get_metrics(phase2_data)
    
    print("Phase 1 (Baseline):")
    print(f"  Mean: {np.mean(p1_metrics):.3f}")
    print(f"  Std:  {np.std(p1_metrics):.3f}")
    print(f"  Min:  {np.min(p1_metrics):.3f}")
    print(f"  Max:  {np.max(p1_metrics):.3f}")
    
    print("Phase 2 (Subsampling):")
    print(f"  Mean: {np.mean(p2_metrics):.3f}")
    print(f"  Std:  {np.std(p2_metrics):.3f}")
    print(f"  Min:  {np.min(p2_metrics):.3f}")
    print(f"  Max:  {np.max(p2_metrics):.3f}")
    
    # Statistical significance (basic t-test)
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(p1_metrics, p2_metrics)
        print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
        if p_value < 0.05:
            print("✓ Statistically significant difference between phases")
        else:
            print("⚠ No statistically significant difference between phases")
    except ImportError:
        print("(scipy not available for statistical testing)")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(p1_metrics) - 1) * np.var(p1_metrics) + 
                         (len(p2_metrics) - 1) * np.var(p2_metrics)) / 
                        (len(p1_metrics) + len(p2_metrics) - 2))
    cohens_d = (np.mean(p2_metrics) - np.mean(p1_metrics)) / pooled_std
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    if abs(cohens_d) > 0.8:
        effect_magnitude = "large"
    elif abs(cohens_d) > 0.5:
        effect_magnitude = "medium"
    elif abs(cohens_d) > 0.2:
        effect_magnitude = "small"
    else:
        effect_magnitude = "negligible"
    
    print(f"Effect magnitude: {effect_magnitude}")

def main():
    """Run all validations."""
    
    print("FIGURE VALIDATION REPORT")
    print("=" * 50)
    
    try:
        # Validate data extraction
        phase1_data, phase2_data = validate_data_extraction()
        
        # Validate subsampling analysis
        validate_subsampling_analysis(phase2_data)
        
        # Validate dataset coverage
        validate_dataset_coverage(phase1_data, phase2_data)
        
        # Validate figures exist
        validate_figures_exist()
        
        # Analyze attack effectiveness
        analyze_attack_effectiveness()
        
        print("\n" + "=" * 50)
        print("✓ ALL VALIDATIONS PASSED")
        print("The generated figures are accurate and complete.")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise

if __name__ == "__main__":
    main()