# Updated Figure Generation for Topology Privacy Leakage Paper

This directory contains scripts and generated figures for the topology privacy leakage paper, specifically focusing on updated versions of Figure 3 and Figure 4 with proper y-axis scaling and comprehensive subsampling analysis.

## Generated Figures

### Figure 3: Comprehensive Subsampling Impact Assessment
- **File**: `analysis/fig3_subsampling_flow.pdf` (and `.png`)
- **Purpose**: Complete analysis of subsampling impact on topology attacks
- **Features**:
  - Four-panel comprehensive analysis
  - Progression line plot with confidence intervals
  - Bar chart comparison with error bars
  - Dataset-specific breakdown
  - Statistical summary table
  - Y-axis starts from 0 for accurate visual comparison

### Figure 4: Dataset Vulnerability Distributions
- **File**: `analysis/fig4_dataset_violin.pdf` (and `.png`)
- **Purpose**: Comparison of dataset vulnerability with proper y-axis scaling
- **Features**:
  - Violin plots showing attack success rate distributions
  - Comparison between MNIST and HAM10000 datasets
  - Phase separation (Baseline vs Subsampling)
  - Y-axis starts from 0 for accurate visual comparison
  - Statistical summaries included

## Scripts Created

### 1. `generate_updated_figures.py`
**Main figure generation script**
- Loads experimental data from Phase 1 (baseline) and Phase 2 (subsampling)
- Extracts attack success metrics and configuration details
- Generates both Figure 3 and Figure 4 with proper formatting
- Ensures y-axis starts from 0 for accurate visual comparison
- Creates both PDF and PNG versions at 300 DPI

### 2. `validate_figures.py`
**Comprehensive validation script**
- Validates data structure and integrity
- Checks subsampling level analysis
- Verifies dataset coverage across phases
- Confirms figure files exist and are non-empty
- Performs statistical analysis of attack effectiveness
- Includes t-tests and effect size calculations

### 3. `figure_generation_summary.py`
**Summary report generator**
- Provides comprehensive overview of generated figures
- Documents key findings and statistics
- Lists file locations and specifications
- Includes recommendations for paper usage
- Validates statistical significance

### 4. `README_figure_generation.md` (this file)
**Documentation and usage guide**

## Experimental Data

### Phase 1 (Baseline)
- **Experiments**: 520 total
- **Datasets**: MNIST (235), HAM10000 (285)
- **Configuration**: No subsampling, various DP settings and topologies
- **Attack Success Rate**: 0.841 ± 0.086

### Phase 2 (Subsampling)
- **Experiments**: 288 total
- **Datasets**: MNIST (144), HAM10000 (144)
- **Subsampling Levels**: Moderate (96), Strong (192)
- **Attack Success Rate**: 0.868 ± 0.060

## Key Findings

1. **Statistical Significance**: T-test shows significant difference between phases (p < 0.001)
2. **Effect Size**: Small but meaningful improvement with subsampling (Cohen's d = 0.345)
3. **Dataset Differences**: 
   - MNIST: 0.848 → 0.888 (+4.7% improvement)
   - HAM10000: 0.836 → 0.848 (+1.4% improvement)
4. **Subsampling Impact**: Both moderate and strong subsampling show similar effectiveness

## Usage Instructions

### To regenerate figures:
```bash
python generate_updated_figures.py
```

### To validate results:
```bash
python validate_figures.py
```

### To generate summary report:
```bash
python figure_generation_summary.py
```

## File Specifications

- **Format**: PDF (vector) and PNG (raster) versions
- **Resolution**: 300 DPI for publication quality
- **Color Scheme**: Publication-ready with clear contrast
- **Typography**: Consistent font sizes (10-16pt)
- **Y-axis**: Properly starts from 0 for accurate visual comparison

## Requirements

- Python 3.7+
- matplotlib
- seaborn
- pandas
- numpy
- scipy (for statistical tests)
- pathlib (built-in)

## Figure Quality Validation

✓ All attack success metrics in valid range [0, 1]  
✓ Y-axis starts from 0 for accurate comparison  
✓ Statistical significance validated (p < 0.001)  
✓ Publication-quality resolution (300 DPI)  
✓ Clear visual distinction between datasets and phases  
✓ Comprehensive data coverage (808 total experiments)  

## Publication Recommendations

1. **Figure 3** (`fig3_subsampling_flow.pdf`): Use in methodology/results section discussing subsampling effectiveness
2. **Figure 4** (`fig4_dataset_violin.pdf`): Use in dataset comparison section
3. Both figures have proper y-axis scaling starting from 0
4. PNG versions available for presentations
5. Statistical significance supports claims about subsampling effectiveness

## Data Sources

- `results_phase1/rerun_attack_results.json`: Baseline experimental results
- `results_phase2/rerun_attack_results.json`: Subsampling experimental results

Generated figures accurately represent 808 experiments across multiple attack strategies, network topologies, and privacy configurations.