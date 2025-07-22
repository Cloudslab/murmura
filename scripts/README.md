# Murmura Analysis and Testing Scripts

This directory contains scripts for running experiments and analyzing results for the EdgeDrift trust monitoring system.

## Experiment Scripts

### `test_enhanced_trust_evaluation.sh`
**Main experimental evaluation script for EdgeDrift paper**

- Tests trust monitoring across multiple node counts (10, 20, 30, 50)
- Compares trust-weighted aggregation vs baseline gossip averaging
- Covers MNIST and CIFAR-10 datasets
- Tests gradient manipulation and label flipping attacks
- Generates comprehensive results with baseline comparisons

**Usage:**
```bash
./test_enhanced_trust_evaluation.sh
```

**Output:** Creates timestamped directory with experimental results and CSV summary.

### `test_cusum_detection.sh` 
**Legacy CUSUM detection script**

- Basic trust monitoring evaluation (10 nodes only)
- Does not include baseline experiments
- Kept for backward compatibility

**Usage:**
```bash
./test_cusum_detection.sh
```

## Analysis Scripts

### `analyze_enhanced_results.py`
**Comprehensive analysis script for enhanced experimental results**

Analyzes results from `test_enhanced_trust_evaluation.sh` including:
- Trust vs baseline accuracy comparisons
- Detection performance metrics (precision, recall, F1)
- Scalability analysis across node counts
- Visualization generation for paper figures

**Usage:**
```bash
python analyze_enhanced_results.py RESULTS_DIR --create-plots
```

**Features:**
- Generates accuracy improvement plots
- Creates detection performance heatmaps
- Exports detailed CSV analysis
- Produces summary reports

### `analyze_existing_cusum_results.py`
**Analysis for legacy CUSUM results**

Analyzes results from original CUSUM detection experiments. Helps understand why accuracy improvements were minimal in earlier experiments.

**Usage:**
```bash
python analyze_existing_cusum_results.py
```

## Workflow

1. **Run Experiments:**
   ```bash
   ./test_enhanced_trust_evaluation.sh
   ```

2. **Analyze Results:**
   ```bash
   python analyze_enhanced_results.py enhanced_trust_results_TIMESTAMP --create-plots
   ```

3. **Review Output:**
   - Check generated plots in `analysis_plots/` directory
   - Review `analysis_report.txt` for summary
   - Use `detailed_analysis.csv` for further processing

## Output Structure

```
enhanced_trust_results_TIMESTAMP/
├── experiment_summary.csv          # Quick experiment overview
├── detailed_analysis.csv          # Comprehensive metrics
├── analysis_report.txt            # Human-readable summary
├── analysis_plots/               # Visualization outputs
│   ├── trust_vs_baseline_accuracy.png
│   ├── detection_recall_vs_nodes.png
│   ├── accuracy_improvement_vs_nodes.png
│   └── detection_f1_heatmap.png
└── *.txt                         # Individual experiment outputs
```

## Key Metrics

- **Accuracy Improvement:** Trust-weighted vs baseline performance
- **Detection Performance:** Precision, recall, F1-score for malicious node detection  
- **Scalability:** Performance across different network sizes
- **Attack Effectiveness:** Relative performance on different attack types