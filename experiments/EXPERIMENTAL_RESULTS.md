# Experimental Results: Evidential Trust-Aware Model Personalization

## Summary

This document summarizes the experimental evaluation of the Evidential Trust aggregation method for decentralized federated learning on the UCI HAR dataset.

## Experiment Configuration

- **Dataset**: UCI HAR (Human Activity Recognition)
- **Nodes**: 10
- **Topology**: Fully Connected
- **Training Rounds**: 50
- **Local Epochs**: 2
- **Batch Size**: 32

## Experiment 1: Baseline Performance (Moderate Non-IID, alpha=0.5)

### Without Attack

| Method | Final Accuracy | Std Dev |
|--------|---------------|---------|
| FedAvg | 96.98% | 2.39% |
| Evidential Trust | 83.86% | 10.02% |

**Observation**: With moderate heterogeneity and no attacks, FedAvg performs better because it fully leverages all peer information. Evidential Trust's conservative filtering is not needed in this benign scenario.

### With Byzantine Attack (20% Gaussian, std=10.0)

| Method | Final Accuracy | Std Dev |
|--------|---------------|---------|
| FedAvg | 19.19% | 7.54% |
| Krum | 64.97% | 22.86% |
| Evidential Trust | 5.68% | 2.99% |

**Observation**: The strong Gaussian noise attack (std=10.0) causes "evidence explosion" in evidential models, making Byzantine nodes appear highly confident (low vacuity). This defeats the uncertainty-based trust mechanism. Krum's geometric filtering is more effective against this specific attack type.

**Recommendation**: Evidential Trust may need additional defenses against attacks that artificially inflate evidence, such as evidence magnitude capping or combining with geometric filtering.

## Experiment 2: Personalization (Extreme Non-IID, alpha=0.1)

### Key Results

| Metric | FedAvg | Evidential Trust | Improvement |
|--------|--------|------------------|-------------|
| Final Accuracy | 85.29% | 92.50% | **+7.21%** |
| Final Std Dev | 19.59% | 14.34% | **-5.25%** |
| Rounds to 80% Accuracy | ~32 | ~3 | **~10x faster** |
| Peak Accuracy | 85.29% (R50) | 95.42% (R45) | **+10.13%** |

### Convergence Analysis

**FedAvg (alpha=0.1)**:
- Slow convergence due to conflicting gradients from heterogeneous nodes
- High variance throughout training (0.26-0.38 in early rounds)
- Final variance still high (0.19) indicating inconsistent personalization

**Evidential Trust (alpha=0.1)**:
- Rapid convergence: 80% accuracy by round 3, 88% by round 5
- Variance decreases steadily (0.38 → 0.14)
- More consistent performance across nodes
- Occasional peaks to 95%+ accuracy

### Key Finding: Personalization Effectiveness

Evidential Trust successfully enables **personalization** in extreme non-IID settings:

1. **Higher Trust for Similar Peers**: Nodes with similar class distributions achieve higher mutual trust through cross-evaluation
2. **Self-Weight for Personalization**: The `self_weight=0.6` parameter allows nodes to retain more local knowledge
3. **Faster Adaptation**: By filtering unhelpful peers early, nodes converge faster to their local optima
4. **Lower Variance**: Consistent high accuracy across nodes indicates effective personalization

## Uncertainty Metrics Evolution

### Vacuity (Epistemic Uncertainty)
- FedAvg: 0.588 → 0.548 (6.8% decrease)
- Evidential Trust: 0.588 → 0.554 (5.8% decrease)

### Dirichlet Strength (Evidence)
- FedAvg: 10.20 → 11.08 (8.6% increase)
- Evidential Trust: 10.20 → 10.91 (7.0% increase)

**Observation**: Evidence accumulation is modest in both cases. The EDL loss function may benefit from modifications to encourage stronger evidence accumulation (see recommendations below).

## Identified Issues and Recommendations

### Issue 1: Low Evidence Accumulation
**Problem**: Dirichlet strength only increases from ~10 to ~11 over 50 rounds, indicating limited evidence accumulation.

**Cause**: The MSE-based evidential loss doesn't directly incentivize high evidence for correct predictions. The original EDL paper includes a variance term that encourages higher strength.

**Recommendation**: Add variance regularization term to the loss:
```python
# In EvidentialLoss.forward():
var_loss = (probs * (1 - probs) / (S + 1)).sum(dim=-1).mean()
loss = mse_loss + lambda_t * kl_loss - beta * var_loss
```

### Issue 2: Vulnerability to Evidence Explosion Attacks
**Problem**: High-noise attacks create models with artificially high evidence, appearing falsely confident.

**Recommendation**: Add evidence magnitude capping or combine uncertainty-based trust with distance-based filtering:
```python
# Cap evidence per class to prevent explosion
evidence = torch.clamp(F.softplus(logits), max=100.0)
```

### Issue 3: Learning Rate Sensitivity
**Problem**: Evidential models trained with lower learning rate (0.001 vs 0.01) for stability, causing slower convergence.

**Recommendation**: Use adaptive learning rate scheduling or gradient clipping instead of fixed low learning rate.

## Conclusions

1. **Personalization Success**: Evidential Trust significantly outperforms FedAvg under extreme non-IID conditions (+7% accuracy, -5% variance, 10x faster convergence)

2. **Trust-Based Peer Selection**: The uncertainty-aware trust mechanism effectively identifies helpful peers in heterogeneous settings

3. **Limitations**: The current implementation is vulnerable to attacks that manipulate evidence magnitude. Additional defenses are recommended.

4. **Evidence Accumulation**: The EDL training objective could be enhanced to encourage stronger evidence accumulation for improved uncertainty estimates.

## Experiment Configurations

All experiment configurations are available in `experiments/configs/`:
- `exp1_baseline_*.yaml`: Baseline comparisons
- `exp2_attack*_*.yaml`: Byzantine attack scenarios
- `exp3_heterog_*.yaml`: Heterogeneity studies
- `exp4_personalization_*.yaml`: Personalization experiments

## Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Run individual experiment
murmura run experiments/configs/exp4_personalization_evidential.yaml

# Run all experiments
python experiments/scripts/run_all_experiments.py --category personalization
```
