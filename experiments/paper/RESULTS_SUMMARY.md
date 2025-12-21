# Paper Experiment Results

Generated: 2025-12-20T09:18:29.246377


## UCI HAR

| Algorithm | Final Acc | Std Dev | Conv. Round | Peak Acc | Vacuity | Entropy |
|-----------|-----------|---------|-------------|----------|---------|---------|
| fedavg | 85.29% | 19.59% | 32 | 85.29% | 0.548 | 1.597 |
| krum | 46.83% | 38.41% | None | 47.37% | 0.552 | 1.701 |
| balance | 83.68% | 24.99% | 10 | 85.16% | 0.551 | 1.635 |
| ubar | 83.45% | 21.90% | 32 | 83.94% | 0.507 | 1.488 |
| sketchguard | 89.68% | 14.35% | 3 | 89.83% | 0.565 | 1.657 |
| evidential_trust | 92.50% | 14.34% | 3 | 95.42% | 0.554 | 1.635 |

## PAMAP2

| Algorithm | Final Acc | Std Dev | Conv. Round | Peak Acc | Vacuity | Entropy |
|-----------|-----------|---------|-------------|----------|---------|---------|
| fedavg | 90.23% | 3.26% | 34 | 90.23% | 0.635 | 2.222 |
| krum | 38.84% | 34.96% | None | 38.84% | 0.552 | 1.997 |
| balance | 85.92% | 30.33% | 21 | 85.92% | 0.574 | 2.050 |
| ubar | 88.65% | 14.70% | 11 | 91.52% | 0.510 | 1.888 |
| sketchguard | 96.99% | 3.52% | 7 | 97.43% | 0.565 | 2.029 |
| evidential_trust | 98.88% | 1.26% | 4 | 98.88% | 0.555 | 1.996 |

## PPG DALIA

| Algorithm | Final Acc | Std Dev | Conv. Round | Peak Acc | Vacuity | Entropy |
|-----------|-----------|---------|-------------|----------|---------|---------|
| fedavg | 66.46% | 7.53% | None | 66.46% | 0.578 | 1.762 |
| krum | 54.46% | 18.43% | None | 58.54% | 0.553 | 1.651 |
| balance | 72.24% | 9.25% | None | 72.24% | 0.576 | 1.760 |
| ubar | 75.09% | 6.11% | None | 75.09% | 0.549 | 1.720 |
| sketchguard | 69.26% | 12.93% | None | 69.26% | 0.577 | 1.765 |
| evidential_trust | 78.77% | 6.39% | None | 78.77% | 0.574 | 1.754 |


## LaTeX Table

```latex
\begin{table*}[t]
\centering
\caption{Model Personalization Performance Across Wearable Datasets (Non-IID, α=0.1)}
\label{tab:personalization_all}
\begin{tabular}{l|cccc|cccc|cccc}
\toprule
 & \multicolumn{4}{c|}{\textbf{UCI HAR}} & \multicolumn{4}{c|}{\textbf{PAMAP2}} & \multicolumn{4}{c}{\textbf{PPG-DaLiA}} \\
\textbf{Algorithm} & Acc & Std & Conv & Peak & Acc & Std & Conv & Peak & Acc & Std & Conv & Peak \\
\midrule
fedavg & 85.3 & 19.6 & 32 & 85.3 & 90.2 & 3.3 & 34 & 90.2 & 66.5 & 7.5 & None & 66.5 \\
krum & 46.8 & 38.4 & None & 47.4 & 38.8 & 35.0 & None & 38.8 & 54.5 & 18.4 & None & 58.5 \\
balance & 83.7 & 25.0 & 10 & 85.2 & 85.9 & 30.3 & 21 & 85.9 & 72.2 & 9.2 & None & 72.2 \\
ubar & 83.5 & 21.9 & 32 & 83.9 & 88.6 & 14.7 & 11 & 91.5 & 75.1 & 6.1 & None & 75.1 \\
sketchguard & 89.7 & 14.3 & 3 & 89.8 & 97.0 & 3.5 & 7 & 97.4 & 69.3 & 12.9 & None & 69.3 \\
evidential_trust & 92.5 & 14.3 & 3 & 95.4 & 98.9 & 1.3 & 4 & 98.9 & 78.8 & 6.4 & None & 78.8 \\
\bottomrule
\end{tabular}
\end{table*}
```