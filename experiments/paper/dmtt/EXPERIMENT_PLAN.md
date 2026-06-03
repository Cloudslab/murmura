# DMTT Experimental Plan

Paper: *Trust-Aware Topology Learning for Dynamic Decentralized Federated Learning under Adversaries*  
Target venue: IEEE TNNLS / TMLR / top ML conference  
Framework: Murmura (distributed ZMQ backend + simulation backend)  
Hardware: single VM, 24 GB VRAM GPU

---

## Quick-reference table

| # | Name | Configs | Backend | Est. wall-clock | Status |
|---|------|---------|---------|-----------------|--------|
| 1 | Main comparison (fraction sweep) | 54 | sim + dist | ~20 h | **ready to run** |
| 2 | Multi-dataset validation (PAMAP2) | 6 | dist | ~3 h | configs TBD |
| 3 | Ablation study | 30 | dist | ~8 h | configs TBD |
| 4 | Trust-score evolution (qualitative) | 1 | dist | 25 min | requires trust logger |
| 5 | Sensitivity sweep (α, mobility) | ~36 | sim + dist | ~12 h | configs TBD |
| 6 | Communication overhead | 3 | dist | 1 h | requires msg-size logger |

All Experiment 1 configs are generated under `exp1_main_comparison/` and can be run
immediately after `generate_configs.py` is executed.

---

## Experiment 1 — Main comparison table

### Goal

Establish a single comprehensive table that shows, across three Byzantine fractions,
how each method performs under topology-liar + Gaussian model poisoning in a dynamic
network.  This is the primary empirical claim of the paper: DMTT maintains honest-node
accuracy under conditions that defeat both static-topology methods and dynamic methods
that lack trust-aware topology filtering.

### Conditions

| Label | Aggregation | Topology | Trust protocol | Backend |
|-------|-------------|----------|----------------|---------|
| C1 — Static FedAvg | FedAvg | Static (fully-conn.) | None | Simulation |
| C2 — Dynamic FedAvg | FedAvg | Dynamic (G^t) | None | Distributed |
| Dyn-Krum | Multi-Krum | Dynamic (G^t) | None | Distributed |
| Dyn-BALANCE | BALANCE | Dynamic (G^t) | None | Distributed |
| Dyn-UBAR | UBAR | Dynamic (G^t) | None | Distributed |
| **C3 — DMTT** | FedAvg + TopB | Dynamic (G^t) | Beta trust + TOPO_CLAIM | Distributed |

C1 is the "worst-case" reference: a method that ignores topology dynamics entirely.
C2 and Dyn-Krum/BALANCE/UBAR represent the best achievable with existing Byzantine-robust
aggregators under dynamic topology.  C3 is the proposed method.

### Attack

All conditions: `topology_liar` + `model_attack_type: gaussian` (noise_std = 10.0).
This is the richest attack in the paper's adversary model.
Topology-liar Byzantine nodes inject false TOPO_CLAIM messages *and* poisoned model updates.
Non-DMTT conditions are only affected by the model-poisoning component (NodeProcess does not
process TOPO_CLAIM), which means the comparison is conservative for DMTT.

### Sweep

- Byzantine fractions: **10 %, 20 %, 30 %** (1, 2, 3 nodes out of N=10)
- Random seeds: **42, 123, 777** (data partitioning, model initialization, mobility)
- Mobility seed = experiment seed across all conditions (same G^t for each seed triplet)

### Metrics (all at round 50)

- `honest_accuracy`: mean test accuracy over honest nodes — **primary metric**
- `mean_accuracy`: mean test accuracy over all nodes
- `convergence_round`: first round where honest_accuracy ≥ 0.80
  (use NaN if never reached — relevant for high-attack conditions)

Report mean ± std across the 3 seeds in the paper table.

### Expected table shape

```
Method              | 10 % Byz           | 20 % Byz           | 30 % Byz
                    | hon.    all        | hon.    all        | hon.    all
--------------------|--------------------|--------------------|-------------------
C1 Static FedAvg    | x.xx±y  x.xx±y    | …                  | …
C2 Dyn FedAvg       | …                  | …                  | …
Dyn-Krum            | …                  | …                  | …
Dyn-BALANCE         | …                  | …                  | …
Dyn-UBAR            | …                  | …                  | …
DMTT (proposed)     | …                  | …                  | …
```

### Config location

```
experiments/paper/dmtt/exp1_main_comparison/
  generate_configs.py     ← generates all 54 YAML files
  run_exp1.sh             ← sequential runner (logs to results/)
  c1_static_fedavg/       ← 9 configs (3 fracs × 3 seeds)
  c2_dynamic_fedavg/      ← 9 configs
  dyn_krum/               ← 9 configs
  dyn_balance/            ← 9 configs
  dyn_ubar/               ← 9 configs
  c3_dmtt/                ← 9 configs
  results/                ← stdout logs (created at run-time)
```

### Hyperparameters (shared across all conditions)

```yaml
num_nodes: 10
rounds: 50
local_epochs: 2
batch_size: 32
lr: 0.01
dataset: uci_har (Dirichlet α=0.5)
mobility: area=100, range=40, speed=8, ensure_connected=True
round_duration_s: 30   # conservative for small UCI HAR MLP on GPU
```

### Compute estimate

| Condition | Per-run time | Runs | Total |
|-----------|-------------|------|-------|
| C1 (simulation) | ~1 min | 9 | ~9 min |
| C2 / baselines / C3 (distributed, 30 s rounds) | ~28 min | 45 | ~21 h |

On a 24 GB GPU with UCI HAR MLP (<<1 MB), 30 s per round gives >20× slack.
Experiments must run sequentially; IPC dirs are unique so parallel runs are technically
possible but GPU contention on 20+ concurrent processes is inadvisable.

---

## Experiment 2 — Multi-dataset validation

### Goal

Confirm that Exp 1 results generalise beyond UCI HAR.  Replicate the hardest condition
(30 % Byzantine, topology-liar) for C1, C2, and C3 on **PAMAP2** (9 clients, 12-class
activity recognition, different feature set).

### Setup

- Conditions: C1, C2, C3 only (C1 as lower bound, C2 as upper bound for baselines)
- Byzantine fraction: 30 % only
- Seeds: 42, 123, 777
- Configs: `experiments/paper/dmtt/exp2_multidata/`

### Expected table shape

```
Dataset     | C1 Static  | C2 Dyn (no trust) | C3 DMTT
------------|------------|-------------------|--------
UCI HAR     | (from Exp1)| (from Exp1)       | (from Exp1)
PAMAP2      | x.xx±y     | x.xx±y            | x.xx±y
```

### Note on PAMAP2 config

PAMAP2 uses 9 nodes, 12 classes, input_dim=4000.  The model config changes accordingly.
See `experiments/paper/pamap2/` for existing baseline configs to copy from.

---

## Experiment 3 — Ablation study

### Goal

Isolate the contribution of each DMTT component.  Without this, reviewers cannot
distinguish which of the four collaboration-score terms drives the result.

### Variants

| Label | What is disabled |
|-------|-----------------|
| DMTT-full | nothing (full system) |
| no-beta-trust | `T_ij^topo = 1.0` always (topology trust frozen at 1) |
| no-edge-conf | `χ_i^t = 1.0` always (accept all topology claims immediately) |
| no-model-compat | `λ1 = 0`, normalise remaining λ (topology/link only) |
| no-link-rel | `λ3 = 0`, normalise remaining λ (model/topology only) |
| uniform-q | `λ1=λ2=λ3=0.33`, `λ4=0` |

### Setup

- Byzantine fraction: 30 % (most discriminative)
- Dataset: UCI HAR
- Seeds: 42, 123, 777
- Total: 6 variants × 3 seeds = 18 distributed runs (~8 h)
- Configs: `experiments/paper/dmtt/exp3_ablation/`

### Implementation note

`no-beta-trust` and `no-edge-conf` require small changes to `DMTTNodeState`
(freeze `alpha/beta` at initial values or set `chi` to 1.0 unconditionally).
Add a `disable_beta_trust: bool` and `disable_edge_conf: bool` flag to `DMTTConfig`
before running this experiment.

---

## Experiment 4 — Trust-score evolution

### Goal

Demonstrate mechanistically that `T_ij^topo` correctly separates honest from Byzantine
nodes over training rounds.  This is the main qualitative evidence for the theoretical
claims in Section 5.

### What to collect

For a single C3 run (seed 42, 30 % Byzantine), log per-round:
- `T_ij^topo` for each (honest node i, Byzantine source j) pair → should converge low
- `T_ij^topo` for each (honest node i, honest source j) pair → should stay high
- Edge confidence `χ_i^t(e)` for a fabricated edge vs. a real edge

### Implementation

Add a per-round dump of `DMTTNodeState.alpha`, `beta`, and derived `T_topo` to the
METRICS message sent to the monitor.  The monitor's ordering buffer already handles
out-of-order delivery — the trust dict just becomes an extra key in the metrics payload.

### Output

A single 2-panel figure:
- Left: `T_ij^topo` vs. round for honest→honest and honest→Byzantine edges (mean ± std)
- Right: `χ_i^t(e)` vs. round for a fabricated edge and a real edge

---

## Experiment 5 — Sensitivity analysis

### Goal

Show robustness of DMTT across (a) non-IID severity and (b) mobility parameters.

### (a) Non-IID heterogeneity sweep

Compare C2 vs. C3 at α ∈ {0.1, 0.3, 0.5, 1.0} with 30 % Byzantine.
This directly illustrates Corollary 1 — DMTT's advantage grows with heterogeneity
because τ_scr² < τ_raw² becomes more pronounced.

Conditions: C2, C3 × 4 α values × 3 seeds = 24 distributed runs.
Configs: `experiments/paper/dmtt/exp5_sensitivity/heterogeneity/`

### (b) Mobility parameter sweep

Fix 30 % Byzantine and α=0.5.  Vary:
- `comm_range` ∈ {20, 30, 40, 50} (sparse → dense graph)
- `max_speed` ∈ {4, 8, 12} (low → high churn)

Conditions: C2, C3 × (4 + 3) values × 1 seed = 14 distributed runs.
Configs: `experiments/paper/dmtt/exp5_sensitivity/mobility/`

---

## Experiment 6 — Communication overhead

### Goal

Quantify the additional cost of DMTT's TOPO_CLAIM channel and trust computation.

### What to measure

Per round, per node:
- Total bytes sent (MODEL_STATE + TOPO_CLAIM, if applicable)
- Wall-clock time for Steps 6-8 in DMTTNodeProcess (trust update + collaborator selection)

### Implementation

Add lightweight byte counters to `messaging.py` encode functions and a timer around
the trust-update block in `DMTTNodeProcess`.  Report as additional keys in METRICS.

### Output

A small table comparing C2 and C3 on bytes/round and ms/round (trust overhead only).

---

## Run instructions (Experiment 1)

### Setup

```bash
cd ~/Projects/Research/Murmura
source .venv/bin/activate   # or: conda activate murmura

# Generate all 54 config files
python experiments/paper/dmtt/exp1_main_comparison/generate_configs.py

# Verify: should show 54 YAML files
find experiments/paper/dmtt/exp1_main_comparison -name "*.yaml" | wc -l
```

### Running

```bash
# Sequential run — logs each experiment to results/<name>.log
bash experiments/paper/dmtt/exp1_main_comparison/run_exp1.sh 2>&1 | tee run_exp1_master.log
```

The script runs simulation experiments first (fast), then distributed experiments.
Total expected wall-clock: ~21 hours on a 24 GB GPU machine.

### Resuming after interruption

The run script skips any experiment whose result log already contains "Final accuracy".
Re-running `run_exp1.sh` after an interruption will pick up where it left off.

### Monitoring a distributed run

While a distributed experiment is running, tail the node logs:

```bash
# In a second terminal
ls /tmp/murmura_*/  # shows active IPC socket dirs
tail -f run_exp1_master.log
```

---

## Parsing and aggregating results

After all runs complete, use the existing `run_all_experiments.py` parser pattern to
extract per-round `honest_accuracy` and `mean_accuracy` from each log file.

A dedicated results collector for Exp 1 is at:
`experiments/paper/dmtt/exp1_main_comparison/collect_results.py` (to be written after
first results are in hand, using the same regex pattern as the existing parser).

---

## Notes on round_duration_s

The existing DMTT configs use 120 s.  Exp 1 configs use **30 s** because:
- UCI HAR MLP (561→256→128→6) with batch_size=32 and E=2 local epochs takes < 2 s
  per node even with 10 concurrent processes on one GPU.
- ZMQ IPC model exchange (< 1 MB model) completes in milliseconds.
- 30 s gives > 15× slack for process scheduling, evaluation, and ZMQ overhead.

If experiments stall or nodes miss rounds, increase to 60 s and rerun the affected
conditions.  The `round_duration_s` field is in `distributed.round_duration_s`.
