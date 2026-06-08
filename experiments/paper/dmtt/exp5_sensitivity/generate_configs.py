#!/usr/bin/env python3
"""Generate configs for Experiment 5 (sensitivity analysis).

Part (a) — Non-IID heterogeneity sweep (alpha ∈ {0.1, 0.3, 0.5, 1.0}):
  C2 vs. C3 × 4 alpha × 3 seeds = 24 distributed runs.
  Written to: heterogeneity/c2_dynamic_fedavg/  and  heterogeneity/c3_dmtt/

Part (b) — Mobility parameter sweep (30 % Byzantine, alpha=0.5, seed=42):
  comm_range ∈ {20, 30, 40, 50}  — sparse to dense connectivity
  max_speed  ∈ {4, 8, 12}        — low to high churn
  C2 vs. C3 × (4 + 3) values × 1 seed = 14 distributed runs.
  Uses synthetic MobilityConfig (random walk) so comm_range / max_speed are
  directly controllable.
  Written to: mobility/c2_dynamic_fedavg/  and  mobility/c3_dmtt/

Usage:
  python experiments/paper/dmtt/exp5_sensitivity/generate_configs.py
"""

from pathlib import Path

HERE = Path(__file__).parent

SEEDS       = [42, 123, 777]
BYZ_FRACTION = 0.3
NUM_NODES   = 10
TRACE_PATH  = "data/sociopatterns/primary_school_N10_R30s.csv"

TOPOLOGY_BLOCK = f"""\
topology:
  type: "fully"
  num_nodes: {NUM_NODES}
  seed: 12345
"""

AGG_FEDAVG = """\
aggregation:
  algorithm: "fedavg"
  params: {}
"""

TRAINING_BLOCK = """\
training:
  local_epochs: 2
  batch_size: 32
  lr: 0.01
  max_samples: null
"""

MODEL_BLOCK = """\
model:
  factory: "examples.wearables.uci_har"
  params:
    input_dim: 561
    hidden_dims: [256, 128]
    num_classes: 6
    dropout: 0.3
"""

DISTRIBUTED_TEMPLATE = """\
backend: distributed

distributed:
  transport: ipc
  ipc_dir: "/tmp/murmura_{tag}"
  round_duration_s: 30.0
  startup_grace_s: 5.0
"""

TRACE_MOBILITY = f"""\
trace_mobility:
  trace_path: "{TRACE_PATH}"
  num_nodes: {NUM_NODES}
  ensure_connected: true
"""

DMTT_BLOCK = """\
dmtt:
  budget_B: 5
  rho: 0.1
  lambda_forget: 0.98
  w_d: 2.0
  w_c: 0.5
  w_x: 1.0
  tau_U: 0.3
  eta: 5.0
  w_a: 0.7
  tau_u: 0.5
  lambda1: 0.1
  lambda2: 0.6
  lambda3: 0.2
  lambda4: 0.1
"""

ATTACK_BLOCK = """\
attack:
  enabled: true
  type: "topology_liar"
  percentage: 0.3
  params:
    model_attack_type: "gaussian"
    noise_std: 10.0
"""


def data_block(alpha: float) -> str:
    return (
        f"data:\n"
        f"  adapter: \"wearables.uci_har\"\n"
        f"  params:\n"
        f"    data_path: \"wearables_datasets/UCI HAR Dataset\"\n"
        f"    split: \"train\"\n"
        f"    partition_method: \"dirichlet\"\n"
        f"    alpha: {alpha}\n"
    )


def mobility_block(comm_range: float, max_speed: float) -> str:
    return (
        f"mobility:\n"
        f"  area_size: 100.0\n"
        f"  comm_range: {comm_range}\n"
        f"  max_speed: {max_speed}\n"
        f"  seed: 42\n"
        f"  ensure_connected: true\n"
    )


def make_experiment_block(name: str, seed: int) -> str:
    return (
        f"experiment:\n"
        f"  name: \"{name}\"\n"
        f"  seed: {seed}\n"
        f"  rounds: 50\n"
        f"  verbose: true\n"
    )


def write_config(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


# --------------------------------------------------------------------------
# Part (a) — heterogeneity sweep
# --------------------------------------------------------------------------

ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0]


def gen_het_c2() -> int:
    out_dir = HERE / "heterogeneity" / "c2_dynamic_fedavg"
    count = 0
    for alpha in ALPHA_VALUES:
        alpha_str = str(alpha).replace(".", "p")
        for seed in SEEDS:
            tag  = f"exp5a_c2_a{alpha_str}_byz30_s{seed}"
            name = f"Exp5a-C2-alpha{alpha}-byz30-s{seed}"
            blocks = [
                f"# Exp 5a — C2 Dynamic FedAvg, alpha={alpha}, Byzantine 30 %, seed {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_FEDAVG,
                ATTACK_BLOCK,
                TRAINING_BLOCK,
                data_block(alpha),
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY,
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
            count += 1
    return count


def gen_het_c3() -> int:
    out_dir = HERE / "heterogeneity" / "c3_dmtt"
    count = 0
    for alpha in ALPHA_VALUES:
        alpha_str = str(alpha).replace(".", "p")
        for seed in SEEDS:
            tag  = f"exp5a_c3_a{alpha_str}_byz30_s{seed}"
            name = f"Exp5a-C3-alpha{alpha}-byz30-s{seed}"
            blocks = [
                f"# Exp 5a — C3 DMTT, alpha={alpha}, Byzantine 30 %, seed {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_FEDAVG,
                ATTACK_BLOCK,
                TRAINING_BLOCK,
                data_block(alpha),
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY,
                DMTT_BLOCK,
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
            count += 1
    return count


# --------------------------------------------------------------------------
# Part (b) — mobility parameter sweep (seed=42 only, synthetic random walk)
# --------------------------------------------------------------------------

COMM_RANGES = [20, 30, 40, 50]
MAX_SPEEDS  = [4, 8, 12]
MOB_SEED    = 42


def gen_mob_c2() -> int:
    out_dir = HERE / "mobility" / "c2_dynamic_fedavg"
    count   = 0
    # comm_range sweep (fix max_speed=8)
    for cr in COMM_RANGES:
        tag  = f"exp5b_c2_cr{cr}_sp8_s{MOB_SEED}"
        name = f"Exp5b-C2-cr{cr}-sp8-s{MOB_SEED}"
        blocks = [
            f"# Exp 5b — C2 Dynamic FedAvg, comm_range={cr}, max_speed=8\n",
            make_experiment_block(name, MOB_SEED),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            ATTACK_BLOCK,
            TRAINING_BLOCK,
            data_block(0.5),
            MODEL_BLOCK,
            DISTRIBUTED_TEMPLATE.format(tag=tag),
            mobility_block(cr, 8),
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
        count += 1
    # max_speed sweep (fix comm_range=40)
    for sp in MAX_SPEEDS:
        tag  = f"exp5b_c2_cr40_sp{sp}_s{MOB_SEED}"
        name = f"Exp5b-C2-cr40-sp{sp}-s{MOB_SEED}"
        blocks = [
            f"# Exp 5b — C2 Dynamic FedAvg, comm_range=40, max_speed={sp}\n",
            make_experiment_block(name, MOB_SEED),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            ATTACK_BLOCK,
            TRAINING_BLOCK,
            data_block(0.5),
            MODEL_BLOCK,
            DISTRIBUTED_TEMPLATE.format(tag=tag),
            mobility_block(40, sp),
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
        count += 1
    return count


def gen_mob_c3() -> int:
    out_dir = HERE / "mobility" / "c3_dmtt"
    count   = 0
    # comm_range sweep
    for cr in COMM_RANGES:
        tag  = f"exp5b_c3_cr{cr}_sp8_s{MOB_SEED}"
        name = f"Exp5b-C3-cr{cr}-sp8-s{MOB_SEED}"
        blocks = [
            f"# Exp 5b — C3 DMTT, comm_range={cr}, max_speed=8\n",
            make_experiment_block(name, MOB_SEED),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            ATTACK_BLOCK,
            TRAINING_BLOCK,
            data_block(0.5),
            MODEL_BLOCK,
            DISTRIBUTED_TEMPLATE.format(tag=tag),
            mobility_block(cr, 8),
            DMTT_BLOCK,
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
        count += 1
    # max_speed sweep
    for sp in MAX_SPEEDS:
        tag  = f"exp5b_c3_cr40_sp{sp}_s{MOB_SEED}"
        name = f"Exp5b-C3-cr40-sp{sp}-s{MOB_SEED}"
        blocks = [
            f"# Exp 5b — C3 DMTT, comm_range=40, max_speed={sp}\n",
            make_experiment_block(name, MOB_SEED),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            ATTACK_BLOCK,
            TRAINING_BLOCK,
            data_block(0.5),
            MODEL_BLOCK,
            DISTRIBUTED_TEMPLATE.format(tag=tag),
            mobility_block(40, sp),
            DMTT_BLOCK,
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
        count += 1
    return count


if __name__ == "__main__":
    print("Generating Experiment 5 configs (sensitivity analysis)...")
    n_het_c2  = gen_het_c2()
    n_het_c3  = gen_het_c3()
    n_mob_c2  = gen_mob_c2()
    n_mob_c3  = gen_mob_c3()

    total = n_het_c2 + n_het_c3 + n_mob_c2 + n_mob_c3
    print(f"  heterogeneity C2: {n_het_c2}")
    print(f"  heterogeneity C3: {n_het_c3}")
    print(f"  mobility C2:      {n_mob_c2}")
    print(f"  mobility C3:      {n_mob_c3}")
    print(f"\nDone — {total} configs written to {HERE}")
