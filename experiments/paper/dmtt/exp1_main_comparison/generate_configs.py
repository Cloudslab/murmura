#!/usr/bin/env python3
"""Generate all 54 YAML configs for Experiment 1 (main comparison table).

Conditions:
  c1_static_fedavg  — Static FedAvg, simulation backend (C1 baseline)
  c2_dynamic_fedavg — Dynamic FedAvg, no trust, distributed backend (C2)
  dyn_krum          — Dynamic Krum, distributed backend
  dyn_balance       — Dynamic BALANCE, distributed backend
  dyn_ubar          — Dynamic UBAR, distributed backend
  c3_dmtt           — Full DMTT, distributed backend (proposed)

Sweep:
  Byzantine fractions: 10%, 20%, 30%  (1, 2, 3 nodes out of 10)
  Seeds:               42, 123, 777

Total: 6 conditions × 3 fractions × 3 seeds = 54 configs.

Usage:
  # First prepare the contact trace (run once):
  python scripts/prepare_sociopatterns.py --dataset primary_school --num_nodes 10 --round_duration 30 --min-contacts 3

  # Then generate configs:
  python experiments/paper/dmtt/exp1_main_comparison/generate_configs.py
"""

from pathlib import Path

HERE = Path(__file__).parent

# Path to preprocessed contact-trace CSV (output of prepare_sociopatterns.py)
# Update this if you change --num_nodes or --round_duration in the prepare script.
TRACE_PATH = "data/sociopatterns/primary_school_N10_R30s_mc3.csv"

SEEDS = [42, 123, 777]
BYZ_FRACTIONS = [0.1, 0.2, 0.3]

# Shared fixed blocks
DATA_BLOCK = """\
data:
  adapter: "wearables.uci_har"
  params:
    data_path: "wearables_datasets/UCI HAR Dataset"
    split: "train"
    partition_method: "dirichlet"
    alpha: 0.5
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

TRAINING_BLOCK = """\
training:
  local_epochs: 2
  batch_size: 32
  lr: 0.01
  max_samples: null
"""

TOPOLOGY_BLOCK = """\
topology:
  type: "fully"
  num_nodes: 10
  seed: 12345
"""

# Real contact-trace mobility block (replaces synthetic random walk)
# trace_path is a relative path from the repo root.
TRACE_MOBILITY_TEMPLATE = """\
trace_mobility:
  trace_path: "{trace_path}"
  num_nodes: 10
  ensure_connected: true
"""

# Distributed block — ipc_dir is templated in
DISTRIBUTED_TEMPLATE = """\
backend: distributed

distributed:
  transport: ipc
  ipc_dir: "/tmp/murmura_{tag}"
  round_duration_s: 30.0
  startup_grace_s: 5.0
"""

# DMTT block — fixed hyperparameters from the paper config
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

# Aggregation blocks (static parts only; Krum's `f` is templated)
AGG_FEDAVG = """\
aggregation:
  algorithm: "fedavg"
  params: {}
"""

AGG_KRUM_TEMPLATE = """\
aggregation:
  algorithm: "krum"
  params:
    f: {f}
"""

AGG_BALANCE = """\
aggregation:
  algorithm: "balance"
  params:
    gamma: 0.5
    kappa: 1.0
    alpha: 0.5
    min_neighbors: 1
"""

AGG_UBAR = """\
aggregation:
  algorithm: "ubar"
  params:
    rho: 0.5
    alpha: 0.5
    min_neighbors: 1
"""


def byz_nodes(fraction: float, total: int = 10) -> int:
    return round(fraction * total)


def byz_label(fraction: float) -> str:
    return f"byz{int(fraction * 100):02d}"


def make_attack_block(fraction: float) -> str:
    return (
        f"attack:\n"
        f"  enabled: true\n"
        f"  type: \"topology_liar\"\n"
        f"  percentage: {fraction}\n"
        f"  params:\n"
        f"    model_attack_type: \"gaussian\"\n"
        f"    noise_std: 10.0\n"
    )


def make_experiment_block(name: str, seed: int) -> str:
    return (
        f"experiment:\n"
        f"  name: \"{name}\"\n"
        f"  seed: {seed}\n"
        f"  rounds: 50\n"
        f"  verbose: true\n"
    )


def write_config(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


# ---------- C1: Static FedAvg (simulation backend) ----------

def gen_c1_static_fedavg() -> None:
    out_dir = HERE / "c1_static_fedavg"
    for frac in BYZ_FRACTIONS:
        for seed in SEEDS:
            tag = f"c1_static_{byz_label(frac)}_s{seed}"
            name = f"C1-StaticFedAvg-{byz_label(frac)}-s{seed}"
            blocks = [
                f"# C1 — Static FedAvg baseline (simulation)\n"
                f"# Byzantine fraction: {int(frac*100)} %  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_FEDAVG,
                make_attack_block(frac),
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                "backend: simulation\n",
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  c1_static_fedavg: {len(SEEDS)*len(BYZ_FRACTIONS)} configs")


# ---------- C2: Dynamic FedAvg (distributed, no trust) ----------

def gen_c2_dynamic_fedavg() -> None:
    out_dir = HERE / "c2_dynamic_fedavg"
    for frac in BYZ_FRACTIONS:
        for seed in SEEDS:
            tag = f"c2_dyn_fedavg_{byz_label(frac)}_s{seed}"
            name = f"C2-DynFedAvg-{byz_label(frac)}-s{seed}"
            blocks = [
                f"# C2 — Dynamic FedAvg, no trust (distributed)\n"
                f"# Byzantine fraction: {int(frac*100)} %  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_FEDAVG,
                make_attack_block(frac),
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY_TEMPLATE.format(trace_path=TRACE_PATH),
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  c2_dynamic_fedavg: {len(SEEDS)*len(BYZ_FRACTIONS)} configs")


# ---------- Dyn-Krum ----------

def gen_dyn_krum() -> None:
    out_dir = HERE / "dyn_krum"
    for frac in BYZ_FRACTIONS:
        f_val = byz_nodes(frac)
        for seed in SEEDS:
            tag = f"dyn_krum_{byz_label(frac)}_s{seed}"
            name = f"DynKrum-{byz_label(frac)}-s{seed}"
            blocks = [
                f"# Dynamic Krum baseline (distributed)\n"
                f"# Byzantine fraction: {int(frac*100)} %  (f={f_val})  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_KRUM_TEMPLATE.format(f=f_val),
                make_attack_block(frac),
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY_TEMPLATE.format(trace_path=TRACE_PATH),
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  dyn_krum: {len(SEEDS)*len(BYZ_FRACTIONS)} configs")


# ---------- Dyn-BALANCE ----------

def gen_dyn_balance() -> None:
    out_dir = HERE / "dyn_balance"
    for frac in BYZ_FRACTIONS:
        for seed in SEEDS:
            tag = f"dyn_balance_{byz_label(frac)}_s{seed}"
            name = f"DynBALANCE-{byz_label(frac)}-s{seed}"
            blocks = [
                f"# Dynamic BALANCE baseline (distributed)\n"
                f"# Byzantine fraction: {int(frac*100)} %  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_BALANCE,
                make_attack_block(frac),
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY_TEMPLATE.format(trace_path=TRACE_PATH),
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  dyn_balance: {len(SEEDS)*len(BYZ_FRACTIONS)} configs")


# ---------- Dyn-UBAR ----------

def gen_dyn_ubar() -> None:
    out_dir = HERE / "dyn_ubar"
    for frac in BYZ_FRACTIONS:
        for seed in SEEDS:
            tag = f"dyn_ubar_{byz_label(frac)}_s{seed}"
            name = f"DynUBAR-{byz_label(frac)}-s{seed}"
            blocks = [
                f"# Dynamic UBAR baseline (distributed)\n"
                f"# Byzantine fraction: {int(frac*100)} %  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_UBAR,
                make_attack_block(frac),
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY_TEMPLATE.format(trace_path=TRACE_PATH),
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  dyn_ubar: {len(SEEDS)*len(BYZ_FRACTIONS)} configs")


# ---------- C3: Full DMTT ----------

def gen_c3_dmtt() -> None:
    out_dir = HERE / "c3_dmtt"
    for frac in BYZ_FRACTIONS:
        for seed in SEEDS:
            tag = f"c3_dmtt_{byz_label(frac)}_s{seed}"
            name = f"C3-DMTT-{byz_label(frac)}-s{seed}"
            blocks = [
                f"# C3 — Full DMTT (distributed)\n"
                f"# Byzantine fraction: {int(frac*100)} %  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_FEDAVG,
                make_attack_block(frac),
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY_TEMPLATE.format(trace_path=TRACE_PATH),
                DMTT_BLOCK,
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  c3_dmtt: {len(SEEDS)*len(BYZ_FRACTIONS)} configs")


if __name__ == "__main__":
    print("Generating Experiment 1 configs...")
    gen_c1_static_fedavg()
    gen_c2_dynamic_fedavg()
    gen_dyn_krum()
    gen_dyn_balance()
    gen_dyn_ubar()
    gen_c3_dmtt()

    total = sum(
        len(list((HERE / d).glob("*.yaml")))
        for d in ["c1_static_fedavg", "c2_dynamic_fedavg",
                  "dyn_krum", "dyn_balance", "dyn_ubar", "c3_dmtt"]
    )
    print(f"\nDone — {total} configs written to {HERE}")
