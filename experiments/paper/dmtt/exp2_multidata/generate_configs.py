#!/usr/bin/env python3
"""Generate 9 YAML configs for Experiment 2 (multi-dataset validation on PAMAP2).

Conditions: C1 (static FedAvg), C2 (dynamic FedAvg), C3 (full DMTT)
Byzantine fraction: 30 % only (hardest condition from Exp 1)
Seeds: 42, 123, 777

Total: 3 conditions × 3 seeds = 9 configs.

Usage:
  # First prepare the 9-node contact trace (run once):
  python scripts/prepare_sociopatterns.py --dataset primary_school --num_nodes 9 --round_duration 30 --min-contacts 3

  # Then generate configs:
  python experiments/paper/dmtt/exp2_multidata/generate_configs.py
"""

from pathlib import Path

HERE = Path(__file__).parent

TRACE_PATH = "data/sociopatterns/primary_school_N9_R30s_mc3.csv"

SEEDS        = [42, 123, 777]
BYZ_FRACTION = 0.3
NUM_NODES    = 9

DATA_BLOCK = """\
data:
  adapter: "wearables.pamap2"
  params:
    data_path: "wearables_datasets/PAMAP2_Dataset"
    partition_method: "dirichlet"
    alpha: 0.5
    window_size: 100
    window_stride: 50
    include_heart_rate: true
    normalize: true
"""

MODEL_BLOCK = """\
model:
  factory: "examples.wearables.pamap2"
  params:
    input_dim: 4000
    hidden_dims: [512, 256, 128]
    num_classes: 12
    dropout: 0.3
"""

TRAINING_BLOCK = """\
training:
  local_epochs: 2
  batch_size: 64
  lr: 0.001
  max_samples: null
"""

TOPOLOGY_BLOCK = f"""\
topology:
  type: "fully"
  num_nodes: {NUM_NODES}
  seed: 12345
"""

TRACE_MOBILITY = f"""\
trace_mobility:
  trace_path: "{TRACE_PATH}"
  num_nodes: {NUM_NODES}
  ensure_connected: true
"""

DISTRIBUTED_TEMPLATE = """\
backend: distributed

distributed:
  transport: ipc
  ipc_dir: "/tmp/murmura_{tag}"
  round_duration_s: 30.0
  startup_grace_s: 5.0
"""

DMTT_BLOCK = """\
dmtt:
  budget_B: 4
  rho: 0.1
  lambda_forget: 0.98
  w_d: 1.0
  w_c: 0.5
  w_x: 5.0
  tau_U: 0.3
  eta: 5.0
  tau_trust: 0.49
  w_a: 0.7
  tau_u: 0.5
  lambda1: 0.1
  lambda2: 0.6
  lambda3: 0.2
  lambda4: 0.1
"""

AGG_FEDAVG = """\
aggregation:
  algorithm: "fedavg"
  params: {}
"""


def make_attack_block() -> str:
    return (
        f"attack:\n"
        f"  enabled: true\n"
        f"  type: \"topology_liar\"\n"
        f"  percentage: {BYZ_FRACTION}\n"
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


def write_config(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def gen_c1_static() -> None:
    out_dir = HERE / "c1_static_fedavg"
    for seed in SEEDS:
        tag  = f"exp2_c1_static_byz30_s{seed}"
        name = f"Exp2-C1-Static-byz30-s{seed}"
        blocks = [
            f"# Exp 2 — C1 Static FedAvg baseline (simulation)\n"
            f"# Dataset: PAMAP2  |  Byzantine: 30 %  |  seed: {seed}\n",
            make_experiment_block(name, seed),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            make_attack_block(),
            TRAINING_BLOCK,
            DATA_BLOCK,
            MODEL_BLOCK,
            "backend: simulation\n",
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  c1_static_fedavg: {len(SEEDS)} configs")


def gen_c2_dynamic() -> None:
    out_dir = HERE / "c2_dynamic_fedavg"
    for seed in SEEDS:
        tag  = f"exp2_c2_dyn_fedavg_byz30_s{seed}"
        name = f"Exp2-C2-DynFedAvg-byz30-s{seed}"
        blocks = [
            f"# Exp 2 — C2 Dynamic FedAvg, no trust (distributed)\n"
            f"# Dataset: PAMAP2  |  Byzantine: 30 %  |  seed: {seed}\n",
            make_experiment_block(name, seed),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            make_attack_block(),
            TRAINING_BLOCK,
            DATA_BLOCK,
            MODEL_BLOCK,
            DISTRIBUTED_TEMPLATE.format(tag=tag),
            TRACE_MOBILITY,
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  c2_dynamic_fedavg: {len(SEEDS)} configs")


def gen_c3_dmtt() -> None:
    out_dir = HERE / "c3_dmtt"
    for seed in SEEDS:
        tag  = f"exp2_c3_dmtt_byz30_s{seed}"
        name = f"Exp2-C3-DMTT-byz30-s{seed}"
        blocks = [
            f"# Exp 2 — C3 Full DMTT (distributed)\n"
            f"# Dataset: PAMAP2  |  Byzantine: 30 %  |  seed: {seed}\n",
            make_experiment_block(name, seed),
            TOPOLOGY_BLOCK,
            AGG_FEDAVG,
            make_attack_block(),
            TRAINING_BLOCK,
            DATA_BLOCK,
            MODEL_BLOCK,
            DISTRIBUTED_TEMPLATE.format(tag=tag),
            TRACE_MOBILITY,
            DMTT_BLOCK,
        ]
        write_config(out_dir / f"{tag}.yaml", blocks)
    print(f"  c3_dmtt: {len(SEEDS)} configs")


if __name__ == "__main__":
    print("Generating Experiment 2 configs (PAMAP2 multi-dataset validation)...")
    gen_c1_static()
    gen_c2_dynamic()
    gen_c3_dmtt()

    total = sum(
        len(list((HERE / d).glob("*.yaml")))
        for d in ["c1_static_fedavg", "c2_dynamic_fedavg", "c3_dmtt"]
    )
    print(f"\nDone — {total} configs written to {HERE}")
    print(
        "\nPrerequisite (run once):\n"
        "  python scripts/prepare_sociopatterns.py "
        "--dataset primary_school --num_nodes 9 --round_duration 30"
    )
