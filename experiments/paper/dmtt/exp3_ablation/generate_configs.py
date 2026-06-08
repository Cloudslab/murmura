#!/usr/bin/env python3
"""Generate 18 YAML configs for Experiment 3 (DMTT ablation study).

Variants (6 × 3 seeds = 18 distributed runs):
  full          — full DMTT (reproduced from Exp 1 for direct comparison)
  no_beta       — disable_beta_trust: T_ij^topo frozen at 1.0
  no_topo       — disable_topo_claims: skip TOPO_CLAIM trust updates
  no_model      — lambda1=0, weights renormalised (no model-compatibility score)
  no_link       — lambda3=0, weights renormalised (no link reliability)
  uniform_q     — lambda1=lambda2=lambda3=1/3, lambda4=0

All runs: 30 % Byzantine, UCI HAR, primary_school trace (N=10), 50 rounds.

Usage:
  python experiments/paper/dmtt/exp3_ablation/generate_configs.py
"""

from pathlib import Path

HERE = Path(__file__).parent
TRACE_PATH = "data/sociopatterns/primary_school_N10_R30s.csv"

SEEDS    = [42, 123, 777]
NUM_NODES = 10

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

ATTACK_BLOCK = """\
attack:
  enabled: true
  type: "topology_liar"
  percentage: 0.3
  params:
    model_attack_type: "gaussian"
    noise_std: 10.0
"""

AGG_FEDAVG = """\
aggregation:
  algorithm: "fedavg"
  params: {}
"""

# Base DMTT hyperparameters (shared across all variants)
_BASE_DMTT = dict(
    budget_B=5, rho=0.1, lambda_forget=0.98,
    w_d=2.0, w_c=0.5, w_x=1.0,
    tau_U=0.3, eta=5.0,
    w_a=0.7, tau_u=0.5,
    lambda1=0.1, lambda2=0.6, lambda3=0.2, lambda4=0.1,
    disable_beta_trust=False, disable_topo_claims=False,
)


def _dmtt_block(**overrides) -> str:
    cfg = {**_BASE_DMTT, **overrides}
    lines = ["dmtt:"]
    for k, v in cfg.items():
        if isinstance(v, bool):
            lines.append(f"  {k}: {'true' if v else 'false'}")
        elif isinstance(v, float):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


# Each variant: (label, description, dmtt_overrides)
VARIANTS = [
    (
        "full",
        "Full DMTT (all components enabled)",
        {},
    ),
    (
        "no_beta_trust",
        "Ablation: disable Beta topology trust (T_ij^topo=1.0)",
        {"disable_beta_trust": True},
    ),
    (
        "no_topo_claims",
        "Ablation: skip TOPO_CLAIM trust updates (accept all claims blindly)",
        {"disable_topo_claims": True},
    ),
    (
        "no_model_compat",
        "Ablation: no model-compatibility score (lambda1=0, renormalized)",
        # lambda1=0; renorm (lambda2+lambda3+lambda4 = 0.3+0.2+0.1 = 0.6)
        {"lambda1": 0.0, "lambda2": 0.5, "lambda3": 0.333, "lambda4": 0.167},
    ),
    (
        "no_link_rel",
        "Ablation: no link reliability (lambda3=0, renormalized)",
        # lambda3=0; renorm (lambda1+lambda2+lambda4 = 0.4+0.3+0.1 = 0.8)
        {"lambda1": 0.5, "lambda2": 0.375, "lambda3": 0.0, "lambda4": 0.125},
    ),
    (
        "uniform_q",
        "Ablation: uniform collaboration weights (lambda1=lambda2=lambda3=1/3, lambda4=0)",
        {"lambda1": 0.333, "lambda2": 0.333, "lambda3": 0.333, "lambda4": 0.0},
    ),
]


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


def gen_ablation() -> None:
    out_dir = HERE / "configs"
    count   = 0
    for variant_label, description, overrides in VARIANTS:
        for seed in SEEDS:
            tag  = f"exp3_{variant_label}_byz30_s{seed}"
            name = f"Exp3-{variant_label}-byz30-s{seed}"
            blocks = [
                f"# Exp 3 — Ablation: {description}\n"
                f"# Byzantine: 30 %  |  seed: {seed}\n",
                make_experiment_block(name, seed),
                TOPOLOGY_BLOCK,
                AGG_FEDAVG,
                ATTACK_BLOCK,
                TRAINING_BLOCK,
                DATA_BLOCK,
                MODEL_BLOCK,
                DISTRIBUTED_TEMPLATE.format(tag=tag),
                TRACE_MOBILITY,
                _dmtt_block(**overrides),
            ]
            write_config(out_dir / f"{tag}.yaml", blocks)
            count += 1
    print(f"  ablation configs: {count}")


if __name__ == "__main__":
    print("Generating Experiment 3 (ablation) configs...")
    gen_ablation()

    total = len(list((HERE / "configs").glob("*.yaml")))
    print(f"\nDone — {total} configs written to {HERE / 'configs'}")
