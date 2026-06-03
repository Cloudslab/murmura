"""Write experiment results to Parquet for fast downstream analysis.

Each experiment run produces one Parquet file with one row per training round.
Config metadata (condition, dataset, aggregation, byzantine fraction, seed, …)
is baked into every row so multiple files can be concatenated with pd.concat()
and sliced immediately without any join.

Usage (called automatically by the CLI when --results-dir is set):

    from murmura.utils.results import write_parquet
    write_parquet(history, config, results_dir=Path("results"))

Analysis example:

    import pandas as pd
    from pathlib import Path

    df = pd.concat([pd.read_parquet(p) for p in Path("results").glob("*.parquet")])
    pivot = (
        df[df["round"] == df["round"].max()]          # final-round rows only
        .groupby(["condition", "byzantine_pct"])
        .agg(
            honest_acc_mean=("honest_accuracy", "mean"),
            honest_acc_std=("honest_accuracy", "std"),
        )
        .round(4)
    )
    print(pivot)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from murmura.config.schema import Config


# ---------------------------------------------------------------------------
# Metadata inference helpers
# ---------------------------------------------------------------------------

def _condition(config: Config) -> str:
    """Infer a short condition label from the config structure."""
    has_dynamic = config.mobility is not None or config.trace_mobility is not None
    if not has_dynamic:
        return "c1_static"
    if config.dmtt is not None:
        return "c3_dmtt"
    return f"dyn_{config.aggregation.algorithm}"


def _mobility_type(config: Config) -> str:
    if config.trace_mobility is not None:
        return "trace"
    if config.mobility is not None:
        return "random_walk"
    return "static"


def _dataset_name(config: Config) -> str:
    # adapter strings look like "wearables.uci_har" or "leaf.femnist"
    return config.data.adapter.split(".")[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_parquet(
    history: Dict[str, Any],
    config: Config,
    results_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """Serialise *history* + config metadata to a Parquet file.

    Args:
        history:     Training history dict as returned by Network.train() or
                     DistributedRunner.run().  Keys: round, mean_accuracy,
                     std_accuracy, honest_accuracy, compromised_accuracy,
                     mean_loss, (optionally mean_vacuity, mean_entropy,
                     mean_strength).
        config:      Validated Config object for this experiment.
        results_dir: Directory to write the Parquet file into (created if absent).
        filename:    Stem of the output file; defaults to config.experiment.name.

    Returns:
        Path to the written Parquet file.
    """
    import pandas as pd

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_rounds = len(history["round"])
    if n_rounds == 0:
        raise ValueError("history contains no rounds — nothing to write.")

    # --- build metadata columns (same value repeated for every round) ---
    meta: Dict[str, Any] = {
        "experiment_name": config.experiment.name,
        "condition":       _condition(config),
        "dataset":         _dataset_name(config),
        "aggregation":     config.aggregation.algorithm,
        "backend":         config.backend,
        "byzantine_pct":   config.attack.percentage if config.attack.enabled else 0.0,
        "seed":            config.experiment.seed,
        "mobility_type":   _mobility_type(config),
        "num_nodes":       config.topology.num_nodes,
    }

    # --- per-round metric columns ---
    def _pad(key: str, n: int) -> list:
        vals = history.get(key, [])
        return list(vals) + [None] * (n - len(vals))

    rows: Dict[str, Any] = {
        **{k: [v] * n_rounds for k, v in meta.items()},
        "round":               history["round"],
        "mean_accuracy":       _pad("mean_accuracy", n_rounds),
        "std_accuracy":        _pad("std_accuracy", n_rounds),
        "honest_accuracy":     _pad("honest_accuracy", n_rounds),
        "byzantine_accuracy":  _pad("compromised_accuracy", n_rounds),
        "mean_loss":           _pad("mean_loss", n_rounds),
        # EDL uncertainty metrics (present for evidential_trust runs only)
        "mean_vacuity":        _pad("mean_vacuity", n_rounds),
        "mean_entropy":        _pad("mean_entropy", n_rounds),
        "mean_strength":       _pad("mean_strength", n_rounds),
    }

    df = pd.DataFrame(rows)

    stem = filename or config.experiment.name
    # sanitise: replace spaces and slashes that would break file paths
    stem = stem.replace(" ", "_").replace("/", "-")
    out_path = results_dir / f"{stem}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    return out_path
