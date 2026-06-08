#!/usr/bin/env python3
"""Download and preprocess a SocioPatterns contact-trace dataset.

Produces a CSV consumed by TraceBasedMobility:
    round,node_i,node_j
    0,0,3
    ...

Usage examples:
    # Primary School — recommended starting point (wearable RFID sensors)
    python scripts/prepare_sociopatterns.py --dataset primary_school

    # Hospital Ward — smaller, faster to prototype
    python scripts/prepare_sociopatterns.py --dataset hospital_ward --num_nodes 10

    # Custom round duration (default 30 s to match FL config)
    python scripts/prepare_sociopatterns.py --dataset primary_school --round_duration 60

    # Keep a specific set of node IDs rather than auto-selecting top-N
    python scripts/prepare_sociopatterns.py --dataset primary_school --node_ids 1540,1541,1542,...

Outputs (written to data/sociopatterns/):
    {dataset}_N{n}_R{r}s.csv   — contact-trace CSV for TraceBasedMobility
    {dataset}_N{n}_R{r}s.json  — metadata (node mapping, stats, round count)
"""

import argparse
import csv
import gzip
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# SocioPatterns datasets are freely available at sociopatterns.org.
# Each entry lists one or more candidate download URLs and the raw CSV format.
#
# Raw format varies slightly per dataset:
#   "t_i_j"       → columns: t  i  j         (no class labels)
#   "t_i_j_ci_cj" → columns: t  i  j  Ci  Cj  (class labels appended)
#
# Time is in seconds from the start of the observation window.
# Temporal resolution is 20 seconds for all SocioPatterns iMote datasets.

DATASETS = {
    "primary_school": {
        "urls": [
            "http://www.sociopatterns.org/files/datasets/contact-primary-school.csv.gz",
            "https://snap.stanford.edu/data/primaryschool.csv.gz",
        ],
        "format": "t_i_j_ci_cj",
        "description": "Primary school RFID contact trace (242 nodes, 2 days, 20 s resolution)",
        "citation": (
            "Stehlé et al. (2011). High-resolution measurements of face-to-face contact "
            "patterns in a primary school. PLOS ONE."
        ),
    },
    "hospital_ward": {
        "urls": [
            "http://www.sociopatterns.org/files/datasets/detailed-contacts-hospital-ward-2010.csv.gz",
        ],
        "format": "t_i_j_ci_cj",
        "description": "Hospital ward RFID contact trace (~75 nodes, ~4 days, 20 s resolution)",
        "citation": (
            "Vanhems et al. (2013). Estimating Potential Infection Transmission Routes "
            "in Hospital Wards Using Wearable Proximity Sensors. PLOS ONE."
        ),
    },
    "high_school_2012": {
        "urls": [
            "http://www.sociopatterns.org/files/datasets/contact-high-school-2012.csv.gz",
        ],
        "format": "t_i_j_ci_cj",
        "description": "High school RFID contact trace (~327 nodes, 5 days, 20 s resolution)",
        "citation": (
            "Fournet & Barrat (2014). Contact patterns among high school students. PLOS ONE."
        ),
    },
}

RAW_DIR  = Path("data/sociopatterns/raw")
OUT_DIR  = Path("data/sociopatterns")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(dataset: str) -> Path:
    """Download the raw gzipped CSV for *dataset*, return local path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{dataset}.csv.gz"

    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return dest

    info = DATASETS[dataset]
    for url in info["urls"]:
        print(f"  Downloading {url} …")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            print(f"  Saved to {dest}")
            return dest
        except Exception as exc:
            print(f"  Failed ({exc}), trying next URL …")

    raise RuntimeError(
        f"Could not download {dataset}.\n"
        "Please download manually from http://www.sociopatterns.org/datasets/ "
        f"and save the .csv.gz file to {dest}"
    )


# ---------------------------------------------------------------------------
# Parse raw CSV
# ---------------------------------------------------------------------------

def parse_raw(gz_path: Path, fmt: str) -> List[Tuple[int, int, int]]:
    """Return list of (t_seconds, node_i, node_j) from the raw gzipped CSV."""
    contacts = []
    with gzip.open(gz_path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t  = int(parts[0])
            ni = int(parts[1])
            nj = int(parts[2])
            if ni != nj:
                contacts.append((t, ni, nj))
    return contacts


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------

def select_nodes(
    contacts: List[Tuple[int, int, int]],
    num_nodes: int,
    node_ids: Optional[List[int]] = None,
) -> List[int]:
    """Return *num_nodes* node IDs to include in the subgraph.

    If *node_ids* is given, use those exactly (after validating they appear in
    the trace).  Otherwise pick the top-*num_nodes* nodes by total contact count
    (number of (t, j) pairs involving each node), which tends to produce a
    well-connected subgraph.
    """
    if node_ids is not None:
        present = {ni for _, ni, nj in contacts} | {nj for _, ni, nj in contacts}
        missing = set(node_ids) - present
        if missing:
            raise ValueError(f"Requested node IDs not found in trace: {missing}")
        return sorted(node_ids)[:num_nodes]

    # Count contacts per node
    count: Dict[int, int] = defaultdict(int)
    for _, ni, nj in contacts:
        count[ni] += 1
        count[nj] += 1

    top = sorted(count, key=lambda n: -count[n])[:num_nodes]
    return sorted(top)


# ---------------------------------------------------------------------------
# Discretise into FL rounds
# ---------------------------------------------------------------------------

def discretise(
    contacts: List[Tuple[int, int, int]],
    selected: List[int],
    round_duration: int,
    min_contacts: int = 0,
) -> Tuple[Dict[int, Set[Tuple[int, int]]], int, int]:
    """Map raw contacts to FL rounds.

    When min_contacts > 0, rounds with fewer than that many edges are dropped
    and the remaining rounds are renumbered 0, 1, 2, … This removes dead
    periods (e.g. before/after school hours) so the trace always has meaningful
    contact density.

    Returns:
        edges_per_round: {round_idx: {(i_remapped, j_remapped), …}}
        total_rounds:    number of rounds in the output (after any filtering)
        dropped_rounds:  number of rounds removed by the min_contacts filter
    """
    # Remap original IDs to 0 … num_nodes-1
    id_map = {orig: new for new, orig in enumerate(selected)}
    selected_set = set(selected)

    # Shift time to start at 0
    times = [t for t, ni, nj in contacts if ni in selected_set and nj in selected_set]
    if not times:
        raise ValueError("No contacts found among the selected nodes.")
    t_min = min(times)

    edges_per_round: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for t, ni, nj in contacts:
        if ni not in selected_set or nj not in selected_set:
            continue
        r  = (t - t_min) // round_duration
        a, b = id_map[ni], id_map[nj]
        if a > b:
            a, b = b, a
        edges_per_round[r].add((a, b))

    raw_total = max(edges_per_round) + 1 if edges_per_round else 0

    if min_contacts > 0:
        # Drop rounds below density threshold and renumber survivors.
        kept: Dict[int, Set[Tuple[int, int]]] = {}
        for old_r in range(raw_total):
            edges = edges_per_round.get(old_r, set())
            if len(edges) >= min_contacts:
                kept[len(kept)] = edges
        dropped = raw_total - len(kept)
        return kept, len(kept), dropped

    return dict(edges_per_round), raw_total, 0


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_csv(
    edges_per_round: Dict[int, Set[Tuple[int, int]]],
    total_rounds: int,
    out_path: Path,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["round", "node_i", "node_j"])
        for r in range(total_rounds):
            for (i, j) in sorted(edges_per_round.get(r, set())):
                writer.writerow([r, i, j])


def write_metadata(
    dataset: str,
    selected: List[int],
    num_nodes: int,
    round_duration: int,
    total_rounds: int,
    edges_per_round: Dict[int, Set[Tuple[int, int]]],
    out_path: Path,
    min_contacts: int = 0,
    dropped_rounds: int = 0,
) -> None:
    edge_counts = [len(edges_per_round.get(r, set())) for r in range(total_rounds)]
    isolated_rounds = sum(
        1 for r in range(total_rounds)
        if any(
            not any(i in (a, b) for a, b in edges_per_round.get(r, set()))
            for i in range(num_nodes)
        )
    )
    meta = {
        "dataset": dataset,
        "description": DATASETS[dataset]["description"],
        "citation": DATASETS[dataset]["citation"],
        "num_nodes": num_nodes,
        "round_duration_s": round_duration,
        "total_rounds": total_rounds,
        "original_node_ids": selected,
        "avg_edges_per_round": round(sum(edge_counts) / max(total_rounds, 1), 2),
        "min_edges_per_round": min(edge_counts, default=0),
        "max_edges_per_round": max(edge_counts, default=0),
        "rounds_with_isolated_nodes": isolated_rounds,
        "min_contacts_filter": min_contacts,
        "dropped_sparse_rounds": dropped_rounds,
        "note": (
            "TraceBasedMobility wraps round_idx modulo total_rounds, so a 50-round "
            "FL experiment reuses the trace from the beginning if total_rounds < 50."
        ),
    }
    with open(out_path, "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="primary_school",
        help="Which SocioPatterns dataset to use (default: primary_school)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=10,
        help="Number of FL nodes to extract (default: 10)",
    )
    parser.add_argument(
        "--round_duration",
        type=int,
        default=30,
        help="FL round duration in seconds; contacts are binned into windows of this size (default: 30)",
    )
    parser.add_argument(
        "--min-contacts",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Drop rounds with fewer than N contact edges; survivors are renumbered 0,1,2,… "
            "This removes dead periods (e.g. before/after school hours) so the trace always "
            "has meaningful connectivity. Adds a _mcN suffix to the output filename. "
            "(default: 0 = keep all rounds)"
        ),
    )
    parser.add_argument(
        "--node_ids",
        type=str,
        default=None,
        help="Comma-separated list of specific original node IDs to use instead of auto-selection",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default=None,
        help="Path to an already-downloaded .csv.gz file. Skips the download step entirely.",
    )
    args = parser.parse_args()

    node_ids = None
    if args.node_ids:
        node_ids = [int(x) for x in args.node_ids.split(",")]

    mc_tag = f"_mc{args.min_contacts}" if args.min_contacts > 0 else ""
    stem = f"{args.dataset}_N{args.num_nodes}_R{args.round_duration}s{mc_tag}"
    csv_path  = OUT_DIR / f"{stem}.csv"
    meta_path = OUT_DIR / f"{stem}.json"

    print(f"Dataset:        {args.dataset}")
    print(f"Nodes:          {args.num_nodes}")
    print(f"Round duration: {args.round_duration} s")
    print(f"Min contacts:   {args.min_contacts} (0 = keep all rounds)")
    print(f"Output CSV:     {csv_path}")
    print()

    print("Step 1/4  Download")
    if args.raw_file:
        gz_path = Path(args.raw_file)
        if not gz_path.exists():
            raise FileNotFoundError(f"--raw-file not found: {gz_path}")
        # Copy into the canonical raw location so the cache check works next time
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = RAW_DIR / f"{args.dataset}.csv.gz"
        if gz_path.resolve() != dest.resolve():
            shutil.copy2(gz_path, dest)
        gz_path = dest
        print(f"  Using provided file: {gz_path}")
    else:
        gz_path = download(args.dataset)

    print("Step 2/4  Parse raw CSV")
    contacts = parse_raw(gz_path, DATASETS[args.dataset]["format"])
    print(f"  {len(contacts):,} raw contact events parsed")

    print("Step 3/4  Select nodes and discretise")
    selected = select_nodes(contacts, args.num_nodes, node_ids)
    print(f"  Selected node IDs: {selected}")
    edges_per_round, total_rounds, dropped = discretise(
        contacts, selected, args.round_duration, args.min_contacts
    )
    print(f"  {total_rounds} rounds  ({total_rounds * args.round_duration / 3600:.1f} h of active trace)")
    if dropped:
        print(f"  Dropped {dropped} sparse rounds (< {args.min_contacts} contacts each)")

    print("Step 4/4  Write output")
    write_csv(edges_per_round, total_rounds, csv_path)
    write_metadata(
        args.dataset, selected, args.num_nodes,
        args.round_duration, total_rounds, edges_per_round, meta_path,
        min_contacts=args.min_contacts, dropped_rounds=dropped,
    )

    print()
    print(f"Done.")
    print(f"  CSV:      {csv_path}")
    print(f"  Metadata: {meta_path}")
    print()
    print("Use in experiment configs:")
    print(f"  trace_mobility:")
    print(f"    trace_path: \"{csv_path}\"")
    print(f"    num_nodes: {args.num_nodes}")
    print(f"    ensure_connected: true")


if __name__ == "__main__":
    main()
