#!/usr/bin/env python3
"""Diagnose ensure_connected fallback and Byzantine-honest edge injection.

Run from the Murmura project root with the venv activated:
    python tmp/ec_diag.py
"""
import csv
import random
from collections import defaultdict

TRACE = "data/sociopatterns/primary_school_N10_R30s.csv"
NUM_NODES = 10
BYZ_SEED = 42
BYZ_COUNT = 3  # 30% of 10

# --- Load trace ---
contact_count: dict = defaultdict(lambda: defaultdict(int))
raw: dict = defaultdict(list)

with open(TRACE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = int(row["round"])
        ni, nj = int(row["node_i"]), int(row["node_j"])
        raw[r].append((ni, nj))
        contact_count[ni][nj] += 1
        contact_count[nj][ni] += 1

# --- Compute fallback peers (most-frequent contact, same as TraceBasedMobility) ---
fallback: dict = {}
for node in range(NUM_NODES):
    peers = contact_count.get(node, {})
    if peers:
        fallback[node] = max(peers, key=lambda p: peers[p])

print("=== Fallback peer for each node ===")
for n in range(NUM_NODES):
    peer_counts = dict(sorted(contact_count[n].items(), key=lambda x: -x[1]))
    print(f"  node {n:2d} -> fallback {fallback.get(n, 'NONE'):>3}  "
          f"contact counts: {peer_counts}")

# --- Identify Byzantine nodes for seed=42 ---
rng = random.Random(BYZ_SEED)
byz = sorted(rng.sample(range(NUM_NODES), BYZ_COUNT))
honest = [n for n in range(NUM_NODES) if n not in byz]
print(f"\n=== Byzantine nodes (seed={BYZ_SEED}, {BYZ_COUNT}/{NUM_NODES}): {byz} ===")
print(f"    Honest nodes: {honest}")

# --- Check isolation and synthetic edges at rounds 28-40 ---
print("\n=== Isolated nodes and synthetic edges (rounds 28-40) ===")
for r in range(28, 41):
    adj: dict = defaultdict(list)
    for ni, nj in raw.get(r, []):
        adj[ni].append(nj)
        adj[nj].append(ni)
    isolated = [n for n in range(NUM_NODES) if not adj[n]]
    synth_edges = [(n, fallback[n]) for n in isolated if n in fallback]
    # Flag dangerous edges (Byzantine <-> Honest)
    dangerous = [
        (a, b) for a, b in synth_edges
        if (a in byz) != (b in byz)  # one Byzantine, one honest
    ]
    marker = " <-- BYZANTINE-HONEST EDGE" if dangerous else ""
    natural_contacts = sum(len(adj[n]) for n in range(NUM_NODES)) // 2
    print(f"  round {r:2d}: natural_contacts={natural_contacts:2d}  "
          f"isolated={isolated}  synth={synth_edges}{marker}")

# --- Summary: which honest nodes have Byzantine fallback peers? ---
print("\n=== Honest nodes with Byzantine fallback peers ===")
for h in honest:
    fp = fallback.get(h)
    tag = " <-- DANGER" if fp in byz else ""
    print(f"  honest node {h} -> fallback {fp}{tag}")

print("\n=== Byzantine nodes with honest fallback peers ===")
for b in byz:
    fp = fallback.get(b)
    tag = " <-- WILL INJECT INTO HONEST NODE" if fp in honest else ""
    print(f"  byzantine node {b} -> fallback {fp}{tag}")
