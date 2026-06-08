#!/usr/bin/env bash
# Run Experiment 4 (trust-score evolution).
#
# Usage (from repo root, with venv active):
#   bash experiments/paper/dmtt/exp4_trust_evolution/run_exp4.sh
#
# Produces:
#   results/trust_log.jsonl          — per-round trust state for each node
#   results/exp4_dmtt_byz30_s42.log  — console log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

CFG="${SCRIPT_DIR}/exp4_dmtt_byz30_s42.yaml"
LOG="${RESULTS_DIR}/exp4_dmtt_byz30_s42.log"

echo "========================================"
echo " Experiment 4 — Trust-score evolution"
echo " Config:  ${CFG}"
echo " Results: ${RESULTS_DIR}"
echo "========================================"
echo ""

# Clean stale IPC dir
IPC_DIR="/tmp/murmura_exp4_trust_byz30_s42"
[[ -d "${IPC_DIR}" ]] && rm -rf "${IPC_DIR}"

murmura run "${CFG}" --results-dir "${RESULTS_DIR}" 2>&1 | tee "${LOG}"

echo ""
echo "Trust log written to: ${RESULTS_DIR}/trust_log.jsonl"
echo "Run analysis:  python experiments/paper/dmtt/exp4_trust_evolution/analyze_exp4.py"
