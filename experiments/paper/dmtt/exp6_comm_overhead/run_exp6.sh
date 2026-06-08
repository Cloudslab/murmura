#!/usr/bin/env bash
# Run Experiment 6 (communication overhead measurement).
#
# Usage (from repo root, with venv active):
#   bash experiments/paper/dmtt/exp6_comm_overhead/run_exp6.sh
#
# Runs 2 configs (C2 baseline + C3 with byte logging) for 20 rounds each.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TOTAL=0
FAILED=0

run_config() {
  local cfg="$1"
  local name
  name="$(basename "${cfg}" .yaml)"
  local log="${RESULTS_DIR}/${name}.log"

  echo "  [RUN ] ${name}"
  local start=$SECONDS

  local ipc_tag
  ipc_tag="$(grep 'ipc_dir' "${cfg}" 2>/dev/null | sed 's/.*"\(.*\)".*/\1/' || true)"
  if [[ -n "${ipc_tag}" && -d "${ipc_tag}" ]]; then
    rm -rf "${ipc_tag}"
  fi

  if murmura run "${cfg}" --results-dir "${RESULTS_DIR}" > "${log}" 2>&1; then
    echo "         done in $(( SECONDS - start ))s"
  else
    echo "  [FAIL] ${name} — see ${log}"
    ((FAILED++)) || true
  fi

  ((TOTAL++)) || true
}

echo "========================================"
echo " Experiment 6 — Communication overhead"
echo " Results → ${RESULTS_DIR}"
echo "========================================"
echo ""

run_config "${SCRIPT_DIR}/c2_comm_byz30_s42.yaml"
run_config "${SCRIPT_DIR}/c3_comm_byz30_s42.yaml"

echo ""
echo "========================================"
echo " Summary: ran ${TOTAL}, failed ${FAILED}"
echo "========================================"
echo "Run analysis:  python experiments/paper/dmtt/exp6_comm_overhead/analyze_exp6.py"

if (( FAILED > 0 )); then
  exit 1
fi
