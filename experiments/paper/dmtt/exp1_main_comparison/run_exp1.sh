#!/usr/bin/env bash
# Run all 54 Experiment 1 configs sequentially.
#
# Usage (from repo root, with venv active):
#   bash experiments/paper/dmtt/exp1_main_comparison/run_exp1.sh
#
# Logs are written to experiments/paper/dmtt/exp1_main_comparison/results/<name>.log
# An experiment is skipped if its log already contains a "Final" accuracy line,
# so re-running this script after an interruption resumes from where it left off.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Order: simulation first (instant), then distributed (slow, sequential)
CONFIG_DIRS=(
  "c1_static_fedavg"
  "c2_dynamic_fedavg"
  "dyn_krum"
  "dyn_balance"
  "dyn_ubar"
  "c3_dmtt"
)

TOTAL=0
SKIPPED=0
FAILED=0

run_config() {
  local cfg="$1"
  local name
  name="$(basename "${cfg}" .yaml)"
  local log="${RESULTS_DIR}/${name}.log"

  # Skip if already completed
  if [[ -f "${log}" ]] && grep -q "Final" "${log}" 2>/dev/null; then
    echo "  [SKIP] ${name} (log exists with results)"
    ((SKIPPED++)) || true
    return 0
  fi

  echo "  [RUN ] ${name}"
  local start=$SECONDS

  # Clean up any stale IPC sockets from a previous partial run
  local ipc_tag
  ipc_tag="$(grep 'ipc_dir' "${cfg}" 2>/dev/null | sed 's/.*"\(.*\)".*/\1/' || true)"
  if [[ -n "${ipc_tag}" && -d "${ipc_tag}" ]]; then
    rm -rf "${ipc_tag}"
  fi

  if murmura run "${cfg}" --results-dir "${RESULTS_DIR}" > "${log}" 2>&1; then
    local elapsed=$(( SECONDS - start ))
    echo "         done in ${elapsed}s"
  else
    echo "  [FAIL] ${name} — see ${log}"
    ((FAILED++)) || true
  fi

  ((TOTAL++)) || true
}

echo "========================================"
echo " Experiment 1 — Main comparison table"
echo " Results → ${RESULTS_DIR}"
echo "========================================"
echo ""

for dir in "${CONFIG_DIRS[@]}"; do
  echo "--- ${dir} ---"
  for cfg in "${SCRIPT_DIR}/${dir}"/*.yaml; do
    run_config "${cfg}"
  done
  echo ""
done

echo "========================================"
echo " Summary"
echo "   Ran:     ${TOTAL}"
echo "   Skipped: ${SKIPPED}"
echo "   Failed:  ${FAILED}"
echo "========================================"

if (( FAILED > 0 )); then
  echo "Some experiments failed. Check logs in ${RESULTS_DIR}/"
  exit 1
fi
