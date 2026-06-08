#!/usr/bin/env bash
# Run all 9 Experiment 2 configs (PAMAP2 multi-dataset validation).
#
# Usage (from repo root, with venv active):
#   bash experiments/paper/dmtt/exp2_multidata/run_exp2.sh
#
# Prerequisite — run once before this script:
#   python scripts/prepare_sociopatterns.py --dataset primary_school --num_nodes 9 --round_duration 30
#   python experiments/paper/dmtt/exp2_multidata/generate_configs.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

CONFIG_DIRS=(
  "c1_static_fedavg"
  "c2_dynamic_fedavg"
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

  if [[ -f "${log}" ]] && grep -q "Final" "${log}" 2>/dev/null; then
    echo "  [SKIP] ${name}"
    ((SKIPPED++)) || true
    return 0
  fi

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
echo " Experiment 2 — PAMAP2 multi-dataset"
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
