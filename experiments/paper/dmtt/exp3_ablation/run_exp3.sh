#!/usr/bin/env bash
# Run all 18 Experiment 3 ablation configs.
#
# Usage (from repo root, with venv active):
#   bash experiments/paper/dmtt/exp3_ablation/run_exp3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

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
echo " Experiment 3 — Ablation study"
echo " Results → ${RESULTS_DIR}"
echo "========================================"
echo ""

for cfg in "${SCRIPT_DIR}/configs"/*.yaml; do
  run_config "${cfg}"
done

echo ""
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
