#!/usr/bin/env bash
# Run all Experiment 5 configs (sensitivity analysis).
#
# Usage (from repo root, with venv active):
#   bash experiments/paper/dmtt/exp5_sensitivity/run_exp5.sh
#
# You can also run each part independently:
#   bash experiments/paper/dmtt/exp5_sensitivity/run_exp5.sh het   # heterogeneity only
#   bash experiments/paper/dmtt/exp5_sensitivity/run_exp5.sh mob   # mobility only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

PART="${1:-all}"  # all | het | mob

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

run_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo "  (no configs in ${dir})"
    return
  fi
  echo "--- ${dir##*/} ---"
  for cfg in "${dir}"/*.yaml; do
    [[ -f "${cfg}" ]] && run_config "${cfg}"
  done
  echo ""
}

echo "========================================"
echo " Experiment 5 — Sensitivity analysis"
echo " Part: ${PART}"
echo " Results → ${RESULTS_DIR}"
echo "========================================"
echo ""

if [[ "${PART}" == "all" || "${PART}" == "het" ]]; then
  echo "=== Part (a): Heterogeneity sweep ==="
  run_dir "${SCRIPT_DIR}/heterogeneity/c2_dynamic_fedavg"
  run_dir "${SCRIPT_DIR}/heterogeneity/c3_dmtt"
fi

if [[ "${PART}" == "all" || "${PART}" == "mob" ]]; then
  echo "=== Part (b): Mobility sweep ==="
  run_dir "${SCRIPT_DIR}/mobility/c2_dynamic_fedavg"
  run_dir "${SCRIPT_DIR}/mobility/c3_dmtt"
fi

echo "========================================"
echo " Summary"
echo "   Ran:     ${TOTAL}"
echo "   Skipped: ${SKIPPED}"
echo "   Failed:  ${FAILED}"
echo "========================================"

if (( FAILED > 0 )); then
  exit 1
fi
