#!/usr/bin/env bash
# run_train.sh — generic launcher for any train_once*.py script.
#
# Usage:
#   bash run_train.sh train_once_old1.py
#   bash run_train.sh train_once_old1.py --cpus 8 --mem 16 --time 20000
#   FORCE_RETRAIN=1 bash run_train.sh train_once_old1.py
#
# Options:
#   --cpus  N   CPU cores to allow (CPUQuota = N*100%)  [default: 15]
#   --mem   G   Memory limit in GB                      [default: 32]
#   --time  S   Timeout in seconds                      [default: 30000]

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse positional argument (python script name)
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_train.sh <train_script.py> [--cpus N] [--mem G] [--time S]"
    exit 1
fi

TRAIN_SCRIPT_NAME="$1"
shift

CPUS=15
MEM_GB=32
TIMEOUT_SEC=30000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpus)  CPUS="$2";        shift 2 ;;
        --mem)   MEM_GB="$2";      shift 2 ;;
        --time)  TIMEOUT_SEC="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/${TRAIN_SCRIPT_NAME}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Detect python
# ---------------------------------------------------------------------------
if command -v python &>/dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python not found in PATH"
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import tensorflow, h5py, sklearn" &>/dev/null; then
    echo "ERROR: Missing dependencies — activate your venv and install requirements first"
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
CPU_QUOTA=$(( CPUS * 100 ))
MEM_LIMIT="${MEM_GB}G"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STEM="${TRAIN_SCRIPT_NAME%.py}"
LOG_FILE="${LOG_DIR}/${STEM}_${TIMESTAMP}.log"

export TF_NUM_INTRAOP_THREADS="${CPUS}"
export TF_NUM_INTEROP_THREADS="1"

cd "${REPO_DIR}"

echo "=========================================================="
echo "  Script      : ${TRAIN_SCRIPT}"
echo "  CPU limit   : ${CPUS} core(s) (CPUQuota=${CPU_QUOTA}%)"
echo "  Memory limit: ${MEM_LIMIT}"
echo "  Timeout     : ${TIMEOUT_SEC} s"
echo "  Log         : ${LOG_FILE}"
echo "=========================================================="

systemd-run \
    --scope \
    --user \
    --unit="ml4phys-${STEM}-${TIMESTAMP}" \
    -p "CPUQuota=${CPU_QUOTA}%" \
    -p "MemoryMax=${MEM_LIMIT}" \
    -p "CPUWeight=50" \
    -p "RuntimeMaxSec=${TIMEOUT_SEC}" \
    -- "${PYTHON_BIN}" "${TRAIN_SCRIPT}" >> "${LOG_FILE}" 2>&1
