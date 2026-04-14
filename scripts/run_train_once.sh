#!/usr/bin/env bash
# run_train_once.sh -> Run train_once.py with cgroup CPU/memory limits
# probably ther is some elegant way to do it but KM3NeT people @ INFN Genova have no slurm account to submit jobs with this cluster and this is a workaround to run the training on the frontend without disturbing to mutch other users

set -euo pipefail

CPUS=15
MEM_GB=32
TIMEOUT_SEC=30000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/train_once.py" ]]; then
    TRAIN_SCRIPT="${SCRIPT_DIR}/train_once.py"
    REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "ERROR: Could not locate train_once.py in ${SCRIPT_DIR}"
    exit 1
fi

LOG_DIR="${REPO_DIR}/logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpus)  CPUS="$2";        shift 2 ;;
        --mem)   MEM_GB="$2";      shift 2 ;;
        --time)  TIMEOUT_SEC="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

if command -v python &>/dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python not found in PATH"
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import tensorflow, h5py, sklearn" &>/dev/null; then
    echo "ERROR: Missing dependencies in current environment"
    echo "Activate your venv and install requirements first"
    exit 1
fi

CPU_QUOTA=$(( CPUS * 100 ))
MEM_LIMIT="${MEM_GB}G"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_once_${TIMESTAMP}.log"

export TF_NUM_INTRAOP_THREADS="${CPUS}"
export TF_NUM_INTEROP_THREADS="1"

cd "${REPO_DIR}"

echo "=========================================================="
echo "  CPU limit   : ${CPUS} core(s) (CPUQuota=${CPU_QUOTA}%)"
echo "  Memory limit: ${MEM_LIMIT}"
echo "  Timeout     : ${TIMEOUT_SEC} s"
echo "  Script      : ${TRAIN_SCRIPT}"
echo "  Log         : ${LOG_FILE}"
echo "=========================================================="

systemd-run \
    --scope \
    --user \
    --unit="ml4phys-train-once-${TIMESTAMP}" \
    -p "CPUQuota=${CPU_QUOTA}%" \
    -p "MemoryMax=${MEM_LIMIT}" \
    -p "CPUWeight=50" \
    -p "RuntimeMaxSec=${TIMEOUT_SEC}" \
    -- "${PYTHON_BIN}" "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
