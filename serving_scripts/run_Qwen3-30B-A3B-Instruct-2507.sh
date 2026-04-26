#!/usr/bin/env bash
#SBATCH --job-name=vllm-host-qwen3-30b
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -euo pipefail

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
VENV_DIR="${REPO_ROOT}/.venv"
echo "REPO_ROOT=${REPO_ROOT}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

PYTHON_PATH="$(command -v python)"
EXPECTED_PYTHON="${VENV_DIR}/bin/python"
echo "Using python: ${PYTHON_PATH}"
if [ "${PYTHON_PATH}" != "${EXPECTED_PYTHON}" ]; then
  echo "Error: python did not resolve to venv interpreter." >&2
  echo "Expected: ${EXPECTED_PYTHON}" >&2
  echo "Got:      ${PYTHON_PATH}" >&2
  exit 1
fi

python -m pip install -U pip
python -m pip install -r "${REPO_ROOT}/requirements/cuda.txt"
python -m pip install -r "${REPO_ROOT}/requirements/build/cuda.txt"
(
  cd "${REPO_ROOT}" || exit 1
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
  python -m pip install -e . ${VLLM_PIP_INSTALL_EXTRA_ARGS:-}
)

export VLLM_TARGET_DEVICE=cuda
# DeepGEMM can fail on some ARC nodes/toolchains; disable by default.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
EP="${EP:-1}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT.sh"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  --additional-config "{\"sharding\":{\"sharding_strategy\":{\"tensor_parallelism\":${TP},\"expert_parallelism\":${EP}}}}" \
  --enforce-eager \
  --disable-custom-all-reduce &
SERVER_PID=$!
echo "Started vLLM server (pid=${SERVER_PID}). Waiting for /health ..."

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_PID}" || true
    exit 1
  fi
  sleep 5
done
echo "Server is healthy. Running ${SERVE_SCRIPT} ..."

HOST=127.0.0.1 PORT="${PORT}" MODEL_ID="${MODEL_ID}" bash "${SERVE_SCRIPT}"
