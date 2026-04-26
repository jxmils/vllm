#!/usr/bin/env bash
#SBATCH --job-name=vllm-host-qwen3-30b
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -euo pipefail

export HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export WORKER_NODES=$(scontrol show hostnames $SLURM_NODELIST | tail -n+2)
export HEAD_NODE_IP=$(dig +short ${HEAD_NODE})
export RAY_PORT=6378
export RAY_ADDRESS=$HEAD_NODE_IP:$RAY_PORT

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
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-${HEAD_NODE_IP}}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
EP="${EP:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT.sh"

SERVER_STEP_PID=""
HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

cleanup() {
  if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
  fi
  if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
    kill "${HEAD_RAY_PID}" 2>/dev/null || true
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
  fi
  for pid in ${WORKER_RAY_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "Starting head node ${HEAD_NODE}..."
srun \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  bash -lc "export VLLM_HOST_IP=${HEAD_NODE_IP}; ray start --block --head --node-ip-address=${HEAD_NODE_IP} --port=${RAY_PORT}" &
HEAD_RAY_PID=$!
sleep 20

if [ -n "${WORKER_NODES}" ]; then
  echo "Starting worker nodes..."
  for WORKER in ${WORKER_NODES}; do
    WORKER_IP=$(dig +short "${WORKER}" | head -n1)
    echo "Starting worker node: ${WORKER} with IP ${WORKER_IP}"
    srun \
      --nodelist "${WORKER}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --gpus-per-task="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      bash -lc "export VLLM_HOST_IP=${WORKER_IP}; ray start --block --address=${HEAD_NODE_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP}" &
    WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
  done
  sleep 20
fi

echo "Checking cluster status..."
srun \
  --overlap \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  ray status

echo "Starting vLLM server on head node..."
srun \
  --overlap \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  bash -lc "source \"${VENV_DIR}/bin/activate\" && python -m vllm.entrypoints.openai.api_server \
  --model \"${MODEL_ID}\" \
  --host \"${HOST}\" \
  --port \"${PORT}\" \
  --distributed-executor-backend ray \
  --tensor-parallel-size \"${TP}\" \
  --enable-expert-parallel \
  --additional-config '{\"sharding\":{\"sharding_strategy\":{\"tensor_parallelism\":${TP},\"expert_parallelism\":${EP}}}}' \
  --enforce-eager \
  --disable-custom-all-reduce" &
SERVER_STEP_PID=$!
echo "Started vLLM server step (pid=${SERVER_STEP_PID}). Waiting for /health ..."

until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi
  sleep 5
done
echo "Server is healthy. Running ${SERVE_SCRIPT} ..."

HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" bash "${SERVE_SCRIPT}"