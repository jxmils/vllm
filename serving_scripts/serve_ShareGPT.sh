#!/usr/bin/env bash
#SBATCH --job-name=vllm-bench-sharegpt
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -euo pipefail

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

python -m pip install -U pip datasets

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/sharegpt_sample.json}"
PROMPTS_OUT="${PROMPTS_OUT:-${REPO_ROOT}/datasets/sharegpt_prompts.jsonl}"
MAX_ROWS="${MAX_ROWS:-100000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-10.0}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-100}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"

mkdir -p "$(dirname "${DATASET_PATH}")"
mkdir -p "$(dirname "${PROMPTS_OUT}")"

# Use the repository dataset preparation script to keep data formatting consistent.
python "${REPO_ROOT}/datasets/prepare_sharegpt_vicuna_prompts.py" \
  --max-rows "${MAX_ROWS}" \
  --raw-out "${DATASET_PATH}" \
  --prompts-out "${PROMPTS_OUT}"

cd "${REPO_ROOT}"

vllm bench serve \
  --backend vllm \
  --host "${HOST}" \
  --port "${PORT}" \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL_ID}" \
  --dataset-name sharegpt \
  --dataset-path "${DATASET_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate "${REQUEST_RATE}" \
  --burstiness "${BURSTINESS}" \
  --seed "${SEED}"
