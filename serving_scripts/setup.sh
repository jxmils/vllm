module purge
module load Anaconda3/2025.06-1
module load CUDA/12.1.1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

PYTHON_PATH="$(command -v python)"
echo "Using python: $PYTHON_PATH"

EXPECTED_PYTHON="$VENV_DIR/bin/python"
if [ "$PYTHON_PATH" != "$EXPECTED_PYTHON" ]; then
  echo "Error: python did not resolve to venv interpreter." >&2
  echo "Expected: $EXPECTED_PYTHON" >&2
  echo "Got:      $PYTHON_PATH" >&2
  exit 1
fi

python -m pip install -U pip
python -m pip install -r requirements/cuda.txt

export VLLM_TARGET_DEVICE=cuda
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --disable-custom-all-reduce