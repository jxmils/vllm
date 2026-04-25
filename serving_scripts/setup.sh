module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

VENV_DIR="${REPO_ROOT}/.venv"
echo "REPO_ROOT=$REPO_ROOT"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_VENV_BASE" -m venv "$VENV_DIR"
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
python -m pip install -r "$REPO_ROOT/requirements/cuda.txt"
python -m pip install -r "$REPO_ROOT/requirements/build/cuda.txt"
(
  cd "$REPO_ROOT" || exit 1
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
  python -m pip install -e . ${VLLM_PIP_INSTALL_EXTRA_ARGS:-}
)

export VLLM_TARGET_DEVICE=cuda
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --disable-custom-all-reduce