module purge
module load Anaconda3/2025.06-1
module load CUDA/12.1.1

if [ -n "${REPO_ROOT_OVERRIDE:-}" ]; then
  REPO_ROOT="${REPO_ROOT_OVERRIDE}"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
  if [ -z "$REPO_ROOT" ]; then
    echo "Error: paste failed (no BASH_SOURCE). 'cd' into the vllm clone and:" >&2
    echo "  source serving_scripts/setup.sh" >&2
    echo "or:  export REPO_ROOT=/path/to/vllm" >&2
    exit 1
  fi
fi
VENV_DIR="${REPO_ROOT}/.venv"
echo "REPO_ROOT=$REPO_ROOT"

pick_python_310() {
  for c in python3.13 python3.12 python3.11 python3.10; do
    p="$(command -v "$c" 2>/dev/null)" || continue
    if "$p" -c "import sys; assert sys.version_info >= (3, 10), sys.version" 2>/dev/null; then
      echo "$p"
      return 0
    fi
  done
  p="$(command -v python3 2>/dev/null || true)"
  if [ -n "$p" ] && "$p" -c "import sys; assert sys.version_info >= (3, 10), sys.version" 2>/dev/null; then
    echo "$p"
    return 0
  fi
  return 1
}
PYTHON_VENV_BASE="$(pick_python_310)" || {
  echo "Error: need Python 3.10+ (Anaconda module loaded?) to match vLLM. Got:" >&2
  command -v python3 2>/dev/null && python3 -V >&2
  exit 1
}
echo "Using base interpreter for venv: $PYTHON_VENV_BASE ($("$PYTHON_VENV_BASE" -V 2>&1))"

if [ -x "$VENV_DIR/bin/python" ] && ! "$VENV_DIR/bin/python" -c "import sys; assert sys.version_info >= (3, 10), sys.version" 2>/dev/null; then
  echo "Removing old venv (Python < 3.10): $VENV_DIR" >&2
  rm -rf "$VENV_DIR"
fi

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
(
  cd "$REPO_ROOT" || exit 1
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
  python -m pip install -e .
)
if ! python -c "import vllm._C" 2>/dev/null; then
  echo "Error: vllm._C is missing after install. Precompiled install did not place it." >&2
  echo "Try (network to wheels.vllm.ai may be required):" >&2
  echo "  export VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=nightly" >&2
  echo "  cd $REPO_ROOT && python -m pip install -e ." >&2
  echo "Or build from source (no precompiled; needs nvcc/CMake; slow):" >&2
  echo "  export VLLM_USE_PRECOMPILED=0" >&2
  echo "  cd $REPO_ROOT && python -m pip install -e ." >&2
  exit 1
fi

export VLLM_TARGET_DEVICE=cuda
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --disable-custom-all-reduce