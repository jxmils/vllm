#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=r32_sp128_sd128_tp4_pp1_ep_qwen3_nsys
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

###############################################################################
# Clean 2-node vLLM/Ray/Nsight debug script
#
# Matches the manual-working direction:
#   - 2 nodes x 2 H100 = 4 GPUs total
#   - TP=4
#   - PP=1
#   - expert parallel enabled
#   - all2all backend = allgather_reducescatter
#   - no DeepEP package mutation
#
# For real traces, increase SBATCH time and final Nsight waits.
###############################################################################

REPO="/data/engs-glass/catz0932/inference-traces/vllm"
VENV_DIR="${REPO}/.venv"
TRACE_BASE="${REPO}/results"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6378}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-64}}"

TP="${TP:-4}"
PP="${PP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-allgather_reducescatter}"

# Debug workload defaults.
SP="${SP:-64}"
SD="${SD:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-100}"

# Short debug waits. Override from sbatch env for real runs.
RAY_HEAD_START_SLEEP_S="${RAY_HEAD_START_SLEEP_S:-25}"
RAY_WORKER_START_SLEEP_S="${RAY_WORKER_START_SLEEP_S:-25}"
RAY_READY_TIMEOUT_S="${RAY_READY_TIMEOUT_S:-240}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-600}"
SERVER_SHUTDOWN_TIMEOUT_S="${SERVER_SHUTDOWN_TIMEOUT_S:-120}"

# For 15-min debug, keep these low. For real profiling use 900/900.
WORKER_NSYS_FINALIZE_WAIT_S="${WORKER_NSYS_FINALIZE_WAIT_S:-0}"
WORKER_NSYS_FLUSH_WAIT_S="${WORKER_NSYS_FLUSH_WAIT_S:-0}"
NSYS_STATS_ENABLE="${NSYS_STATS_ENABLE:-0}"

export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-900}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

export NSYS_ENABLE="${NSYS_ENABLE:-1}"
export NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-1}"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"

mkdir -p "${TRACE_RUN_DIR}/nccl_logs"
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"
mkdir -p "${TRACE_RUN_DIR}/manual_logs"
mkdir -p "${TRACE_RUN_DIR}/responses"
mkdir -p "${TRACE_RUN_DIR}/nsys_stats"

export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

echo "=== clean TP4 PP1 EP vLLM/Ray/Nsight debug run ==="
echo "date=$(date -Is)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "REPO=${REPO}"
echo "MODEL_ID=${MODEL_ID}"
echo "TP=${TP} PP=${PP} GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "SP=${SP} SD=${SD} NUM_PROMPTS=${NUM_PROMPTS}"
echo "NSYS_ENABLE=${NSYS_ENABLE} NSYS_PROFILE_WORKERS=${NSYS_PROFILE_WORKERS}"
echo "WORKER_NSYS_FINALIZE_WAIT_S=${WORKER_NSYS_FINALIZE_WAIT_S}"
echo "WORKER_NSYS_FLUSH_WAIT_S=${WORKER_NSYS_FLUSH_WAIT_S}"
echo "NSYS_STATS_ENABLE=${NSYS_STATS_ENABLE}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
source "${VENV_DIR}/bin/activate"

echo "python=$(which python)"
echo "ray=$(which ray)"
echo "nsys=$(which nsys || true)"
nsys --version || true

HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
WORKER_NODES="$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)"

resolve_ip() {
  local node="$1"
  local ip
  ip="$(getent hosts "${node}" 2>/dev/null | awk '{print $1}' | grep -E '^[0-9]+\.' | head -n1 || true)"
  if [ -z "${ip}" ]; then
    ip="$(ssh "${node}" "hostname -I | awk '{print \$1}'")"
  fi
  printf '%s' "${ip}"
}

iface_for_ip_local() {
  local ip="$1"
  ip -o -4 addr show | awk -v target="${ip}" '
    {
      split($4, a, "/")
      if (a[1] == target) {
        print $2
        exit
      }
    }'
}

HEAD_IP="$(resolve_ip "${HEAD_NODE}")"
WORKER_IPS=""
for n in ${WORKER_NODES}; do
  ip="$(resolve_ip "${n}")"
  WORKER_IPS="${WORKER_IPS} ${ip}"
done
WORKER_IPS="$(echo "${WORKER_IPS}" | xargs)"

HOST="${HEAD_IP}"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

HEAD_IFACE="$(iface_for_ip_local "${HEAD_IP}")"
if [ -z "${HEAD_IFACE}" ]; then
  echo "ERROR: could not find local interface for HEAD_IP=${HEAD_IP}" >&2
  ip -o -4 addr show >&2 || true
  exit 1
fi

export GLOO_SOCKET_IFNAME="${HEAD_IFACE}"
export NCCL_SOCKET_IFNAME="${HEAD_IFACE}"
export VLLM_HOST_IP="${HEAD_IP}"
export RAY_ADDRESS="${RAY_ADDRESS}"

echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
echo "HEAD_IP=${HEAD_IP}"
echo "WORKER_IPS=${WORKER_IPS}"
echo "HEAD_IFACE=${HEAD_IFACE}"
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"

cleanup() {
  set +e
  echo "=== cleanup ==="

  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "stopping vLLM server pid=${SERVER_PID}"
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    sleep 10
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi

  echo "copying any available Nsight files before Ray shutdown"
  copy_all_nsight || true

  echo "stopping Ray"
  for n in ${WORKER_NODES} ${HEAD_NODE}; do
    ssh "${n}" "
      set +e
      source '${VENV_DIR}/bin/activate' 2>/dev/null || true
      ray stop --force || true
      pkill -u '$USER' -f 'ray start --block' || true
      pkill -u '$USER' -f 'gcs_server|raylet|plasma_store|worker_process|RayWorkerProc|Worker_TP|VLLM::EngineCore|vllm.entrypoints.openai.api_server|nsys' || true
    " || true
  done
}
trap cleanup EXIT

copy_all_nsight() {
  set +e
  for n in ${HEAD_NODE} ${WORKER_NODES}; do
    echo "=== copy Nsight from ${n} ==="
    ssh "${n}" "
      set +e
      SHORT=\$(hostname -s)
      SRC=/tmp/vray-${SLURM_JOB_ID}-\${SHORT}/session_latest/logs/nsight
      DEST='${TRACE_RUN_DIR}/ray_worker_nsight/'\${SHORT}
      mkdir -p \"\$DEST\"

      echo SRC=\$SRC
      find \"\$SRC\" -maxdepth 1 -type f \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
        -printf '%p %s bytes\n' 2>/dev/null | sort || true

      cp -av \"\$SRC\"/*.nsys-rep \"\$DEST\"/ 2>/dev/null || true
      cp -av \"\$SRC\"/*.qdstrm \"\$DEST\"/ 2>/dev/null || true

      echo DEST=\$DEST
      find \"\$DEST\" -type f -printf '%p %s bytes\n' 2>/dev/null | sort || true
    " || true
  done
}

start_ray_head() {
  local node="$1"
  local ip="$2"

  echo "=== starting Ray head on ${node} ${ip} ==="

  srun \
    --nodelist "${node}" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-task="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --output="${TRACE_RUN_DIR}/manual_logs/ray_head_${node}.out" \
    --error="${TRACE_RUN_DIR}/manual_logs/ray_head_${node}.err" \
    bash -lc "
      set -euo pipefail
      cd '${REPO}'
      module purge
      module load Anaconda3/2025.06-1
      module load CUDA/12.9.0
      source '${VENV_DIR}/bin/activate'

      export RAY_USAGE_STATS_ENABLED=0
      export RAY_DEDUP_LOGS=0

      export RAY_TMPDIR=/tmp/vray-${SLURM_JOB_ID}-${node}
      export TMPDIR=\${RAY_TMPDIR}/py_tmp
      mkdir -p \${RAY_TMPDIR} \${TMPDIR} \${RAY_TMPDIR}/spill

      echo RAY_TMPDIR=\${RAY_TMPDIR}
      echo python=\$(which python)
      echo ray=\$(which ray)

      exec ray start --block \
        --head \
        --node-ip-address='${ip}' \
        --port='${RAY_PORT}' \
        --temp-dir=\${RAY_TMPDIR} \
        --plasma-directory=/dev/shm \
        --object-store-memory=200000000000 \
        --num-gpus='${GPUS_PER_NODE}' \
        --num-cpus='${CPUS_PER_TASK}'
    " &
  RAY_HEAD_SRUN_PID=$!
}

start_ray_worker() {
  local node="$1"
  local ip="$2"

  echo "=== starting Ray worker on ${node} ${ip} ==="

  srun \
    --nodelist "${node}" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-task="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --output="${TRACE_RUN_DIR}/manual_logs/ray_worker_${node}.out" \
    --error="${TRACE_RUN_DIR}/manual_logs/ray_worker_${node}.err" \
    bash -lc "
      set -euo pipefail
      cd '${REPO}'
      module purge
      module load Anaconda3/2025.06-1
      module load CUDA/12.9.0
      source '${VENV_DIR}/bin/activate'

      export RAY_USAGE_STATS_ENABLED=0
      export RAY_DEDUP_LOGS=0

      export RAY_TMPDIR=/tmp/vray-${SLURM_JOB_ID}-${node}
      export TMPDIR=\${RAY_TMPDIR}/py_tmp
      mkdir -p \${RAY_TMPDIR} \${TMPDIR} \${RAY_TMPDIR}/spill

      echo RAY_TMPDIR=\${RAY_TMPDIR}
      echo python=\$(which python)
      echo ray=\$(which ray)

      exec ray start --block \
        --address='${RAY_ADDRESS}' \
        --node-ip-address='${ip}' \
        --temp-dir=\${RAY_TMPDIR} \
        --plasma-directory=/dev/shm \
        --object-store-memory=200000000000 \
        --num-gpus='${GPUS_PER_NODE}' \
        --num-cpus='${CPUS_PER_TASK}'
    " &
  RAY_WORKER_SRUN_PIDS="${RAY_WORKER_SRUN_PIDS:-} $!"
}

wait_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"

  python - "$host" "$port" "$timeout_s" <<'PY'
import socket, sys, time
host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.time() + timeout
last = None
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"TCP ready: {host}:{port}", flush=True)
            sys.exit(0)
    except Exception as e:
        last = repr(e)
        print(f"TCP not ready: {host}:{port}: {last}", flush=True)
        time.sleep(5)
print(f"ERROR: TCP never ready: {host}:{port}: {last}", file=sys.stderr)
sys.exit(1)
PY
}

wait_ray_resources() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))

  echo "=== waiting for Ray resources ==="

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    set +e
    out="$(ray status 2>&1)"
    rc=$?
    set -e

    echo "${out}" | sed -n '1,120p'

    # We expect total 4 GPUs.
    if [ "${rc}" = "0" ] && echo "${out}" | grep -q "4.0/4.0 GPU"; then
      echo "Ray sees 4 GPUs."
      return 0
    fi

    # Some Ray versions render differently; accept seeing both nodes via debug_state.
    ok=1
    for n in ${HEAD_NODE} ${WORKER_NODES}; do
      if ! ssh "${n}" "grep -R 'GPU: 20000' /tmp/vray-${SLURM_JOB_ID}-\$(hostname -s)/session_latest/logs/debug_state.txt 2>/dev/null | head -1" >/dev/null; then
        ok=0
      fi
    done
    if [ "${ok}" = "1" ]; then
      echo "Ray debug_state sees GPU resources on all nodes."
      return 0
    fi

    sleep 10
  done

  echo "ERROR: Ray resources did not become ready." >&2
  return 1
}

wait_health() {
  local url="$1"
  local timeout_s="$2"
  local deadline=$((SECONDS + timeout_s))

  echo "=== waiting for health: ${url} ==="

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "health OK: ${url}"
      return 0
    fi

    if [ -n "${SERVER_PID:-}" ] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: vLLM server exited before health." >&2
      wait "${SERVER_PID}" || true
      return 1
    fi

    echo "waiting for health..."
    ps -u "${USER}" -f | egrep 'vllm|EngineCore|RayWorkerProc|Worker_TP|ray::' | grep -v grep || true
    sleep 5
  done

  echo "ERROR: health timed out: ${url}" >&2
  return 1
}

###############################################################################
# Start Ray
###############################################################################

# Clean stale processes on allocated nodes only.
for n in ${HEAD_NODE} ${WORKER_NODES}; do
  echo "=== pre-clean ${n} ==="
  ssh "${n}" "
    set +e
    source '${VENV_DIR}/bin/activate' 2>/dev/null || true
    ray stop --force || true
    pkill -u '$USER' -f 'ray start --block' || true
    pkill -u '$USER' -f 'gcs_server|raylet|plasma_store|worker_process|RayWorkerProc|Worker_TP|VLLM::EngineCore|vllm.entrypoints.openai.api_server|nsys' || true
    rm -rf /tmp/vray-${SLURM_JOB_ID}-\$(hostname -s)
  " || true
done

RAY_WORKER_SRUN_PIDS=""

start_ray_head "${HEAD_NODE}" "${HEAD_IP}"
echo "RAY_HEAD_SRUN_PID=${RAY_HEAD_SRUN_PID}"
sleep "${RAY_HEAD_START_SLEEP_S}"

for n in ${WORKER_NODES}; do
  ip="$(resolve_ip "${n}")"
  start_ray_worker "${n}" "${ip}"
done

echo "RAY_WORKER_SRUN_PIDS=${RAY_WORKER_SRUN_PIDS}"
sleep "${RAY_WORKER_START_SLEEP_S}"

wait_tcp "${HEAD_IP}" "${RAY_PORT}" 120
wait_ray_resources "${RAY_READY_TIMEOUT_S}"

###############################################################################
# Launch vLLM on batch/head process
###############################################################################

echo "=== launching vLLM TP=${TP} PP=${PP} EP enabled ==="

VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(
    --ray-workers-use-nsight
    --enable-layerwise-nvtx-tracing
    --enable-logging-iteration-details
  )
fi

echo "vLLM command:"
printf '  %q' python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  "${VLLM_TRACE_FLAGS[@]}" \
  --enable-expert-parallel \
  --all2all-backend "${VLLM_ALL2ALL_BACKEND}" \
  --disable-custom-all-reduce
echo

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  "${VLLM_TRACE_FLAGS[@]}" \
  --enable-expert-parallel \
  --all2all-backend "${VLLM_ALL2ALL_BACKEND}" \
  --disable-custom-all-reduce \
  > "${TRACE_RUN_DIR}/manual_logs/vllm_server.out" \
  2> "${TRACE_RUN_DIR}/manual_logs/vllm_server.err" &

SERVER_PID=$!
echo "SERVER_PID=${SERVER_PID}"

wait_health "http://${HEAD_IP}:${PORT}/health" "${HEALTH_TIMEOUT_S}"

###############################################################################
# Tiny debug request / workload
###############################################################################

echo "=== warmup request ==="
curl -s "http://${HEAD_IP}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"temperature\": 0
  }" | tee "${TRACE_RUN_DIR}/responses/warmup_response.json"
echo

echo "=== measured request ==="
python - <<PY
import json
import urllib.request

url = "http://${HEAD_IP}:${PORT}/v1/completions"
prompt = "Explain why expert parallelism and accelerator networking matter for mixture-of-experts inference. " * ${SP}
payload = {
    "model": "${MODEL_ID}",
    "prompt": prompt,
    "max_tokens": ${SD},
    "temperature": 0,
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    data = r.read().decode()
print(data)
open("${TRACE_RUN_DIR}/responses/measured_response.json", "w").write(data)
PY

###############################################################################
# Shutdown and collect
###############################################################################

echo "=== stopping vLLM ==="
kill -INT "${SERVER_PID}" 2>/dev/null || true

elapsed=0
while kill -0 "${SERVER_PID}" 2>/dev/null && [ "${elapsed}" -lt "${SERVER_SHUTDOWN_TIMEOUT_S}" ]; do
  sleep 1
  elapsed=$((elapsed + 1))
done

if kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "vLLM still alive after ${SERVER_SHUTDOWN_TIMEOUT_S}s; TERM"
  kill -TERM "${SERVER_PID}" 2>/dev/null || true
  sleep 5
fi

if kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "vLLM still alive after TERM; KILL"
  kill -KILL "${SERVER_PID}" 2>/dev/null || true
fi

SERVER_PID=""

echo "=== first Nsight copy while Ray still alive ==="
copy_all_nsight || true

if [ "${WORKER_NSYS_FINALIZE_WAIT_S}" != "0" ]; then
  echo "waiting ${WORKER_NSYS_FINALIZE_WAIT_S}s for worker Nsight finalize while Ray alive"
  sleep "${WORKER_NSYS_FINALIZE_WAIT_S}"
  copy_all_nsight || true
fi

echo "=== stopping Ray ==="
for n in ${WORKER_NODES} ${HEAD_NODE}; do
  ssh "${n}" "
    set +e
    source '${VENV_DIR}/bin/activate' 2>/dev/null || true
    ray stop --force || true
    pkill -u '$USER' -f 'ray start --block' || true
  " || true
done

if [ "${WORKER_NSYS_FLUSH_WAIT_S}" != "0" ]; then
  echo "waiting ${WORKER_NSYS_FLUSH_WAIT_S}s for Nsight flush after Ray stop"
  sleep "${WORKER_NSYS_FLUSH_WAIT_S}"
fi

echo "=== final Nsight copy ==="
copy_all_nsight || true

if [ "${NSYS_STATS_ENABLE}" = "1" ]; then
  echo "=== generating nsys stats ==="
  for rep in "${TRACE_RUN_DIR}"/ray_worker_nsight/*/*.nsys-rep; do
    [ -f "${rep}" ] || continue
    size="$(stat -c '%s' "${rep}" 2>/dev/null || echo 0)"
    if [ "${size}" -lt 4096 ]; then
      echo "skip tiny ${rep} ${size} bytes"
      continue
    fi

    base="$(basename "${rep}" .nsys-rep)"
    echo "nsys stats ${rep}"
    nsys stats \
      --force-overwrite=true \
      --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
      --output "${TRACE_RUN_DIR}/nsys_stats/${base}" \
      "${rep}" \
      2>&1 | tee "${TRACE_RUN_DIR}/nsys_stats/${base}.stats.log" || true
  done
fi

echo "=== final files ==="
find "${TRACE_RUN_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" | sort || true

echo "Done."
trap - EXIT
cleanup || true
exit 0