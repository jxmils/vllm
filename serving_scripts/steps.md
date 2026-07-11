# Two-Node Ray/vLLM Startup Runbook

This runbook records the working manual procedure for starting a two-node Ray cluster for vLLM on:

- Head: `htc-g059`
- Worker: `htc-g060`
- Ray GCS: `10.137.21.59:6378`
- vLLM API port: `8000`
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Target Ray resources after success: `0.0/128.0 CPU`, `0.0/2.0 GPU`

The successful debug run used job `8115442`. Replace `JOB` with the current Slurm job ID for future runs.

---

## 0. Reserve the nodes

Run from the login node:

```bash
cd /data/engs-glass/catz0932/inference-traces/vllm

salloc \
  --reservation=engs-glass5 \
  --nodelist=htc-g[059-060] \
  --nodes=2 \
  --partition=short \
  --gres=gpu:h100:1 \
  --ntasks-per-node=1 \
  --cpus-per-task=64 \
  --mem=512G \
  --time=02:00:00 \
  --account=engs-glass \
  --qos=priority
```

After allocation:

```bash
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NODELIST=$SLURM_NODELIST"
scontrol show hostnames "$SLURM_NODELIST"
sq
```

Expected nodes:

```text
htc-g059
htc-g060
```

---

## 1. Set variables

Run on the head node, `htc-g059`.

Use the current job ID. Example below uses the successful debug job `8115442`.

```bash
export JOB=8115442
export SLURM_JOB_ID=8115442
export SLURM_NODELIST='htc-g[059-060]'

cd /data/engs-glass/catz0932/inference-traces/vllm

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
source .venv/bin/activate

export REPO=/data/engs-glass/catz0932/inference-traces/vllm
export VENV=$REPO/.venv
export TRACE_RUN_DIR=$REPO/results/manual_tp2_2node_sharegpt64_sp256_sd256_${JOB}

mkdir -p "$TRACE_RUN_DIR"/{manual_logs,nccl_logs,responses,ray_worker_nsight,bench_results}
```

Use hardcoded node names and IPs. This avoided the `Malformed host` failure we hit when `HEAD_IP` was empty.

```bash
export HEAD_NODE=htc-g059
export WORKER_NODE=htc-g060

export HEAD_IP=10.137.21.59
export WORKER_IP=10.137.21.60

export RAY_PORT=6378
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

export MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507
export PORT=8000

echo "JOB=$JOB"
echo "TRACE_RUN_DIR=$TRACE_RUN_DIR"
echo "HEAD_NODE=$HEAD_NODE HEAD_IP=$HEAD_IP"
echo "WORKER_NODE=$WORKER_NODE WORKER_IP=$WORKER_IP"
echo "RAY_ADDRESS=$RAY_ADDRESS"
```

Expected:

```text
HEAD_NODE=htc-g059 HEAD_IP=10.137.21.59
WORKER_NODE=htc-g060 WORKER_IP=10.137.21.60
RAY_ADDRESS=10.137.21.59:6378
```

---

## 2. Clean stale Ray/vLLM state

Run from `htc-g059`.

```bash
for N in "$HEAD_NODE" "$WORKER_NODE"; do
  ssh "$N" "
    set +e
    cd '$REPO'

    module purge
    module load Anaconda3/2025.06-1
    module load CUDA/12.9.0
    source '$VENV/bin/activate'

    ray stop --force || true

    pkill -u '$USER' -f 'ray start --block' || true
    pkill -u '$USER' -f 'gcs_server|raylet|plasma_store|worker_process|RayWorkerProc|Worker_TP|VLLM::EngineCore|vllm.entrypoints.openai.api_server|nsys|dashboard_agent|runtime_env_agent|log_monitor.py|monitor.py' || true

    rm -rf /tmp/vray-${JOB}-\$(hostname -s)

    echo === \$(hostname -s) remaining Ray processes ===
    ps -u '$USER' -f | egrep 'ray start|gcs_server|raylet|dashboard_agent|runtime_env_agent' | grep -v grep || true
  "
done
```

If another user's Ray processes appear in `ray stop` output, ignore them unless they occupy the ports we need. Our cleanup uses `pkill -u "$USER"` to avoid killing other users' jobs.

---

## 3. Start Ray head on `htc-g059`

Run on `htc-g059`.

```bash
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0
export RAY_raylet_start_wait_time_s=300

export RAY_TMPDIR=/tmp/vray-${JOB}-${HEAD_NODE}
export TMPDIR=$RAY_TMPDIR/py_tmp

rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$RAY_TMPDIR/spill"

: > "$TRACE_RUN_DIR/manual_logs/ray_head.out"
: > "$TRACE_RUN_DIR/manual_logs/ray_head.err"
```

Start the head.

```bash
nohup ray start --block \
  --head \
  --include-dashboard=false \
  --node-ip-address="$HEAD_IP" \
  --port="$RAY_PORT" \
  --temp-dir="$RAY_TMPDIR" \
  --object-store-memory=80000000000 \
  --dashboard-agent-listen-port=52365 \
  --dashboard-agent-grpc-port=52366 \
  --metrics-export-port=0 \
  --num-gpus=1 \
  --num-cpus=64 \
  >> "$TRACE_RUN_DIR/manual_logs/ray_head.out" \
  2>> "$TRACE_RUN_DIR/manual_logs/ray_head.err" \
  < /dev/null &

export RAY_HEAD_PID=$!
echo "RAY_HEAD_PID=$RAY_HEAD_PID"
```

Check the head.

```bash
sleep 45

tail -100 "$TRACE_RUN_DIR/manual_logs/ray_head.out"
tail -100 "$TRACE_RUN_DIR/manual_logs/ray_head.err"

ray status --address="$RAY_ADDRESS" || true
```

Expected head-only status:

```text
Ray runtime started

Resources:
 0.0/64.0 CPU
 0.0/1.0 GPU
```

---

## 4. Start Ray worker on `htc-g060`

Open a second terminal on `htc-g060`.

Set variables on the worker.

```bash
export JOB=8115442
export SLURM_JOB_ID=8115442
export SLURM_NODELIST='htc-g[059-060]'

export REPO=/data/engs-glass/catz0932/inference-traces/vllm
export VENV=$REPO/.venv
export TRACE_RUN_DIR=$REPO/results/manual_tp2_2node_sharegpt64_sp256_sd256_${JOB}

export HEAD_NODE=htc-g059
export WORKER_NODE=htc-g060

export HEAD_IP=10.137.21.59
export WORKER_IP=10.137.21.60

export RAY_PORT=6378
export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

cd "$REPO"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
source "$VENV/bin/activate"

mkdir -p "$TRACE_RUN_DIR"/{manual_logs,nccl_logs,responses,ray_worker_nsight,bench_results}

echo "JOB=$JOB"
echo "TRACE_RUN_DIR=$TRACE_RUN_DIR"
echo "WORKER_NODE=$WORKER_NODE WORKER_IP=$WORKER_IP"
echo "RAY_ADDRESS=$RAY_ADDRESS"
```

Clean only our worker-side Ray attempt.

```bash
pkill -u "$USER" -f 'ray start --block|raylet|plasma_store|worker_process|RayWorkerProc|dashboard_agent|runtime_env_agent|log_monitor.py' || true

export RAY_TMPDIR=/tmp/vray-${JOB}-${WORKER_NODE}

rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR" "$RAY_TMPDIR/py_tmp" "$RAY_TMPDIR/spill"

ps -u "$USER" -f | egrep 'ray start|raylet|dashboard_agent|runtime_env_agent' | grep -v grep || true
```

Start the worker.

```bash
export TMPDIR=$RAY_TMPDIR/py_tmp
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0
export RAY_raylet_start_wait_time_s=300

: > "$TRACE_RUN_DIR/manual_logs/ray_worker.out"
: > "$TRACE_RUN_DIR/manual_logs/ray_worker.err"

nohup ray start --block \
  --address="$RAY_ADDRESS" \
  --node-ip-address="$WORKER_IP" \
  --temp-dir="$RAY_TMPDIR" \
  --object-store-memory=80000000000 \
  --dashboard-agent-listen-port=52465 \
  --dashboard-agent-grpc-port=52466 \
  --metrics-export-port=52467 \
  --num-gpus=1 \
  --num-cpus=64 \
  >> "$TRACE_RUN_DIR/manual_logs/ray_worker.out" \
  2>> "$TRACE_RUN_DIR/manual_logs/ray_worker.err" \
  < /dev/null &

export RAY_WORKER_PID=$!
echo "RAY_WORKER_PID=$RAY_WORKER_PID"
```

Check the worker.

```bash
sleep 60

echo "=== worker out ==="
tail -100 "$TRACE_RUN_DIR/manual_logs/ray_worker.out"

echo "=== worker err ==="
tail -100 "$TRACE_RUN_DIR/manual_logs/ray_worker.err"

echo "=== worker processes ==="
ps -u "$USER" -f | egrep 'ray start --block|raylet|dashboard_agent|runtime_env_agent' | grep -v grep || true

echo "=== status ==="
ray status --address="$RAY_ADDRESS" || true
```

Expected successful status:

```text
Node status
---------------------------------------------------------------
Active:
 1 node_...
 1 node_...

Resources
---------------------------------------------------------------
Total Usage:
 0.0/128.0 CPU
 0.0/2.0 GPU
 0B/1.80TiB memory
 0B/149.01GiB object_store_memory
```

---

## 5. What fixed the one-GPU problem

The symptom was:

```text
Resources:
 0.0/64.0 CPU
 0.0/1.0 GPU
```

The head was healthy, but the worker was not registering.

### Root cause 1: worker dashboard agent port file timeout

The first worker failure was:

```text
Timed out waiting for file ... dashboard_agent_listen_port_...
```

Raylet started, but crashed while waiting for the dashboard agent port file. Because raylet died locally on `htc-g060`, the head never saw the second node.

Fix:

```bash
--dashboard-agent-listen-port=52465
--dashboard-agent-grpc-port=52466
```

Also, do not pass `--include-dashboard=false` to the worker. Ray rejects that flag unless starting the head node.

### Root cause 2: worker metrics export port file timeout

After fixing dashboard ports, the worker failed again with:

```text
Timed out waiting for file ... metrics_export_port_...
```

Fix:

```bash
--metrics-export-port=52467
```

On this Ray version, using `--metrics-export-port=0` on the worker caused raylet to wait for a metrics port file that was not produced reliably.

### Root cause 3: `HEAD_IP` became empty

One head restart failed with:

```text
ValueError: Malformed host:
Failed to convert :6378 to host:port
```

This happened because `HEAD_IP` was empty, making:

```bash
RAY_ADDRESS=:6378
```

Fix: hardcode the IPs for this allocation:

```bash
export HEAD_IP=10.137.21.59
export WORKER_IP=10.137.21.60
export RAY_ADDRESS=${HEAD_IP}:6378
```

### Stability knobs

Use a smaller object store:

```bash
--object-store-memory=80000000000
```

This avoids Ray defaulting near `200 GB`, which is more likely to stress `/dev/shm`.

Use isolated temp dirs:

```bash
/tmp/vray-${JOB}-${HEAD_NODE}
/tmp/vray-${JOB}-${WORKER_NODE}
```

Use one GPU per node despite the SSH shell seeing all GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0
--num-gpus=1
```

---

## 6. Proof of success

Successful worker output:

```text
Ray runtime started.
```

Successful final status:

```text
Active:
 1 node_c1cde583054d150144bf3bda45dac32b6d52aa54a8d4b33076c91ea7
 1 node_34e54b0d3f59cefb712ef8a1a703cf46658055b6e2190741d1b72b88

Resources:
 0.0/128.0 CPU
 0.0/2.0 GPU
 0B/1.80TiB memory
 0B/149.01GiB object_store_memory
```

---

## 7. Do not launch vLLM until this is true

Before starting vLLM, run from either node:

```bash
ray status --address="10.137.21.59:6378"
```

Do not proceed unless it shows:

```text
0.0/2.0 GPU
```

---

## 8. Next step: start vLLM from the head only

Run vLLM only from `htc-g059`. Do not start vLLM from the worker.

The Ray cluster is ready once the final status shows two active nodes and two GPUs.


unset CUDA_VISIBLE_DEVICES

export CACHE_ROOT=/data/engs-glass/catz0932/cache
mkdir -p "$CACHE_ROOT"/{xdg,hf,hf/hub,hf/transformers,hf/datasets,vllm,vllm/flashinfer_autotune,flashinfer,flashinfer/0.6.12,triton,torchinductor,cuda,tmp}

export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export HF_HOME=$CACHE_ROOT/hf
export HUGGINGFACE_HUB_CACHE=$CACHE_ROOT/hf/hub
export TRANSFORMERS_CACHE=$CACHE_ROOT/hf/transformers
export HF_DATASETS_CACHE=$CACHE_ROOT/hf/datasets
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$CACHE_ROOT/vllm/flashinfer_autotune
export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer
export FLASHINFER_WORKSPACE_DIR=$CACHE_ROOT/flashinfer/0.6.12
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export CUDA_CACHE_PATH=$CACHE_ROOT/cuda
export TMPDIR=$CACHE_ROOT/tmp

export VLLM_HOST_IP=$HEAD_IP
export HOST=$HEAD_IP
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0

export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_STATS_INTERVAL=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL,P2P,TUNING
export NCCL_DEBUG_FILE=$TRACE_RUN_DIR/nccl_logs/nccl_%h_%p.log
export NCCL_SOCKET_IFNAME=eno12399
export GLOO_SOCKET_IFNAME=eno12399

: > "$TRACE_RUN_DIR/manual_logs/vllm_server.out"
: > "$TRACE_RUN_DIR/manual_logs/vllm_server.err"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HEAD_IP" \
  --port "$PORT" \
  --distributed-executor-backend ray \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --ray-workers-use-nsight \
  --enable-layerwise-nvtx-tracing \
  --enable-logging-iteration-details \
  --disable-custom-all-reduce \
  > "$TRACE_RUN_DIR/manual_logs/vllm_server.out" \
  2> "$TRACE_RUN_DIR/manual_logs/vllm_server.err" \
  < /dev/null &

export VLLM_PID=$!
echo "VLLM_PID=$VLLM_PID"



for i in $(seq 1 720); do
  if curl -fsS "http://${HEAD_IP}:${PORT}/health" >/dev/null; then
    echo "vLLM healthy at http://${HEAD_IP}:${PORT}"
    break
  fi

  echo "waiting for vLLM health $i/720"

  if [ $((i % 12)) -eq 0 ]; then
    echo "=== vLLM stderr tail ==="
    tail -120 "$TRACE_RUN_DIR/manual_logs/vllm_server.err" || true

    echo "=== vLLM stdout tail ==="
    tail -120 "$TRACE_RUN_DIR/manual_logs/vllm_server.out" || true

    echo "=== Ray status ==="
    ray status --address="$RAY_ADDRESS" || true
  fi

  sleep 5
done


export JOB=8115761
export REPO=/data/engs-glass/catz0932/inference-traces/vllm
export VENV=$REPO/.venv
export TRACE_RUN_DIR=$REPO/results/manual_tp8_2node_sharegpt64_sp256_sd256_${JOB}

export HEAD_IP=10.137.21.59
export RAY_ADDRESS=10.137.21.59:6378
export MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507
export PORT=8000

cd "$REPO"
module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
source "$VENV/bin/activate"

ray status --address="$RAY_ADDRESS"
echo "PORT=$PORT"


# works



export JOB=8115761
export REPO=/data/engs-glass/catz0932/inference-traces/vllm
export VENV=$REPO/.venv
export TRACE_RUN_DIR=$REPO/results/manual_tp8_2node_sharegpt64_sp256_sd256_${JOB}

export HEAD_IP=10.137.21.59
export RAY_ADDRESS=10.137.21.59:6378
export MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507
export PORT=8000

cd "$REPO"
module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
source "$VENV/bin/activate"

ray status --address="$RAY_ADDRESS"
echo "PORT=$PORT"

unset CUDA_VISIBLE_DEVICES

mkdir -p "$TRACE_RUN_DIR"/{manual_logs,nccl_logs,responses,ray_worker_nsight,bench_results}

export VLLM_HOST_IP=$HEAD_IP
export HOST=$HEAD_IP
export PYTHONNOUSERSITE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_STATS_INTERVAL=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL,P2P,TUNING
export NCCL_DEBUG_FILE=$TRACE_RUN_DIR/nccl_logs/nccl_%h_%p.log
export NCCL_SOCKET_IFNAME=eno12399
export GLOO_SOCKET_IFNAME=eno12399

: > "$TRACE_RUN_DIR/manual_logs/vllm_server.out"
: > "$TRACE_RUN_DIR/manual_logs/vllm_server.err"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HEAD_IP" \
  --port 8000 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --ray-workers-use-nsight \
  --enable-layerwise-nvtx-tracing \
  --enable-logging-iteration-details \
  --disable-custom-all-reduce \
  > "$TRACE_RUN_DIR/manual_logs/vllm_server.out" \
  2> "$TRACE_RUN_DIR/manual_logs/vllm_server.err" \
  < /dev/null &

export VLLM_PID=$!
echo "VLLM_PID=$VLLM_PID"

for i in $(seq 1 720); do
  if curl -fsS "http://10.137.21.59:8000/health" >/dev/null; then
    echo "vLLM healthy at http://10.137.21.59:8000"
    break
  fi

  echo "waiting for vLLM health $i/720"

  if [ $((i % 12)) -eq 0 ]; then
    echo "=== vLLM stderr tail ==="
    tail -120 "$TRACE_RUN_DIR/manual_logs/vllm_server.err" || true

    echo "=== vLLM stdout tail ==="
    tail -120 "$TRACE_RUN_DIR/manual_logs/vllm_server.out" || true

    echo "=== Ray status ==="
    ray status --address="$RAY_ADDRESS" || true
  fi

  sleep 5
done