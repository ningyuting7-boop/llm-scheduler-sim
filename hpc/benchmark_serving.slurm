#!/bin/bash
#SBATCH --job-name=tie_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/tie_bench_%j.out
#SBATCH --error=logs/tie_bench_%j.err

# Phase 3 (docs/Phase3_Scheduling_Evaluation_Plan.md): benchmark TIE
# scheduling against its controls on real vLLM serving real Qwen3-8B.
#
# Every arm runs inside this one job, sequentially, because each arm needs
# its own server process and queueing for the GPU once per arm would
# dominate the wall time. The GPU sits idle for a few minutes between arms
# while servers start and stop; that is cheaper than five queue waits.
#
# One A100 is enough for every arm including the ones that load the
# predictor: vLLM allocates KV cache before the scheduler is constructed
# (vllm/v1/engine/core.py:109 vs :133), so DeBERTa lives in the
# (1 - gpu_memory_utilization) remainder -- 7.2GB on an 80GB card. See plan
# doc section 4. GPU model is pinned to a100 because phase 1's
# generate_labels.slurm showed that scheduling far faster on this cluster
# than an unpinned gpu:1 request.
#
# ARMS defaults to the three that share a GPU load and therefore form the
# primary comparison (plan doc section 5.1). Override to add the P1 arms:
#   sbatch --export=ALL,ARMS="fcfs fcfs_pred sjf tie tie_oracle" ...

set -euo pipefail

module load anaconda3/2024.06 cuda/12.1.1
source activate /scratch/ning.yuti/envs/tie-repro

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

export HF_HOME=/scratch/ning.yuti/hf_cache
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Clusters commonly set http_proxy for outbound access, and both curl and
# aiohttp will route a request to 127.0.0.1 through it, which fails. Exempt
# loopback for everything below, including `vllm bench serve`.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.88}"
DATASET="${DATASET:-data/benchmark_prompts.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATES="${REQUEST_RATES:-2 4 8 16 32}"
ARMS="${ARMS:-fcfs_pred sjf tie}"
RESULT_DIR="results/bench_${SLURM_JOB_ID}"

# Predictor config, read by vllm_tie/predictor.py.
export TIE_MODEL_DIR="${TIE_MODEL_DIR:-checkpoints/predictor_full}"
export TIE_PREDICTOR_GPU=0
export TIE_MAX_PREDICT_BATCH="${TIE_MAX_PREDICT_BATCH:-32}"
export TIE_SCORE_SEED="${TIE_SCORE_SEED:-0}"   # reproducible Monte Carlo scores

mkdir -p "$RESULT_DIR"
echo "=== phase 3 benchmark: job $SLURM_JOB_ID ==="
echo "model=$MODEL arms='$ARMS' rates='$REQUEST_RATES' num_prompts=$NUM_PROMPTS"
echo "max_num_seqs=$MAX_NUM_SEQS gpu_util=$GPU_UTIL results=$RESULT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Per-arm server configuration. GPU_UTIL and MAX_NUM_SEQS are deliberately
# identical across arms: differing KV cache capacity would make the latency
# comparison meaningless (plan doc section 5.1).
configure_arm() {
    unset TIE_BETA TIE_MODE TIE_ORACLE_CSV
    SCHED_ARGS=()
    case "$1" in
        fcfs)        ;;  # stock vLLM, no scheduler flags at all
        fcfs_pred)   SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.FCFSWithPredictorScheduler) ;;
        sjf)         SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.TIEScheduler)
                     export TIE_BETA=0 ;;                      # score = E[X], no CVaR term
        tie)         SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.TIEScheduler) ;;
        tie_oracle)  SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.TIEScheduler)
                     export TIE_MODE=oracle
                     export TIE_ORACLE_CSV="${TIE_ORACLE_CSV:-data/benchmark_oracle_labels.csv}" ;;
        *)           echo "unknown arm: $1" >&2; return 1 ;;
    esac
}

health_ok() {
    # Python rather than curl: curl is not guaranteed present on a compute
    # node, and a missing binary inside a silenced `if` is indistinguishable
    # from a server that never came up.
    python - "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for_server() {
    # vLLM takes minutes to load an 8B model and profile memory; poll rather
    # than guess a sleep.
    for _ in $(seq 1 180); do
        if health_ok; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server died during startup -- see $SERVER_LOG" >&2
            tail -40 "$SERVER_LOG" >&2
            return 1
        fi
        sleep 5
    done
    echo "server failed to become healthy within 15 minutes" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        # SIGINT, not SIGKILL: the scheduler's shutdown() prints the
        # prediction counters (how many requests were scheduled before their
        # prediction landed), which is needed to tell a real TIE run from one
        # that degenerated to FCFS.
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 2
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    unset SERVER_PID
}
trap stop_server EXIT

for ARM in $ARMS; do
    echo ""
    echo "================ arm: $ARM ================"
    configure_arm "$ARM"

    SERVER_LOG="$RESULT_DIR/server_${ARM}.log"
    vllm serve "$MODEL" \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len 8192 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --no-enable-prefix-caching \
        "${SCHED_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "arm $ARM failed to start; skipping" >&2
        stop_server
        continue
    fi
    echo "server up (pid $SERVER_PID)"

    for RATE in $REQUEST_RATES; do
        echo "--- $ARM @ ${RATE} req/s ---"
        vllm bench serve \
            --model "$MODEL" \
            --port "$PORT" \
            --dataset-name custom \
            --dataset-path "$DATASET" \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$RATE" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,90,95,99 \
            --save-result \
            --result-dir "$RESULT_DIR" \
            --result-filename "${ARM}_rate${RATE}.json" \
            2>&1 | tee "$RESULT_DIR/bench_${ARM}_rate${RATE}.log"
    done

    stop_server
    # Let the GPU drain before the next arm claims its memory.
    sleep 20
done

echo ""
echo "=== done; results in $RESULT_DIR ==="
ls -la "$RESULT_DIR"
