#!/bin/bash
# Phase 3 smoke test: start one arm, push a small burst through it, and check
# the things the CPU-only gate check (scripts/verify_vllm_integration.py)
# cannot see.
#
#   usage: bash scripts/smoke_test.sh [arm] [num_prompts] [request_rate]
#   arms:  fcfs | fcfs_pred | sjf | tie | tie_oracle   (default: tie)
#
# Three questions, none of which a successful startup alone answers:
#
#   1. Does the predictor fit? vLLM allocates KV cache before the scheduler is
#      constructed, so DeBERTa has to live in the (1 - gpu_memory_utilization)
#      remainder. If that is too tight it OOMs at load time, not at startup.
#   2. Do the scores mean anything? Every request enters the queue at a
#      placeholder 2048 and is rescored asynchronously. If predictions land
#      after requests are already scheduled, the queue is running FCFS no
#      matter what the predictor says -- the run would look fine and measure
#      nothing. That is what popped_before_prediction counts.
#   3. Does the queue ever have depth? Scheduling policy only decides
#      admission into the running batch; below saturation the waiting queue
#      is empty and every arm behaves identically.
#
# Request rate defaults deliberately high (32/s) for a burst: the point here
# is to force queueing, not to measure latency.

set -uo pipefail

ARM="${1:-tie}"
NUM_PROMPTS="${2:-60}"
RATE="${3:-32}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.88}"
DATASET="${DATASET:-data/benchmark_prompts.jsonl}"
LOG="/tmp/smoke_${ARM}_$$.log"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TIE_MODEL_DIR="${TIE_MODEL_DIR:-checkpoints/predictor_full}"
export TIE_SCORE_SEED="${TIE_SCORE_SEED:-0}"
export TIE_PREDICTOR_GPU=0

# Clusters commonly set http_proxy/https_proxy for outbound access, and both
# curl and aiohttp will happily route a request to 127.0.0.1 through it,
# which fails. Exempt loopback for everything downstream, including
# `vllm bench serve`.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# Health probe via Python rather than curl: curl is not guaranteed present on
# a compute node, and a missing binary inside a silenced `if` would look
# exactly like a server that never came up.
health_ok() {
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

unset TIE_BETA TIE_MODE TIE_ORACLE_CSV
SCHED_ARGS=()
case "$ARM" in
    fcfs)       ;;
    fcfs_pred)  SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.FCFSWithPredictorScheduler) ;;
    sjf)        SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.TIEScheduler); export TIE_BETA=0 ;;
    tie)        SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.TIEScheduler) ;;
    tie_oracle) SCHED_ARGS=(--scheduler-cls vllm_tie.scheduler.TIEScheduler)
                export TIE_MODE=oracle
                export TIE_ORACLE_CSV="${TIE_ORACLE_CSV:-data/benchmark_oracle_labels.csv}" ;;
    *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

[[ -f "$DATASET" ]] || { echo "missing $DATASET -- run scripts/export_benchmark_dataset.py" >&2; exit 2; }

echo "=== smoke test: arm=$ARM prompts=$NUM_PROMPTS rate=${RATE}/s ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "server log: $LOG"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len 8192 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --no-enable-prefix-caching \
    "${SCHED_ARGS[@]}" \
    > "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        # SIGINT so shutdown() runs and prints the final counters.
        kill -INT "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
        kill -9 "$SERVER_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

echo -n "waiting for server"
UP=0
for _ in $(seq 1 120); do
    if health_ok; then UP=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "SERVER DIED DURING STARTUP. Last 60 lines:"
        tail -60 "$LOG"
        exit 1
    fi
    echo -n "."
    sleep 5
done
echo ""

if [[ $UP -ne 1 ]]; then
    echo "server never became healthy -- diagnosing:"
    echo "  startup completed?  $(grep -c 'Application startup complete' "$LOG") match(es)"
    echo "  proxy vars:         $(env | grep -i '^[a-z_]*proxy=' | tr '\n' ' ')"
    echo "  listening sockets:"
    (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep ":${PORT}" || echo "    nothing on port $PORT"
    echo "  /health route registered?"
    grep -E "Route: /health" "$LOG" || echo "    not in the route list"
    echo "  raw probe:"
    python - "$PORT" <<'PY'
import sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5) as r:
        print("    status", r.status)
except Exception as exc:
    print("    failed:", type(exc).__name__, exc)
PY
    echo "  tail of server log:"
    tail -30 "$LOG"
    exit 1
fi
echo "server up (pid $SERVER_PID)"

echo ""
echo "--- predictor startup lines ---"
grep -E "^\[TIE\]" "$LOG" | head -10 || echo "(none -- expected for the stock fcfs arm)"

echo ""
echo "--- pushing $NUM_PROMPTS prompts at ${RATE}/s ---"
vllm bench serve \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$RATE" \
    --percentile-metrics ttft,e2el --metric-percentiles 50,99 \
    2>&1 | tail -25

echo ""
echo "--- queue depth seen by vLLM (is it saturating?) ---"
grep -oE "Waiting: [0-9]+" "$LOG" | sort -u -t' ' -k2 -n | tail -5 || echo "(no Waiting lines)"

cleanup
sleep 3

echo ""
echo "=== verdict ==="
FAIL=0
if grep -qE "Traceback|CUDA out of memory" "$LOG"; then
    echo "FAIL: traceback or OOM in server log"
    grep -E "Traceback|CUDA out of memory" -A 12 "$LOG" | head -40
    FAIL=1
fi

if [[ "$ARM" != "fcfs" ]]; then
    grep -q "predictor loaded" "$LOG" \
        && echo "PASS: predictor loaded" \
        || { echo "FAIL: predictor never loaded"; FAIL=1; }

    STATS=$(grep "final stats" "$LOG" | tail -1)
    if [[ -n "$STATS" ]]; then
        echo "final stats: ${STATS#*final stats: }"
    else
        echo "WARN: no final stats line (server may not have shut down cleanly)"
    fi
fi

if [[ "$ARM" == "tie" || "$ARM" == "sjf" || "$ARM" == "tie_oracle" ]]; then
    # The measurement that decides whether the run means anything: if every
    # request was scheduled before its score arrived, this arm silently ran
    # FCFS.
    POPPED=$(grep -oE "popped=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
    UNPRED=$(grep -oE "popped_before_prediction=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
    if [[ -n "${POPPED:-}" && -n "${UNPRED:-}" && "$POPPED" -gt 0 ]]; then
        PCT=$(( 100 * UNPRED / POPPED ))
        echo "scheduled before their prediction landed: $UNPRED / $POPPED (${PCT}%)"
        if [[ "$PCT" -ge 90 ]]; then
            echo "FAIL: this arm is effectively running FCFS -- scores arrive too late to matter"
            FAIL=1
        elif [[ "$PCT" -ge 40 ]]; then
            echo "WARN: a large fraction bypassed scoring; results will understate the policy"
        else
            echo "PASS: most requests were scored before being scheduled"
        fi
    else
        echo "WARN: could not read scheduling counters from the log"
    fi
fi

if [[ "$ARM" == "tie_oracle" ]]; then
    grep -q "oracle table loaded" "$LOG" \
        && echo "PASS: oracle table loaded" \
        || { echo "FAIL: oracle table not loaded"; FAIL=1; }
fi

echo ""
[[ $FAIL -eq 0 ]] && echo "SMOKE TEST PASSED ($ARM)" || echo "SMOKE TEST FAILED ($ARM) -- full log: $LOG"
echo "full server log kept at: $LOG"
exit $FAIL
