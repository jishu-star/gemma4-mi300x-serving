#!/usr/bin/env bash
# Reproduce the result matrix against a running server (./serve.sh first).
#
#   ./bench/run_bench.sh                 all cells
#   ./bench/run_bench.sh long32k_c1      one cell
#
# Cells are (workload, concurrency) pairs. ShareGPT cells need the dataset; long/aa cells are
# synthetic and need nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-gemma4-vllm}
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
DOCKER=${DOCKER:-docker}
OUT=${OUT:-bench/results}; mkdir -p "$OUT"
SG=${SG:-$HOME/ShareGPT_V3_unfiltered_cleaned_split.json}

run() {  # name workload concurrency nprompts extra...
  local N=$1 CONC=$3 NP=$4; shift 4
  echo "=== $N (concurrency $CONC, $NP prompts)"
  $DOCKER exec "$NAME" bash -c "mkdir -p /tmp/b && vllm bench serve \
    --model $MODEL --port 8000 --backend openai-chat --endpoint /v1/chat/completions \
    --num-prompts $NP --max-concurrency $CONC --num-warmups $(( CONC > 2 ? CONC : 2 )) \
    --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir /tmp/b --result-filename $N.json $*" 2>&1 \
    | grep -E "Successful requests|Benchmark duration|Total generated tokens|Output token throughput|Mean TTFT|Median TTFT" \
    | sed 's/^/  /'
  $DOCKER cp "$NAME:/tmp/b/$N.json" "$OUT/$N.json" >/dev/null 2>&1
}

# NOTE ON WARM-UP: a freshly started server compiles Triton kernel variants lazily. The first
# measured run of any cell can absorb a one-off ~1.6 s compile, which on a ~24 s benchmark reads as
# a spurious ~6% loss. Discard the first pass, or treat medians across >= 3 runs as the result.
sharegpt() { [ -f "$SG" ] || { echo "  (skipped: ShareGPT dataset not at $SG)"; return; }
  $DOCKER cp "$SG" "$NAME:/tmp/sharegpt.json" >/dev/null 2>&1
  run "$1" chat "$2" "$3" --dataset-name sharegpt --dataset-path /tmp/sharegpt.json --ignore-eos; }
synth() { run "$1" chat "$2" "$3" --dataset-name random --random-input-len "$4" --random-output-len "$5" --ignore-eos; }

CELL=${1:-all}
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c1 ]   && sharegpt sharegpt_c1   1   100
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c32 ]  && sharegpt sharegpt_c32  32  500
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c128 ] && sharegpt sharegpt_c128 128 1000
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c256 ] && sharegpt sharegpt_c256 256 1000
[ "$CELL" = all ] || [ "$CELL" = aa_c1 ]         && synth    aa_c1         1   10   9921 1500
[ "$CELL" = all ] || [ "$CELL" = long8k_c1 ]     && synth    long8k_c1     1   20   8192 1024
[ "$CELL" = all ] || [ "$CELL" = long32k_c1 ]    && synth    long32k_c1    1   10  32768 1024
echo "=== results in $OUT"
