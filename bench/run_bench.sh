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
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
OUT=${OUT:-bench/results}; mkdir -p "$OUT"
SG=${SG:-$HOME/ShareGPT_V3_unfiltered_cleaned_split.json}

# Draft acceptance is read as a DELTA across each cell. vLLM's spec-decode counters are cumulative
# since server start, so reading them once at the end mixes every workload that has run -- which is
# exactly how a 2.0-on-random-tokens cell and a 2.8-on-ShareGPT cell average into a meaningless 2.67.
accept_counters() {
  curl -fsS "http://$HOST:$PORT/metrics" 2>/dev/null | awk '
    /^vllm:spec_decode_num_accepted_tokens_total/ {a=$2}
    /^vllm:spec_decode_num_draft_tokens_total/    {d=$2}
    END {printf "%.0f %.0f", a+0, d+0}'
}

run() {  # name workload concurrency nprompts extra...
  local N=$1 CONC=$3 NP=$4; shift 4
  echo "=== $N (concurrency $CONC, $NP prompts)"
  local before after
  before=$(accept_counters)
  timeout -k 30 5400 $DOCKER exec "$NAME" bash -c "mkdir -p /tmp/b && HF_HUB_OFFLINE=1 vllm bench serve \
    --model $MODEL --port 8000 --backend openai-chat --endpoint /v1/chat/completions \
    --num-prompts $NP --max-concurrency $CONC --num-warmups $(( CONC > 2 ? CONC : 2 )) \
    --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --save-detailed --result-dir /tmp/b --result-filename $N.json $*" 2>&1 \
    | grep -E "Successful requests|Benchmark duration|Total generated tokens|Output token throughput|Mean TTFT|Median TTFT" \
    | sed 's/^/  /'
  after=$(accept_counters)
  $DOCKER cp "$NAME:/tmp/b/$N.json" "$OUT/$N.json" >/dev/null 2>&1
  # acceptance length = 1 + accepted/proposals, proposals = drafted / num_speculative_tokens
  echo "$before $after ${NSPEC:-3}" | awk '{
    da=$3-$1; dd=$4-$2;
    if (dd>0) printf "  acceptance (this cell only): %.3f\n", 1 + da/(dd/$5);
    else      print  "  acceptance (this cell only): n/a (no spec-decode counters)" }'
}

# NOTE ON WARM-UP: a freshly started server compiles Triton kernel variants lazily. The first
# measured run of any cell can absorb a one-off ~1.6 s compile, which on a ~24 s benchmark reads as
# a spurious ~6% loss. Discard the first pass, or treat medians across >= 3 runs as the result.
sharegpt() { [ -f "$SG" ] || { echo "  (skipped: ShareGPT dataset not at $SG)"; return; }
  $DOCKER cp "$SG" "$NAME:/tmp/sharegpt.json" >/dev/null 2>&1
  run "$1" chat "$2" "$3" --dataset-name sharegpt --dataset-path /tmp/sharegpt.json --ignore-eos; }
synth() { run "$1" chat "$2" "$3" --dataset-name random --random-input-len "$4" --random-output-len "$5" --ignore-eos; }

# aa_c1 needs REAL PROSE, not random tokens. The draft model cannot predict random tokens, so
# acceptance collapses to ~2.0 against ~2.7 on real text, and since single-stream throughput scales
# close to linearly with acceptance the cell under-reports by ~25%. Measured: 283.8 tok/s on random
# tokens vs 371.7 on prose, with acceptance 2.00 vs 2.71 -- the whole gap.
# Supply a JSONL of ~10K-token prose passages via AA_PROMPTS; the reference set is real book prose
# and is not redistributable here.
aa_cell() {
  local remote=/tmp/aa.jsonl
  if [ -n "${AA_PROMPTS:-}" ] && [ -f "${AA_PROMPTS:-}" ]; then
    $DOCKER cp "$AA_PROMPTS" "$NAME:$remote" >/dev/null 2>&1
  elif ! $DOCKER exec "$NAME" test -s "$remote" 2>/dev/null; then
    echo "  building aa_c1 prompts from public-domain prose (once)"
    $DOCKER cp "$(dirname "$0")/make_aa_prompts.py" "$NAME:/tmp/mk.py" >/dev/null 2>&1
    if ! $DOCKER exec "$NAME" bash -lc "HF_HUB_OFFLINE=0 python3 /tmp/mk.py --out $remote -n 24" 2>&1 | sed 's/^/    /'; then
      echo "=== aa_c1 SKIPPED — could not build prompts"
      echo "    No network for the public-domain download? Supply your own:"
      echo "      AA_PROMPTS=/path/to/prompts.jsonl ./bench/run_bench.sh aa_c1"
      echo "    (one {\"prompt\": \"...\"} per line, ~10K tokens of real prose each)"
      echo "    Random tokens would report ~25% low: draft acceptance collapses 2.71 -> 2.00."
      return
    fi
  fi
  run aa_c1 longin 1 10 --dataset-name custom --dataset-path "$remote" \
      --custom-output-len 1500 --ignore-eos
}

CELL=${1:-all}
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c1 ]   && sharegpt sharegpt_c1   1   100
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c32 ]  && sharegpt sharegpt_c32  32  500
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c128 ] && sharegpt sharegpt_c128 128 1000
[ "$CELL" = all ] || [ "$CELL" = sharegpt_c256 ] && sharegpt sharegpt_c256 256 1000
[ "$CELL" = all ] || [ "$CELL" = aa_c1 ]         && aa_cell
# output-len 256, NOT 1024. At 32K the ~1.6 s prefill dominates, so output length sets the number:
# 256 -> ~109 tok/s, 1024 -> ~199 tok/s, with identical per-token time. Both are "correct"; only
# 256 matches the reference table.
[ "$CELL" = all ] || [ "$CELL" = long8k_c1 ]     && synth    long8k_c1     1   20   8192  256
[ "$CELL" = all ] || [ "$CELL" = long32k_c1 ]    && synth    long32k_c1    1   10  32768  256
echo "=== results in $OUT"

# Compare against the reference table so a gap is visible immediately rather than hand-computed.
/usr/bin/env python3 - "$OUT" <<'PY' 2>/dev/null || true
import json, os, sys
REF = {"sharegpt_c1":445.2,"sharegpt_c32":4315.7,"sharegpt_c128":8644.9,
       "sharegpt_c256":10469.5,"aa_c1":371.7,"long8k_c1":277.2,"long32k_c1":109.0}
FLOOR = {"sharegpt_c1":0.4,"sharegpt_c32":3.0,"sharegpt_c128":0.7,"sharegpt_c256":10.0,
         "aa_c1":0.4,"long8k_c1":3.0,"long32k_c1":3.0}
out = sys.argv[1]
print(f"\n{'cell':<16}{'measured':>10}{'reference':>11}{'delta':>8}{'floor':>7}  verdict")
print("-"*62)
for c, r in REF.items():
    p = os.path.join(out, f"{c}.json")
    if not os.path.exists(p):
        print(f"{c:<16}{'-':>10}{r:>11.1f}{'':>8}{'':>7}  not run")
        continue
    d = json.load(open(p))
    v = d.get("output_throughput") or 0.0
    dl = 100*(v-r)/r
    f = FLOOR[c]
    verdict = "ok" if abs(dl) <= f else ("ABOVE" if dl > 0 else "BELOW reference")
    print(f"{c:<16}{v:>10.1f}{r:>11.1f}{dl:>7.1f}%{f:>6.1f}%  {verdict}")
print("\nReference = one MI300X, client inside the container. A cell outside its floor is worth")
print("chasing with ./bench/diagnose.sh; note the first run of a fresh server can absorb a ~1.6 s")
print("Triton compile, which on a short cell reads as ~6%.")
PY
