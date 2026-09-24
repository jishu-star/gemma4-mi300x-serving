#!/usr/bin/env bash
# Why is this machine slower than the README numbers?
#
# The reference figures were measured on one specific MI300X box with the client INSIDE the
# container. This prints the handful of things that actually move single-stream throughput, each
# next to its reference value, so a gap can be attributed instead of guessed at.
#
# Run it with the server already up (./serve.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-gemma4-vllm}; HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
DOCKER=${DOCKER:-docker}
ok(){ printf '  %-34s %s\n' "$1" "$2"; }

echo "=== 1. GPU ==="
$DOCKER exec "$NAME" bash -c 'rocm-smi --showproductname --showpower --showclocks 2>/dev/null' 2>/dev/null \
  | grep -iE "card series|card model|power cap|sclk|mclk" | head -8 | sed 's/^/  /' \
  || echo "  (rocm-smi unavailable)"
ok "reference" "AMD Instinct MI300X (gfx942), 304 CU"
echo "  NOTE: a different SKU (MI325X, MI300A) or a lower power cap changes everything below."

echo
echo "=== 2. Are the kernel overrides actually live? ==="
# A mount that silently failed is the single easiest way to lose double-digit percent.
V=/usr/local/lib/python3.12/dist-packages/vllm
for f in "kernels/triton_unified_attention.py:$V/v1/attention/ops/triton_unified_attention.py" \
         "kernels/triton_attn.py:$V/v1/attention/backends/triton_attn.py" \
         "kernels/triton_moe.py:$V/model_executor/layers/fused_moe/experts/triton_moe.py" \
         "kernels/fused_act_quant.py:$V/nx_fused_act_quant.py"; do
  L=${f%%:*}; C=${f#*:}
  h1=$(md5sum "$L" 2>/dev/null | cut -c1-8)
  h2=$($DOCKER exec "$NAME" md5sum "$C" 2>/dev/null | cut -c1-8)
  [ "$h1" = "$h2" ] && ok "$(basename $L)" "MOUNTED ok ($h1)" || ok "$(basename $L)" "*** MISMATCH host=$h1 container=$h2 ***"
done

echo
echo "=== 3. Speculative decoding — the biggest single-stream lever ==="
AL=$(curl -fsS "http://$HOST:$PORT/metrics" 2>/dev/null \
      | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | awk '{s[$1]=$2} END{
        a=0;d=0; for(k in s){ if(k ~ /accepted/) a=s[k]; if(k ~ /draft/) d=s[k] }
        if(d>0) printf "%.3f", 1 + a/(d/3); else print "n/a" }')
ok "acceptance length" "${AL:-n/a}"
ok "reference" "~2.71  (1.0 = drafter not working at all)"
echo "  Throughput scales ~linearly with this. 2.71 -> 2.10 is -23% on its own."

echo
echo "=== 4. Host dispatch — 920 launches per decode step at concurrency 1 ==="
ok "CPU model" "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs)"
ok "cores / threads" "$(nproc 2>/dev/null)"
ok "governor" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "  920 dispatches x 2.31 us = ~2.1 ms of pure launch cost per step. A slow core, a low core"
echo "  count, or a 'powersave' governor is charged directly against single-stream throughput."
echo "  Set 'performance' if this says powersave."

echo
echo "=== 5. CUDA graphs and MoE backend ==="
$DOCKER logs "$NAME" 2>&1 | grep -iE "Capturing cudagraphs|graph capturing finished|Using .* MoE backend|enforce_eager" \
  | tail -4 | sed 's/^/  /' || echo "  (nothing in log)"
ok "expected" "cudagraphs captured; TRITON Fp8 MoE; enforce_eager NOT set"

echo
echo "=== 6. Client placement ==="
echo "  The README numbers were measured with the client INSIDE the container (pure localhost)."
echo "  Benchmarking from another host adds ~1 RTT to TTFT. It should NOT change tok/s materially;"
echo "  if it does, suspect the link or the client, not the server."
