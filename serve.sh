#!/usr/bin/env bash
# Serve Gemma-4-26B-A4B on a single AMD MI300X with the tuned kernel stack.
#
# Starts an OpenAI-compatible server on http://127.0.0.1:8000.
# Requires: ROCm host with /dev/kfd and /dev/dri, docker, and the model in the HF cache.
#
#   ./serve.sh              start
#   ./serve.sh stop         stop and remove the container
#   ./serve.sh logs         follow the server log
set -uo pipefail
cd "$(dirname "$0")"; ROOT=$(pwd)

NAME=${NAME:-gemma4-vllm}
PORT=${PORT:-8000}
# 127.0.0.1 by default ON PURPOSE. --network host means binding 0.0.0.0 publishes this server on
# every interface of the machine, and vLLM has NO authentication unless API_KEY is set below. An
# open endpoint is free use of your GPU by anyone who can reach the port, and they can read every
# prompt sent through it. For remote access prefer an SSH tunnel (see README); if you must bind
# externally, set API_KEY and firewall the port.
HOST=${HOST:-127.0.0.1}
API_KEY=${API_KEY:-}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
VLLM_CACHE=${VLLM_CACHE:-$HOME/.cache/vllm}
DOCKER=${DOCKER:-docker}   # preflight may rewrite this to "sudo -n docker"

# Pinned by digest: the kernel overrides below are written against this vLLM build's internals
# (vllm 0.29.0). A different image can change those module paths and silently disable the overrides.
IMAGE=${IMAGE:-vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1}
V=/usr/local/lib/python3.12/dist-packages/vllm

MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
REV=${REV:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}
DRAFT=${DRAFT:-google/gemma-4-26B-A4B-it-assistant}
DRAFT_REV=${DRAFT_REV:-6e5aaaf4c42b98394530b8fda2e95cadd65c151c}

usage() {
  cat <<USAGE
Serve Gemma-4-26B-A4B on one AMD MI300X.

  ./serve.sh            preflight, then start; waits until /health answers
  ./serve.sh check      preflight only — run this first on a new machine
  ./serve.sh stop       stop and remove the container
  ./serve.sh logs       follow the server log
  ./serve.sh status     is it up, and what is it serving

Environment overrides:
  PORT=8000  HOST=127.0.0.1  API_KEY=  NAME=gemma4-vllm
  HF_CACHE=~/.cache/huggingface   VLLM_CACHE=~/.cache/vllm

HOST defaults to 127.0.0.1 deliberately. See "Accessing the server from another
machine" in README.md before changing it — there is no auth unless API_KEY is set.
USAGE
}

preflight() {
  local fail=0
  step(){ printf '  %-42s' "$1"; }
  pass(){ echo "OK${1:+  ($1)}"; }
  bad(){  echo "FAIL  $1"; fail=1; }

  step "docker"
  if ! command -v docker >/dev/null 2>&1; then
    bad "docker not installed"
  elif $DOCKER info >/dev/null 2>&1; then
    pass "${DOCKER}"
  elif sudo -n docker info >/dev/null 2>&1; then
    # many hosts require root for the daemon socket; use it rather than failing
    DOCKER="sudo -n docker"; pass "via sudo"
  else
    bad "cannot reach the docker daemon as $(id -un), and passwordless sudo is unavailable.
      Either add yourself to the docker group (newgrp docker) or run: DOCKER='sudo docker' ./serve.sh"
  fi

  step "AMD GPU devices (/dev/kfd, /dev/dri)"
  if [ -e /dev/kfd ] && [ -d /dev/dri ]; then pass
  else bad "missing — this needs a ROCm host with an AMD Instinct GPU"; fi

  step "GPU is MI300X (gfx942)"
  local arch
  arch=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || echo "")
  if [ "$arch" = "gfx942" ]; then pass "$arch"
  elif [ -n "$arch" ]; then echo "WARN  found $arch, tuned for gfx942 — expect different numbers"
  else echo "WARN  rocminfo not on PATH; skipping (the container has its own)"; fi

  step "disk free for image + weights (~160 GB)"
  local avail; avail=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$avail" ] && [ "$avail" -ge 160 ]; then pass "${avail}G"
  elif [ -n "$avail" ]; then bad "only ${avail}G free under $HOME"
  else echo "WARN  could not read"; fi

  step "model in HF cache"
  if [ -d "$HF_CACHE/hub/models--google--gemma-4-26B-A4B-it" ]; then pass
  else
    bad "not found under $HF_CACHE"
    echo "      download both the model and its MTP drafter:"
    echo "        pip install -U huggingface_hub"
    echo "        hf download $MODEL --revision $REV"
    echo "        hf download $DRAFT --revision $DRAFT_REV"
  fi

  step "MTP drafter in HF cache"
  if [ -d "$HF_CACHE/hub/models--google--gemma-4-26B-A4B-it-assistant" ]; then pass
  else bad "not found — speculative decoding needs it; see above"; fi

  step "kernel override files present"
  local missing=0 f
  for f in triton_attn.py triton_unified_attention.py triton_attention_helpers.py \
           vocab_parallel_embedding.py modular_kernel.py triton_moe.py \
           fused_act_quant.py fused_moe.py; do
    [ -f "$ROOT/kernels/$f" ] || { missing=$((missing+1)); }
  done
  [ "$missing" = 0 ] && pass "8 files" || bad "$missing missing from kernels/ — incomplete clone?"

  step "port $PORT free"
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    bad "something is already listening on $PORT (./serve.sh stop, or set PORT=)"
  else pass; fi

  return $fail
}

case "${1:-start}" in
  -h|--help|help) usage; exit 0 ;;
  check)  echo "=== preflight ==="; preflight && { echo; echo "ready — run ./serve.sh"; exit 0; } || { echo; echo "fix the FAIL lines above"; exit 1; } ;;
  stop)   $DOCKER rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"; exit 0 ;;
  logs)   exec $DOCKER logs -f "$NAME" ;;
  status)
    if [ -n "$($DOCKER ps -q -f name="^${NAME}$" 2>/dev/null)" ]; then
      echo "container: up"
      curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
        | python3 -c "import json,sys;[print('serving :',m['id'],'| max_model_len',m.get('max_model_len')) for m in json.load(sys.stdin).get('data',[])]" \
        2>/dev/null || echo "serving : not answering yet"
    else echo "container: down"; fi
    exit 0 ;;
  start) ;;
  *) echo "unknown command: $1"; echo; usage; exit 1 ;;
esac

echo "=== preflight ==="
preflight || { echo; echo "preflight failed — fix the above, or ./serve.sh check for detail"; exit 1; }
echo

if [ "$HOST" = "0.0.0.0" ] && [ -z "$API_KEY" ]; then
  echo "WARNING: binding 0.0.0.0 with no API_KEY — this endpoint is open to anyone who can reach"
  echo "         port $PORT. Set API_KEY=... or use an SSH tunnel instead (see README)."
  echo "         Continuing in 5s; Ctrl-C to abort."
  sleep 5
fi

if [ -n "$($DOCKER ps -q -f name="^${NAME}$" 2>/dev/null)" ]; then
  echo "$NAME is already running. './serve.sh stop' first."; exit 1
fi

# cudagraph capture sizes: the tail above 256 is what makes the large-batch cells work.
CG='[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,768,896,1024,1152,1280,1408,1536,1792,2048,2560,3072]'

set -x
$DOCKER run -d --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host --network host --shm-size 16g --security-opt label=disable \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$VLLM_CACHE:/root/.cache/vllm" \
  -v "$ROOT/moe_configs:/tuned_moe:ro" \
  -v "$ROOT/kernels/triton_attn.py:$V/v1/attention/backends/triton_attn.py:ro" \
  -v "$ROOT/kernels/triton_unified_attention.py:$V/v1/attention/ops/triton_unified_attention.py:ro" \
  -v "$ROOT/kernels/triton_attention_helpers.py:$V/v1/attention/ops/triton_attention_helpers.py:ro" \
  -v "$ROOT/kernels/vocab_parallel_embedding.py:$V/model_executor/layers/vocab_parallel_embedding.py:ro" \
  -v "$ROOT/kernels/modular_kernel.py:$V/model_executor/layers/fused_moe/modular_kernel.py:ro" \
  -v "$ROOT/kernels/triton_moe.py:$V/model_executor/layers/fused_moe/experts/triton_moe.py:ro" \
  -v "$ROOT/kernels/fused_act_quant.py:$V/nx_fused_act_quant.py:ro" \
  -v "$ROOT/kernels/fused_moe.py:$V/model_executor/layers/fused_moe/fused_moe.py:ro" \
  -e TRITON_CACHE_DIR=/root/.cache/vllm/triton \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 \
  -e VLLM_TUNED_CONFIG_FOLDER=/tuned_moe \
  -e NX_ADAPTIVE_SEGMENTS=1 -e NX_SEG_CHOICES=16,64,128 -e NX_MAX_ADAPTIVE_QLEN=4 \
  -e NX_SPEC_GEOMETRY=1 -e NX_VERIFY_SPLIT=1 \
  -e BH6_FP8_LM_HEAD=1 -e NX_FUSED_ACTQUANT=1 \
  -e NX_PF_BM_512=32 -e NX_PF_WARPS_512=4 -e NX_PF_BM_512_MINSEQ=4096 \
  -e NX_PF_TILE_512=16 -e NX_PF_TILE_512_MINSEQ=16384 \
  -e NX_ATTN_BM256=128 -e NX_ATTN_WARPS_256=4 -e NX_ATTN_BM256_MINSEQ=4096 \
  "$IMAGE" "$MODEL" \
  ${API_KEY:+--api-key "$API_KEY"} \
  --host "$HOST" --port "$PORT" \
  --revision "$REV" --tokenizer-revision "$REV" \
  --language-model-only --generation-config vllm \
  --dtype bfloat16 --quantization fp8_per_channel --kv-cache-dtype auto \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --max-model-len 131072 --max-num-seqs 256 --max-num-batched-tokens 32768 \
  --enable-chunked-prefill --no-enable-prefix-caching --seed 0 \
  --no-enforce-eager --optimization-level 2 --performance-mode balanced \
  --compilation-config "{\"cudagraph_capture_sizes\":$CG}" \
  --speculative-config "{\"model\":\"$DRAFT\",\"revision\":\"$DRAFT_REV\",\"num_speculative_tokens\":3}"
set +x

echo "waiting for /health ..."
for i in $(seq 1 240); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "ready after ${i}0s"; exit 0
  fi
  if [ -z "$($DOCKER ps -q -f name="^${NAME}$")" ]; then
    echo "container exited:"; $DOCKER logs --tail 40 "$NAME"; exit 1
  fi
  sleep 10
done
echo "timed out; last log:"; $DOCKER logs --tail 40 "$NAME"; exit 1
