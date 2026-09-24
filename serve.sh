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
DOCKER=${DOCKER:-docker}

# Pinned by digest: the kernel overrides below are written against this vLLM build's internals
# (vllm 0.29.0). A different image can change those module paths and silently disable the overrides.
IMAGE=${IMAGE:-vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1}
V=/usr/local/lib/python3.12/dist-packages/vllm

MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
REV=${REV:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}
DRAFT=${DRAFT:-google/gemma-4-26B-A4B-it-assistant}
DRAFT_REV=${DRAFT_REV:-6e5aaaf4c42b98394530b8fda2e95cadd65c151c}

case "${1:-start}" in
  stop) $DOCKER rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME"; exit 0 ;;
  logs) exec $DOCKER logs -f "$NAME" ;;
esac

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
