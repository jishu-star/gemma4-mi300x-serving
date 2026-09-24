#!/usr/bin/env bash
# One completion against a running server. Fastest check that the stack is alive and correct.
set -uo pipefail
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
curl -sS -m 120 "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "google/gemma-4-26B-A4B-it",
  "messages": [{"role": "user", "content": "What is 17 times 23? Answer with just the number."}],
  "max_tokens": 64, "temperature": 0}' | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("answer :", d["choices"][0]["message"]["content"].strip())
print("expect : 391")
print("usage  :", d["usage"])'
