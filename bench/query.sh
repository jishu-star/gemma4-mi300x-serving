#!/usr/bin/env bash
# Query the server FROM ANOTHER MACHINE. The only script here that is meant to run off-box.
#
#   ./bench/query.sh "What is 17 times 23?"                    # localhost (or an SSH tunnel)
#   HOST=10.0.0.5 ./bench/query.sh "Hello"                     # direct, server bound 0.0.0.0
#   HOST=10.0.0.5 API_KEY=abc123 ./bench/query.sh "Hello"      # direct, server started with API_KEY
#   ./bench/query.sh -s "Write a haiku about GPUs"             # stream tokens as they arrive
#
# Needs only curl and python3 — no vLLM, no GPU, no clone of the model.
#
# If the server is bound to 127.0.0.1 (the default) open a tunnel first, then use the default HOST:
#   ssh -N -L 8000:127.0.0.1:8000 user@gpu-host
set -uo pipefail
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
MAX_TOKENS=${MAX_TOKENS:-256}; TEMPERATURE=${TEMPERATURE:-0}
STREAM=0
[ "${1:-}" = "-s" ] && { STREAM=1; shift; }
PROMPT=${*:-"What is 17 times 23? Answer with just the number."}
URL="http://$HOST:$PORT/v1/chat/completions"
AUTH=(); [ -n "${API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $API_KEY")

if ! curl -fsS -m 10 ${AUTH[@]+"${AUTH[@]}"} "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "cannot reach http://$HOST:$PORT" >&2
  echo "  - server bound to 127.0.0.1? open a tunnel: ssh -N -L $PORT:127.0.0.1:$PORT user@gpu-host" >&2
  echo "  - bound to 0.0.0.0 but started with API_KEY? pass API_KEY=... to this script" >&2
  echo "  - firewall may be blocking port $PORT" >&2
  exit 1
fi

BODY=$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],
 "max_tokens":int(sys.argv[3]),"temperature":float(sys.argv[4]),"stream":sys.argv[5]=="1"}))' \
 "$MODEL" "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" "$STREAM")

if [ "$STREAM" = 1 ]; then
  curl -sS -N -m 600 "$URL" ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -u -c '
import json,sys
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: "): continue
    if line=="data: [DONE]": print(); break
    try: d=json.loads(line[6:])
    except: continue
    t=(d.get("choices") or [{}])[0].get("delta",{}).get("content")
    if t: sys.stdout.write(t)'
else
  curl -sS -m 600 "$URL" ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d:
    print("server error:", d["error"], file=sys.stderr); raise SystemExit(1)
print(d["choices"][0]["message"]["content"].strip())
u = d.get("usage") or {}
pt, ct = u.get("prompt_tokens", "?"), u.get("completion_tokens", "?")
print("[%s in / %s out]" % (pt, ct), file=sys.stderr)'
fi
