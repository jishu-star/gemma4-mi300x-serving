# Gemma-4-26B-A4B on a single MI300X — tuned vLLM serving stack

An OpenAI-compatible server for `google/gemma-4-26B-A4B-it` on one AMD Instinct MI300X,
**2.0–2.4× stock vLLM** across the workload matrix, built from Triton kernel overrides mounted into
a pinned vLLM image. No vLLM rebuild required — every override is a Python file that Triton
JIT-compiles at runtime.

```bash
git clone https://github.com/jishu-star/gemma4-mi300x-serving
cd gemma4-mi300x-serving

./serve.sh check      # preflight: docker, GPU, disk, weights, port — run this first
./serve.sh            # start; waits until /health answers
./bench/smoke.sh      # one completion — expects "391"
./bench/run_bench.sh  # reproduce the results matrix
./serve.sh status     # what is it serving
./serve.sh logs       # follow the server log
./serve.sh stop
```

`./serve.sh check` tells you what is missing before anything is downloaded or launched, including
the exact `hf download` commands for the weights. It also detects whether your docker daemon needs
`sudo` and uses it rather than failing.

## Requirements

- AMD Instinct MI300X (gfx942), ROCm host exposing `/dev/kfd` and `/dev/dri`
- Docker, ~160 GB free for the image + model
- The model in your HF cache (`HF_CACHE`, default `~/.cache/huggingface`):
  - `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  - `google/gemma-4-26B-A4B-it-assistant` @ `6e5aaaf4c42b98394530b8fda2e95cadd65c151c` (MTP drafter)

The vLLM image is **pinned by digest**. The overrides are written against that build's internals;
a different image can move those module paths and silently disable them.

## Results

Single MI300X. `tok/s` is output-token throughput. Stock = unmodified vLLM, same image and model.
H100 = the same model and quantization on an H100 at matched `--max-model-len 131072`.

| cell | stock | H100 | **this stack** | ×stock | ×H100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sharegpt_c1 | 230.2 | 366.1 | **445.2** | 1.93× | 1.22× |
| sharegpt_c32 | 1764.5 | 4418.5 | **4315.7** | 2.45× | 0.98× |
| sharegpt_c128 | 3984.1 | 6972.1 | **8644.9** | 2.17× | 1.24× |
| sharegpt_c256 | 5325.1 | 9077.0 | **10469.5** | 1.97× | 1.15× |
| aa_c1 (10K in / 1.5K out) | 163.5 | 355.9 | **371.7** | 2.27× | 1.04× |
| long8k_c1 | 144.6 | 281.0 | **277.2** | 1.92× | 0.99× |
| long32k_c1 | 65.0 | 160.0 | **109.0** | 1.68× | 0.68× |

32K time-to-first-token: **2160 → 1585 ms (−27%)**.

**Cell definitions matter more than they look.** `long8k_c1`/`long32k_c1` use
`--random-output-len 256`; at 32K a ~1.6 s prefill dominates, so running 1024 instead reports
~199 tok/s rather than ~109 — same per-token time, different amortisation. And `aa_c1` needs **real
prose**: on random tokens the drafter's acceptance falls from ~2.71 to ~2.00 and the cell reports
~25% low (measured 283.8 vs 371.7). `bench/run_bench.sh` encodes the reference definitions and
builds `aa_c1`'s prompts for you on first use, from public-domain prose
(`bench/make_aa_prompts.py`, Project Gutenberg). Offline or want your own corpus:

```bash
python3 bench/make_aa_prompts.py --src your_book.txt --out aa.jsonl
AA_PROMPTS=aa.jsonl ./bench/run_bench.sh aa_c1
```

Absolute tok/s shifts a little with the passage; acceptance lands in the right regime, which is what
the cell is measuring.

All figures are a single verification run of the shipped configuration, taken after a discarded
warm-up pass (see *Benchmarking notes* — the warm-up read 8614.6 on c128 against 8644.9 measured).
Every ShareGPT cell sits inside its noise floor relative to the previous release.

### What this does NOT improve

**Artificial Analysis Output Speed is unchanged: 388.2 vs 388.0 tok/s.** That metric is
`output_tokens / (total_time − TTFT)` — it subtracts TTFT by construction, so the long-context
prefill work here cannot move it. AA's separately published TTFT does improve, 0.38 → 0.33 s.
Quoted honestly: the 2.28× on `aa_c1` comes from decode-side work (speculative decoding, FP8,
adaptive split-KV, MoE fusion); the long-context gains come from prefill work that AA's headline
number excludes.

`sharegpt_c32` sits at 0.98× H100 — the one cell where this stack does not lead.
`long32k_c1` at 0.70× is explained in [docs/TUNING.md](docs/TUNING.md#the-lds-wall): a hard
shared-memory capacity limit, not a tuning gap.

## What's in the stack

| component | what it does |
| --- | --- |
| `--quantization fp8_per_channel` | avoids the ROCm per-tensor downgrade |
| MTP n=3 speculative decoding | separate assistant draft model, acceptance ≈ 2.7 |
| adaptive split-KV segments | segment count targets CU saturation; the single largest win (+16.8%) |
| FP8 LM head | removes a 1.476 GB BF16 read every step |
| fused GELU-tanh + per-token FP8 quant | one kernel instead of two, −30 dispatches/step |
| MoE output aliasing + cached padding tensor | −70 dispatches/step |
| long-context prefill tiling | `BLOCK_M`/tile tuning on both attention layer types, gated to ≥4096 context |
| extended cudagraph capture sizes | up to 3072 |
| tuned MoE GEMM config | `moe_configs/`, E=128 N=704 |

Full derivation, measurements and the negative results in [docs/TUNING.md](docs/TUNING.md).

## If your numbers are lower than the table

Run `./bench/diagnose.sh` with the server up. It checks, in order of how much each can cost you:

1. **GPU SKU and power cap** — the table is one MI300X (gfx942, 304 CU). A different SKU or a lower
   cap changes everything.
2. **Whether the kernel overrides are actually live** — it md5s each file inside the container
   against the repo copy. A silently failed mount is the easiest way to lose double digits, and the
   server starts perfectly happily without them.
3. **Acceptance length** — reference ~2.71. Single-stream throughput scales close to linearly with
   it, so a drafter that failed to load (1.0) or is accepting 2.1 is −23% on its own. This is the
   first thing to check for a large single-stream gap.
4. **Host dispatch** — a decode step issues **920 kernel launches**; at the measured 2.31 µs floor
   that is ~2.1 ms of pure launch cost per step at concurrency 1. A low core count or a `powersave`
   governor is charged straight against throughput.
5. **CUDA graphs and MoE backend** — graphs captured, `TRITON Fp8 MoE`, `enforce_eager` unset.

A one-off Triton compile explains ~6% on a short benchmark, no more. A larger gap is one of the
above, not warm-up.

## Benchmarking notes

Two things will mislead you if you measure carelessly:

1. **Warm-up.** A fresh server compiles Triton kernel variants lazily. The first measured run of a
   cell can absorb a one-off ~1.6 s compile; on a ~24 s benchmark that reads as a spurious ~6% loss.
   Discard the first pass or take medians over ≥3 runs.
2. **Noise floors.** Measured on this hardware, these are the limits below which a difference means
   nothing: `sharegpt_c1` 0.4%, `aa_c1` 0.36%, `sharegpt_c128` 0.7%, `long8k`/`long32k` 3%,
   `sharegpt_c256` 4–10%. Ranking changes below these is measuring noise.

## Licence

Apache-2.0. The files in `kernels/` are modified from the
[vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0) and retain their SPDX headers;
see [NOTICE](NOTICE).

## Querying it from another machine

`bench/query.sh` is the client — the only script here meant to run off-box. It needs just `curl` and
`python3`: no vLLM, no GPU, no model download.

```bash
./bench/query.sh "What is 17 times 23?"                 # localhost, or through an SSH tunnel
HOST=10.0.0.5 ./bench/query.sh "Hello"                  # server bound 0.0.0.0
HOST=10.0.0.5 API_KEY=abc ./bench/query.sh "Hello"      # server started with API_KEY
./bench/query.sh -s "Write a haiku about GPUs"          # stream tokens as they arrive
```

It is an OpenAI-compatible endpoint, so any OpenAI client works the same way:

```python
from openai import OpenAI
c = OpenAI(base_url="http://<host>:8000/v1", api_key="<API_KEY or 'none'>")
print(c.chat.completions.create(
    model="google/gemma-4-26B-A4B-it",
    messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

## Accessing the server from another machine

By default the server binds `127.0.0.1` — local only. There is **no authentication** in vLLM unless
you set one, and `--network host` means binding `0.0.0.0` publishes the endpoint on every interface.
An open endpoint is free use of your GPU by anyone who can reach the port, and they can read every
prompt sent through it.

**Preferred — SSH tunnel.** Nothing is exposed; the server stays bound to localhost:

```bash
# on your laptop
ssh -N -L 8000:127.0.0.1:8000 user@gpu-host
# then use http://127.0.0.1:8000 locally, as if it were on your machine
curl http://127.0.0.1:8000/v1/models
```

**If you must bind externally**, set a key and firewall the port:

```bash
HOST=0.0.0.0 API_KEY="$(openssl rand -hex 32)" ./serve.sh
# clients then send:  Authorization: Bearer <that key>
```

### Does remote access cost inference speed?

- **Token generation rate: no measurable effect.** Binding `0.0.0.0` is the same server and the same
  kernels. With streaming over a persistent connection every token is delayed by the same one-way
  latency, so the gaps between tokens — which is what tok/s measures — are unchanged. You get a
  constant offset, not a compounding one.
- **TTFT: +~1 round-trip.** Sub-millisecond on a LAN (invisible against 1535 ms at 32K context);
  50–100 ms across a continent, which matters most for the short-prompt cells where TTFT is ~29 ms.
- **Throughput at concurrency: unaffected**, unless your link saturates — at ~8,650 tok/s on
  `sharegpt_c128` the JSON stream is on the order of tens of Mbit/s, so a 1 Gbit link is ample.

Every number in the results matrix above was measured **with the client inside the container**, i.e.
pure localhost. They measure the serving stack, not your network path.
