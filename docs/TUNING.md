# How the stack was tuned

Every number here was measured on one MI300X against a matched control — the same kernel files with
the feature's flag off — not against a remembered baseline.

## The model, and why it is awkward

`gemma-4-26B-A4B-it` is 30 layers: **25 sliding** (head_dim 256, window 1024, 8 KV heads) and
**5 full-attention** (`global_head_dim` **512**, only **2** KV heads). MoE is 128 experts, top-8,
`moe_intermediate_size` 704. Activation is GELU-tanh, embeddings are tied, logits softcap at 30.

Two of those are the whole story for performance: head_dim **512** attention is far outside what
vendor kernels are tuned for, and **GELU-tanh** is not the activation any shipped fusion covers.

## Wins, with measured increments

On the AA single-stream harness (3 reps, floor 0.35%) unless noted.

| change | effect |
| --- | --- |
| `fp8_per_channel` | 269.1 → 288.9 — escapes the ROCm per-tensor downgrade |
| cudagraph capture to 3072 | +16.3% on c256 |
| MTP n=3 | acceptance 2.28 → 2.79, 289.5 → 303.1 |
| FP8 LM head | 303.1 → 312.8 — a 1.476 GB BF16 read per step |
| `NX_SEG_CHOICES=16,64,128` | 312.8 → 317.6 |
| MoE output aliasing | ~40 fewer copy dispatches/step |
| **adaptive split-KV on the MTP path** | **319.6 → 370.9 (+16.8%)** |
| fused GELU-tanh + FP8 quant | 374.5 → 383.9, 30 fewer dispatches/step |
| cached padding tensor | 383.9 → 389.2, 30 fewer dispatches/step |
| exp2 prefill softmax | 32K TTFT −5.6% |
| long-context prefill tiling | long32k +26.2%, 32K TTFT −29% |

### The largest single win was a gate that never fired

`triton_attn.py` gated the adaptive split-KV segment picker on `max_query_len == 1`, i.e. plain
decode. **An MTP n=3 step has `max_query_len = 4`**, so the gate never fired and the fixed 16
segments were always used. The picker targets CU saturation — `segments ≈ CU_count/(num_seqs·kv_heads)`:

- c=1 → 64 segments → 512 workgroups, against 128 on a 304-CU GPU (58% idle)
- c≥32 → resolves to 16, i.e. unchanged

That arithmetic predicted the shape of the result *before* the matrix ran: c32/c128/c256 moved
+0.1/+0.2/+0.2% while aa_c1 went +14.4%, long8k +11.5%, long32k +7.5%.

## The LDS wall

The remaining gap to H100 at 32K context is **not** a tuning gap.

Raising `BLOCK_M` is the only parameter that changes arithmetic intensity on this kernel — the tile
size cancels exactly:

```
FLOP/byte = (BLOCK_M · T · d · 4) / (T · d · 4) = BLOCK_M
```

But on the head-512 layers, `BLOCK_M = 64` does not launch:

```
OutOfResources: shared memory, Required: 69632, Hardware limit: 65536
```

The Q tile alone at BM=64 is `64 × 512 × 2 = 65,536 B` — **the entire 64 KB LDS of a gfx942 CU**,
leaving nothing for the S tile. The requirement is identical at 4 and 8 warps, because LDS is
allocated per workgroup: this is not register pressure. No `TILE_SIZE` reduction recovers it.

**H100 runs `tile_m = 64` at this head size because Hopper has 228 KB of shared memory per SM —
3.6× more.** `BLOCK_M = 32` is the ceiling *for this kernel structure*.

> **Correction (measured, same day).** An earlier version of this section said the 32K gap was
> "not closable from our side". That was an overreach. AMD's own AITER unified attention — also a
> Triton kernel — hits the identical wall unpatched (`Required: 131072`, i.e. `2 × TILE(64) × 512 ×
> 2`), but once its tile is reduced to fit it reaches a **31% better 32K TTFT than our fully tuned
> kernel** (1465 vs 1585 ms) and **+71% throughput over stock Triton** on `long32k`, untuned.
> The 64 KB LDS limit is real and all three ROCm backends hit it; what it bounds is a particular
> tiling, not the problem. The 0.68× H100 figure is this kernel's ceiling, not the hardware's.

The 25 **sliding** layers are head_dim 256, so their Q tile is half as wide and `BLOCK_M = 128`
fits (`128 × 256 × 2 = 65,536`). Measured per dispatch at 32K: 6.889 → 2.954 ms (**−57.1%**).

## Traps worth knowing

**`max_seqlen_k` is not the context length.** A gate census over one `sharegpt_c128` run found it
reads **131072 — the full `max-model-len`** — on 5,192 of 26,578 calls (the MTP decode calls,
`q_len = 4`), against a true ShareGPT context of ≤ 2048.

**`TILE_SIZE_PREFILL` is not prefill-only.** The 2D launch path uses it for **decode** as well; only
the 3D segmented path reads `TILE_SIZE_DECODE`.

Together those made a context-only gate fire 1,028 times on `sharegpt_c128` and run its *decode* at
TILE=16, costing a real, reproducible 0.81%. Three rounds of statistics — replication, order
reversal, and a never-fire arm — failed to find it, and two proposed explanations were disproved.
One run of a kernel counting its own gate firings settled it. Every gate in this repo therefore
requires **both** a query-length and a context-length condition.

**An inactive knob is not a free knob.** `sharegpt_c128` is host-dispatch-bound, so four
`os.environ.get` calls per attention call (1.548 µs measured) are charged against it. All NX knobs
are resolved once at import, never in the per-layer path.

## Negative results — do not retry these

| tried | outcome |
| --- | --- |
| `BLOCK_M = 64` or 128 on head-512 | impossible: LDS, see above |
| KV tile 64 on head-512 | LDS 131,072 B — the double-buffered K/V staging buffer |
| 8 warps with `BLOCK_M = 32` | 199.5 vs 149.3 ms/dispatch — *destroys* the BLOCK_M gain |
| sliding `BLOCK_M = 64` as a middle point | same acceptance loss as 128 with less gain; strictly dominated |
| fp8 KV cache to relieve LDS | ROCm leaves Q in bf16 (`supports_quant_query_input = is_cuda()`), so the Q tile does not shrink; measured *slower* |
| MoE tuned-config file for M=4 | the file path uses nearest-key lookup, the default path uses exact M — no file can express "defaults except M=4"; mounting any file costs 5–7% |
| AITER assembly MoE | cannot serve this model in any precision — no `Gelu_tanh` in `ActivationType` |
| `ROCM_ATTN`, `ROCM_AITER_FA` backends | reject `head_size` 512 outright (supported list caps at 256) |
| `ROCM_AITER_UNIFIED_ATTN` unpatched | loads, then `OutOfResources: Required 131072` — its gfx942 tile is 2× what fits at head 512 |

## Measurement discipline

- A control must measure **every cell the arm could regress**. A control narrower than the matrix
  you report is not a control.
- Acceptance length is a separate gate. Under greedy drafting the accept test is an argmax
  comparison — a discrete test — so numerical perturbation invisible to perplexity shows up as a
  throughput cliff. It has caught three changes that throughput called neutral.
- A reproducible effect with no mechanism means an assumption is wrong, not the measurement.
  Instrument; do not infer.
