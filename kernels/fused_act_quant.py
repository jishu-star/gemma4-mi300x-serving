"""Fused GELU-tanh-and-mul + per-token FP8 quantisation, in Triton.

WHY. In the MoE path vLLM runs the gated activation and the per-token activation quantisation as two
separate kernels (triton_moe.py: `self.activation(...)` then `moe_kernel_quantize_input(...)`). On the
current shipped config that is 30 + 30 dispatches per step at ~2.5 us and ~2.68 us each, of which
92% and 86% is the measured per-dispatch floor (2.31 us on this box, after subtracting 1.89 us of
rocprofv3 overhead established by a traced-vs-untraced A/B: 6.311 ms vs 4.604 ms per step).
Fusing them removes 30 dispatches ~= 69 us of a 4604 us step ~= 1.5%.

vLLM already ships exactly this fusion for the OTHER activation: `ops.silu_and_mul_per_block_quant`,
taken when `activation == SILU and block_shape == [128,128]`. Gemma-4 uses GELU_TANH with PER-TOKEN
(not per-block) scales, so it falls to the unfused branch. The C++ op cannot be extended without
rebuilding vLLM; a Triton kernel is JIT-compiled and needs no rebuild.

NUMERICS. This must reproduce the two-kernel path, not approximate it:
  * GELU-tanh is the exact tanh approximation vLLM's CUDA kernel uses,
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    computed in fp32 regardless of input dtype, matching `gelu_tanh_and_mul`'s float accumulation.
  * The per-token scale is amax/FP8_MAX over the ROW, the same reduction
    `dynamic_per_token_scaled_fp8_quant` performs, and the same direction (scale = amax/FP8_MAX,
    q = x/scale) so the returned scale is a DEQUANT scale as the caller expects.
  * FP8_MAX is 240.0 for float8_e4m3fnuz (gfx942), NOT 448.0 as on e4m3fn. Using 448 would silently
    clip every value above 240 into saturation. The caller passes the platform dtype; we derive the
    max from it rather than hardcoding.
  * A zero row would give scale 0 and produce NaN on divide, so amax is clamped to a small epsilon,
    which is what the reference kernel does.

Acceptance length is the canary for all of this: if the fused result differs numerically from the
two-kernel path, draft/target agreement moves and that is visible long before any quality metric.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_mul_quant_kernel(
    in_ptr,              # [M, 2N] gated input: [gate | up]
    out_q_ptr,           # [M, N]  fp8 output
    out_s_ptr,           # [M, 1]  fp32 dequant scale
    M,
    N: tl.constexpr,     # output width (half the input width)
    in_stride_m,
    out_stride_m,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    # Compute the activation and its row amax. N is small for Gemma-4 (704), so a single
    # BLOCK_N >= N tile covers the row and the value is recomputed once rather than staged to LDS.
    mask = offs < N
    g = tl.load(in_ptr + row * in_stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(in_ptr + row * in_stride_m + N + offs, mask=mask, other=0.0).to(tl.float32)
    # exact tanh-approximation GELU, fp32
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    # tl.math.tanh does not exist in this Triton build (AttributeError at compile time), and the
    # libdevice tanh is CUDA-only. Use the stable identity
    #     tanh(z) = sign(z) * (1 - 2 / (exp(2|z|) + 1))
    # which avoids the exp(2z) overflow that the naive (e^2z-1)/(e^2z+1) form hits for large positive
    # z; for |z| ~ 20 the naive form is already inf/inf = NaN, and GELU inputs do reach that range.
    t = tl.exp(-2.0 * tl.abs(inner))
    tanh_abs = (1.0 - t) / (1.0 + t)
    tanh_inner = tl.where(inner >= 0.0, tanh_abs, -tanh_abs)
    act = 0.5 * g * (1.0 + tanh_inner)
    val = act * u
    row_max = tl.max(tl.abs(tl.where(mask, val, 0.0)), axis=0)

    scale = tl.maximum(row_max, 1e-12) / FP8_MAX
    q = val / scale
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    tl.store(out_q_ptr + row * out_stride_m + offs, q.to(out_q_ptr.dtype.element_ty), mask=mask)
    tl.store(out_s_ptr + row, scale)


def gelu_tanh_and_mul_per_token_quant(x: torch.Tensor, quant_dtype: torch.dtype):
    """x: [M, 2N] -> (q: [M, N] quant_dtype, scale: [M, 1] fp32).

    Returns the same (tensor, scale) pair shape as moe_kernel_quantize_input's per-token path so the
    caller is unchanged.
    """
    assert x.dim() == 2 and x.shape[1] % 2 == 0, f"expected [M, 2N], got {tuple(x.shape)}"
    M, two_n = x.shape
    N = two_n // 2
    finfo = torch.finfo(quant_dtype)
    fp8_max = float(finfo.max)          # 240.0 on e4m3fnuz, 448.0 on e4m3fn

    q = torch.empty((M, N), dtype=quant_dtype, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M == 0:
        return q, s
    BLOCK_N = triton.next_power_of_2(N)
    _gelu_tanh_mul_quant_kernel[(M,)](
        x, q, s, M, N,
        x.stride(0), q.stride(0),
        FP8_MAX=fp8_max, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return q, s
