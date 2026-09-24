# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import get_dtype_size, is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_tensor,
    get_num_attention_heads_from_layers,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    MAX_UNIFORM_DECODE_QUERY_LEN,
    _supports_uniform_decode,
    unified_attention,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)

logger = init_logger(__name__)


# constants
import os as _nx_os
MIN_LAUNCH_GRID_SIZE_2D = int(_nx_os.environ.get("NX_MIN_LAUNCH_GRID_2D", "128"))  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = int(_nx_os.environ.get("NX_PAR_SOFTMAX_SEGMENTS", "16"))  # Parallel tiled softmax segments

# NX adaptive segments (README_ADAPTIVE.md). The 3D decode grid is (q_blocks, kv_heads, segments) and in decode q_blocks
# ~= num_seqs, so a fixed 16 leaves 1x8x16 = 128 workgroups of 304 CUs at batch 1 (42%) while oversubscribing at batch 16.
# Measured at batch 1 / 32K: 64 segments = -3.3% e2e, and 128 is no better (results/summary_bs1_in32768.txt). The kernels
# index the partial buffers with the segment count as a constexpr multiplier, so each choice needs its own buffers: we
# pre-allocate a small set and select per step. NX_ADAPTIVE_SEGMENTS=0 restores the shipped fixed-16 behaviour.
NX_ADAPTIVE_SEGMENTS = _nx_os.environ.get("NX_ADAPTIVE_SEGMENTS", "1") == "1"
# NX: largest max_query_len for which the adaptive picker runs. 1 = upstream behaviour
# (plain decode only); 4 = also the MTP n=3 uniform verify step.
NX_MAX_ADAPTIVE_QLEN = int(_nx_os.environ.get("NX_MAX_ADAPTIVE_QLEN", "1"))
NX_SEG_CHOICES = tuple(int(x) for x in _nx_os.environ.get("NX_SEG_CHOICES", "16,64,128").split(","))
NX_TILE_SIZE = 16  # decode tile in triton_unified_attention (q.element_size() >= 2)
NX_SPEC_GEOMETRY = _nx_os.environ.get("NX_SPEC_GEOMETRY", "0") == "1"  # PR #56996; default off so arms A/B cleanly
NX_VERIFY_SPLIT = _nx_os.environ.get("NX_VERIFY_SPLIT", "0") == "1"   # PR #56148; must match the kernel module
NX_KV_ALIAS = _nx_os.environ.get("NX_KV_ALIAS", "0") == "1"           # K==V alias probe (Gemma attention_k_eq_v)


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    causal: bool | torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None
    is_uniform_decode: bool = False


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    # Step-dependent fields reference persistent input buffers directly.
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # Compatible with models with non-uniform per-layer head counts.
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        # NX (vLLM PR #56996): take the geometry from THIS attention group's kv_cache_spec, not from model_config.
        # model_config.get_num_kv_heads()/get_head_size() are model-wide and collapse heterogeneous layers with max, so
        # on Gemma-4 every builder sees kv_heads=8 / head_size=512 even though the 5 full-attention layers have 2 KV
        # heads and the 25 sliding layers have head_size 256. Two consequences we measured: seq_threshold_3D became
        # 128//8 = 16 instead of 128//2 = 64 for the full-attention group, so batches 17-64 dropped to the 2D path with
        # a grid far below MIN_LAUNCH_GRID_SIZE_2D; and the softmax scratch was sized at head 512 for every group.
        # Upstream measured this on gemma-4-26B-A4B: full-attention decode latency at ctx 4096 improved +60.2% at
        # batch 24, +50.9% at 32, +35.7% at 48, +23.1% at 64, with batches 8/16/96/128 unchanged within 1.3%.
        if NX_SPEC_GEOMETRY:
            self.num_heads_kv = kv_cache_spec.num_kv_heads
            self.headdim = kv_cache_spec.head_size
        else:
            self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
            self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        # NX (PR #56148): under speculation the 3D scratch is indexed by query token, so a verify batch of num_seqs
        # sequences needs num_seqs * query_len rows. query_len = 1 + num_spec (x2 when the drafter drafts in parallel).
        max_query_len_3d = 1
        _spec = vllm_config.speculative_config
        if _spec is not None:
            _qlen = 1 + (_spec.num_speculative_tokens or 0) * (
                2 if getattr(_spec, "parallel_drafting", False) else 1
            )
            _mns = vllm_config.scheduler_config.max_num_seqs
            if _supports_uniform_decode(_qlen, _mns, _mns * _qlen):
                max_query_len_3d = _qlen
        self.max_query_len_3d = max_query_len_3d
        max_num_tokens_3d = (
            min(self.seq_threshold_3D, vllm_config.scheduler_config.max_num_seqs) * max_query_len_3d
        )
        self.max_num_tokens_3d = max_num_tokens_3d
        self.softmax_segm_output = torch.empty(
            (
                max_num_tokens_3d,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (max_num_tokens_3d, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (max_num_tokens_3d, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.rswa_window = model_config.rswa_window
        self.persistent_rswa_prefix_lens: torch.Tensor | None = None
        if self.rswa_window is not None:
            self.persistent_rswa_prefix_lens = torch.empty(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=device,
            )

        # NX: one pre-allocated buffer set per segment choice; the shipped buffers above are the "16" entry.
        self.nx_segm_buffers = {self.num_par_softmax_segments: (
            self.softmax_segm_output, self.softmax_segm_max, self.softmax_segm_expsum)}
        self.nx_capturing = False
        if NX_ADAPTIVE_SEGMENTS:
            try:
                self.nx_num_cus = torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                self.nx_num_cus = 304
            for _s in NX_SEG_CHOICES:
                if _s in self.nx_segm_buffers:
                    continue
                self.nx_segm_buffers[_s] = (
                    torch.empty((max_num_tokens_3d, self.num_heads_q, _s, headdim_padded),
                                dtype=torch.float32, device=device),
                    torch.empty((max_num_tokens_3d, self.num_heads_q, _s), dtype=torch.float32, device=device),
                    torch.empty((max_num_tokens_3d, self.num_heads_q, _s), dtype=torch.float32, device=device),
                )
            # Hot-path precompute. build() runs once per attention group per step (12 groups here), so anything done
            # per call is multiplied by 12 and lands directly in TPOT. The first version did sorted()+list
            # comprehensions inside the picker and cost ~90 us/step -- measured as -1.34% throughput on sharegpt_c1,
            # a cell where the selection never even changes (490-token sequences = 30 KV tiles, so the cap pins S=16).
            # Everything below is therefore hoisted here and the per-step path is int ops plus one tuple index.
            self._nx_choices = tuple(sorted(self.nx_segm_buffers))
            self._nx_min_choice = self._nx_choices[0]
            self._nx_max_choice = self._nx_choices[-1]
            # Segments the CU target wants, indexed by decode batch size (0..seq_threshold_3D).
            self._nx_by_nseq = tuple(
                self._nx_cu_target(n) for n in range(self.seq_threshold_3D + 1)
            )
            # Buffer sets in the same order, so the per-step path never hashes.
            self._nx_bufs_by_segs = {c: self.nx_segm_buffers[c] for c in self._nx_choices}
            # Below this many KV tiles the cap pins us to the smallest choice -- the common short-context case, which
            # we can then answer without touching the choice list at all.
            self._nx_tiles_floor = self._nx_min_choice
            _mb = sum(b.numel() * b.element_size() for bs in self.nx_segm_buffers.values() for b in bs) / 2**20
            logger.info(
                "NX adaptive KV segments on: choices=%s, CUs=%d, q_heads=%d kv_heads=%d threshold_3D=%d, "
                "buffers %.1f MB, segments by decode batch %s",
                sorted(self.nx_segm_buffers), self.nx_num_cus, self.num_heads_q, self.num_heads_kv,
                self.seq_threshold_3D, _mb,
                {n: self._nx_pick_segments(n, None) for n in (1, 2, 4, 8, 16, self.seq_threshold_3D)},
            )

    def _nx_cu_target(self, num_seqs: int) -> int:
        """Segments so that num_seqs x kv_heads x segments ~ the CU count. Init-time only; see _nx_by_nseq."""
        if num_seqs <= 0:
            return self.num_par_softmax_segments
        choices = sorted(self.nx_segm_buffers)
        want = -(-self.nx_num_cus // max(1, num_seqs * self.num_heads_kv))     # ceil
        want = 1 << max(0, (want - 1)).bit_length()                            # up to a power of two (tl.arange needs one)
        return min((c for c in choices if c >= want), default=max(choices))

    def _nx_pick_segments(self, num_seqs: int, max_seq_len: int | None) -> int:
        """Per-step choice. Must stay cheap: 12 attention groups x every decode step."""
        if not NX_ADAPTIVE_SEGMENTS or num_seqs <= 0:
            return self.num_par_softmax_segments
        segs = self._nx_by_nseq[num_seqs] if num_seqs < len(self._nx_by_nseq) else self._nx_min_choice
        if max_seq_len is not None and segs > self._nx_min_choice:
            # Never ask for more segments than there are KV tiles to split: those workgroups early-return but still cost
            # a dispatch and a slot in the reduce. This clamp must round DOWN -- rounding up to the next available choice
            # would re-introduce exactly the over-segmentation it exists to prevent.
            tiles = int(max_seq_len) // NX_TILE_SIZE
            if tiles < self._nx_tiles_floor:
                return self._nx_min_choice          # common short-context case, answered without scanning choices
            if tiles < segs:
                for c in self._nx_choices:          # largest choice <= tiles; the list is 3 long
                    if c <= tiles:
                        segs = c
                    else:
                        break
        return segs

    def _nx_segment_fields(self, common_attn_metadata, max_query_len, max_seq_len) -> dict:
        """The 3D path only runs for decode-only batches at or below seq_threshold_3D; pick segments for that case."""
        segs = self.num_par_softmax_segments
        # NX: allow the adaptive segment picker on the MTP verify path too.
        # It was gated on max_query_len == 1, i.e. plain decode. Under MTP n=3 the uniform decode step
        # has max_query_len = 4, so we always fall back to the FIXED num_par_softmax_segments (16)
        # regardless of how much KV there actually is to split.
        # Safe by construction: `segs` is passed to BOTH kernel_unified_attention and reduce_segments
        # through the attention metadata, so they cannot disagree about how many segments are live --
        # which is exactly the failure that killed the winsplit arm (see winsplit/FINDING.md).
        # The 3-D path only runs for uniform-decode batches anyway (_supports_uniform_decode), so
        # widening this gate to the spec-decode query length does not admit any new shape.
        _nx_uniform_q = max_query_len <= NX_MAX_ADAPTIVE_QLEN
        if NX_ADAPTIVE_SEGMENTS and _nx_uniform_q:
            n = int(getattr(common_attn_metadata, "num_reqs", 0) or 0)
            if 0 < n <= self.seq_threshold_3D:
                # Under full CUDA graphs the metadata is built once per capture size and replayed with the chosen
                # segment count and buffer pointers baked in; only the step-dependent tensors are refreshed. So the
                # choice must be a pure function of the capture size -- during capture we drop the seq-len cap, whose
                # input is the dummy seq_lens (build_for_cudagraph_capture overwrites them with 1 straight after).
                segs = self._nx_pick_segments(n, None if self.nx_capturing else max_seq_len)
        out, mx, es = self.nx_segm_buffers[segs]
        return {"seq_threshold_3D": self.seq_threshold_3D, "num_par_softmax_segments": segs,
                "softmax_segm_output": out, "softmax_segm_max": mx, "softmax_segm_expsum": es}

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        self.nx_capturing = True          # NX: make the segment choice depend only on the capture size
        try:
            attn_metadata = self.build(0, common_attn_metadata)
        finally:
            self.nx_capturing = False
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        capture_seq_len = attn_metadata.max_query_len if attn_metadata.is_uniform_decode else 1
        attn_metadata.seq_lens.fill_(capture_seq_len)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # NX (PR #56148): the multi-query 3D path is only safe on a batch that is provably a uniform decode -- every
        # active sequence contributing exactly max_query_len rows, none of them prefill. Adaptive verification makes
        # query lengths non-uniform by design, so it disqualifies the batch.
        is_uniform_decode = False
        if NX_VERIFY_SPLIT and max_query_len > 1:
            _spec = self.vllm_config.speculative_config
            _pref = common_attn_metadata.is_prefilling
            if (
                _supports_uniform_decode(max_query_len, num_reqs, num_actual_tokens)
                and _pref is not None
                and not (_spec is not None and getattr(_spec, "enable_adaptive_verification", False))
            ):
                _starts = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
                _lens = _starts[1:] - _starts[:-1]
                _rows = min(num_reqs, len(_pref))
                is_uniform_decode = bool(
                    torch.any(_lens > 0)
                    and torch.all((_lens == 0) | (_lens == max_query_len))
                    and torch.all((_lens[:_rows] == 0) | ~_pref[:_rows])
                    and torch.all(_lens[_rows:] == 0)
                )
                if is_uniform_decode and not getattr(self, "_nx_logged_verify", False):
                    # One-time proof the verify path actually engaged, so the A/B is not read off timings alone.
                    self._nx_logged_verify = True
                    logger.info(
                        "NX verify split ENGAGED: query_len=%d num_reqs=%d segments=%d threshold_3D=%d kv_heads=%d",
                        max_query_len, num_reqs, self.num_par_softmax_segments,
                        self.seq_threshold_3D, self.num_heads_kv,
                    )

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            is_uniform_decode=is_uniform_decode,
            **self._nx_segment_fields(common_attn_metadata, max_query_len, max_seq_len),
        )

        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None:
            attn_metadata.mm_prefix_range = mm_ranges
            attn_metadata.mm_prefix_range_tensor = compute_mm_prefix_range_tensor(
                mm_ranges, num_reqs, seq_lens.device
            )

        rswa_prefix_lens = common_attn_metadata.rswa_prefix_lens
        if self.rswa_window is not None and rswa_prefix_lens is not None:
            assert self.persistent_rswa_prefix_lens is not None
            rswa_prefix_lens = rswa_prefix_lens.to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            persistent_prefix_lens = self.persistent_rswa_prefix_lens[:num_reqs]
            persistent_prefix_lens.copy_(rswa_prefix_lens[:num_reqs])
            attn_metadata.rswa_prefix_lens = persistent_prefix_lens
            attn_metadata.rswa_window = self.rswa_window

        return attn_metadata

    def update_draft_decode_metadata(self, _metadata: TritonAttentionMetadata) -> None:
        pass


class TritonAttentionBackend(AttentionBackend):
    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """Per-token-head modes pack inline fp32 scales after each head's
        data, so the content is (data + one scale) per K/V side."""
        mode = spec.kv_quant_mode
        if spec.state_content_bytes is not None or not mode.is_per_token_head:
            return spec
        hs_k, hs_v = spec.head_size, spec.head_size_v
        if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            hs_k, hs_v = hs_k // 2, hs_v // 2
        scale_bytes = get_dtype_size(torch.float32)
        content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
        return replace(spec, state_content_bytes=content)

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int4_per_token_head",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded content dimension.

        The KV cache is packed as logical shape
        ``(num_blocks, nkv, block_size, 2 * (hs + pad))`` where
        ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The content dim holds
        ``[K(hs) | K_scale(pad) | V(hs) | V_scale(pad)]`` per (head, slot); the
        last ``pad`` elements of each half hold one float32 scale.  We create
        strided float32 views over those bytes.  ``kv_cache`` must be the
        packed logical tensor (call before any transpose), but may have HND or
        NHD physical strides.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, nkv, block_size, content = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        padded_hs = content // 2
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        def to_f32_units(elements: int) -> int:
            nbytes = elements * dtype_sz
            assert nbytes % 4 == 0
            return nbytes // 4

        # Actual strides (in float32 units) from the tensor. The logical cache
        # may be physically NHD, so do not assume C-contiguous HND layout.
        strides = kv_cache.stride()
        block_f32 = to_f32_units(strides[0])
        head_f32 = to_f32_units(strides[1])
        slot_f32 = to_f32_units(strides[2])
        # Scale sits at byte offset hs within each (K, then V) content half.
        base_off_f32 = to_f32_units(kv_cache.storage_offset())
        k_scale_off_f32 = base_off_f32 + to_f32_units(hs)
        v_scale_off_f32 = base_off_f32 + to_f32_units(padded_hs + hs)

        # K scales (first content half)
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=k_scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        # V scales (second content half)
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=v_scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton attention backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
                    f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported on {dev} (compute capability "
                    f"{cap_str}); bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.chunk_lookback = chunk_lookback
        self.supports_quant_query_input = current_platform.is_cuda()

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

        # Enable tensor descriptors for Q/K/V load/store on platforms that
        # benefit from HW 2D block reads (Intel XPU).  The dead branch
        # is eliminated at Triton compile time, so other platforms see
        # zero cost when TD is off.
        #
        # ``VLLM_TRITON_USE_TD`` is tri-state:
        #   - unset (None): auto-select (TD on for XPU, off elsewhere),
        #   - ``1``: force TD on regardless of platform,
        #   - ``0``: force TD off regardless of platform (useful for A/B).
        td_override = envs.VLLM_TRITON_USE_TD
        if td_override is None:
            self.use_td = current_platform.is_xpu()
        else:
            self.use_td = td_override

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # KV cache arrives in logical (B, H, N, 2*hs) order.
        # Per-token-head quantized KV cache: handled by the core unified
        # kernel, which dequantizes per-(token, head) inline via constexpr
        # branches (INT8 / FP8) and dispatches to the packed INT4 kernel.
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            q_descale = k_descale = v_descale = None
        # FP8 per-tensor / auto path (original flow).
        else:
            kv_cache = kv_cache.transpose(1, 2)
            hs = self.head_size
            key_cache, value_cache = kv_cache.split(hs, dim=-1)
            # NX: K==V alias probe. Gemma-4 sets `attention_k_eq_v` -- the checkpoint has k_proj but NO v_proj for the
            # 5 full-attention layers, and vLLM synthesises V by duplicating K (models/gemma4.py:1635-1642). So the two
            # halves of the page hold identical bytes. Pointing the V view at the K view is therefore BITWISE exact,
            # and makes the kernel's V load hit cache lines the K load just brought in, instead of a second HBM stream.
            # Global layers are 76% of KV bytes at 32K and 93% at 128K, so this targets 38%/46% of all KV traffic.
            # head_size >= 512 identifies them (sliding layers use 256). Probe only: the V half of the page is still
            # allocated and still written -- proving the read-side saving before touching the cache spec.
            if NX_KV_ALIAS and hs >= 512:
                value_cache = key_cache
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                and key_cache.dtype != self.fp8_dtype
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            q_descale = (
                layer._q_scale
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and query.dtype == self.fp8_dtype
                )
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            is_uniform_decode=attn_metadata.is_uniform_decode,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
        )

        return output

    def _pth_key_value_caches(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token-head K/V cache views (ensures scale caches; FP8 retyped)."""
        self._ensure_scale_caches(kv_cache)
        padded_hs = kv_cache.shape[-1] // 2
        key_cache, value_cache = kv_cache.transpose(1, 2).split(padded_hs, dim=-1)
        if self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        return key_cache, value_cache

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # Quantized KV cache is not supported for encoder attention.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
            sinks=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                k_scale_cache,
                v_scale_cache,
                slot_mapping,
                kv_quant_mode=self._kv_quant_mode,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
