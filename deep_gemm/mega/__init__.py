import torch
import types
from typing import Tuple, Optional
from ..utils.math import align, per_token_cast_to_fp8, unpack_ue8m0_from_int

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    arch_major = torch.cuda.get_device_capability()[0]
    is_fp8_weights = (l1_weights[0].dtype == torch.float8_e4m3fn)

    if arch_major == 9 and is_fp8_weights:
        # SM90 FP8 path: no interleaving or UTCCP transpose needed
        return l1_weights, l2_weights

    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def _fp8_mega_moe_composed(y: torch.Tensor,
                           l1_weights: Tuple[torch.Tensor, torch.Tensor],
                           l2_weights: Tuple[torch.Tensor, torch.Tensor],
                           sym_buffer: SymmBuffer,
                           cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                           activation_clamp: Optional[float] = None,
                           fast_math: bool = True):
    """Composed (non-fused) MegaMoE for SM90/Hopper with block-scaled FP8 weights.

    Chains existing m_grouped FP8 GEMMs with a SwiGLU activation step.
    Single-rank only (multi-rank requires the fused kernel or deep_ep).
    """
    num_tokens = y.size(0)
    hidden = y.size(1)
    num_experts = sym_buffer.num_experts
    num_topk = sym_buffer.num_topk
    intermediate_hidden = sym_buffer.intermediate_hidden
    num_ranks = sym_buffer.group.size()

    assert num_ranks == 1, "Composed SM90 MegaMoE only supports single-rank"
    num_experts_per_rank = num_experts

    x_fp8_raw = sym_buffer.x[:num_tokens]
    x_sf_raw = sym_buffer.x_sf[:num_tokens]
    topk_idx = sym_buffer.topk_idx[:num_tokens]
    topk_weights_tensor = sym_buffer.topk_weights[:num_tokens]

    l1_w, l1_w_sf = l1_weights
    l2_w, l2_w_sf = l2_weights

    # Convert input from UE8M0 per-32 scales to FP32 per-128 scales for SM90 GEMM.
    # Dequantize from per-32 UE8M0, then re-quantize with per-128 FP32 scales.
    if x_sf_raw.dtype == torch.int:
        sf_per32 = unpack_ue8m0_from_int(x_sf_raw)  # [num_tokens, hidden/32] float
        x_dequant = x_fp8_raw.float()
        gran_k = 32
        x_view = x_dequant.view(num_tokens, -1, gran_k)
        sf_expanded = sf_per32.unsqueeze(2).expand_as(x_view)
        x_dequant = (x_view * sf_expanded).view(num_tokens, hidden).bfloat16()
        x_fp8, x_sf = per_token_cast_to_fp8(x_dequant, use_ue8m0=False, gran_k=128)
    else:
        x_fp8 = x_fp8_raw
        x_sf = x_sf_raw

    # --- Step 1: Dispatch (scatter tokens by expert) ---
    flat_topk_idx = topk_idx.reshape(-1)
    flat_token_idx = torch.arange(num_tokens, device=y.device).unsqueeze(1).expand(
        -1, num_topk).reshape(-1)
    flat_topk_weights = topk_weights_tensor.reshape(-1)

    valid_mask = flat_topk_idx >= 0
    valid_expert_idx = flat_topk_idx[valid_mask]
    valid_token_idx = flat_token_idx[valid_mask]
    valid_topk_weights = flat_topk_weights[valid_mask]

    sorted_order = torch.argsort(valid_expert_idx, stable=True)
    sorted_expert_idx = valid_expert_idx[sorted_order]
    sorted_token_idx = valid_token_idx[sorted_order]
    sorted_topk_weights = valid_topk_weights[sorted_order]

    num_recv_tokens = sorted_token_idx.size(0)
    if num_recv_tokens == 0:
        y.zero_()
        return

    # Build grouped layout: cumulative token counts per expert (psum format)
    expert_counts = torch.bincount(
        sorted_expert_idx.int(), minlength=num_experts_per_rank
    )[:num_experts_per_rank].to(torch.int)

    # Alignment: pad each expert's token count to the alignment boundary
    mk_alignment = _C.get_theoretical_mk_alignment_for_contiguous_layout(None, None)
    _C.set_mk_alignment_for_contiguous_layout(mk_alignment)

    # Build padded grouped layout for the GEMM (each expert aligned)
    padded_expert_counts = ((expert_counts + mk_alignment - 1) // mk_alignment * mk_alignment).to(torch.int)
    padded_grouped_layout = torch.cumsum(padded_expert_counts, dim=0).to(torch.int)
    total_padded_tokens = padded_grouped_layout[-1].item()

    # Build padded recv_x: [total_padded_tokens, hidden] FP8, zero-padded per expert
    recv_x_fp8 = torch.zeros((total_padded_tokens, hidden),
                             dtype=torch.float8_e4m3fn, device=y.device)
    recv_x_sf = torch.zeros((total_padded_tokens, hidden // 128),
                            dtype=x_sf.dtype, device=y.device)
    recv_topk_weights = torch.zeros(total_padded_tokens, dtype=torch.float32,
                                    device=y.device)

    # Copy tokens into padded positions
    src_offset = 0
    for e in range(num_experts_per_rank):
        count = expert_counts[e].item()
        dst_offset = (padded_grouped_layout[e - 1].item() if e > 0 else 0)
        if count > 0:
            idx = sorted_token_idx[src_offset:src_offset + count]
            recv_x_fp8[dst_offset:dst_offset + count] = x_fp8[idx]
            recv_x_sf[dst_offset:dst_offset + count] = x_sf[idx]
            recv_topk_weights[dst_offset:dst_offset + count] = sorted_topk_weights[src_offset:src_offset + count]
            src_offset += count

    # --- Step 2: L1 GEMM ---
    l1_y = torch.zeros((total_padded_tokens, intermediate_hidden * 2),
                       dtype=torch.bfloat16, device=y.device)
    if total_padded_tokens > 0:
        _C.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (recv_x_fp8, recv_x_sf), l1_weights, l1_y, padded_grouped_layout,
            None, None, None, '', False, True, None)

    # --- Step 3: SwiGLU + FP8 requant ---
    gate = l1_y[:, :intermediate_hidden].float()
    up = l1_y[:, intermediate_hidden:].float()

    clamp = activation_clamp if activation_clamp is not None else float('inf')
    if clamp < float('inf'):
        gate = gate.clamp(-clamp, clamp)
        up = up.clamp(-clamp, clamp)

    if fast_math:
        silu_gate = gate * torch.sigmoid(gate)
    else:
        silu_gate = torch.nn.functional.silu(gate)

    activation_out = silu_gate * up * recv_topk_weights.unsqueeze(1)

    l2_acts_fp8, l2_acts_sf = per_token_cast_to_fp8(
        activation_out.bfloat16(), use_ue8m0=False, gran_k=128)

    # --- Step 4: L2 GEMM ---
    l2_y = torch.zeros((total_padded_tokens, hidden),
                       dtype=torch.bfloat16, device=y.device)
    if total_padded_tokens > 0:
        _C.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (l2_acts_fp8, l2_acts_sf), l2_weights, l2_y, padded_grouped_layout,
            None, None, None, '', False, True, None)

    # --- Step 5: Combine (gather + reduce) ---
    # Collect unpadded results and scatter-add back to output
    unpadded_results = torch.zeros((num_recv_tokens, hidden),
                                   dtype=torch.bfloat16, device=y.device)
    src_offset = 0
    for e in range(num_experts_per_rank):
        count = expert_counts[e].item()
        dst_offset = (padded_grouped_layout[e - 1].item() if e > 0 else 0)
        if count > 0:
            unpadded_results[src_offset:src_offset + count] = l2_y[dst_offset:dst_offset + count]
            src_offset += count

    y.zero_()
    y.index_add_(0, sorted_token_idx, unpadded_results)

    # Update stats if provided
    if cumulative_local_expert_recv_stats is not None:
        cumulative_local_expert_recv_stats.copy_(
            torch.cumsum(expert_counts, dim=0).to(torch.int))


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    # Detect architecture and weight type to dispatch
    arch_major = torch.cuda.get_device_capability()[0]
    l1_w = l1_weights[0] if isinstance(l1_weights, tuple) else l1_weights
    is_fp8_weights = (l1_w.dtype == torch.float8_e4m3fn)

    # SM90 with FP8 weights: use composed path
    if arch_major == 9 and is_fp8_weights:
        _fp8_mega_moe_composed(
            y, l1_weights, l2_weights, sym_buffer,
            cumulative_local_expert_recv_stats,
            activation_clamp, fast_math)
        return

    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )
