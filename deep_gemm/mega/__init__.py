import torch
import types
from typing import Tuple, Optional
from ..utils.math import align

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
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        # FP8 block-scaled weights (SM90 path): no interleaving or UTCCP transpose needed
        return l1_weights, l2_weights
    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def _sm90_fp8_mega_moe_composed(y: torch.Tensor,
                                l1_weights: Tuple[torch.Tensor, torch.Tensor],
                                l2_weights: Tuple[torch.Tensor, torch.Tensor],
                                sym_buffer: SymmBuffer,
                                cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                                activation_clamp: Optional[float] = None,
                                fast_math: bool = True):
    """SM90 block-scaled FP8 MegaMoE: composed dispatch → grouped-GEMM → SwiGLU → GEMM → combine."""
    from ..utils.math import unpack_ue8m0_from_int, per_token_cast_to_fp8

    num_tokens = y.size(0)
    hidden = y.size(1)
    l1_w, l1_sf = l1_weights
    l2_w, l2_sf = l2_weights
    num_experts_per_rank = l1_w.size(0)
    intermediate_hidden = l2_w.size(2)
    num_topk = sym_buffer.num_topk
    mk_alignment = _C.get_mk_alignment_for_contiguous_layout()

    # Read inputs from SymmBuffer
    x_fp8 = sym_buffer.x[:num_tokens]
    x_sf_packed = sym_buffer.x_sf[:num_tokens]
    topk_idx = sym_buffer.topk_idx[:num_tokens]
    topk_weights_tensor = sym_buffer.topk_weights[:num_tokens]

    # Dequant: FP8 + per-32 packed-UE8M0 → BF16
    sf_unpacked = unpack_ue8m0_from_int(x_sf_packed)
    x_bf16 = (x_fp8.float().view(num_tokens, -1, 32) *
              sf_unpacked.view(num_tokens, -1, 1)).view(num_tokens, hidden).to(torch.bfloat16)

    # Token routing: sort by expert
    flat_topk_idx = topk_idx.view(-1)
    flat_token_idx = torch.arange(num_tokens, device=y.device).unsqueeze(1).expand(-1, num_topk).reshape(-1)
    flat_topk_pos = torch.arange(num_topk, device=y.device).unsqueeze(0).expand(num_tokens, -1).reshape(-1)

    valid_mask = (flat_topk_idx >= 0) & (flat_topk_idx < num_experts_per_rank)
    valid_expert_idx = flat_topk_idx[valid_mask]
    valid_token_idx = flat_token_idx[valid_mask]
    valid_topk_pos = flat_topk_pos[valid_mask]

    sort_order = valid_expert_idx.argsort(stable=True)
    sorted_expert_idx = valid_expert_idx[sort_order]
    sorted_token_idx = valid_token_idx[sort_order]
    sorted_topk_pos = valid_topk_pos[sort_order]

    total_valid = sorted_token_idx.size(0)
    if total_valid == 0:
        y.zero_()
        return

    # Per-expert counts and aligned layout
    expert_counts = torch.zeros(num_experts_per_rank, dtype=torch.int, device=y.device)
    for e in range(num_experts_per_rank):
        expert_counts[e] = (sorted_expert_idx == e).sum().int()

    aligned_counts = ((expert_counts + mk_alignment - 1) // mk_alignment * mk_alignment).int()
    total_aligned = aligned_counts.sum().item()

    # Build psum_layout: actual cumulative token positions within aligned tensor
    psum_layout = torch.zeros(num_experts_per_rank, dtype=torch.int, device=y.device)
    routed_positions = torch.zeros(total_valid, dtype=torch.long, device=y.device)
    offset = 0
    pos = 0
    for e in range(num_experts_per_rank):
        cnt = expert_counts[e].item()
        psum_layout[e] = offset + cnt
        routed_positions[pos:pos + cnt] = torch.arange(cnt, device=y.device) + offset
        pos += cnt
        offset += aligned_counts[e].item()

    # Build routed activation tensor (aligned, zero-padded)
    routed_x_bf16 = torch.zeros((total_aligned, hidden), dtype=torch.bfloat16, device=y.device)
    routed_x_bf16[routed_positions] = x_bf16[sorted_token_idx]

    # Requant for SM90: per-token, per-128K fp32 scales
    routed_x_fp8, routed_x_sf = per_token_cast_to_fp8(routed_x_bf16, use_ue8m0=False, gran_k=128)

    # L1 GEMM: [total_aligned, hidden] @ [E, 2*I, hidden].T → [total_aligned, 2*I]
    l1_y = torch.empty((total_aligned, intermediate_hidden * 2), dtype=torch.bfloat16, device=y.device)
    _C.m_grouped_fp8_gemm_nt_contiguous(
        (routed_x_fp8, routed_x_sf), (l1_w, l1_sf),
        l1_y, psum_layout, use_psum_layout=True)

    # SwiGLU: gate * sigmoid(gate) * up * topk_weight
    gate = l1_y[:, :intermediate_hidden].float()
    up = l1_y[:, intermediate_hidden:].float()

    if activation_clamp is not None and activation_clamp < float('inf'):
        gate = gate.clamp(-activation_clamp, activation_clamp)
        up = up.clamp(-activation_clamp, activation_clamp)

    swiglu_out = gate * torch.sigmoid(gate) * up

    # Apply topk weights per routed token
    routed_topk_wts = torch.zeros(total_aligned, dtype=torch.float, device=y.device)
    routed_topk_wts[routed_positions] = topk_weights_tensor[sorted_token_idx, sorted_topk_pos]
    swiglu_out = swiglu_out * routed_topk_wts.unsqueeze(1)

    # Requant SwiGLU output for L2
    swiglu_bf16 = swiglu_out.to(torch.bfloat16)
    swiglu_fp8, swiglu_sf = per_token_cast_to_fp8(swiglu_bf16, use_ue8m0=False, gran_k=128)

    # L2 GEMM: [total_aligned, I] @ [E, hidden, I].T → [total_aligned, hidden]
    l2_y = torch.empty((total_aligned, hidden), dtype=torch.bfloat16, device=y.device)
    _C.m_grouped_fp8_gemm_nt_contiguous(
        (swiglu_fp8, swiglu_sf), (l2_w, l2_sf),
        l2_y, psum_layout, use_psum_layout=True)

    # Combine: scatter back and reduce across topk
    y.zero_()
    y.index_add_(0, sorted_token_idx, l2_y[routed_positions])

    # Update cumulative expert stats
    if cumulative_local_expert_recv_stats is not None:
        cumulative_local_expert_recv_stats.add_(expert_counts)


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    # Dispatch on weight dtype: FP8 weights → SM90 composed path
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        _sm90_fp8_mega_moe_composed(
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
