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
        # FP8 weights (SM90 block-scaled): no interleave or UTCCP transpose needed
        return l1_weights, l2_weights
    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def _sm90_fp8_mega_moe(y: torch.Tensor,
                       l1_weights: Tuple[torch.Tensor, torch.Tensor],
                       l2_weights: Tuple[torch.Tensor, torch.Tensor],
                       sym_buffer: SymmBuffer,
                       cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                       activation_clamp: Optional[float] = None,
                       fast_math: bool = True):
    from ..utils.math import unpack_ue8m0_from_int

    num_tokens = y.shape[0]
    hidden = y.shape[1]
    device = y.device

    x_fp8 = sym_buffer.x[:num_tokens]
    x_sf_packed = sym_buffer.x_sf[:num_tokens]
    topk_idx = sym_buffer.topk_idx[:num_tokens]
    topk_wts = sym_buffer.topk_weights[:num_tokens]

    l1_w, l1_w_sf = l1_weights
    l2_w, l2_w_sf = l2_weights

    num_experts_per_rank = l1_w.shape[0]
    intermediate_hidden_x2 = l1_w.shape[1]
    intermediate_hidden = l2_w.shape[2]
    num_topk = topk_idx.shape[1]

    # Unpack per-32 UE8M0 scales (used after gather to avoid dequanting all tokens)
    x_sf = unpack_ue8m0_from_int(x_sf_packed)
    num_sf_groups = hidden // 32

    # Flatten topk assignments and filter valid ones
    flat_topk_idx = topk_idx.reshape(-1)
    flat_token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_topk).reshape(-1)
    flat_wts = topk_wts.reshape(-1)

    valid = (flat_topk_idx >= 0) & (flat_topk_idx < num_experts_per_rank)
    valid_expert_idx = flat_topk_idx[valid]
    valid_token_idx = flat_token_idx[valid]
    valid_wts = flat_wts[valid]
    num_valid = valid_expert_idx.shape[0]

    if num_valid == 0:
        y.zero_()
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.zero_()
        return

    # Sort by expert to create grouped layout
    sorted_indices = torch.argsort(valid_expert_idx, stable=True)
    sorted_expert_idx = valid_expert_idx[sorted_indices]
    sorted_token_idx = valid_token_idx[sorted_indices]
    sorted_wts = valid_wts[sorted_indices]

    # Expert token counts
    expert_counts = torch.zeros(num_experts_per_rank, dtype=torch.int32, device=device)
    bc = torch.bincount(sorted_expert_idx.int(), minlength=num_experts_per_rank)
    expert_counts[:bc.shape[0]] = bc[:num_experts_per_rank].int()

    mk_alignment = _C.get_theoretical_mk_alignment_for_contiguous_layout()
    _C.set_mk_alignment_for_contiguous_layout(mk_alignment)

    # Build psum-style grouped layout with alignment padding between experts
    aligned_counts = ((expert_counts + mk_alignment - 1) // mk_alignment * mk_alignment)
    aligned_counts = aligned_counts * (expert_counts > 0).int()
    cum_aligned = torch.zeros(num_experts_per_rank + 1, dtype=torch.int32, device=device)
    cum_aligned[1:] = aligned_counts.cumsum(0)
    expert_starts = cum_aligned[:-1]
    total_aligned = cum_aligned[-1].item()

    # psum grouped_layout[e] = start_of_expert_e + count_of_expert_e
    grouped_layout = (expert_starts + expert_counts).int()

    # Map sorted token positions to aligned data array positions
    cum_counts = expert_counts.cumsum(0)
    expert_start_in_sorted = (cum_counts - expert_counts).long()
    within_pos = torch.arange(num_valid, device=device) - expert_start_in_sorted[sorted_expert_idx.long()]
    aligned_positions = expert_starts[sorted_expert_idx.long()].long() + within_pos

    # Gather FP8 bytes and per-32 scales, then rescale to per-128 (avoids BF16 intermediate)
    gathered_fp8_raw = torch.zeros((total_aligned, hidden), dtype=torch.float8_e4m3fn, device=device)
    gathered_fp8_raw[aligned_positions] = x_fp8[sorted_token_idx]
    gathered_sf32 = torch.zeros((total_aligned, num_sf_groups), dtype=torch.float32, device=device)
    gathered_sf32[aligned_positions] = x_sf[:, :num_sf_groups][sorted_token_idx]

    # Convert per-32 scales to per-128: max of 4 adjacent per-32 scales
    sf32_view = gathered_sf32.view(total_aligned, -1, 4)
    gathered_sf = sf32_view.amax(dim=2).clamp(min=1e-4)

    # Rescale FP8 values from per-32 to per-128 scale domain
    ratios = sf32_view / gathered_sf.unsqueeze(2)
    fp8_view = gathered_fp8_raw.float().view(total_aligned, -1, 4, 32)
    gathered_fp8 = (fp8_view * ratios.unsqueeze(3)).to(torch.float8_e4m3fn).reshape(total_aligned, hidden)

    # L1 grouped GEMM: FP8 activations × FP8 weights → BF16
    l1_out = torch.empty((total_aligned, intermediate_hidden_x2), dtype=torch.bfloat16, device=device)
    _C.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (gathered_fp8, gathered_sf), l1_weights, l1_out, grouped_layout,
        disable_ue8m0_cast=True, use_psum_layout=True)

    # SwiGLU in BF16, then requant to FP8 with per-128 block scales
    gate = l1_out[:, :intermediate_hidden]
    up = l1_out[:, intermediate_hidden:]

    if activation_clamp is not None and activation_clamp < float('inf'):
        gate = gate.float().clamp(max=activation_clamp).bfloat16()
        up = up.float().clamp(-activation_clamp, activation_clamp).bfloat16()

    act_bf16 = torch.nn.functional.silu(gate) * up
    padded_ih = (intermediate_hidden + 127) // 128 * 128
    if padded_ih != intermediate_hidden:
        act_f32 = torch.zeros((total_aligned, padded_ih), dtype=torch.float32, device=device)
        act_f32[:, :intermediate_hidden] = act_bf16.float()
    else:
        act_f32 = act_bf16.float()
    act_view = act_f32.view(total_aligned, -1, 128)
    act_sf = act_view.abs().amax(dim=2).clamp(1e-4) / 448.0
    act_fp8 = (act_view * (1.0 / act_sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(total_aligned, padded_ih)[:, :intermediate_hidden].contiguous()

    # L2 grouped GEMM: FP8 intermediate × FP8 weights → BF16
    l2_out = torch.empty((total_aligned, hidden), dtype=torch.bfloat16, device=device)
    _C.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (act_fp8, act_sf), l2_weights, l2_out, grouped_layout,
        disable_ue8m0_cast=True, use_psum_layout=True)

    # Scatter results back weighted by topk
    y.zero_()
    sorted_wts_bf16 = sorted_wts.unsqueeze(1).bfloat16()
    y.index_add_(0, sorted_token_idx, l2_out[aligned_positions] * sorted_wts_bf16)

    if cumulative_local_expert_recv_stats is not None:
        cumulative_local_expert_recv_stats.copy_(expert_counts.cumsum(0).int())


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    arch_major = torch.cuda.get_device_capability()[0]
    w_dtype = l1_weights[0].dtype

    if arch_major == 9 and w_dtype == torch.float8_e4m3fn:
        _sm90_fp8_mega_moe(y, l1_weights, l2_weights, sym_buffer,
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
