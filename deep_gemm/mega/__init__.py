import torch
import torch.nn.functional as F
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
    # FP8 e4m3 weights (SM90 block-scaled path): no interleaving or UTCCP transpose needed
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        return l1_weights, l2_weights
    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def _sm90_fp8_mega_moe_composed(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    activation_clamp: Optional[float],
    fast_math: bool
):
    """SM90 composed MegaMoE: dispatch → FP8 GEMM → SwiGLU → FP8 GEMM → combine."""
    num_tokens = y.size(0)
    hidden = y.size(1)
    num_topk = sym_buffer.num_topk
    intermediate_hidden = sym_buffer.intermediate_hidden
    mk_alignment = _C.get_mk_alignment_for_contiguous_layout()

    l1_w, l1_w_sf = l1_weights
    l2_w, l2_w_sf = l2_weights
    num_experts_per_rank = l1_w.size(0)

    # --- Read inputs from SymmBuffer ---
    x_fp8 = sym_buffer.x[:num_tokens]
    x_sf_packed = sym_buffer.x_sf[:num_tokens]
    topk_idx = sym_buffer.topk_idx[:num_tokens]
    topk_wt = sym_buffer.topk_weights[:num_tokens]

    # --- Dequant FP8 activations (per-32 UE8M0) to BF16 ---
    sf_float = unpack_ue8m0_from_int(x_sf_packed)
    x_bf16 = (x_fp8.float().view(num_tokens, -1, 32) * sf_float.view(num_tokens, -1, 1)).view(num_tokens, hidden).bfloat16()

    # --- EP Dispatch (single-rank): group tokens by expert ---
    tok_ids = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    tok_ids = tok_ids.unsqueeze(1).expand(-1, num_topk).reshape(-1)
    exp_ids = topk_idx.reshape(-1)
    wts = topk_wt.reshape(-1)

    valid = exp_ids >= 0
    tok_ids = tok_ids[valid]
    exp_ids = exp_ids[valid]
    wts = wts[valid]

    sort_idx = torch.argsort(exp_ids, stable=True)
    tok_ids = tok_ids[sort_idx]
    exp_ids = exp_ids[sort_idx]
    wts = wts[sort_idx]

    num_dispatched = tok_ids.size(0)

    # Count tokens per expert
    counts_list = []
    for e in range(num_experts_per_rank):
        counts_list.append(int((exp_ids == e).sum().item()))

    if num_dispatched == 0:
        y.zero_()
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.zero_()
        return

    # Build aligned psum layout with mk_alignment padding between groups
    psum = torch.empty(num_experts_per_rank, dtype=torch.int, device='cuda')
    dst_starts = []
    dst_start = 0
    for i in range(num_experts_per_rank):
        c = counts_list[i]
        dst_starts.append(dst_start)
        psum[i] = dst_start + c
        dst_start = align(dst_start + c, mk_alignment)
    total_aligned = dst_start

    if cumulative_local_expert_recv_stats is not None:
        cumulative_local_expert_recv_stats.copy_(psum)

    # --- Scatter tokens to aligned expert-contiguous buffer ---
    dispatched_bf16 = torch.zeros((total_aligned, hidden), dtype=torch.bfloat16, device='cuda')
    weight_buffer = torch.zeros(total_aligned, dtype=torch.float32, device='cuda')
    src_pos = 0
    for i in range(num_experts_per_rank):
        c = counts_list[i]
        if c > 0:
            dispatched_bf16[dst_starts[i]:dst_starts[i]+c] = x_bf16[tok_ids[src_pos:src_pos+c]]
            weight_buffer[dst_starts[i]:dst_starts[i]+c] = wts[src_pos:src_pos+c]
        src_pos += c

    # --- L1 GEMM via SM90 FP8 grouped kernel ---
    l1_a_fp8, l1_a_sf = per_token_cast_to_fp8(dispatched_bf16, use_ue8m0=False, gran_k=128)
    l1_n = intermediate_hidden * 2
    l1_out = torch.empty((total_aligned, l1_n), dtype=torch.bfloat16, device='cuda')
    _C.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (l1_a_fp8, l1_a_sf), (l1_w, l1_w_sf),
        l1_out, psum, use_psum_layout=True, disable_ue8m0_cast=True
    )

    # --- SwiGLU: clamp gate/up individually (matching SM100 kernel), then apply topk_weights ---
    gate = l1_out[:, :intermediate_hidden]
    up = l1_out[:, intermediate_hidden:]
    if activation_clamp is not None:
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(-activation_clamp, activation_clamp)
    act = (F.silu(gate) * up) * weight_buffer.unsqueeze(-1)

    # --- L2 GEMM via SM90 FP8 grouped kernel ---
    l2_a_fp8, l2_a_sf = per_token_cast_to_fp8(act.bfloat16(), use_ue8m0=False, gran_k=128)
    l2_out = torch.empty((total_aligned, hidden), dtype=torch.bfloat16, device='cuda')
    _C.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (l2_a_fp8, l2_a_sf), (l2_w, l2_w_sf),
        l2_out, psum, use_psum_layout=True, disable_ue8m0_cast=True
    )

    # --- Combine: scatter back to original positions ---
    y.zero_()
    src_pos = 0
    for i in range(num_experts_per_rank):
        c = counts_list[i]
        if c > 0:
            y.index_add_(0, tok_ids[src_pos:src_pos+c], l2_out[dst_starts[i]:dst_starts[i]+c])
        src_pos += c



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    # SM90 block-scaled FP8 path: composed pipeline using existing grouped GEMMs
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        _sm90_fp8_mega_moe_composed(
            y, l1_weights, l2_weights, sym_buffer,
            cumulative_local_expert_recv_stats,
            activation_clamp, fast_math
        )
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
