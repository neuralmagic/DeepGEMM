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

    x_sf_fp32 = unpack_ue8m0_from_int(x_sf_packed)
    x_dequant = x_fp8.float() * x_sf_fp32.repeat_interleave(32, dim=1)[:, :hidden]

    def _dequant_expert_weight(w_fp8_e, w_sf_e):
        n, k = w_fp8_e.shape
        if w_sf_e.dtype == torch.int32:
            sf = unpack_ue8m0_from_int(w_sf_e)
        else:
            sf = w_sf_e
        sf_n, sf_k = sf.shape
        if sf_n < n:
            sf = sf.repeat_interleave(n // sf_n, dim=0)[:n]
        if sf_k < k:
            sf = sf.repeat_interleave(k // sf_k, dim=1)[:, :k]
        return w_fp8_e.float() * sf

    y.zero_()

    for e in range(num_experts_per_rank):
        mask = (topk_idx == e)
        if not mask.any():
            continue
        token_indices, topk_positions = mask.nonzero(as_tuple=True)
        expert_wts = topk_wts[token_indices, topk_positions]
        expert_x = x_dequant[token_indices]

        l1_w_e = _dequant_expert_weight(l1_w[e], l1_w_sf[e])
        l1_out = expert_x @ l1_w_e.T

        gate = l1_out[:, :intermediate_hidden_x2 // 2]
        up = l1_out[:, intermediate_hidden_x2 // 2:]

        if activation_clamp is not None and activation_clamp < float('inf'):
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(-activation_clamp, activation_clamp)

        activated = torch.nn.functional.silu(gate) * up
        activated = activated * expert_wts.unsqueeze(1)

        l2_w_e = _dequant_expert_weight(l2_w[e], l2_w_sf[e])
        l2_out = activated @ l2_w_e.T

        y.index_add_(0, token_indices, l2_out.bfloat16())

    if cumulative_local_expert_recv_stats is not None:
        expert_counts = torch.zeros(num_experts_per_rank, dtype=torch.int, device=device)
        valid_idx = topk_idx[(topk_idx >= 0) & (topk_idx < num_experts_per_rank)]
        if valid_idx.numel() > 0:
            bc = torch.bincount(valid_idx.int(), minlength=num_experts_per_rank)
            expert_counts[:bc.shape[0]] = bc[:num_experts_per_rank].int()
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
