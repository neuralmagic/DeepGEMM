import random
import torch

import deep_gemm
from deep_gemm.testing import (
    get_arch_major, test_filter
)


def _check_sm120_hc_prenorm_output_layout(num_splits) -> None:
    for padding in (0, 8):
        m, n, k = 13, 24, 128
        a = torch.arange(m, device='cuda').remainder(3).add(1).to(torch.bfloat16)[:, None].expand(m, k).contiguous()
        b = torch.ones((n, k), dtype=torch.float32, device='cuda')
        splits = num_splits or 1
        storage = torch.full((splits, m + 2, n + padding), -7, dtype=torch.float32, device='cuda')
        d = storage[:, 1:m + 1, :n]
        sqr = torch.empty((splits, m), dtype=torch.float32, device='cuda')
        if num_splits is not None:
            try:
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr, num_splits=num_splits)
            except RuntimeError as exc:
                assert 't.stride(0) == t.size(-2) * t.size(-1)' in str(exc)
            else:
                raise AssertionError('Non-contiguous split batches must be rejected')
            assert torch.all(storage == -7)
            storage = torch.full((splits * m * n + 32,), -7, dtype=torch.float32, device='cuda')
            d = storage[16:-16].view(splits, m, n)
        deep_gemm.tf32_hc_prenorm_gemm(a, b, d[0] if num_splits is None else d,
                                      sqr[0] if num_splits is None else sqr, num_splits=num_splits)
        rows = torch.arange(m).remainder(3).add(1).float()
        assert torch.equal(d.cpu(), (rows[None, :, None] * (k // splits)).expand(splits, m, n))
        assert torch.equal(sqr.cpu(), (rows.square()[None, :] * (k // splits)).expand(splits, m))
        if num_splits is None:
            assert torch.all(storage[:, [0, -1]] == -7)
            assert torch.all(storage[:, 1:m + 1, n:] == -7)
        else:
            assert torch.all(storage[:16] == -7) and torch.all(storage[-16:] == -7)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_hc_prenorm_output_row_stride() -> None:
    _check_sm120_hc_prenorm_output_layout(None)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_hc_prenorm_reject_noncontiguous_split_layout() -> None:
    _check_sm120_hc_prenorm_output_layout(2)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_sm120_hc_prenorm_output_row_stride()
    test_sm120_hc_prenorm_reject_noncontiguous_split_layout()
