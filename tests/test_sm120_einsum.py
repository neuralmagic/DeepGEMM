import random
import torch

import deep_gemm
from deep_gemm.testing import (
    get_arch_major, test_filter
)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_batched_output_row_stride() -> None:
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        for expr in ('bhd,hdr->bhr', 'bhr,hdr->bhd'):
            for padding in (0, 8):
                a = torch.ones((128, 2, 128), dtype=torch.bfloat16, device='cuda')
                a[:, 1].fill_(2)
                b_shape = (2, 128, 32) if expr == 'bhd,hdr->bhr' else (2, 32, 128)
                b = torch.ones(b_shape, dtype=torch.bfloat16, device='cuda')
                storage = torch.full((130, 2, 32 + padding), -7, dtype=torch.bfloat16, device='cuda')
                d = storage[1:129, :, :32]
                deep_gemm.einsum(expr, a, b, d)
                assert torch.all(d[:, 0] == 128)
                assert torch.all(d[:, 1] == 256)
                assert torch.all(storage[[0, -1]] == -7)
                assert torch.all(storage[1:129, :, 32:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_reduction_empty_dimensions_and_fp32_accumulation_contract() -> None:
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        for m, n in ((0, 128), (128, 0), (128, 128)):
            a = torch.ones((1, m, 128), dtype=torch.bfloat16, device='cuda')
            b = torch.ones((1, n, 128), dtype=torch.bfloat16, device='cuda')
            for dtype in (torch.float32, torch.bfloat16):
                storage = torch.full((m * n + 32,), -7, dtype=dtype, device='cuda')
                d = storage[16:16 + m * n].view(m, n)
                d.fill_(3)
                deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=d if dtype == torch.float32 else None)
                assert torch.all(d == (131 if dtype == torch.float32 else 128))
                assert torch.all(storage[:16] == -7) and torch.all(storage[-16:] == -7)
            d = torch.empty((m, n), dtype=torch.float32, device='cuda')
            try:
                deep_gemm.einsum('bmk,bnk->mn', a, b, d)
            except RuntimeError:
                pass
            else:
                raise AssertionError('FP32 reduction must require C=D')
    finally:
        deep_gemm.set_num_sms(old_sms)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_sm120_batched_output_row_stride()
    test_sm120_reduction_empty_dimensions_and_fp32_accumulation_contract()
