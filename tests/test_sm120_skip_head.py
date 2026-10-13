import random

import pytest
import torch

import deep_gemm
from deep_gemm.testing import (
    get_arch_major, test_filter
)


_SKIP_HEAD_CASES = [
    pytest.param(16, 128, (8, 0, 8), 2, 1, id='scalar-control'),
    pytest.param(16, 128, (8, 8, 8), 2, 1, id='scalar-gap'),
    pytest.param(256, 2048, (128, 64, 128), 2, 1, id='split1-gap'),
    pytest.param(256, 2048, (128, 0, 128), 32, 2, id='split2-control'),
    pytest.param(256, 2048, (128, 64, 128), 32, 2, id='split2-gap'),
    pytest.param(256, 512, (128, 64, 128), 2, 1, id='packed-sf-control'),
]


def _skip_head_inputs(n, k, splits, out_dtype, fp32_b_scale=False):
    a = torch.ones((1, k), device='cuda').to(torch.float8_e4m3fn)
    b = torch.ones((n, k), device='cuda').to(torch.float8_e4m3fn)
    sfa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(
        torch.ones((1, k // 128), device='cuda'))
    sfb = torch.ones((n, k // 128), device='cuda')
    if not fp32_b_scale:
        sfb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfb)
    left, mid, right = splits
    heads = n // (left + right)
    width = heads * (left + mid + right)
    storage = torch.full((width + 128,), -19, dtype=out_dtype, device='cuda')
    d = storage[64:64 + width].view(1, width)
    d.fill_(-7)
    expected = torch.full((1, heads, left + mid + right), -7, dtype=out_dtype)
    expected[:, :, :left] = k
    expected[:, :, left + mid:] = k
    return (a, sfa), (b, sfb), d, storage, expected.reshape(1, width)


def _assert_skip_head_output(d, storage, expected):
    assert torch.equal(d.cpu(), expected)
    assert torch.equal(storage[:64].cpu(), torch.full((64,), -19, dtype=d.dtype))
    assert torch.equal(storage[-64:].cpu(), torch.full((64,), -19, dtype=d.dtype))


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('out_dtype', [torch.bfloat16, torch.float32], ids=['bf16', 'fp32'])
@pytest.mark.parametrize('n,k,splits,sms,expected_split', _SKIP_HEAD_CASES)
def test_sm120_skip_head_output_mapping_and_middle_preservation(n, k, splits, sms, expected_split, out_dtype):
    original_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(sms)
        a, b, d, storage, expected = _skip_head_inputs(n, k, splits, out_dtype)

        def run():
            deep_gemm.fp8_gemm_nt_skip_head_mid(
                a, b, d, splits, recipe=(1, 1, 128), disable_ue8m0_cast=True)

        for _ in range(3):
            d.fill_(-7)
            run()
            torch.cuda.synchronize()
            _assert_skip_head_output(d, storage, expected)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            d.fill_(-7)
            run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for _ in range(3):
            d.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            _assert_skip_head_output(d, storage, expected)
    finally:
        deep_gemm.set_num_sms(original_sms)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('out_dtype', [torch.bfloat16, torch.float32], ids=['bf16', 'fp32'])
def test_sm120_skip_head_reject_fp32_scale_without_cast(out_dtype):
    original_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(2)
        a, b, d, storage, _ = _skip_head_inputs(256, 512, (128, 64, 128), out_dtype, fp32_b_scale=True)
        torch.cuda.synchronize()
        before = storage.cpu().clone()
        with pytest.raises(RuntimeError, match=r'(?is)Assertion error.*(scalar_type|dtype|scaling|sfb)'):
            deep_gemm.fp8_gemm_nt_skip_head_mid(
                a, b, d, (128, 64, 128), recipe=(1, 1, 128), disable_ue8m0_cast=True)
        torch.cuda.synchronize()
        assert torch.equal(storage.cpu(), before)
    finally:
        deep_gemm.set_num_sms(original_sms)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    for out_dtype in (torch.bfloat16, torch.float32):
        for case in _SKIP_HEAD_CASES:
            test_sm120_skip_head_output_mapping_and_middle_preservation(*case.values, out_dtype)
        test_sm120_skip_head_reject_fp32_scale_without_cast(out_dtype)
