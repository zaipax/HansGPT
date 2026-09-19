"""GPU2 regression: eager vs compiled first-order CUTLASS output and gradients."""

import os

import torch
import torch.nn.functional as F

from hansgpt_research.cvae_fixed_step import install_xformers


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("Select physical GPU2")
    torch.manual_seed(20260918)
    for causal, scale in ((False, None), (True, 128**-0.5)):
        q = torch.randn(2, 4, 64, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        grad = torch.randn_like(q)
        install_xformers()
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
        install_xformers(opaque_backward=True)

        def attention(q, k, v, causal=causal, scale=scale):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

        compiled = torch.compile(attention, fullgraph=True)
        actual = compiled(q, k, v)
        actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for left, right in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(left, right, rtol=1e-3, atol=1e-3)
        print(
            f"PASS causal={causal} scale={scale}; forward exact, gradients within 1e-3", flush=True
        )


if __name__ == "__main__":
    main()

