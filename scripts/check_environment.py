"""Verify the server-side uv, PyTorch, CUDA, and Transformers environment."""

from __future__ import annotations

import argparse
import json
import platform

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu", action="store_true", help="Check CPU preprocessing without GPU use"
    )
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16"), default="auto")
    args = parser.parse_args()
    if args.cpu:
        result = torch.ones((16, 16)) @ torch.ones((16, 16))
        report = {
            "mode": "cpu_preprocessing",
            "cpu_matmul_finite": bool(torch.isfinite(result).all().item()),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not start model extraction or training")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    native_bf16 = properties.major >= 8 and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not native_bf16:
        raise RuntimeError("This GPU does not support native BF16; use FP16")
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and native_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    left = torch.randn((1024, 1024), device=device, dtype=dtype, requires_grad=True)
    right = torch.randn((1024, 1024), device=device, dtype=dtype)
    result = left @ right
    result.float().square().mean().backward()
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(result).all().item())
    gradient_finite = bool(torch.isfinite(left.grad).all().item())
    if not finite or not gradient_finite:
        raise RuntimeError("GPU forward/backward produced nonfinite values")

    report = {
        "matmul_finite": finite,
        "gradient_finite": gradient_finite,
        "dtype": str(dtype),
        "native_bf16": native_bf16,
        "compiled_architectures": torch.cuda.get_arch_list(),
        "compute_capability": [properties.major, properties.minor],
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "gpu": properties.name,
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
