"""Verify the server-side uv, PyTorch, CUDA, and Transformers environment."""

from __future__ import annotations

import json
import platform

import torch
import transformers


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not start model extraction or training")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    result = left @ right
    torch.cuda.synchronize(device)

    report = {
        "bf16_matmul_finite": bool(torch.isfinite(result).all().item()),
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
