"""GPU / VRAM queries. Imported after torch is installed."""

from __future__ import annotations


def torch_module():
    import torch

    return torch


def cuda_devices() -> list[dict]:
    torch = torch_module()
    found: list[dict] = []
    if not torch.cuda.is_available():
        return found
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        found.append(
            {
                "index": index,
                "name": props.name,
                "total": int(props.total_memory),
            }
        )
    return found


def best_cuda_index() -> int | None:
    devices = cuda_devices()
    if not devices:
        return None
    return max(devices, key=lambda item: item["total"])["index"]


def memory_info(index: int) -> tuple[int, int]:
    torch = torch_module()
    free, total = torch.cuda.mem_get_info(index)
    used = int(total) - int(free)
    return used, int(total)


def format_bytes(num: int) -> str:
    gib = num / (1024**3)
    return f"{gib:.2f} GB"


def describe_runtime() -> dict:
    torch = torch_module()
    devices = cuda_devices()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": devices,
    }
