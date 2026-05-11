"""Device detection: CUDA available + per-stage routing."""

import torch


def detect_device():
    """
    Probe CUDA availability and return a dict of per-stage device choices.

    On CUDA, all stages (YOLO, gain comp, warp, mask, cost, composite)
    route to "cuda". On failure, everything routes to "cpu".
    """
    info = {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_mem_gb": None,
        "yolo_device": "cpu",
        "composite_device": "cpu",
        "warp_device": "cpu",
        "cost_device": "cpu",
        "mask_device": "cpu",
        "gain_device": "cpu",
    }
    if not torch.cuda.is_available():
        return info
    try:
        probe = torch.zeros(1, device="cuda")
        _ = probe + 1
        del probe
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[device] CUDA reported available but probe failed: {e}")
        return info

    info["cuda_available"] = True
    info["device"] = "cuda"
    info["yolo_device"] = "cuda"
    info["composite_device"] = "cuda"
    info["warp_device"] = "cuda"
    info["cost_device"] = "cuda"
    info["mask_device"] = "cuda"
    info["gain_device"] = "cuda"
    try:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_mem_gb"] = props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return info
