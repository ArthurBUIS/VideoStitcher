"""
Environment probe for host applications (e.g. the Portals Electron agent).

A host that wants to spawn `video_stitcher_seam_gpu.py --io pipe` can first run

    python tools/check_env.py

and parse the single JSON line on stdout to decide whether the sidecar is
runnable and to surface actionable errors to the operator. Exit code 0 means
all required imports succeeded; 1 means at least one is missing.

Output shape:
    {"ok": true, "python": "3.13.11", "cuda": false,
     "required": {"cv2": "4.13.0", ...}, "optional": {"numba": null, ...}}
"""

import json
import platform
import sys

REQUIRED_MODULES = ("cv2", "numpy", "torch", "ultralytics")
OPTIONAL_MODULES = ("numba", "transformers", "PIL")


def probe_module_version(module_name):
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main():
    required = {name: probe_module_version(name) for name in REQUIRED_MODULES}
    optional = {name: probe_module_version(name) for name in OPTIONAL_MODULES}

    cuda_available = False
    if required["torch"] is not None:
        import torch

        cuda_available = bool(torch.cuda.is_available())

    result = {
        "ok": all(version is not None for version in required.values()),
        "python": platform.python_version(),
        "cuda": cuda_available,
        "required": required,
        "optional": optional,
    }
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
