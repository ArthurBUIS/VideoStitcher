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

import importlib.metadata
import importlib.util
import json
import platform
import sys

REQUIRED_MODULES = ("cv2", "numpy", "torch", "ultralytics")
OPTIONAL_MODULES = ("numba", "transformers", "PIL")

# Map import name -> distribution name, where they differ, so we can read
# the version from installed package metadata without importing the module.
DIST_NAME_OVERRIDES = {"cv2": "opencv-python", "PIL": "Pillow"}


def probe_module_version(module_name):
    """Report a module's version WITHOUT importing it.

    The host runs this as a preflight before every stitching session, so
    the cost matters: actually importing torch + ultralytics + transformers
    takes ~20s+ cold on the GPU machines and was racing the host's spawn
    timeout. `find_spec` only checks that the module is importable (present
    on sys.path with a valid loader); the version comes from install
    metadata. Returns None when the module isn't installed.
    """
    try:
        if importlib.util.find_spec(module_name) is None:
            return None
    except Exception:
        # find_spec can raise if a parent package is broken; treat as absent.
        return None
    dist_name = DIST_NAME_OVERRIDES.get(module_name, module_name)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main():
    required = {name: probe_module_version(name) for name in REQUIRED_MODULES}
    optional = {name: probe_module_version(name) for name in OPTIONAL_MODULES}

    # torch is the one module we must actually import: cuda availability
    # can't be read from metadata. This is the probe's only heavy import.
    cuda_available = False
    if required["torch"] is not None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            # torch is present per metadata but fails to load (e.g. a
            # broken CUDA/DLL install). Surface it as "not importable" so
            # the host reports a missing/broken dependency rather than
            # claiming the environment is fine.
            required["torch"] = None

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
