"""Prebuilt fused cost-volume extension (mvsf_kernels.cu), named by the source hash; built once on first use."""
import hashlib
import importlib.util
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "mvsf_kernels.cu"
_EXT = None


def _name():
    return "mvsf_kernels_" + hashlib.sha1(_SRC.read_bytes()).hexdigest()[:12]


def build():
    from torch.utils.cpp_extension import load
    name = _name()
    d = _HERE / "prebuilt" / name
    d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0;12.0+PTX")
    load(name=name, sources=[str(_SRC)], build_directory=str(d), extra_cuda_cflags=["-O3"], verbose=False)


def load():
    global _EXT
    if _EXT is not None:
        return _EXT
    name = _name()
    d = _HERE / "prebuilt" / name
    so = sorted(d.glob(f"{name}*.so")) if d.exists() else []
    if not so:
        build()
        so = sorted(d.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so[0])
    _EXT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_EXT)
    return _EXT
