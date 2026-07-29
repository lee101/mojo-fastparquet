"""ctypes loader for the Mojo Parquet codec library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_FASTPARQUET_LIB") or os.path.join(
    ROOT, "dist", "libmojo-fastparquet.so"
)

I = ctypes.c_int64
U = ctypes.c_uint64

_SIGNATURES = {
    "mfp_width_from_max_int": ([I], I),
    "mfp_encode_unsigned_varint": ([U, I, I], I),
    "mfp_read_unsigned_varint": ([I, I, I], U),
    "mfp_encode_bitpacked": ([I, I, I, I, I], I),
    "mfp_decode_hybrid": ([I, I, I, I, I, I, I], I),
    "mfp_read_rle": ([I, I, I, I, I, I, I, I], I),
    "mfp_read_bitpacked": ([I, I, I, I, I, I, I, I], I),
    "mfp_pack_bool": ([I, I, I, I], I),
    "mfp_unpack_bool": ([I, I, I, I], I),
    "mfp_delta_binary_unpack": ([I, I, I, I, I, I], I),
    "mfp_byte_array_offsets": ([I, I, I, I, I], I),
    "mfp_unpack_byte_array": ([I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_FASTPARQUET_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_FASTPARQUET_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Build the shared library if it is missing or older than its source."""
    if os.environ.get("MOJO_FASTPARQUET_LIB") and os.path.exists(LIB) and not force:
        return LIB
    source = os.path.join(SRC, "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None
_py_loaded: ctypes.PyDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def pylib() -> ctypes.PyDLL:
    global _py_loaded
    if _py_loaded is None:
        _py_loaded = ctypes.PyDLL(build())
        argtypes, restype = _SIGNATURES["mfp_unpack_byte_array"]
        fn = _py_loaded.mfp_unpack_byte_array
        fn.argtypes = argtypes
        fn.restype = restype
    return _py_loaded


def addr(array: np.ndarray, offset: int = 0) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if offset < 0 or offset > array.nbytes:
        raise ValueError("FFI buffer offset is out of bounds")
    return int(array.ctypes.data) + offset


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
