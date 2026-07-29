"""mojo-fastparquet versus fastparquet on identical Parquet codec buffers."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_fastparquet as mfp  # noqa: E402
from mojo_fastparquet import cencoding as mc  # noqa: E402
from fastparquet import cencoding as fc  # noqa: E402
from fastparquet import encoding as fe  # noqa: E402


def best_time(fn, reps: int = 5) -> float:
    fn()
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(42)
    rows: list[tuple[str, float, float]] = []

    n = 5_000_000
    width = 10
    values = rng.integers(0, 1 << width, n, dtype=np.int32)
    encoded_size = 16 + (n * width + 7) // 8
    mojo_encoded = np.empty(encoded_size, dtype=np.uint8)
    fast_encoded = np.empty(encoded_size, dtype=np.uint8)
    mo = mc.NumpyIO(mojo_encoded)
    fo = fc.NumpyIO(fast_encoded)

    def mojo_encode():
        mo.seek(0)
        mc.encode_bitpacked(values, width, mo)

    def fast_encode():
        fo.seek(0)
        fc.encode_bitpacked(values, width, fo)

    mt = best_time(mojo_encode)
    ft = best_time(fast_encode)
    assert bytes(mo.so_far()) == bytes(fo.so_far())
    rows.append(("hybrid bit-pack encode, 5M uint10", mt, ft))

    payload = mojo_encoded[: mo.tell()]
    mojo_decoded = np.empty(n, dtype=np.int32)
    fast_decoded = np.empty(n, dtype=np.int32)

    def mojo_decode():
        source = mc.NumpyIO(payload)
        output = mc.NumpyIO(mojo_decoded.view(np.uint8))
        mc.read_rle_bit_packed_hybrid(source, width, payload.size, output)

    def fast_decode():
        source = fc.NumpyIO(payload)
        output = fc.NumpyIO(fast_decoded.view(np.uint8))
        fc.read_rle_bit_packed_hybrid(source, width, payload.size, output)

    mt = best_time(mojo_decode)
    ft = best_time(fast_decode)
    assert np.array_equal(mojo_decoded, values)
    assert np.array_equal(fast_decoded, values)
    rows.append(("hybrid bit-pack decode, 5M uint10", mt, ft))

    boolean_count = 20_000_000
    booleans = rng.integers(0, 2, boolean_count, dtype=np.uint8)
    bool_payload = np.packbits(booleans, bitorder="little").tobytes()
    mt = best_time(lambda: mfp.read_plain_boolean(bool_payload, boolean_count))
    ft = best_time(lambda: fe.read_plain_boolean(bool_payload, boolean_count))
    assert np.array_equal(
        mfp.read_plain_boolean(bool_payload, boolean_count),
        fe.read_plain_boolean(bool_payload, boolean_count),
    )
    rows.append(("PLAIN boolean decode, 20M values", mt, ft))

    delta_values = np.cumsum(
        rng.integers(-8, 9, 1_000_000, dtype=np.int64), dtype=np.int64
    )
    delta_payload = mfp.delta_binary_pack(delta_values)
    mojo_delta = np.empty(delta_values.size, dtype=np.int64)
    fast_delta = np.empty(delta_values.size, dtype=np.int64)

    def mojo_delta_decode():
        source = mc.NumpyIO(np.frombuffer(delta_payload, dtype=np.uint8))
        output = mc.NumpyIO(mojo_delta.view(np.uint8))
        mc.delta_binary_unpack(source, output, longval=1)

    def fast_delta_decode():
        source = fc.NumpyIO(np.frombuffer(delta_payload, dtype=np.uint8))
        output = fc.NumpyIO(fast_delta.view(np.uint8))
        fc.delta_binary_unpack(source, output, longval=1)

    mt = best_time(mojo_delta_decode)
    ft = best_time(fast_delta_decode)
    assert np.array_equal(mojo_delta, delta_values)
    assert np.array_equal(fast_delta, delta_values)
    rows.append(("delta binary decode, 1M int64", mt, ft))

    strings = [f"value-{i % 1000:04d}".encode() for i in range(200_000)]
    byte_payload = mfp.pack_byte_array(strings)
    mt = best_time(lambda: mfp.unpack_byte_array(byte_payload, len(strings)))
    ft = best_time(lambda: fe.unpack_byte_array(byte_payload, len(strings)))
    assert mfp.unpack_byte_array(byte_payload, len(strings)).tolist() == strings
    rows.append(("PLAIN byte-array decode, 200K", mt, ft))

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Kernel | Mojo | fastparquet | Mojo speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, fast_time in rows:
        speedup = fast_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{fast_time * 1000:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
