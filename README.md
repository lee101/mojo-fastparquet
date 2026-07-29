# mojo-fastparquet

`mojo-fastparquet` is a standalone Mojo port of the compute-heavy primitive
codecs beneath Python
[`fastparquet`](https://github.com/dask/fastparquet). It is useful when an
application already owns Parquet page buffers and needs to encode or decode
levels, dictionary indices, primitive PLAIN values, or delta-packed integers.
It is not a second DataFrame engine.

The Python package is `mojo_fastparquet`. Its low-level
`mojo_fastparquet.cencoding` module mirrors the covered names and signatures
from `fastparquet.cencoding`, including `NumpyIO`, so existing codec-level code
only needs an import change.

## Coverage

Implemented and tested:

- RLE/bit-packed hybrid decoding, including mixed RLE and bit-packed runs,
  1-byte definition/repetition levels, and 4-byte dictionary indices
- hybrid bit-packed encoding with optional Data Page V1 length prefix
- unsigned varints and bit-width calculation
- `DELTA_BINARY_PACKED` signed int32/int64 decoding, plus a conforming encoder
- PLAIN BOOLEAN packing and unpacking
- PLAIN INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, and
  FIXED_LEN_BYTE_ARRAY
- checked BYTE_ARRAY length scanning and UTF-8 conversion

The low-level `encode_rle_bp` remains byte-for-byte compatible with
fastparquet. The high-level `encode_rle_bit_packed_hybrid` additionally pads
the final group to eight values as required by the Parquet format. Decoding is
bounds-checked and reports truncated or invalid input instead of reading
beyond the supplied buffer.

Not implemented:

- Parquet file/page framing, Thrift metadata, schema evolution, or pandas
  conversion
- compression codecs
- nested-list Dremel assembly and dictionary materialization
- DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, and BYTE_STREAM_SPLIT

Use upstream fastparquet or PyArrow for those file-level features. This
project can accelerate the covered buffers inside a larger reader or writer.

## Install and build

The repository pins the tested Mojo nightly and carries all Python test
dependencies in Pixi:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-fastparquet.so`. Set
`MOJO_FASTPARQUET_LIB` to load a copy from another location.

## Usage

This example encodes and decodes dictionary indices, then round-trips a
BYTE_ARRAY PLAIN payload:

```python
import numpy as np
import mojo_fastparquet as mfp

indices = np.array([3, 1, 3, 0, 2, 2, 1, 3, 0], dtype=np.int32)
encoded = mfp.encode_rle_bit_packed_hybrid(indices, width=2)
decoded = mfp.decode_rle_bit_packed_hybrid(
    encoded, width=2, count=len(indices)
)
assert np.array_equal(decoded, indices)

plain = mfp.encode_plain([b"alpha", b"beta"], mfp.Type.BYTE_ARRAY)
assert mfp.read_plain(plain, mfp.Type.BYTE_ARRAY, 2).tolist() == [
    b"alpha", b"beta"
]
```

Run it after `pixi install` with `pixi run python example.py`, or use the
repository activation environment, which adds `python/` to `PYTHONPATH`.

## Benchmarks

Measured on 2026-07-29 on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64,
using fastparquet 2026.5.0. Each row is the best of five warmed runs on the
same preallocated input and output buffers. The only supported benchmark
command is `pixi run bench`; it takes a machine-wide lock.

| Kernel | Mojo | fastparquet | Mojo speedup |
|---|---:|---:|---:|
| hybrid bit-pack encode, 5M uint10 | 19.20 ms | 14.95 ms | 0.78x |
| hybrid bit-pack decode, 5M uint10 | 10.47 ms | 20.91 ms | 2.00x |
| PLAIN boolean decode, 20M values | 3.34 ms | 2.60 ms | 0.78x |
| delta binary decode, 1M int64 | 2.65 ms | 14.39 ms | 5.43x |
| PLAIN byte-array decode, 200K | 12.58 ms | 13.46 ms | 1.07x |

On this run Mojo was slower for bit-pack encoding and boolean decoding, and
faster for the other three measured kernels. These are local best-case kernel
timings, not end-to-end Parquet file-read results.

## How it works

All native kernels live in one Mojo compilation unit to avoid repeated build
cost. Python loads the shared library with `ctypes`. Numeric arrays remain
owned by NumPy; the FFI passes their addresses as 64-bit integers, and each
exported Mojo function reconstructs a mutable `UnsafePointer` internally.
BYTE_ARRAY decoding holds the GIL while Mojo creates `bytes` or `str` objects
through the CPython API and writes their references directly into a
preallocated NumPy object array, avoiding intermediate offset arrays and
Python lists.

Parquet bit streams are little-endian within each byte. The hybrid decoder
uses a rolling 64-bit reservoir for widths up to 32, writes either packed
`uint8` levels or native-endian `int32` indices, and returns both consumed and
produced positions to the `NumpyIO` wrapper. Bit-pack encoding validates with
SIMD and uses four workers above one million values. BOOLEAN unpacking expands
multiple source bytes per SIMD operation and handles remaining bytes and bits
with scalar tails. Delta miniblocks use the same reservoir and accumulate
zigzag-decoded minimum deltas directly into caller-owned int32 or int64 arrays.
PLAIN numeric reads stay zero-copy NumPy views.

The pytest suite compares encoded bytes, decoded arrays, dtypes, stream
positions, and error behavior with the installed upstream fastparquet. Wide
delta values and malformed buffers also have independent conformance vectors
covering cases where upstream assumes padded memory.
