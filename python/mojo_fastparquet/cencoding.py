"""Mojo implementations of fastparquet's primitive Cython codec API."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from ._lib import addr, lib


class NumpyIO:
    """Read or write a contiguous byte buffer while tracking its position."""

    def __init__(self, data: Any):
        array = np.asarray(data)
        if not array.flags.c_contiguous:
            raise ValueError("NumpyIO requires a C-contiguous buffer")
        self.data = array.view(np.uint8).reshape(-1)
        self.loc = 0
        self.nbytes = self.data.nbytes

    @property
    def len(self) -> int:
        return self.nbytes

    def read(self, x: int = -1) -> np.ndarray:
        if x < 1:
            x = self.nbytes - self.loc
        end = min(self.loc + x, self.nbytes)
        result = self.data[self.loc : end]
        self.loc = end
        return result

    def read_byte(self) -> int:
        result = int(self.data[self.loc])
        self.loc += 1
        return result

    def read_int(self) -> int:
        if self.nbytes - self.loc < 4:
            return 0
        result = struct.unpack_from("<i", self.data, self.loc)[0]
        self.loc += 4
        return result

    def write(self, data: Any) -> None:
        raw = np.frombuffer(memoryview(data), dtype=np.uint8)
        end = self.loc + raw.size
        self.data[self.loc : end] = raw
        self.loc = end

    def write_byte(self, value: int) -> None:
        if self.loc < self.nbytes:
            self.data[self.loc] = value
            self.loc += 1

    def write_int(self, value: int) -> None:
        if self.nbytes - self.loc >= 4:
            self.data[self.loc : self.loc + 4] = np.frombuffer(
                struct.pack("<i", value), dtype=np.uint8
            )
            self.loc += 4

    def tell(self) -> int:
        return self.loc

    def seek(self, loc: int, whence: int = 0) -> int:
        if whence == 0:
            self.loc = loc
        elif whence == 1:
            self.loc += loc
        elif whence == 2:
            self.loc = self.nbytes + loc
        else:
            raise ValueError("invalid whence")
        self.loc = min(max(self.loc, 0), self.nbytes)
        return self.loc

    def so_far(self) -> np.ndarray:
        return self.data[: self.loc]


def width_from_max_int(value: int) -> int:
    return int(lib().mfp_width_from_max_int(int(value)))


def encode_unsigned_varint(x: int, o: NumpyIO) -> None:
    if x < 0 or x > 2**64 - 1:
        raise OverflowError("unsigned varint is outside uint64")
    written = lib().mfp_encode_unsigned_varint(
        x, addr(o.data, o.loc), o.nbytes - o.loc
    )
    if written < 0:
        raise BufferError("output buffer is too small")
    o.loc += written


def read_unsigned_var_int(file_obj: NumpyIO) -> int:
    consumed = np.zeros(1, dtype=np.int64)
    value = lib().mfp_read_unsigned_varint(
        addr(file_obj.data, file_obj.loc),
        file_obj.nbytes - file_obj.loc,
        addr(consumed),
    )
    if consumed[0] < 0:
        raise EOFError("truncated unsigned varint")
    file_obj.loc += int(consumed[0])
    return int(value)


def encode_bitpacked(values: Any, width: int, o: NumpyIO) -> None:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iub":
        raise TypeError("bit-packed values must be integers")
    if raw.size and (raw.min() < 0 or raw.max() > np.iinfo(np.int32).max):
        raise OverflowError("bit-packed value is outside int32")
    data = np.ascontiguousarray(raw, dtype=np.int32)
    _require_writable(o)
    written = lib().mfp_encode_bitpacked(
        addr(data), data.size, width, addr(o.data, o.loc), o.nbytes - o.loc
    )
    if written == -1:
        raise BufferError("output buffer is too small")
    if written == -2:
        raise ValueError(f"value does not fit in {width} bits")
    o.loc += written


def encode_rle_bp(data: Any, width: int, o: NumpyIO, withlength: int = 0) -> None:
    start = o.tell()
    if withlength:
        o.seek(4, 1)
    encode_bitpacked(data, width, o)
    if withlength:
        end = o.tell()
        o.seek(start)
        o.write_int(end - start - 4)
        o.seek(end)


def _decode_call(
    name: str,
    file_obj: NumpyIO,
    o: NumpyIO,
    itemsize: int,
    *args: int,
) -> int:
    if itemsize not in (1, 4):
        raise ValueError("itemsize must be 1 or 4")
    _require_writable(o)
    capacity = (o.nbytes - o.loc) // itemsize
    consumed = np.zeros(1, dtype=np.int64)
    fn = getattr(lib(), name)
    produced = fn(
        addr(file_obj.data, file_obj.loc),
        file_obj.nbytes - file_obj.loc,
        *args,
        addr(o.data, o.loc),
        capacity,
        itemsize,
        addr(consumed),
    )
    if produced == -2:
        raise EOFError("truncated Parquet encoded stream")
    if produced < 0:
        raise ValueError("invalid Parquet encoded stream")
    file_obj.loc += int(consumed[0])
    o.loc += produced * itemsize
    return produced


def read_rle(
    file_obj: NumpyIO,
    header: int,
    bit_width: int,
    o: NumpyIO,
    itemsize: int = 4,
) -> None:
    _decode_call("mfp_read_rle", file_obj, o, itemsize, header, bit_width)


def read_bitpacked(
    file_obj: NumpyIO,
    header: int,
    width: int,
    o: NumpyIO,
    itemsize: int = 4,
) -> None:
    _decode_call("mfp_read_bitpacked", file_obj, o, itemsize, header, width)


def read_rle_bit_packed_hybrid(
    io_obj: NumpyIO,
    width: int,
    length: int | bool,
    o: NumpyIO,
    itemsize: int = 4,
) -> None:
    if length is False:
        length = io_obj.read_int()
    if length < 0 or length > io_obj.nbytes - io_obj.loc:
        raise EOFError("hybrid stream length exceeds input")
    view = NumpyIO(io_obj.data[io_obj.loc : io_obj.loc + int(length)])
    _decode_call("mfp_decode_hybrid", view, o, itemsize, width)
    io_obj.loc += view.loc


def read_bitpacked1(file_obj: NumpyIO, count: int, o: NumpyIO) -> None:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require_writable(o)
    actual = min(count, o.nbytes - o.loc)
    needed = (count + 7) // 8
    if file_obj.nbytes - file_obj.loc < needed:
        raise EOFError("truncated boolean stream")
    result = lib().mfp_unpack_bool(
        addr(file_obj.data, file_obj.loc),
        needed,
        actual,
        addr(o.data, o.loc),
    )
    if result < 0:
        raise EOFError("truncated boolean stream")
    file_obj.loc += needed
    o.loc += actual


def write_bitpacked1(file_obj: NumpyIO, count: int, o: NumpyIO) -> None:
    if count < 0:
        raise ValueError("count must be non-negative")
    _require_writable(o)
    needed = (count + 7) // 8
    if file_obj.nbytes - file_obj.loc < count:
        raise EOFError("input boolean stream is too short")
    if o.nbytes - o.loc < needed:
        raise BufferError("output buffer is too small")
    result = lib().mfp_pack_bool(
        addr(file_obj.data, file_obj.loc),
        count,
        addr(o.data, o.loc),
        o.nbytes - o.loc,
    )
    if result < 0:
        raise BufferError("output buffer is too small")
    file_obj.loc += count
    o.loc += result


def delta_binary_unpack(file_obj: NumpyIO, o: NumpyIO, longval: int = 0) -> int:
    _require_writable(o)
    itemsize = 8 if longval else 4
    capacity = (o.nbytes - o.loc) // itemsize
    consumed = np.zeros(1, dtype=np.int64)
    produced = lib().mfp_delta_binary_unpack(
        addr(file_obj.data, file_obj.loc),
        file_obj.nbytes - file_obj.loc,
        addr(o.data, o.loc),
        capacity,
        int(bool(longval)),
        addr(consumed),
    )
    if produced == -2:
        raise EOFError("truncated DELTA_BINARY_PACKED stream")
    if produced < 0:
        raise ValueError("invalid DELTA_BINARY_PACKED stream")
    file_obj.loc += int(consumed[0])
    o.loc += produced * itemsize
    return 0


def encode_rle_bit_packed_hybrid(
    values: Any, width: int | None = None, with_length: bool = False
) -> bytes:
    """Return a Parquet hybrid stream, using fastparquet's bit-packed form."""
    data = np.ascontiguousarray(values, dtype=np.int32)
    if width is None:
        width = width_from_max_int(int(data.max(initial=0)))
    padded_size = ((data.size + 7) // 8) * 8
    if padded_size != data.size:
        padded = np.zeros(padded_size, dtype=np.int32)
        padded[: data.size] = data
        data = padded
    size = 16 + (data.size * width + 7) // 8
    storage = np.empty(size + (4 if with_length else 0), dtype=np.uint8)
    output = NumpyIO(storage)
    encode_rle_bp(data, width, output, int(with_length))
    return bytes(output.so_far())


def _require_writable(io_obj: NumpyIO) -> None:
    if not io_obj.data.flags.writeable:
        raise TypeError("output buffer must be writable")


def decode_rle_bit_packed_hybrid(
    data: Any,
    width: int,
    count: int,
    *,
    length_prefix: bool = False,
    dtype: Any = np.int32,
) -> np.ndarray:
    """Decode a complete hybrid stream into an int32 or uint8 array."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.int32)):
        raise ValueError("dtype must be uint8 or int32")
    source = NumpyIO(np.frombuffer(memoryview(data), dtype=np.uint8))
    result = np.empty(count, dtype=dtype)
    output = NumpyIO(result.view(np.uint8))
    length: int | bool = False if length_prefix else source.nbytes
    read_rle_bit_packed_hybrid(source, width, length, output, dtype.itemsize)
    return result[: output.tell() // dtype.itemsize]
