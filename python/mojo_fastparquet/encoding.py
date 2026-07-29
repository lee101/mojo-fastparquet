"""PLAIN and DELTA_BINARY_PACKED Parquet helpers."""

from __future__ import annotations

import struct
from typing import Any, Iterable

import numpy as np

from . import cencoding
from ._lib import addr, lib, pylib
from .parquet_thrift import Type

DECODE_TYPEMAP = {
    Type.INT32: np.dtype("<i4"),
    Type.INT64: np.dtype("<i8"),
    Type.INT96: np.dtype("S12"),
    Type.FLOAT: np.dtype("<f4"),
    Type.DOUBLE: np.dtype("<f8"),
}


def read_plain_boolean(raw_bytes: Any, count: int, out: np.ndarray | None = None):
    data = np.frombuffer(memoryview(raw_bytes), dtype=np.uint8)
    result = np.empty(count, dtype=bool) if out is None else out
    cencoding.read_bitpacked1(
        cencoding.NumpyIO(data), count, cencoding.NumpyIO(result.view(np.uint8))
    )
    return result[:count]


def _byte_array_offsets(raw_bytes: Any, n: int) -> tuple[np.ndarray, np.ndarray, memoryview]:
    view = memoryview(raw_bytes)
    data = np.frombuffer(view, dtype=np.uint8)
    starts = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    consumed = lib().mfp_byte_array_offsets(
        addr(data), data.size, n, addr(starts), addr(lengths)
    )
    if consumed == -2:
        raise ValueError("corrupt BYTE_ARRAY PLAIN data")
    if consumed < 0:
        raise ValueError("invalid BYTE_ARRAY count")
    return starts, lengths, view


def unpack_byte_array(raw_bytes: Any, n: int, utf: bool = False):
    view = memoryview(raw_bytes)
    data = np.frombuffer(view, dtype=np.uint8)
    result = np.empty(n, dtype=object)
    consumed = pylib().mfp_unpack_byte_array(
        addr(data),
        data.size,
        n,
        addr(result),
        int(utf),
    )
    if consumed == -2:
        raise ValueError("corrupt BYTE_ARRAY PLAIN data")
    if consumed < 0:
        raise ValueError("invalid BYTE_ARRAY count")
    return result


def pack_byte_array(values: Iterable[bytes | str]) -> bytes:
    parts: list[bytes] = []
    for value in values:
        raw = value.encode() if isinstance(value, str) else bytes(value)
        if len(raw) > 2**31 - 1:
            raise OverflowError("BYTE_ARRAY value is too large")
        parts.extend((struct.pack("<i", len(raw)), raw))
    return b"".join(parts)


def read_plain(
    raw_bytes: Any,
    type_: int | Type,
    count: int,
    width: int = 0,
    utf: bool = False,
    stat: bool = False,
):
    type_ = Type(type_)
    if type_ in DECODE_TYPEMAP:
        return np.frombuffer(memoryview(raw_bytes), dtype=DECODE_TYPEMAP[type_], count=count)
    if type_ == Type.FIXED_LEN_BYTE_ARRAY:
        if count == 1:
            width = len(raw_bytes)
        return np.frombuffer(memoryview(raw_bytes), dtype=np.dtype(f"S{width}"), count=count)
    if type_ == Type.BOOLEAN:
        return read_plain_boolean(raw_bytes, count)
    if type_ == Type.BYTE_ARRAY:
        if stat:
            value = bytes(raw_bytes).decode() if utf else bytes(raw_bytes)
            return np.array([value], dtype=object)
        return unpack_byte_array(raw_bytes, count, utf=utf)
    raise ValueError(f"unsupported Parquet physical type {type_}")


def encode_plain(values: Any, type_: int | Type, width: int = 0, utf: bool = False) -> bytes:
    """Encode values using Parquet PLAIN physical encoding."""
    type_ = Type(type_)
    if type_ == Type.BOOLEAN:
        source = np.ascontiguousarray(values, dtype=np.uint8)
        storage = np.empty((source.size + 7) // 8, dtype=np.uint8)
        cencoding.write_bitpacked1(
            cencoding.NumpyIO(source), source.size, cencoding.NumpyIO(storage)
        )
        return storage.tobytes()
    if type_ == Type.BYTE_ARRAY:
        return pack_byte_array(values)
    if type_ == Type.FIXED_LEN_BYTE_ARRAY:
        data = np.asarray(values, dtype=f"S{width}")
        return data.tobytes()
    if type_ in DECODE_TYPEMAP:
        return np.asarray(values, dtype=DECODE_TYPEMAP[type_]).tobytes()
    raise ValueError(f"unsupported Parquet physical type {type_}")


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 63)


def _pack_raw(values: list[int], width: int) -> bytes:
    if width == 0:
        return b""
    out = bytearray((len(values) * width + 7) // 8)
    bit = 0
    for value in values:
        byte = bit // 8
        shift = bit & 7
        word = value << shift
        take = (width + shift + 7) // 8
        for j in range(take):
            if byte + j < len(out):
                out[byte + j] |= (word >> (8 * j)) & 255
        bit += width
    return bytes(out)


def delta_binary_pack(
    values: Any, *, block_size: int = 128, miniblocks: int = 4
) -> bytes:
    """Encode signed integers using Parquet DELTA_BINARY_PACKED."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iub":
        raise TypeError("DELTA_BINARY_PACKED values must be integers")
    if raw.size and (
        raw.min() < np.iinfo(np.int64).min or raw.max() > np.iinfo(np.int64).max
    ):
        raise OverflowError("DELTA_BINARY_PACKED value is outside int64")
    data = np.ascontiguousarray(raw, dtype=np.int64)
    if data.size == 0:
        raise ValueError("DELTA_BINARY_PACKED requires at least one value")
    if block_size <= 0 or miniblocks <= 0 or block_size % miniblocks:
        raise ValueError("block_size must be divisible by miniblocks")
    per = block_size // miniblocks
    if per % 8:
        raise ValueError("values per miniblock must be divisible by 8")
    result = bytearray()
    result += _varint(block_size)
    result += _varint(miniblocks)
    result += _varint(data.size)
    result += _varint(_zigzag(int(data[0])))
    deltas = np.diff(data)
    for block_start in range(0, data.size, block_size):
        block = [int(x) for x in deltas[block_start : block_start + block_size]]
        minimum = min(block, default=0)
        adjusted = [x - minimum for x in block]
        adjusted.extend([0] * (block_size - len(adjusted)))
        result += _varint(_zigzag(minimum))
        widths = []
        for start in range(0, block_size, per):
            widths.append(max(adjusted[start : start + per], default=0).bit_length())
        result += bytes(widths)
        for mini, bit_width in enumerate(widths):
            start = mini * per
            result += _pack_raw(adjusted[start : start + per], bit_width)
    return bytes(result)


def delta_binary_decode(data: Any, count: int, *, dtype: Any = np.int64) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise ValueError("dtype must be int32 or int64")
    source = cencoding.NumpyIO(np.frombuffer(memoryview(data), dtype=np.uint8))
    result = np.empty(count, dtype=dtype)
    output = cencoding.NumpyIO(result.view(np.uint8))
    cencoding.delta_binary_unpack(source, output, int(dtype.itemsize == 8))
    return result[: output.tell() // dtype.itemsize]
