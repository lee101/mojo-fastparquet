"""Primitive stream codecs compared directly with fastparquet.cencoding."""

import struct

import numpy as np
import pytest

import mojo_fastparquet.cencoding as ours

upstream = pytest.importorskip("fastparquet.cencoding")


@pytest.mark.parametrize(
    "value", [0, 1, 2, 3, 7, 8, 255, 256, 2**31 - 1, 2**63 - 1]
)
def test_width_from_max_int(value):
    assert ours.width_from_max_int(value) == upstream.width_from_max_int(value)


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 2**32, 2**63 - 1, 2**64 - 1])
def test_unsigned_varint_parity(value):
    left = np.zeros(16, dtype=np.uint8)
    right = np.zeros(16, dtype=np.uint8)
    lo = ours.NumpyIO(left)
    ro = upstream.NumpyIO(right)
    ours.encode_unsigned_varint(value, lo)
    upstream.encode_unsigned_varint(value, ro)
    assert bytes(lo.so_far()) == bytes(ro.so_far())
    source = ours.NumpyIO(left[: lo.tell()])
    assert ours.read_unsigned_var_int(source) == value
    assert source.tell() == lo.tell()


def test_empty_unsigned_varint_is_rejected():
    with pytest.raises(EOFError, match="truncated"):
        ours.read_unsigned_var_int(ours.NumpyIO(np.empty(0, dtype=np.uint8)))


def test_overflowing_unsigned_varint_is_rejected():
    payload = np.array([0xFF] * 9 + [0x02], dtype=np.uint8)
    with pytest.raises(EOFError, match="truncated"):
        ours.read_unsigned_var_int(ours.NumpyIO(payload))


@pytest.mark.parametrize("width", [0, 1, 2, 3, 7, 8, 9, 16, 24])
@pytest.mark.parametrize("count", [1, 7, 8, 9, 33])
def test_encode_bitpacked_byte_parity(width, count):
    rng = np.random.default_rng(width * 100 + count)
    high = 1 if width == 0 else min(2**width, 2**31)
    values = np.zeros(count, dtype=np.int32) if width == 0 else rng.integers(
        0, high, count, dtype=np.int32
    )
    left = np.zeros(1024, dtype=np.uint8)
    right = np.zeros(1024, dtype=np.uint8)
    lo = ours.NumpyIO(left)
    ro = upstream.NumpyIO(right)
    ours.encode_bitpacked(values, width, lo)
    upstream.encode_bitpacked(values, width, ro)
    assert bytes(lo.so_far()) == bytes(ro.so_far())


def test_encode_rle_bp_length_prefix_parity():
    values = np.arange(29, dtype=np.int32) % 11
    left = np.zeros(128, dtype=np.uint8)
    right = np.zeros(128, dtype=np.uint8)
    lo = ours.NumpyIO(left)
    ro = upstream.NumpyIO(right)
    ours.encode_rle_bp(values, 4, lo, withlength=1)
    upstream.encode_rle_bp(values, 4, ro, withlength=1)
    assert bytes(lo.so_far()) == bytes(ro.so_far())
    assert struct.unpack_from("<I", left)[0] == lo.tell() - 4


@pytest.mark.parametrize("width", [1, 2, 3, 8, 12, 16, 24])
def test_hybrid_decode_upstream_stream(width):
    rng = np.random.default_rng(width)
    values = rng.integers(0, min(2**width, 2**31), 257, dtype=np.int32)
    encoded = np.zeros(4096, dtype=np.uint8)
    eo = upstream.NumpyIO(encoded)
    upstream.encode_rle_bp(values, width, eo)
    payload = encoded[: eo.tell()]

    left = np.empty(values.size, dtype=np.int32)
    right = np.empty(values.size, dtype=np.int32)
    li = ours.NumpyIO(payload)
    ri = upstream.NumpyIO(payload)
    lo = ours.NumpyIO(left.view(np.uint8))
    ro = upstream.NumpyIO(right.view(np.uint8))
    ours.read_rle_bit_packed_hybrid(li, width, payload.size, lo)
    upstream.read_rle_bit_packed_hybrid(ri, width, payload.size, ro)
    assert lo.tell() == ro.tell()
    assert np.array_equal(left[: lo.tell() // 4], right[: ro.tell() // 4])
    assert np.array_equal(left[: values.size], values)


def test_width_31_round_trip_reference():
    values = np.array([0, 1, 2**20, 2**30, 2**31 - 1] * 9, dtype=np.int32)
    payload = ours.encode_rle_bit_packed_hybrid(values, 31)
    decoded = ours.decode_rle_bit_packed_hybrid(payload, 31, values.size)
    assert np.array_equal(decoded, values)


def test_mixed_rle_and_bitpacked_stream():
    run = bytes([12 << 1, 5])
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    packed = np.zeros(16, dtype=np.uint8)
    po = upstream.NumpyIO(packed)
    upstream.encode_bitpacked(values, 3, po)
    payload = np.frombuffer(run + bytes(po.so_far()), dtype=np.uint8)

    left = np.empty(20, dtype=np.int32)
    right = np.empty(20, dtype=np.int32)
    lo = ours.NumpyIO(left.view(np.uint8))
    ro = upstream.NumpyIO(right.view(np.uint8))
    ours.read_rle_bit_packed_hybrid(ours.NumpyIO(payload), 3, payload.size, lo)
    upstream.read_rle_bit_packed_hybrid(upstream.NumpyIO(payload), 3, payload.size, ro)
    assert np.array_equal(left, right)
    assert np.array_equal(left, np.r_[np.full(12, 5), values])


def test_itemsize_one_level_stream():
    values = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 4, dtype=np.int32)
    payload = np.zeros(64, dtype=np.uint8)
    encoded = upstream.NumpyIO(payload)
    upstream.encode_bitpacked(values, 1, encoded)
    result = np.empty(values.size, dtype=np.uint8)
    output = ours.NumpyIO(result)
    ours.read_rle_bit_packed_hybrid(
        ours.NumpyIO(payload[: encoded.tell()]), 1, encoded.tell(), output, itemsize=1
    )
    assert np.array_equal(result, values)


def test_direct_rle_and_bitpacked_positions():
    rle_input = np.array([0x34, 0x12], dtype=np.uint8)
    rle_result = np.empty(5, dtype=np.int32)
    ri = ours.NumpyIO(rle_input)
    ro = ours.NumpyIO(rle_result.view(np.uint8))
    ours.read_rle(ri, 10, 16, ro)
    assert ri.tell() == 2
    assert np.array_equal(rle_result, np.full(5, 0x1234))

    values = np.arange(8, dtype=np.int32)
    buf = np.zeros(32, dtype=np.uint8)
    stream = upstream.NumpyIO(buf)
    upstream.encode_bitpacked(values, 3, stream)
    source = ours.NumpyIO(buf[1 : stream.tell()])
    result = np.empty(8, dtype=np.int32)
    output = ours.NumpyIO(result.view(np.uint8))
    ours.read_bitpacked(source, 3, 3, output)
    assert np.array_equal(result, values)


def test_boolean_bitpack_round_trip():
    rng = np.random.default_rng(4)
    values = rng.integers(0, 2, 101, dtype=np.uint8)
    packed = np.empty((values.size + 7) // 8, dtype=np.uint8)
    ours.write_bitpacked1(
        ours.NumpyIO(values), values.size, ours.NumpyIO(packed)
    )
    decoded = np.empty(values.size, dtype=np.uint8)
    ours.read_bitpacked1(
        ours.NumpyIO(packed), values.size, ours.NumpyIO(decoded)
    )
    assert np.array_equal(decoded, values)


@pytest.mark.parametrize("count", [1, 7, 8, 31, 32, 33, 39, 40, 41])
def test_boolean_unpack_simd_and_scalar_tails(count):
    rng = np.random.default_rng(count)
    values = rng.integers(0, 2, count, dtype=np.uint8)
    payload = np.packbits(values, bitorder="little")
    decoded = np.empty(count, dtype=np.uint8)
    ours.read_bitpacked1(
        ours.NumpyIO(payload), count, ours.NumpyIO(decoded)
    )
    assert np.array_equal(decoded, values)


@pytest.mark.parametrize("count", [999_992, 1_000_000])
def test_encode_parallel_threshold_parity(count):
    values = np.arange(count, dtype=np.int32) & 1023
    left = np.empty(16 + count * 10 // 8, dtype=np.uint8)
    right = np.empty_like(left)
    lo = ours.NumpyIO(left)
    ro = upstream.NumpyIO(right)
    ours.encode_bitpacked(values, 10, lo)
    upstream.encode_bitpacked(values, 10, ro)
    assert bytes(lo.so_far()) == bytes(ro.so_far())


def test_encode_validation_simd_tail():
    values = np.zeros(9, dtype=np.int32)
    values[-1] = 8
    with pytest.raises(ValueError, match="does not fit"):
        ours.encode_bitpacked(values, 3, ours.NumpyIO(np.empty(32, dtype=np.uint8)))


def test_encode_rejects_silent_int32_narrowing():
    values = np.array([0, 2**32 + 1], dtype=np.uint64)
    with pytest.raises(OverflowError, match="outside int32"):
        ours.encode_bitpacked(values, 32, ours.NumpyIO(np.empty(32, dtype=np.uint8)))


def test_decode_rejects_non_integer_output_dtype():
    with pytest.raises(ValueError, match="uint8 or int32"):
        ours.decode_rle_bit_packed_hybrid(b"\x03\x00", 1, 8, dtype=np.float32)


def test_read_clamps_position_and_readonly_output_is_rejected():
    source = ours.NumpyIO(np.arange(3, dtype=np.uint8))
    assert source.read(10).tolist() == [0, 1, 2]
    assert source.tell() == 3
    output = ours.NumpyIO(np.frombuffer(bytes(4), dtype=np.uint8))
    with pytest.raises(TypeError, match="writable"):
        ours.read_bitpacked1(ours.NumpyIO(np.array([0], dtype=np.uint8)), 1, output)


def test_truncated_stream_raises():
    output = ours.NumpyIO(np.empty(16, dtype=np.int32).view(np.uint8))
    with pytest.raises(EOFError):
        ours.read_rle_bit_packed_hybrid(
            ours.NumpyIO(np.array([3], dtype=np.uint8)), 8, 1, output
        )


def test_high_level_length_prefixed_round_trip():
    values = np.arange(113, dtype=np.int32) % 17
    payload = ours.encode_rle_bit_packed_hybrid(values, 5, with_length=True)
    decoded = ours.decode_rle_bit_packed_hybrid(
        payload, 5, values.size, length_prefix=True
    )
    assert np.array_equal(decoded, values)
    upstream_result = np.empty(values.size, dtype=np.int32)
    upstream.read_rle_bit_packed_hybrid(
        upstream.NumpyIO(np.frombuffer(payload[4:], dtype=np.uint8)),
        5,
        len(payload) - 4,
        upstream.NumpyIO(upstream_result.view(np.uint8)),
    )
    assert np.array_equal(upstream_result, values)
