"""PLAIN and DELTA_BINARY_PACKED behavior compared with fastparquet."""

import numpy as np
import pytest

import mojo_fastparquet as mfp

fp_encoding = pytest.importorskip("fastparquet.encoding")
fp_cencoding = pytest.importorskip("fastparquet.cencoding")
fp_thrift = pytest.importorskip("fastparquet.parquet_thrift")


@pytest.mark.parametrize(
    ("type_name", "values"),
    [
        ("INT32", np.array([-10, 0, 20, 2**30], dtype=np.int32)),
        ("INT64", np.array([-2**50, 0, 2**50], dtype=np.int64)),
        ("FLOAT", np.array([-1.5, 0, np.inf, np.nan], dtype=np.float32)),
        ("DOUBLE", np.array([-1.5, 0, np.inf, np.nan], dtype=np.float64)),
        ("INT96", np.array([b"abcdefghijkl", b"mnopqrstuvwx"], dtype="S12")),
    ],
)
def test_plain_fixed_width_parity(type_name, values):
    type_code = getattr(fp_thrift.Type, type_name)
    encoded = mfp.encode_plain(values, getattr(mfp.Type, type_name))
    left = mfp.read_plain(encoded, type_code, len(values))
    right = fp_encoding.read_plain(encoded, type_code, len(values))
    assert left.dtype == right.dtype
    if values.dtype.kind == "f":
        assert np.array_equal(left, right, equal_nan=True)
    else:
        assert np.array_equal(left, right)


def test_plain_boolean_parity():
    values = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    encoded = mfp.encode_plain(values, mfp.Type.BOOLEAN)
    left = mfp.read_plain(encoded, mfp.Type.BOOLEAN, values.size)
    right = fp_encoding.read_plain(encoded, fp_thrift.Type.BOOLEAN, values.size)
    assert encoded == bytes([0x4D, 0x07])
    assert np.array_equal(left, right)
    assert np.array_equal(left, values)


@pytest.mark.parametrize("utf", [False, True])
def test_plain_byte_array_parity(utf):
    values = ["", "parquet", "λ", "a\x00b"] if utf else [b"", b"parquet", b"\xff", b"a\x00b"]
    encoded = mfp.pack_byte_array(values)
    left = mfp.unpack_byte_array(encoded, len(values), utf=utf)
    right = fp_encoding.unpack_byte_array(encoded, len(values), utf=utf)
    assert left.tolist() == right.tolist()
    assert mfp.read_plain(
        encoded, mfp.Type.BYTE_ARRAY, len(values), utf=utf
    ).tolist() == right.tolist()


def test_plain_fixed_len_byte_array_parity():
    values = np.array([b"abc", b"x", b"1234"], dtype="S4")
    encoded = mfp.encode_plain(values, mfp.Type.FIXED_LEN_BYTE_ARRAY, width=4)
    left = mfp.read_plain(encoded, mfp.Type.FIXED_LEN_BYTE_ARRAY, 3, width=4)
    right = fp_encoding.read_plain(
        encoded, fp_thrift.Type.FIXED_LEN_BYTE_ARRAY, 3, width=4
    )
    assert np.array_equal(left, right)


def test_byte_array_corruption_is_rejected():
    with pytest.raises(ValueError, match="corrupt"):
        mfp.unpack_byte_array(b"\x05\x00\x00\x00ab", 1)
    with pytest.raises(ValueError, match="corrupt"):
        mfp.unpack_byte_array(b"\xff\xff\xff\xff", 1)


def test_byte_array_empty_and_partial_corruption():
    result = mfp.unpack_byte_array(b"", 0)
    assert result.dtype == object
    assert result.size == 0
    with pytest.raises(ValueError, match="corrupt"):
        mfp.unpack_byte_array(
            b"\x01\x00\x00\x00a\x05\x00\x00\x00ab", 2
        )


def test_byte_array_invalid_utf8_parity():
    payload = mfp.pack_byte_array([b"\xff"])
    with pytest.raises(UnicodeDecodeError):
        mfp.unpack_byte_array(payload, 1, utf=True)


@pytest.mark.parametrize(
    "values",
    [
        np.array([5], dtype=np.int64),
        np.arange(1000, dtype=np.int64),
        np.full(300, -17, dtype=np.int64),
        np.cumsum(np.random.default_rng(0).integers(-100, 101, 777, dtype=np.int64)),
    ],
)
def test_delta_binary_packed_upstream_parity(values):
    payload = mfp.delta_binary_pack(values)
    ours = mfp.delta_binary_decode(payload, values.size)
    theirs = np.empty(values.size, dtype=np.int64)
    fp_cencoding.delta_binary_unpack(
        fp_cencoding.NumpyIO(np.frombuffer(payload, dtype=np.uint8)),
        fp_cencoding.NumpyIO(theirs.view(np.uint8)),
        longval=1,
    )
    assert np.array_equal(ours, values)
    assert np.array_equal(theirs, values)


def test_delta_binary_packed_int32_parity():
    values = np.cumsum(
        np.random.default_rng(2).integers(-20, 20, 513, dtype=np.int32),
        dtype=np.int32,
    )
    payload = mfp.delta_binary_pack(values)
    ours = mfp.delta_binary_decode(payload, values.size, dtype=np.int32)
    theirs = np.empty(values.size, dtype=np.int32)
    fp_cencoding.delta_binary_unpack(
        fp_cencoding.NumpyIO(np.frombuffer(payload, dtype=np.uint8)),
        fp_cencoding.NumpyIO(theirs.view(np.uint8)),
        longval=0,
    )
    assert np.array_equal(ours, theirs)
    assert np.array_equal(ours, values)


def test_delta_wide_miniblock_reference_vector():
    values = np.array([10, -5, 2**40, -(2**40)], dtype=np.int64)
    payload = mfp.delta_binary_pack(values)
    assert mfp.delta_binary_decode(payload, values.size).tolist() == values.tolist()


def test_delta_truncation_is_rejected():
    payload = mfp.delta_binary_pack(np.arange(200, dtype=np.int64))
    with pytest.raises(EOFError):
        mfp.delta_binary_decode(payload[:-1], 200)


def test_delta_pack_rejects_silent_int64_narrowing():
    with pytest.raises(OverflowError, match="outside int64"):
        mfp.delta_binary_pack(np.array([0, 2**63], dtype=np.uint64))
