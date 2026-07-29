"""Mojo-accelerated Parquet primitive encoding and decoding."""

from . import cencoding, encoding, parquet_thrift
from .cencoding import (
    NumpyIO,
    decode_rle_bit_packed_hybrid,
    encode_rle_bit_packed_hybrid,
    width_from_max_int,
)
from .encoding import (
    delta_binary_decode,
    delta_binary_pack,
    encode_plain,
    pack_byte_array,
    read_plain,
    read_plain_boolean,
    unpack_byte_array,
)
from .parquet_thrift import Type

__all__ = [
    "NumpyIO",
    "Type",
    "cencoding",
    "decode_rle_bit_packed_hybrid",
    "delta_binary_decode",
    "delta_binary_pack",
    "encode_plain",
    "encode_rle_bit_packed_hybrid",
    "encoding",
    "pack_byte_array",
    "parquet_thrift",
    "read_plain",
    "read_plain_boolean",
    "unpack_byte_array",
    "width_from_max_int",
]

__version__ = "0.1.0"
