"""Parquet primitive codecs exposed through a small C ABI."""

from std.ffi import external_call
from std.memory import bitcast
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime ObjPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def _varint_size(value: UInt64) -> Int:
    var x = value
    var n = 1
    while x > 127:
        x >>= 7
        n += 1
    return n


def _put_varint(dst: BPtr, pos: Int, capacity: Int, value: UInt64) -> Int:
    var x = value
    var p = pos
    while x > 127:
        if p >= capacity:
            return -1
        dst[p] = UInt8((x & 127) | 128)
        p += 1
        x >>= 7
    if p >= capacity:
        return -1
    dst[p] = UInt8(x)
    return p + 1


def _varint_length(src: BPtr, n: Int, pos: Int) -> Int:
    var p = pos
    var count = 0
    while p < n and count < 10:
        var byte = src[p]
        p += 1
        count += 1
        if (byte & 128) == 0:
            if count == 10 and byte > 1:
                return -1
            return count
    return -1


def _varint_value(src: BPtr, pos: Int, length: Int) -> UInt64:
    var value = UInt64(0)
    for i in range(length):
        value |= UInt64(src[pos + i] & 127) << UInt64(7 * i)
    return value


def _zigzag(value: UInt64) -> Int64:
    return Int64(value >> 1) ^ -Int64(value & 1)


def _mask(width: Int) -> UInt64:
    if width == 0:
        return UInt64(0)
    if width >= 64:
        return UInt64(0xffffffffffffffff)
    return (UInt64(1) << UInt64(width)) - 1


def _write_value(dst32: I32Ptr, dst8: BPtr, pos: Int, value: UInt64, itemsize: Int):
    if itemsize == 1:
        dst8[pos] = UInt8(value)
    else:
        dst32[pos] = Int32(value)


def _read_word(src: BPtr, pos: Int, available: Int) -> UInt64:
    var word = UInt64(0)
    var take = min(available, 8)
    for i in range(take):
        word |= UInt64(src[pos + i]) << UInt64(8 * i)
    return word


def _pack_values(values: I32Ptr, dst: BPtr, n: Int, width: Int):
    var bits = UInt64(0)
    var used = 0
    var pos = 0
    var mask = _mask(width)
    for i in range(n):
        bits |= (UInt64(values[i]) & mask) << UInt64(used)
        used += width
        while used >= 8:
            dst[pos] = UInt8(bits & 255)
            pos += 1
            bits >>= 8
            used -= 8
    if used != 0:
        dst[pos] = UInt8(bits & 255)


def _pack_width_10(values: I32Ptr, dst: BPtr, groups: Int):
    for group in range(groups):
        var base = group * 8
        var low = (
            UInt64(values[base])
            | (UInt64(values[base + 1]) << 10)
            | (UInt64(values[base + 2]) << 20)
            | (UInt64(values[base + 3]) << 30)
            | (UInt64(values[base + 4]) << 40)
            | (UInt64(values[base + 5]) << 50)
            | ((UInt64(values[base + 6]) & 15) << 60)
        )
        var high = UInt16(
            (UInt32(values[base + 6]) >> 4)
            | (UInt32(values[base + 7]) << 6)
        )
        (dst + group * 10).bitcast[UInt64]().store[alignment=1](0, low)
        (dst + group * 10 + 8).bitcast[UInt16]().store[alignment=1](0, high)


def _values_fit_width(values: I32Ptr, n: Int, width: Int) -> Bool:
    comptime W = simd_width_of[DType.int32]()
    var i = 0
    var limit = Int64(_mask(width))
    while i + W <= n:
        var chunk = values.load[width=W](i)
        if chunk.reduce_min()[0] < 0 or Int64(chunk.reduce_max()[0]) > limit:
            return False
        i += W
    while i < n:
        if values[i] < 0 or Int64(values[i]) > limit:
            return False
        i += 1
    return True


def _expand_bool_bytes(src: BPtr, dst: BPtr, nbytes: Int):
    comptime W = simd_width_of[DType.uint64]()
    comptime UNROLL = 4
    var i = 0
    while i + W * UNROLL <= nbytes:
        comptime for j in range(UNROLL):
            var offset = i + j * W
            var value = src.load[width=W](offset).cast[DType.uint64]()
            value = (value | (value << 28)) & 0x0000000f0000000f
            value = (value | (value << 14)) & 0x0003000300030003
            value = (value | (value << 7)) & 0x0101010101010101
            dst.store[alignment=1](
                offset * 8, bitcast[DType.uint8, W * 8](value)
            )
        i += W * UNROLL
    while i + W <= nbytes:
        var value = src.load[width=W](i).cast[DType.uint64]()
        value = (value | (value << 28)) & 0x0000000f0000000f
        value = (value | (value << 14)) & 0x0003000300030003
        value = (value | (value << 7)) & 0x0101010101010101
        dst.store[alignment=1](i * 8, bitcast[DType.uint8, W * 8](value))
        i += W
    while i < nbytes:
        var value = UInt64(src[i])
        value = (value | (value << 28)) & 0x0000000f0000000f
        value = (value | (value << 14)) & 0x0003000300030003
        value = (value | (value << 7)) & 0x0101010101010101
        dst.store[alignment=1](
            i * 8, bitcast[DType.uint8, 8](SIMD[DType.uint64, 1](value))
        )
        i += 1


@export("mfp_width_from_max_int")
def mfp_width_from_max_int(value: Int64) abi("C") -> Int:
    if value <= 0:
        return 0
    var x = UInt64(value)
    var width = 0
    while x != 0:
        width += 1
        x >>= 1
    return width


@export("mfp_encode_unsigned_varint")
def mfp_encode_unsigned_varint(value: UInt64, dst_addr: Int, capacity: Int) abi("C") -> Int:
    if capacity <= 0 or dst_addr == 0:
        return -1
    var dst = BPtr(unsafe_from_address=dst_addr)
    return _put_varint(dst, 0, capacity, value)


@export("mfp_read_unsigned_varint")
def mfp_read_unsigned_varint(src_addr: Int, n: Int, consumed_addr: Int) abi("C") -> UInt64:
    if consumed_addr == 0:
        return UInt64(0)
    var consumed = I64Ptr(unsafe_from_address=consumed_addr)
    if n <= 0 or src_addr == 0:
        consumed[0] = -1
        return UInt64(0)
    var src = BPtr(unsafe_from_address=src_addr)
    var length = _varint_length(src, n, 0)
    consumed[0] = Int64(length)
    if length < 0:
        return UInt64(0)
    return _varint_value(src, 0, length)


@export("mfp_encode_bitpacked")
def mfp_encode_bitpacked(
    values_addr: Int,
    n: Int,
    width: Int,
    dst_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if (
        n < 0
        or width < 0
        or width > 32
        or capacity <= 0
        or values_addr == 0
        or dst_addr == 0
    ):
        return -1
    var values = I32Ptr(unsafe_from_address=values_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var groups = (n + 7) // 8
    var pos = _put_varint(dst, 0, capacity, UInt64((groups << 1) | 1))
    if pos < 0:
        return -1
    var payload = (n * width + 7) // 8
    if pos + payload > capacity:
        return -1
    if not _values_fit_width(values, n, width):
        return -2
    var whole = n - (n % 8)
    var whole_groups = whole // 8
    if width == 10:
        _pack_width_10(values, dst + pos, whole_groups)
    else:
        _pack_values(values, dst + pos, whole, width)
    if whole < n:
        _pack_values(
            values + whole,
            dst + pos + whole_groups * width,
            n - whole,
            width,
        )
    return pos + payload


@export("mfp_decode_hybrid")
def mfp_decode_hybrid(
    src_addr: Int,
    nbytes: Int,
    width: Int,
    dst_addr: Int,
    dst_count: Int,
    itemsize: Int,
    consumed_addr: Int,
) abi("C") -> Int:
    if (
        nbytes < 0
        or width < 0
        or width > 32
        or dst_count < 0
        or consumed_addr == 0
    ):
        return -1
    if itemsize != 1 and itemsize != 4:
        return -1
    if nbytes == 0 or dst_count == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst8 = BPtr(unsafe_from_address=dst_addr)
    var dst32 = I32Ptr(unsafe_from_address=dst_addr)
    var consumed = I64Ptr(unsafe_from_address=consumed_addr)
    var pos = 0
    var produced = 0
    var mask = _mask(width)
    while pos < nbytes and produced < dst_count:
        var header_len = _varint_length(src, nbytes, pos)
        if header_len < 0:
            return -2
        var header = _varint_value(src, pos, header_len)
        pos += header_len
        if (header & 1) == 0:
            var count = Int(header >> 1)
            var byte_width = (width + 7) // 8
            if pos + byte_width > nbytes:
                return -2
            var value = UInt64(0)
            for j in range(byte_width):
                value |= UInt64(src[pos + j]) << UInt64(8 * j)
            pos += byte_width
            var take = min(count, dst_count - produced)
            for j in range(take):
                _write_value(dst32, dst8, produced + j, value, itemsize)
            produced += take
        else:
            var count = Int(header >> 1) * 8
            var payload = (count * width + 7) // 8
            var take = min(count, dst_count - produced)
            if pos + payload > nbytes:
                payload = (take * width + 7) // 8
                if pos + payload > nbytes:
                    return -2
            var packed = UInt64(0)
            var available = 0
            var cursor = 0
            for j in range(take):
                while available < width:
                    packed |= UInt64(src[pos + cursor]) << UInt64(available)
                    cursor += 1
                    available += 8
                _write_value(dst32, dst8, produced + j, packed & mask, itemsize)
                packed >>= UInt64(width)
                available -= width
            produced += take
            pos += payload
    consumed[0] = Int64(pos)
    return produced


@export("mfp_read_rle")
def mfp_read_rle(
    src_addr: Int,
    nbytes: Int,
    header: Int,
    width: Int,
    dst_addr: Int,
    dst_count: Int,
    itemsize: Int,
    consumed_addr: Int,
) abi("C") -> Int:
    if (
        nbytes < 0
        or header < 0
        or width < 0
        or width > 32
        or dst_count < 0
        or consumed_addr == 0
    ):
        return -1
    if itemsize != 1 and itemsize != 4:
        return -1
    if dst_count == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst8 = BPtr(unsafe_from_address=dst_addr)
    var dst32 = I32Ptr(unsafe_from_address=dst_addr)
    var consumed = I64Ptr(unsafe_from_address=consumed_addr)
    var byte_width = (width + 7) // 8
    if byte_width > nbytes:
        return -2
    var value = UInt64(0)
    for j in range(byte_width):
        value |= UInt64(src[j]) << UInt64(8 * j)
    var take = min(header >> 1, dst_count)
    for j in range(take):
        _write_value(dst32, dst8, j, value, itemsize)
    consumed[0] = Int64(byte_width)
    return take


@export("mfp_read_bitpacked")
def mfp_read_bitpacked(
    src_addr: Int,
    nbytes: Int,
    header: Int,
    width: Int,
    dst_addr: Int,
    dst_count: Int,
    itemsize: Int,
    consumed_addr: Int,
) abi("C") -> Int:
    if (
        nbytes < 0
        or header < 0
        or width < 0
        or width > 32
        or dst_count < 0
        or consumed_addr == 0
    ):
        return -1
    if itemsize != 1 and itemsize != 4:
        return -1
    if dst_count == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst8 = BPtr(unsafe_from_address=dst_addr)
    var dst32 = I32Ptr(unsafe_from_address=dst_addr)
    var consumed = I64Ptr(unsafe_from_address=consumed_addr)
    var count = (header >> 1) * 8
    var payload = (count * width + 7) // 8
    if payload > nbytes:
        return -2
    var take = min(count, dst_count)
    var mask = _mask(width)
    for j in range(take):
        var bit_pos = j * width
        var byte_pos = bit_pos // 8
        var shift = bit_pos & 7
        var word = _read_word(src, byte_pos, payload - byte_pos)
        _write_value(dst32, dst8, j, (word >> UInt64(shift)) & mask, itemsize)
    consumed[0] = Int64(payload)
    return take


@export("mfp_pack_bool")
def mfp_pack_bool(src_addr: Int, n: Int, dst_addr: Int, capacity: Int) abi("C") -> Int:
    if n < 0 or capacity < (n + 7) // 8:
        return -1
    if n == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var nbytes = (n + 7) // 8
    for i in range(nbytes):
        dst[i] = UInt8(0)
    for i in range(n):
        if src[i] != 0:
            dst[i // 8] |= UInt8(1 << (i & 7))
    return nbytes


@export("mfp_unpack_bool")
def mfp_unpack_bool(src_addr: Int, nbytes: Int, count: Int, dst_addr: Int) abi("C") -> Int:
    if count < 0 or nbytes < (count + 7) // 8:
        return -1
    if count == 0:
        return 0
    if src_addr == 0 or dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var whole = count // 8
    _expand_bool_bytes(src, dst, whole)
    for i in range(whole * 8, count):
        dst[i] = (src[i // 8] >> UInt8(i & 7)) & 1
    return count


@export("mfp_delta_binary_unpack")
def mfp_delta_binary_unpack(
    src_addr: Int,
    nbytes: Int,
    dst_addr: Int,
    dst_count: Int,
    longval: Int,
    consumed_addr: Int,
) abi("C") -> Int:
    if (
        nbytes <= 0
        or dst_count < 0
        or src_addr == 0
        or consumed_addr == 0
    ):
        return -1
    if dst_count > 0 and dst_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst32 = I32Ptr(unsafe_from_address=dst_addr)
    var dst64 = I64Ptr(unsafe_from_address=dst_addr)
    var consumed = I64Ptr(unsafe_from_address=consumed_addr)
    var pos = 0

    var size = _varint_length(src, nbytes, pos)
    if size < 0:
        return -2
    var block_size = Int(_varint_value(src, pos, size))
    pos += size
    size = _varint_length(src, nbytes, pos)
    if size < 0:
        return -2
    var miniblocks = Int(_varint_value(src, pos, size))
    pos += size
    size = _varint_length(src, nbytes, pos)
    if size < 0:
        return -2
    var total = Int(_varint_value(src, pos, size))
    pos += size
    size = _varint_length(src, nbytes, pos)
    if size < 0:
        return -2
    var value = _zigzag(_varint_value(src, pos, size))
    pos += size
    if block_size <= 0 or miniblocks <= 0 or block_size % miniblocks != 0:
        return -3
    var target = min(total, dst_count)
    var per_miniblock = block_size // miniblocks
    var produced = 0
    while produced < target:
        size = _varint_length(src, nbytes, pos)
        if size < 0:
            return -2
        var min_delta = _zigzag(_varint_value(src, pos, size))
        pos += size
        if pos + miniblocks > nbytes:
            return -2
        var widths_pos = pos
        pos += miniblocks
        for mini in range(miniblocks):
            var width = Int(src[widths_pos + mini])
            if width > 64:
                return -3
            var payload = (per_miniblock * width + 7) // 8
            if pos + payload > nbytes:
                return -2
            var take = min(per_miniblock, target - produced)
            if width <= 56:
                var packed = UInt64(0)
                var available = 0
                var cursor = 0
                for j in range(take):
                    while available < width:
                        packed |= UInt64(src[pos + cursor]) << UInt64(available)
                        cursor += 1
                        available += 8
                    var extra = packed & _mask(width)
                    packed >>= UInt64(width)
                    available -= width
                    if longval != 0:
                        dst64[produced] = value
                    else:
                        dst32[produced] = Int32(value)
                    value += min_delta + Int64(extra)
                    produced += 1
            else:
                for j in range(take):
                    var bit_pos = j * width
                    var byte_pos = bit_pos // 8
                    var shift = bit_pos & 7
                    var word = _read_word(src, pos + byte_pos, payload - byte_pos)
                    var extra = word >> UInt64(shift)
                    if shift != 0 and width > 64 - shift and byte_pos + 8 < payload:
                        extra |= UInt64(src[pos + byte_pos + 8]) << UInt64(64 - shift)
                    extra &= _mask(width)
                    if longval != 0:
                        dst64[produced] = value
                    else:
                        dst32[produced] = Int32(value)
                    value += min_delta + Int64(extra)
                    produced += 1
            pos += payload
            if produced >= target:
                consumed[0] = Int64(pos)
                return produced
    consumed[0] = Int64(pos)
    return produced


@export("mfp_byte_array_offsets")
def mfp_byte_array_offsets(
    src_addr: Int,
    nbytes: Int,
    count: Int,
    starts_addr: Int,
    lengths_addr: Int,
) abi("C") -> Int:
    if nbytes < 0 or count < 0:
        return -1
    if count == 0:
        return 0
    if src_addr == 0 or starts_addr == 0 or lengths_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var starts = I64Ptr(unsafe_from_address=starts_addr)
    var lengths = I32Ptr(unsafe_from_address=lengths_addr)
    var pos = 0
    for i in range(count):
        if pos + 4 > nbytes:
            return -2
        var length = (
            Int(src[pos])
            | (Int(src[pos + 1]) << 8)
            | (Int(src[pos + 2]) << 16)
            | (Int(src[pos + 3]) << 24)
        )
        pos += 4
        if (src[pos - 1] & 128) != 0 or pos + length > nbytes:
            return -2
        starts[i] = Int64(pos)
        lengths[i] = Int32(length)
        pos += length
    return pos


@export("mfp_unpack_byte_array")
def mfp_unpack_byte_array(
    src_addr: Int,
    nbytes: Int,
    count: Int,
    objects_addr: Int,
    utf: Int,
) abi("C") -> Int:
    if nbytes < 0 or count < 0:
        return -1
    if count == 0:
        return 0
    if src_addr == 0 or objects_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var objects = ObjPtr(unsafe_from_address=objects_addr)
    var pos = 0
    for i in range(count):
        if pos + 4 > nbytes:
            return -2
        var length = (
            Int(src[pos])
            | (Int(src[pos + 1]) << 8)
            | (Int(src[pos + 2]) << 16)
            | (Int(src[pos + 3]) << 24)
        )
        pos += 4
        if (src[pos - 1] & 128) != 0 or pos + length > nbytes:
            return -2
        var value: Int
        if utf != 0:
            value = external_call["PyUnicode_DecodeUTF8", Int](
                src_addr + pos, length, Int(0)
            )
        else:
            value = external_call["PyBytes_FromStringAndSize", Int](
                src_addr + pos, length
            )
        if value == 0:
            return -3
        var previous = objects[i]
        objects[i] = value
        external_call["Py_DecRef", NoneType](previous)
        pos += length
    return pos
