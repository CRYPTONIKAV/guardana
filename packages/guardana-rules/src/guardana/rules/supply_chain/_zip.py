"""Where a zip's central directory starts by its end records, read with bounded reads.

Python's `zipfile` finds the directory by where the end record sits and shifts every member
offset by any difference from the offset the record declares; the miniz reader inside
`torch.load` uses the declared offsets as they are. When the two differ, the readers see
different members under the same names, so a caller compares this with `ZipFile.start_dir`.
"""

import struct
import zipfile
from pathlib import Path
from typing import IO

_END = struct.Struct("<4s4H2LH")
_END_SIGNATURE = b"PK\x05\x06"
_LOCATOR = struct.Struct("<4sLQL")
_LOCATOR_SIGNATURE = b"PK\x06\x07"
_END64 = struct.Struct("<4sQ2H2L4Q")
_END64_SIGNATURE = b"PK\x06\x06"
_MAX_COMMENT = 0xFFFF
_UNSET32 = 0xFFFFFFFF
_SIZE, _OFFSET = 5, 6
_SIZE64, _OFFSET64 = 8, 9


def _read_at(handle: IO[bytes], offset: int, size: int) -> bytes:
    handle.seek(offset)
    return handle.read(size)


def _end_record(handle: IO[bytes], file_size: int) -> tuple[int, tuple[int, ...]] | None:
    """Find the end record where `zipfile` looks for it: last in the file, or before a comment."""
    if file_size < _END.size:
        return None
    tail = _read_at(handle, file_size - _END.size, _END.size)
    if tail.startswith(_END_SIGNATURE) and tail.endswith(b"\x00\x00"):
        return file_size - _END.size, _END.unpack(tail)
    window_start = max(file_size - _MAX_COMMENT - 1 - _END.size, 0)
    window = _read_at(handle, window_start, file_size - window_start)
    at = window.rfind(_END_SIGNATURE)
    if at < 0 or len(window) - at < _END.size:
        return None
    return window_start + at, _END.unpack(window[at : at + _END.size])


def _zip64_record(handle: IO[bytes], end_at: int) -> tuple[int, ...] | None:
    """Read the ZIP64 end record that a locator before the end record names, as `zipfile` does."""
    if end_at < _LOCATOR.size:
        return None
    locator = _read_at(handle, end_at - _LOCATOR.size, _LOCATOR.size)
    if not locator.startswith(_LOCATOR_SIGNATURE):
        return None
    declared_at = _LOCATOR.unpack(locator)[2]
    adjacent_at = end_at - _LOCATOR.size - _END64.size
    records = []
    for at in dict.fromkeys((declared_at, adjacent_at)):
        record = _read_at(handle, at, _END64.size) if at >= 0 else b""
        if len(record) == _END64.size and record.startswith(_END64_SIGNATURE):
            records.append(_END64.unpack(record))
    if not records:
        raise ValueError("ZIP64 locator names no ZIP64 end record")
    # Some `zipfile` releases read the record the locator names and others the one before
    # the locator, so two that declare different directories have no single reading.
    if len({(record[_SIZE64], record[_OFFSET64]) for record in records}) > 1:
        raise ValueError("ZIP64 end records declare different directories")
    return records[0]


def declared_directory_start(path: Path) -> int | None:
    """Return the directory offset the end records declare; None when they cannot be read.

    None also when a ZIP64 record and the plain end record declare different directories,
    since readers differ on which of the two they trust.
    """
    try:
        with path.open("rb") as handle:
            file_size = handle.seek(0, 2)
            found = _end_record(handle, file_size)
            if found is None:
                return None
            end_at, end = found
            end64 = _zip64_record(handle, end_at)
    except (OSError, ValueError, struct.error):
        return None
    if end64 is None:
        return end[_OFFSET]
    for plain, wide in ((end[_SIZE], end64[_SIZE64]), (end[_OFFSET], end64[_OFFSET64])):
        if plain not in (_UNSET32, wide):
            return None
    return end64[_OFFSET64]


_LOCAL = struct.Struct("<4s5H3L2H")
_LOCAL_SIGNATURE = b"PK\x03\x04"
_DATA_DESCRIPTOR_FLAG = 0x08
_UTF8_NAME_FLAG = 0x800


def member_disagreement(handle: IO[bytes], info: zipfile.ZipInfo) -> str | None:
    """Return how zip readers could read a member's bytes differently; None when they agree.

    `zipfile` reads a stored member's compressed size and the miniz reader in `torch.load`
    its size, and readers differ on which of the local header and the directory entry they
    trust, so each disagreement is named rather than read one way.
    """
    if info.compress_type == zipfile.ZIP_STORED and info.compress_size != info.file_size:
        return "is stored with a compressed size that differs from its size"
    header = _read_at(handle, info.header_offset, _LOCAL.size)
    if len(header) < _LOCAL.size or not header.startswith(_LOCAL_SIGNATURE):
        return "has no local header where its directory entry points"
    _sig, _version, flags, method, _time, _date, crc, packed, size, name_length, extra_length = (
        _LOCAL.unpack(header)
    )
    name = _read_at(handle, info.header_offset + _LOCAL.size, name_length)
    extra = _read_at(handle, info.header_offset + _LOCAL.size + name_length, extra_length)
    encoding = "utf-8" if info.flag_bits & _UTF8_NAME_FLAG else "cp437"
    if (
        name != info.orig_filename.encode(encoding)
        or flags != info.flag_bits
        or method != info.compress_type
    ):
        return "has a local header whose name, flags or method differ from its directory entry"
    if _repeats_a_field(info.extra) or _repeats_a_field(extra):
        return "repeats an extra field, whose copies zip readers choose between differently"
    pairs = ((crc, info.CRC), (packed, info.compress_size), (size, info.file_size))
    if not flags & _DATA_DESCRIPTOR_FLAG and any(
        local not in (central, _UNSET32) for local, central in pairs
    ):
        return "has a local header whose sizes or checksum differ from its directory entry"
    return None


_EXTRA_HEADER = struct.Struct("<HH")


def _repeats_a_field(extra: bytes) -> bool:
    """Whether an extra-field block holds two fields of one kind, such as two ZIP64 fields."""
    kinds: set[int] = set()
    while len(extra) >= _EXTRA_HEADER.size:
        kind, size = _EXTRA_HEADER.unpack_from(extra)
        if kind in kinds:
            return True
        kinds.add(kind)
        extra = extra[_EXTRA_HEADER.size + size :]
    return False
