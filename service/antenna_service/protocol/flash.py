from __future__ import annotations

from antenna_service.protocol.crc import append_crc_be, verify_crc_be


PAGE_SIZE = 256
SHORT_FRAME_LENGTH = 22
LONG_FRAME_LENGTH = 286
COMMAND_TYPE = 0x0A
PAGE_WRITE_OPCODE = 0x32
PAGE_READ_OPCODE = 0x6B
MAX_ADDRESS = 0xFFFFFF
LAST_PAGE_ADDRESS = 0xFFFF00


def _validate_identity(tile_id: int, address: int) -> None:
    if not 0 <= tile_id <= 0xFF:
        raise ValueError("tile_id 必须在 0..255 范围内")
    if not 0 <= address <= LAST_PAGE_ADDRESS or address % PAGE_SIZE:
        raise ValueError("FLASH 页首地址必须在 0x000000..0xFFFF00 范围内并按 0x100 对齐")


def encode_page_write(tile_id: int, address: int, page: bytes) -> bytes:
    """Compile exactly one physical FLASH page write request.

    ``page`` must already contain the business-layer zero padding.  This transport
    encoder never guesses the effective payload length and never inserts metadata
    into the 256-byte page data area.
    """

    _validate_identity(tile_id, address)
    if len(page) != PAGE_SIZE:
        raise ValueError("FLASH 页写数据必须恰好为 256 字节")
    prefix = (
        b"\xAA\x55"
        + LONG_FRAME_LENGTH.to_bytes(2, "big")
        + bytes((COMMAND_TYPE, tile_id, 0, PAGE_WRITE_OPCODE))
        + address.to_bytes(3, "big")
    )
    return append_crc_be(prefix + page + bytes(17))


def encode_page_read(tile_id: int, address: int) -> bytes:
    """Compile one page read request; address is a byte address, not a page number."""

    _validate_identity(tile_id, address)
    prefix = (
        b"\xAA\x55"
        + SHORT_FRAME_LENGTH.to_bytes(2, "big")
        + bytes((COMMAND_TYPE, tile_id, 0, PAGE_READ_OPCODE))
        + address.to_bytes(3, "big")
    )
    return append_crc_be(prefix + bytes(9))


def response_has_identity(frame: bytes, opcode: int, tile_id: int, address: int) -> bool:
    """Return whether a received frame belongs to the outstanding FLASH request.

    CRC and reserved bytes are deliberately checked by ``decode_page_response``.
    Matching identity first lets the serial waiter reject an invalid correlated
    response immediately instead of accidentally consuming another command's frame.
    """

    expected_length = SHORT_FRAME_LENGTH if opcode == PAGE_WRITE_OPCODE else LONG_FRAME_LENGTH
    return (
        len(frame) == expected_length
        and frame[:2] == b"\xAA\x55"
        and int.from_bytes(frame[2:4], "big") == expected_length
        and frame[4] == COMMAND_TYPE
        and frame[5] == tile_id
        and frame[7] == opcode
        and int.from_bytes(frame[8:11], "big") == address
    )


def decode_page_response(frame: bytes, opcode: int, tile_id: int, address: int) -> bytes:
    """Validate one correlated response and return its page data for a read response."""

    _validate_identity(tile_id, address)
    if opcode not in {PAGE_WRITE_OPCODE, PAGE_READ_OPCODE}:
        raise ValueError("未知 FLASH 操作码")
    if not response_has_identity(frame, opcode, tile_id, address):
        raise ValueError("FLASH 应答长度、操作码、tile_id 或地址与请求不匹配")
    if not verify_crc_be(frame):
        raise ValueError("FLASH 应答 CRC 错误")
    if frame[6] != 0:
        raise ValueError("FLASH 应答保留字段非零")
    if opcode == PAGE_WRITE_OPCODE:
        if any(frame[11:20]):
            raise ValueError("FLASH 页写应答保留字段非零")
        return b""
    if any(frame[267:284]):
        raise ValueError("FLASH 页读应答保留字段非零")
    return frame[11:267]
