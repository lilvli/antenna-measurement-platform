from __future__ import annotations


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xor 0."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def append_crc_be(data: bytes) -> bytes:
    return data + crc16_ccitt_false(data).to_bytes(2, "big")


def verify_crc_be(frame: bytes) -> bool:
    return len(frame) >= 2 and crc16_ccitt_false(frame[:-2]) == int.from_bytes(frame[-2:], "big")

