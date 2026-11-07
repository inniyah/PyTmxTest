"""PNG RGBA de 16 bits por canal (PIL no los escribe). Lectura limitada a lo que escribimos."""

import struct
import zlib
from pathlib import Path

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_rgba16(path: Path, pixels: np.ndarray):
    """``pixels``: array (alto, ancho, 4) de uint16."""
    h, w, ch = pixels.shape
    assert ch == 4 and pixels.dtype == np.uint16
    rows = np.ascontiguousarray(pixels.astype(">u2")).reshape(h, w * 4)     # 4 valores por píxel
    raw = np.hstack([np.zeros((h, 1), np.uint8), rows.view(np.uint8).reshape(h, w * 8)])
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0)
    Path(path).write_bytes(_SIGNATURE + _chunk(b"IHDR", ihdr)
                           + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
                           + _chunk(b"IEND", b""))


def read_rgba16(path: Path) -> np.ndarray:
    """Lee un PNG escrito por ``write_rgba16`` (16 bits, RGBA, sin filtros)."""
    data = Path(path).read_bytes()
    assert data[:8] == _SIGNATURE
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", body[:10])
            assert depth == 16 and color == 6
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * 8)
    assert (raw[:, 0] == 0).all(), "solo se admiten filas sin filtro"
    return raw[:, 1:].copy().view(">u2").reshape(h, w, 4).astype(np.uint16)
