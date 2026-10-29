"""Length-prefixed JSON frames for host <-> kernel messaging.

The kernel is a separate Python process, so the boundary between "code the model
wrote" and "harness state" is a byte stream. Framing keeps that stream
unambiguous: a 4-byte big-endian length prefix followed by exactly that many
UTF-8 bytes of JSON. Message boundaries survive any interleaving of writes, and
the size ceiling is enforced in both directions so one runaway ``print`` cannot
make the peer allocate without bound.

Callables never travel over this channel -- only JSON. Kernel-side bindings are
therefore closures defined in :mod:`tau_coding.kernel.child` that ask the host
for a completion, rather than callables handed down from it.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Mapping
from typing import Any, BinaryIO, Protocol

MAX_FRAME_BYTES = 1024 * 1024
"""Hard ceiling for one frame, enforced by sender and receiver alike."""

_HEADER = struct.Struct(">I")
_HEADER_BYTES = _HEADER.size


class FrameError(RuntimeError):
    """Raised when a frame cannot be encoded, decoded, or fits no budget."""


class FrameTooLargeError(FrameError):
    """Raised when a frame would exceed :data:`MAX_FRAME_BYTES`."""


class SupportsSendAll(Protocol):
    """The one socket method the blocking sender needs."""

    def sendall(self, data: bytes) -> object: ...


def encode_frame(payload: Mapping[str, Any]) -> bytes:
    """Serialize ``payload`` into one framed message.

    Encoding is strict: a value that is not JSON-serializable is reported rather
    than quietly coerced, because the caller -- model code asking for a
    structured result -- needs to learn that its payload was not data.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FrameError(f"frame payload is not JSON-serializable: {exc}") from exc
    body = text.encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise FrameTooLargeError(f"frame is {len(body)} bytes, limit is {MAX_FRAME_BYTES}")
    return _HEADER.pack(len(body)) + body


def decode_frame(body: bytes) -> dict[str, Any]:
    """Decode one JSON frame body into a mapping."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"malformed frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrameError(f"frame payload must be an object, got {type(payload).__name__}")
    return payload


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one frame from an asyncio stream.

    Returns ``None`` on a clean end of stream. A half-read frame -- the peer died
    mid-message -- is an error, not an EOF, so a torn write can never be mistaken
    for a graceful shutdown.
    """
    try:
        header = await reader.readexactly(_HEADER_BYTES)
    except asyncio.IncompleteReadError as exc:
        if exc.partial:
            raise FrameError("truncated frame header") from exc
        return None
    length = _HEADER.unpack(header)[0]
    if length > MAX_FRAME_BYTES:
        raise FrameTooLargeError(f"incoming frame is {length} bytes, limit is {MAX_FRAME_BYTES}")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise FrameError("truncated frame body") from exc
    return decode_frame(body)


async def write_frame(writer: asyncio.StreamWriter, payload: Mapping[str, Any]) -> None:
    """Encode and drain one frame, surfacing a dead peer as :class:`FrameError`."""
    writer.write(encode_frame(payload))
    try:
        await writer.drain()
    except (ConnectionError, OSError) as exc:
        raise FrameError(f"peer closed the channel: {exc}") from exc


def read_frame_sync(stream: BinaryIO) -> dict[str, Any] | None:
    """Blocking counterpart of :func:`read_frame`, for the kernel child."""
    header = stream.read(_HEADER_BYTES)
    if not header:
        return None
    if len(header) < _HEADER_BYTES:
        raise FrameError("truncated frame header")
    length = _HEADER.unpack(header)[0]
    if length > MAX_FRAME_BYTES:
        raise FrameTooLargeError(f"incoming frame is {length} bytes, limit is {MAX_FRAME_BYTES}")
    body = stream.read(length)
    if len(body) < length:
        raise FrameError("truncated frame body")
    return decode_frame(body)


def send_frame_sync(sock: SupportsSendAll, payload: Mapping[str, Any]) -> None:
    """Blocking counterpart of :func:`write_frame`, for the kernel child."""
    sock.sendall(encode_frame(payload))


__all__ = [
    "FrameError",
    "FrameTooLargeError",
    "MAX_FRAME_BYTES",
    "SupportsSendAll",
    "decode_frame",
    "encode_frame",
    "read_frame",
    "read_frame_sync",
    "send_frame_sync",
    "write_frame",
]
