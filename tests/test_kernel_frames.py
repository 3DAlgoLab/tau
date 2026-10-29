"""Frame codec tests: the byte boundary between model code and harness state."""

from __future__ import annotations

import asyncio
from io import BytesIO

import pytest

from tau_coding.kernel.frames import (
    MAX_FRAME_BYTES,
    FrameError,
    FrameTooLargeError,
    decode_frame,
    encode_frame,
    read_frame,
    read_frame_sync,
    send_frame_sync,
    write_frame,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_frame_round_trips_through_the_codec() -> None:
    payload = {"op": "exec", "cell_id": "c1", "code": "print('한국어')"}
    assert decode_frame(encode_frame(payload)[4:]) == payload


def test_encoder_is_strict_about_non_data() -> None:
    """A payload the wire cannot express is an error, not a silent coercion."""
    with pytest.raises(FrameError, match="not JSON-serializable"):
        encode_frame({"op": "done", "ret": {1, 2}})


def test_encoder_refuses_an_oversized_frame() -> None:
    with pytest.raises(FrameTooLargeError):
        encode_frame({"op": "out", "text": "x" * (MAX_FRAME_BYTES + 10)})


def test_nan_is_refused_rather_than_written_as_bare_json() -> None:
    """``NaN`` parses in Python but is not valid JSON; the kernel must not invent it."""
    with pytest.raises(FrameError):
        encode_frame({"ms": float("nan")})


def test_sync_reader_returns_none_on_a_clean_end_of_stream() -> None:
    assert read_frame_sync(BytesIO(b"")) is None


def test_sync_reader_round_trips_two_back_to_back_frames() -> None:
    stream = BytesIO(encode_frame({"n": 1}) + encode_frame({"n": 2}))
    assert read_frame_sync(stream) == {"n": 1}
    assert read_frame_sync(stream) == {"n": 2}
    assert read_frame_sync(stream) is None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"\x00\x00", "truncated frame header"),
        (encode_frame({"n": 1})[:-2], "truncated frame body"),
    ],
)
def test_a_torn_frame_is_an_error_not_an_eof(body: bytes, message: str) -> None:
    """A half-written frame must never look like a graceful shutdown."""
    with pytest.raises(FrameError, match=message):
        read_frame_sync(BytesIO(body))


def test_sync_reader_refuses_an_oversized_announced_length() -> None:
    stream = BytesIO(b"\xff\xff\xff\xff" + b"x")
    with pytest.raises(FrameTooLargeError):
        read_frame_sync(stream)


def test_decode_rejects_a_non_object_payload() -> None:
    with pytest.raises(FrameError, match="must be an object"):
        decode_frame(b"[1, 2]")


def test_decode_rejects_invalid_json() -> None:
    with pytest.raises(FrameError, match="malformed frame"):
        decode_frame(b"{oops")


def test_send_frame_sync_writes_through_the_lock_step_codec() -> None:
    class _Sink:
        def __init__(self) -> None:
            self.written = bytearray()

        def sendall(self, data: bytes) -> int:
            self.written.extend(data)
            return len(data)

    sink = _Sink()
    send_frame_sync(sink, {"op": "ready"})
    assert decode_frame(bytes(sink.written[4:])) == {"op": "ready"}


async def test_async_reader_handles_frames_and_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame({"op": "out", "text": "hi"}))
    reader.feed_eof()
    assert await read_frame(reader) == {"op": "out", "text": "hi"}
    assert await read_frame(reader) is None


async def test_async_reader_reports_a_torn_body() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame({"op": "done"})[:-3])
    reader.feed_eof()
    with pytest.raises(FrameError, match="truncated frame body"):
        await read_frame(reader)


async def test_write_frame_surfaces_a_dead_peer() -> None:
    class _DeadWriter:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            raise ConnectionResetError("peer gone")

    with pytest.raises(FrameError, match="peer closed the channel"):
        await write_frame(_DeadWriter(), {"op": "exec"})  # type: ignore[arg-type]
