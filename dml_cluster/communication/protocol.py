"""JSON-lines protocol helpers shared by the leader and workers."""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

PROTOCOL_LIMIT = 64 * 1024 * 1024
MAX_BINARY_PAYLOAD = 512 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised when a peer sends invalid protocol data."""


async def send_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
    writer.write(payload + b"\n")
    await writer.drain()


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await reader.readline()
    if not line:
        raise EOFError("peer closed the connection")
    try:
        message = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON message: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("protocol message must be a JSON object")
    return message


async def send_binary(writer: asyncio.StreamWriter, payload: bytes) -> None:
    if len(payload) > MAX_BINARY_PAYLOAD:
        raise ProtocolError(f"binary payload too large: {len(payload)} bytes")
    writer.write(struct.pack(">Q", len(payload)))
    writer.write(payload)
    await writer.drain()


async def read_binary(
    reader: asyncio.StreamReader,
    expected_size: int | None = None,
) -> bytes:
    header = await reader.readexactly(8)
    size = struct.unpack(">Q", header)[0]
    if size > MAX_BINARY_PAYLOAD:
        raise ProtocolError(f"binary payload too large: {size} bytes")
    if expected_size is not None and size != expected_size:
        raise ProtocolError(f"expected {expected_size} binary bytes, got {size}")
    return await reader.readexactly(size)
