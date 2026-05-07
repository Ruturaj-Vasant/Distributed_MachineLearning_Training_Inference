from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from .protocol import send_message


async def close_writer(
    writer: asyncio.StreamWriter,
    final_message: dict[str, Any] | None = None,
) -> None:
    if writer.is_closing():
        return
    with contextlib.suppress(Exception):
        if final_message is not None:
            await send_message(writer, final_message)
    writer.close()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(writer.wait_closed(), timeout=2.0)


def format_peer(peer: object) -> str:
    if isinstance(peer, tuple) and len(peer) >= 2:
        return f"{peer[0]}:{peer[1]}"
    return str(peer)
