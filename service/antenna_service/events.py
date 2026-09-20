from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any


class EventBus:
    """Small in-process fan-out bus used by the REST API and WebSocket clients."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.sequence = 0

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self.sequence += 1
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
            "sequence": self.sequence,
        }
        for queue in tuple(self._subscribers):
            if queue.full():
                # A slow client must resynchronize rather than silently display a
                # heatmap with missing samples while its connection appears healthy.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "stream.resync", "sequence": self.sequence})
            queue.put_nowait(event)
        return event
