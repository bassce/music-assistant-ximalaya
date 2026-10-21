# Copyright (c) 2026 ha_ximalaya contributors
"""Reuse authorization only across range requests within one playback capability."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .protocol import PermissionDenied

if TYPE_CHECKING:
    from . import XimalayaProvider

RESOLUTION_TTL = 20


def playback_identity(provider: XimalayaProvider) -> tuple:
    """Invalidate transient work when its client, account or quality changes."""
    client = provider._client
    credentials = getattr(client, "credentials", {})
    return (
        id(client),
        credentials.get("account_id"),
        credentials.get("account_key"),
        getattr(client, "audio_quality", None),
    )


class PlaybackResolver:
    """Coalesce concurrent ranges; failures and later playbacks are never cached."""

    def __init__(self, provider: XimalayaProvider) -> None:
        """Scope the resolver to a single raw chapter-relay session."""
        self.provider = provider
        self.lock = asyncio.Lock()
        self.identity = playback_identity(provider)
        self.entries: dict[str, tuple[float, tuple[str, float]]] = {}

    async def resolve(self, ident: str) -> tuple[str, float]:
        """Resolve a short-lived media ticket without publishing it."""
        async with self.lock:
            identity = playback_identity(self.provider)
            if identity != self.identity:
                self.entries.clear()
                self.identity = identity
            now = time.monotonic()
            self.entries = {key: value for key, value in self.entries.items() if value[0] > now}
            if cached := self.entries.get(ident):
                return cached[1]
            result = await self.provider._client.resolve(ident)
            if identity != playback_identity(self.provider):
                raise PermissionDenied("Playback account changed during authorization")
            if len(self.entries) >= 4:
                self.entries.pop(next(iter(self.entries)))
            self.entries[ident] = (time.monotonic() + RESOLUTION_TTL, result)
            return result

    async def refresh(self, ident: str) -> None:
        """Require fresh authorization after a failed transfer or changed resource."""
        async with self.lock:
            self.entries.pop(ident, None)
