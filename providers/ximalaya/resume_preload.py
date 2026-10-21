# Copyright (c) 2026 ha_ximalaya contributors
"""Prepare bounded PCM during the existing local-resume handshake."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
import asyncio
import time
from contextlib import aclosing
from typing import TYPE_CHECKING

from music_assistant_models.errors import AudioError

from .book_pcm import BYTES_PER_SECOND, chapter_audio
from .playback_source import playback_identity

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant_models.streamdetails import StreamDetails

    from . import XimalayaProvider
    from .resume_audio import ResumeGate

PREPARE_TTL = 20
MAX_PREPARATIONS = 4
PCM_CHUNKS = 8  # Two seconds, plus at most one quarter-second producer chunk.


def timeline(stream: StreamDetails) -> tuple:
    """Compare actual chapter identities and durations, not just book IDs."""
    chapters = stream.data.get("chapters") if isinstance(stream.data, dict) else None
    return tuple((str(row["id"]), row["duration"]) for row in chapters.rows) if chapters else ()


class PreparedPCM:
    """One decoder whose bounded output can be handed to exactly one consumer."""

    def __init__(
        self, provider: XimalayaProvider, stream: StreamDetails, item: str, target: int
    ) -> None:
        """Start preparation without blocking the silent startup handshake."""
        self.provider = provider
        self.origin = stream
        self.item = item
        self.target = target
        self.identity = playback_identity(provider)
        self.timeline = timeline(stream)
        self.expires = time.monotonic() + PREPARE_TTL
        self.handoff = False
        self.closed = False
        self.chunks: asyncio.Queue[bytes | None] = asyncio.Queue(PCM_CHUNKS)
        self.error: Exception | None = None
        self.task = asyncio.create_task(self.produce())

    async def produce(self) -> None:
        """Keep the same decoder alive across handoff; never seek twice."""
        source = chapter_audio(self.provider, self.origin, self.target)
        try:
            async with aclosing(source):
                async for chunk in source:
                    for offset in range(0, len(chunk), BYTES_PER_SECOND // 4):
                        await self.chunks.put(chunk[offset : offset + BYTES_PER_SECOND // 4])
        except Exception:
            # Decoder/HTTP errors may contain private playback capabilities.
            self.error = AudioError("断点音频预备失败，已保留播放断点")
        finally:
            if not asyncio.current_task().cancelling():
                await self.chunks.put(None)

    async def close(self) -> None:
        """Close the official FFmpeg helper and release buffered PCM."""
        self.closed = True
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while not self.chunks.empty():
            self.chunks.get_nowait()
        self.chunks.put_nowait(None)

    async def audio(self) -> AsyncGenerator[bytes]:
        """Transfer prepared audio and then continue decoding with backpressure."""
        started = time.monotonic()
        first = True
        try:
            while (chunk := await self.chunks.get()) is not None:
                if playback_identity(self.provider) != self.identity:
                    raise AudioError("播放账号或音质已变化，请重新播放")
                if first:
                    self.provider.logger.info(
                        "本地断点预备音频命中，交接首段 PCM 耗时 %s ms",
                        round((time.monotonic() - started) * 1000),
                    )
                    first = False
                yield chunk
            if self.error:
                raise self.error
            if self.closed:
                raise AudioError("断点音频预备已取消")
        finally:
            await self.close()


class ResumePreloader:
    """Own speculative decoders without changing MA positions or buffers."""

    def __init__(self, provider: XimalayaProvider) -> None:
        """Bound preparation across all queues of this provider."""
        self.provider = provider
        self.pending: dict[str, PreparedPCM] = {}
        self.live: set[PreparedPCM] = set()
        self.cleanup: set[asyncio.Task] = set()

    def discard(self, queue_id: str, origin: StreamDetails | None = None) -> None:
        """Cancel unused preparation after a command, timeout or failed handoff."""
        prepared = self.pending.get(queue_id)
        if prepared and (origin is None or prepared.origin is origin):
            self.pending.pop(queue_id)
            self.provider.logger.debug(
                "释放未交接断点预备（已授权交接=%s，来源清理=%s）",
                prepared.handoff,
                origin is not None,
            )
            task = asyncio.create_task(prepared.close())
            self.cleanup.add(task)
            task.add_done_callback(self.cleanup.discard)
            task.add_done_callback(lambda _: self.live.discard(prepared))

    def prepare(self, stream: StreamDetails, gate: ResumeGate) -> None:
        """Only prepare the current foreground book and its checked bookmark."""
        self.discard(stream.queue_id)
        queue = self.provider.mass.player_queues.get(stream.queue_id)
        item = queue.current_item if queue else None
        busy = bool(queue and getattr(queue, "extra_attributes", {}).get("play_action_in_progress"))
        # prepare_gate already proved that this stream belongs to the queue.
        # MA can prepare it while current_item still points at the outgoing item.
        incoming = gate.item if busy else None
        if item and item.streamdetails is stream:
            incoming = item.queue_item_id
        if (
            not incoming
            or not isinstance(getattr(stream, "path", None), list)
            or len(self.live) >= MAX_PREPARATIONS
        ):
            return
        prepared = PreparedPCM(self.provider, stream, incoming, int(gate.start + gate.target))
        self.pending[stream.queue_id] = prepared
        self.live.add(prepared)
        self.provider.logger.info("开始预备本地断点音频（整书位置 %s 秒）", prepared.target)

    def allow_handoff(self, stream: StreamDetails) -> None:
        """Authorize handoff only before the existing observer's native resume seek."""
        if (prepared := self.pending.get(stream.queue_id)) and prepared.origin is stream:
            prepared.handoff = True
            self.provider.logger.debug("已授权本地断点音频交接")

    def take(self, stream: StreamDetails, position: int) -> PreparedPCM | None:
        """Match MA's actual decoder position; a different seek gets fresh audio."""
        prepared = self.pending.get(stream.queue_id)
        if prepared is None:
            return None
        queue = self.provider.mass.player_queues.get(stream.queue_id)
        item = queue.current_item if queue else None
        busy = bool(queue and getattr(queue, "extra_attributes", {}).get("play_action_in_progress"))
        if busy and (item is None or item.streamdetails is not stream):
            item = next(
                (
                    candidate
                    for candidate in self.provider.mass.player_queues.items(
                        queue.queue_id, limit=queue.items
                    )
                    if candidate.streamdetails is stream
                ),
                None,
            )
        if (
            prepared.handoff
            and prepared.expires > time.monotonic()
            and item
            and item.queue_item_id == prepared.item
            and item.streamdetails is stream
            and stream.item_id == prepared.origin.item_id
            and position == prepared.target
            and timeline(stream) == prepared.timeline
            # Native background seeks have no stable HTTP caller context.
            # Bookmarks/entitlements belong to the actual API account below.
            and playback_identity(self.provider) == prepared.identity
            and (busy or str(queue.state) != "paused")
        ):
            return self.pending.pop(stream.queue_id)
        self.provider.logger.debug(
            "断点预备交接不匹配（授权=%s，播放项=%s，音频对象=%s，位置=%s，目录=%s）",
            prepared.handoff,
            bool(item and item.queue_item_id == prepared.item),
            bool(item and item.streamdetails is stream),
            position == prepared.target,
            timeline(stream) == prepared.timeline,
        )
        self.discard(stream.queue_id)
        return None

    def prune(self, queue) -> None:
        """Use the existing observer tick to cancel obsolete preparations."""
        prepared = self.pending.get(queue.queue_id)
        if prepared is None:
            return
        item = queue.current_item
        busy = bool(getattr(queue, "extra_attributes", {}).get("play_action_in_progress"))
        if (
            time.monotonic() >= prepared.expires
            or playback_identity(self.provider) != prepared.identity
            or (
                not busy
                and (
                    not item
                    or item.queue_item_id != prepared.item
                    or str(queue.state) in ("idle", "paused")
                    or (item.streamdetails is not prepared.origin and not prepared.handoff)
                )
            )
        ):
            self.provider.logger.debug(
                "断点预备观察器清理（忙碌=%s，状态=%s，播放项=%s，来源=%s）",
                busy,
                str(queue.state),
                bool(item and item.queue_item_id == prepared.item),
                bool(item and item.streamdetails is prepared.origin),
            )
            self.discard(queue.queue_id)

    async def close(self) -> None:
        """Release both pending and adopted decoders on provider unload."""
        await asyncio.gather(*(prepared.close() for prepared in self.live))
        await asyncio.gather(*self.cleanup, return_exceptions=True)
        self.pending.clear()
        self.live.clear()
