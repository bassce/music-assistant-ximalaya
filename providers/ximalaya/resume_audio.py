# Copyright (c) 2026 ha_ximalaya contributors
"""Keep a chapter's opening audio out of MA while its local seek is pending."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
import asyncio
import math
import time
from bisect import bisect_right
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from music_assistant_models.errors import AudioError

from .book_pcm import BYTES_PER_SECOND, chapter_audio

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

    from . import XimalayaProvider
    from .local_chapter_resume import LocalChapterResume


@dataclass(frozen=True)
class ResumeGate:
    """A checked bookmark belongs only to this particular source stream."""

    track: str
    start: float
    target: float
    item: str | None = None


def gate_for(stream: StreamDetails) -> ResumeGate | None:
    """Return this stream's pending chapter-resume guard."""
    data = getattr(stream, "data", None)
    return data.get("resume_gate") if isinstance(data, dict) else None


async def prepare_gate(monitor: LocalChapterResume, stream: StreamDetails) -> ResumeGate | None:
    """Inspect the requested position before allowing any chapter audio through."""
    provider = monitor.provider
    queue = provider.mass.player_queues.get(stream.queue_id)
    if not queue or not isinstance(stream.data, dict):
        return None
    chapters = stream.data.get("chapters")
    if not chapters:
        return None
    item = next(
        (
            item
            for item in provider.mass.player_queues.items(queue.queue_id, limit=queue.items)
            if item.streamdetails is stream
        ),
        None,
    )
    # Background prefetch/analysis must not control the current playback.
    if not item or (item is not queue.current_item and not monitor.busy(queue)):
        return None
    requested = stream.seek_position
    index = max(0, bisect_right(chapters.starts, requested) - 1)
    if index + 1 < len(chapters.rows) and 0 < chapters.starts[index + 1] - requested <= 1:
        index += 1  # MA's explicit chapter selection subtracts/floors 500 ms.
    row = chapters.rows[index]
    start = chapters.starts[index]
    track = str(row["id"])
    settings = provider._skip_configs.get(stream.item_id)
    head, tail = (
        settings[1] if settings and provider.config.get_value("skip_sync", True) else (0, 0)
    )
    if head + tail >= row["duration"]:
        head, tail = 0, 0
    if requested - start > head + 5:
        return None  # An explicit seek within a chapter retains the requested position.
    previous = monitor.samples.get(queue.queue_id)
    if previous and previous.identity == (item.queue_item_id, track):
        return None  # Rewinding within the current chapter is intentional.
    skip = provider._skip_attempts.get(queue.queue_id)
    if (
        previous
        and previous.item == item.queue_item_id
        and previous.index + 1 == index
        and previous.offset >= previous.duration - previous.tail - 2
        and skip
        and skip[0] == (item.queue_item_id, math.ceil(start + head))
        and time.monotonic() - skip[1] < 5
    ):
        return None  # Our natural tail skipper must not resume the next chapter's old point.
    # Fail before audio on a lookup error; never replace a good bookmark with the
    # chapter opening just because its local store is temporarily unavailable.
    async with asyncio.timeout(5):
        await monitor.flush_track(track)
        record = (await provider._client.chapter_progress(track)).get("record")
    if not record or record.get("chapter_id") != track or record.get("fully_played"):
        return None
    position = record.get("position_ms")
    if type(position) is not int or not 2000 < position < row["duration"] * 1000:
        return None
    target = max(head, position / 1000)
    if target <= max(0, requested - start) + 1:
        return None
    gate = ResumeGate(track, start, target, item.queue_item_id)
    stream.data["resume_gate"] = gate
    return gate


async def silent_stream(preroll: int = 8) -> AsyncGenerator[bytes]:
    """Use exactly the same declared PCM format as the chapter decoder."""
    yield bytes(BYTES_PER_SECOND * preroll)
    while True:
        await asyncio.sleep(0.25)
        yield bytes(BYTES_PER_SECOND // 4)


async def book_audio(
    provider: XimalayaProvider, stream: StreamDetails, seek_position: int
) -> AsyncGenerator[bytes]:
    """Supply chapter PCM through MA's official custom-stream interface."""
    if isinstance(stream.data, dict) and (token := stream.data.get("relay_session")):
        provider._audio_cache.pcm_session(token, stream.queue_id).idle()
    monitor = getattr(provider, "_local_chapter_resume", None)
    try:
        gate = await prepare_gate(monitor, stream) if monitor else None
    except Exception:
        raise AudioError("无法读取本地分集断点，请稍后重试") from None
    if gate:
        if preloader := getattr(provider, "_resume_preloader", None):
            preloader.prepare(stream, gate)
        provider.logger.info(
            "章节 %s 本地续播至 %.1f 秒，定位期间输出静音", gate.track, gate.target
        )
        # MA may decode from zero and discard the first <=60 seconds internally.
        # Provide that bounded prefix promptly so its startup buffer can become ready.
        source = silent_stream(min(68, max(8, int(stream.seek_position - seek_position) + 8)))
        try:
            async with asyncio.timeout(25), aclosing(source):
                async for chunk in source:
                    yield chunk
        except TimeoutError:
            raise AudioError("本地分集断点定位超时，请重新播放") from None
        return
    preloader = getattr(provider, "_resume_preloader", None)
    prepared = preloader.take(stream, seek_position) if preloader else None
    source = prepared.audio() if prepared else chapter_audio(provider, stream, seek_position)
    try:
        async with aclosing(source):
            async for chunk in source:
                yield chunk
    finally:
        if prepared:
            preloader.live.discard(prepared)
