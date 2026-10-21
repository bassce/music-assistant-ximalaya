# Copyright (c) 2026 ha_ximalaya contributors
"""Seek each original chapter before decoding; join bounded PCM through the Provider API."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
import math
import time
from bisect import bisect_right
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING

from music_assistant_models.enums import ContentType
from music_assistant_models.errors import AudioError
from music_assistant_models.media_items import AudioFormat

from music_assistant.helpers.ffmpeg import get_ffmpeg_stream

from .playback_sessions import hold_session, stream_token

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

    from . import XimalayaProvider

# Each chapter has an independent decoder. Source codecs may change at chapter
# boundaries; the bytes delivered to MA always have this declared PCM format.
SAMPLE_RATE = 44100
FRAME_BYTES = 4
BYTES_PER_SECOND = SAMPLE_RATE * FRAME_BYTES
ROUNDING_SECONDS = 1.1


def pcm_format() -> AudioFormat:
    """Return a fresh descriptor; MA's FFmpeg helper may update input descriptors."""
    return AudioFormat(
        content_type=ContentType.PCM_S16LE, sample_rate=SAMPLE_RATE, bit_depth=16, channels=2
    )


async def chapter_audio(
    provider: XimalayaProvider, stream: StreamDetails, seek_position: int
) -> AsyncGenerator[bytes]:
    """Honor MA's requested absolute position without changing its clock or metadata."""
    with hold_session(provider, stream_token(stream)):
        async with aclosing(_chapter_audio(provider, stream, seek_position)) as source:
            async for chunk in source:
                yield chunk


async def _chapter_audio(
    provider: XimalayaProvider, stream: StreamDetails, seek_position: int
) -> AsyncGenerator[bytes]:
    """Decode the complete timeline under the caller's playback-session lease."""
    chapters = stream.data.get("chapters") if isinstance(stream.data, dict) else None
    if (
        chapters is None
        or not isinstance(stream.path, list)
        or len(stream.path) != len(chapters.rows)
        or not math.isfinite(seek_position)
        or not 0 <= seek_position <= chapters.duration
    ):
        raise AudioError("有声书音频时间轴无效")
    if seek_position == chapters.duration:
        return
    index = max(0, bisect_right(chapters.starts, seek_position) - 1)
    lead_in = 0
    if index + 1 < len(chapters.rows):
        gap = chapters.starts[index + 1] - seek_position
        if 0 < gap <= 1:
            # MA floors its chapter selection after subtracting 500 ms. Do not
            # decode the previous chapter's tail; preserve that second on the
            # original timeline with silence, only after the next source opens.
            lead_in = round(gap * SAMPLE_RATE) * FRAME_BYTES
            index += 1
    for number in range(index, len(chapters.rows)):
        row = chapters.rows[number]
        offset = max(0, seek_position - chapters.starts[number])
        seconds = row["duration"] - offset
        expected = round(seconds * SAMPLE_RATE) * FRAME_BYTES
        # The existing authenticated, range-capable relay remains responsible for
        # entitlement checks, expiring URLs, recovery and bounded read-ahead.
        # Signed upstream URLs never enter FFmpeg or MA's stream details.
        url = stream.path[number].path + "&raw=1"
        source = get_ffmpeg_stream(
            audio_input=url,
            input_format=AudioFormat(content_type=ContentType.M4A),
            output_format=pcm_format(),
            extra_input_args=["-ss", str(offset), "-rw_timeout", "15000000"],
            extra_output_args=["-t", str(seconds)],
            chunk_size=BYTES_PER_SECOND // 4,
        )
        received = 0
        started = time.monotonic()
        try:
            async with aclosing(source):
                async for chunk in source:
                    if not received:
                        provider.logger.info(
                            "章节 %s 从 %.3f 秒直接解码，首段 PCM 耗时 %s ms",
                            row["id"],
                            offset,
                            round((time.monotonic() - started) * 1000),
                        )
                        if lead_in:
                            yield bytes(lead_in)
                            lead_in = 0
                    received += len(chunk)
                    if received > expected:
                        raise AudioError("章节解码超出声明时间轴")
                    yield chunk
        except AudioError, OSError, TimeoutError:
            # Avoid propagating decoder stderr containing a playback capability.
            raise AudioError("章节音频读取或解码失败，已保留播放断点") from None
        missing = expected - received
        if not received or received % FRAME_BYTES or missing > ROUNDING_SECONDS * BYTES_PER_SECOND:
            # A truncated trial/network transfer must not become minutes of
            # silence or advance to the next chapter as if it were completed.
            raise AudioError("章节音频未完整到达，已保留播放断点")
        # Catalog durations are rounded to seconds. Only compensate that small
        # difference, keeping every subsequent chapter at its original start.
        if missing:
            yield bytes(missing)
