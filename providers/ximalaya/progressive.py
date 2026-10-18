"""Bounded fragmented MP4 output; preserve edit lists and the source timeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio_cache import CacheSession
    from .normalization import Entry, Format, Normalizer, Target

import asyncio
import struct
import time

from aiohttp import web

from .protocol import PreparationError

HEADER_LIMIT = 256 * 1024
CHUNK_BYTES = 65536
BUFFER_CHUNKS = 8
CONSUMER_TIMEOUT = 135


def boxes(
    data: bytes | bytearray, start: int = 0, end: int | None = None
) -> Iterator[tuple[int, int, bytes]]:
    """Walk complete boxes produced by our encoder, never arbitrary filesystem paths."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        if size < 8:
            raise ValueError("Invalid MP4 box")
        if pos + size > end:
            break
        yield pos, pos + size, kind
        pos += size


def header(data: bytes | bytearray, duration: float) -> tuple[bytes, float] | None:
    """Patch known duration before delivery; retain AAC priming edit information."""
    moov = next(((a, b) for a, b, kind in boxes(data) if kind == b"moov"), None)
    if moov is None:
        return None
    result = bytearray(data)
    movie_scale, audio_scale, delay = 0, 0, 0

    def visit(start: int, end: int) -> None:
        nonlocal movie_scale, audio_scale, delay
        for a, b, kind in boxes(result, start, end):
            body = a + 8
            if kind in (b"mvhd", b"mdhd"):
                version = result[body]
                if version not in (0, 1):
                    raise ValueError("Invalid MP4 version")
                offset = body + (20 if version else 12)
                scale = struct.unpack_from(">I", result, offset)[0]
                if not scale:
                    raise ValueError("Invalid MP4 timescale")
                struct.pack_into(
                    ">Q" if version else ">I", result, offset + 4, round(duration * scale)
                )
                if kind == b"mvhd":
                    movie_scale = scale
                else:
                    audio_scale = scale
            elif kind == b"tkhd":
                version = result[body]
                if not movie_scale or version not in (0, 1):
                    raise ValueError("Missing movie timescale")
                struct.pack_into(
                    ">Q" if version else ">I",
                    result,
                    body + (28 if version else 20),
                    round(duration * movie_scale),
                )
            elif kind == b"elst":
                version = result[body]
                count = struct.unpack_from(">I", result, body + 4)[0]
                if version not in (0, 1) or count != 1:
                    raise ValueError("Unsupported MP4 edit list")
                delay = struct.unpack_from(
                    ">q" if version else ">i", result, body + (16 if version else 12)
                )[0]
                if delay < 0:
                    raise ValueError("Invalid priming edit")
            elif kind in (b"trak", b"mdia", b"edts"):
                visit(body, b)

    visit(moov[0] + 8, moov[1])
    if not movie_scale or not audio_scale:
        raise ValueError("Missing MP4 timescale")
    return bytes(result), delay / audio_scale


def first_fragment_end(data: bytes | bytearray) -> int | None:
    """Return the end of the first complete audio fragment."""
    fragment = False
    for _, end, kind in boxes(data):
        if kind == b"moof":
            fragment = True
        elif kind == b"mdat" and fragment:
            return end
    return None


def fragment_end_ticks(data: bytes | bytearray) -> int:
    """Read the last coded sample timestamp from one encoder-owned moof."""
    end_time = 0
    for a, b, kind in boxes(data):
        if kind != b"moof":
            continue
        for start, end, container in boxes(data, a + 8, b):
            if container != b"traf":
                continue
            default, timestamp = 0, None
            runs = []
            for x, _y, name in boxes(data, start + 8, end):
                body = x + 8
                flags = int.from_bytes(data[body + 1 : body + 4])
                if name == b"tfhd":
                    offset = body + 8
                    if flags & 1:
                        offset += 8
                    if flags & 2:
                        offset += 4
                    if flags & 8:
                        default = struct.unpack_from(">I", data, offset)[0]
                elif name == b"tfdt":
                    timestamp = struct.unpack_from(">Q" if data[body] else ">I", data, body + 4)[0]
                elif name == b"trun":  # codespell:ignore trun
                    runs.append((body, flags))
            if timestamp is None:
                raise ValueError("Missing fragment timestamp")
            for body, flags in runs:
                count = struct.unpack_from(">I", data, body + 4)[0]
                offset = body + 8 + (4 if flags & 1 else 0) + (4 if flags & 4 else 0)
                for _ in range(count):
                    duration = (
                        struct.unpack_from(">I", data, offset)[0] if flags & 0x100 else default
                    )
                    if duration <= 0:
                        raise ValueError("Missing sample duration")
                    timestamp += duration
                    offset += sum(4 for flag in (0x100, 0x200, 0x400, 0x800) if flags & flag)
                if offset > len(data):
                    raise ValueError("Truncated fragment metadata")
            end_time = max(end_time, timestamp)
    return end_time


class GrowingAudio:
    """Stream one encoder through a bounded queue and discard consumed audio."""

    def __init__(
        self,
        normalizer: Normalizer,
        key: tuple[str, str],
        job: str,
        relay: CacheSession,
        audio: Format,
        target: Target,
        *,
        prefetch: bool,
    ) -> None:
        """Initialize per-instance state."""
        self.normalizer, self.key, self.job, self.relay = normalizer, key, job, relay
        self.audio, self.target = audio, target
        self.prefetch = prefetch
        self.size = 0
        self.readers = 0
        self.served = False
        self.failure_reported = False
        self.error: PreparationError | None = None
        self.cancelled = False
        self.ready = asyncio.Event()
        self.done = asyncio.Event()
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=BUFFER_CHUNKS)
        self.started = time.monotonic()
        self.task = asyncio.create_task(self._encode())

    async def start(self) -> None:
        """Prepare the client or stream for use."""
        await self.ready.wait()
        if self.error:
            raise self.error

    def cancel(self) -> None:
        """Cancel ongoing work without waiting for completion."""
        self.cancelled = True
        self.task.cancel()
        while not self.queue.empty():
            self.queue.get_nowait()

    async def close(self) -> None:
        """Cancel work and release buffered resources."""
        if not self.done.is_set():
            self.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()

    async def report_failure(self) -> None:
        """Preserve progress once when a consumed stream fails."""
        if self.error and self.served and not self.cancelled and not self.failure_reported:
            self.failure_reported = True
            await self.normalizer.provider._chapter_failed(
                self.key[0], self.key[1], str(self.error)
            )

    async def serve(
        self,
        request: web.Request,
        entry: Entry,
        progress: Callable[[int, float], None] | None = None,
    ) -> web.StreamResponse:
        """Send a forward-only stream; each subsequent read creates a new encoder."""
        if self.served:
            return await self.normalizer.serve(request, self.key, progress)
        if self.error:
            return web.Response(status=503, text="Audio preparation failed")
        response = web.StreamResponse(
            headers={
                "Content-Type": "audio/mp4",
                "Cache-Control": "no-store",
                "Accept-Ranges": "none",
            }
        )
        if request.method == "HEAD":
            await response.prepare(request)
            return response
        self.served = True
        self.prefetch = False
        entry.readers += 1
        self.readers += 1
        position = 0
        try:
            await response.prepare(request)
            while not self.done.is_set() or not self.queue.empty():
                if self.error:
                    raise ConnectionError("Audio encoder failed")
                try:
                    data = await asyncio.wait_for(self.queue.get(), 0.5)
                except TimeoutError:
                    if request.transport is None or request.transport.is_closing():
                        break
                    continue
                await response.write(data)
                position += len(data)
                entry.touched = time.monotonic()
                if progress:
                    total = (
                        self.size
                        if self.done.is_set()
                        else max(self.size, self.audio.duration * self.target.bitrate / 8)
                    )
                    progress(position, total)
            if self.error:
                raise ConnectionError("Audio encoder failed")
            await response.write_eof()
        except ConnectionError, RuntimeError:
            response.force_close()
            if request.transport:
                request.transport.close()
        finally:
            entry.readers -= 1
            self.readers -= 1
            await self.close()
            await self.report_failure()
        return response

    async def _put(self, data: bytes | bytearray) -> None:
        for offset in range(0, len(data), CHUNK_BYTES):
            async with asyncio.timeout(CONSUMER_TIMEOUT):
                await self.queue.put(bytes(data[offset : offset + CHUNK_BYTES]))
            self.size += min(CHUNK_BYTES, len(data) - offset)

    async def _encode(self) -> None:  # noqa: PLR0915 -- Keep encoder process ownership and shutdown in one scope.

        n = self.normalizer
        process = None
        try:
            args = n._encode_args(f"{n.base}/input/{self.job}", self.audio, self.target)
            creation = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *args,
                    "-movflags",
                    "+frag_keyframe+delay_moov+default_base_moof+skip_trailer",
                    "-frag_duration",
                    "1000000",
                    "-flush_packets",
                    "1",
                    "-f",
                    "mp4",
                    "pipe:1",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            )
            try:
                process = await asyncio.shield(creation)
            except asyncio.CancelledError:
                process = await creation
                raise
            assert process.stdout is not None
            pending = bytearray()
            priming = None
            last_tick = 0
            while True:
                try:
                    prefix = await process.stdout.readexactly(8)
                except asyncio.IncompleteReadError as err:
                    if err.partial:
                        raise PreparationError("Truncated audio fragment") from None
                    break
                size, kind = struct.unpack(">I4s", prefix)
                if size < 8 or (kind != b"mdat" and size > HEADER_LIMIT):
                    raise PreparationError("Invalid audio fragment size")
                if kind != b"mdat":
                    block = prefix + await process.stdout.readexactly(size - 8)
                    if kind == b"moof":
                        last_tick = fragment_end_ticks(block)
                    if priming is None:
                        pending.extend(block)
                        if len(pending) > HEADER_LIMIT:
                            raise PreparationError("Audio header exceeds buffer limit")
                        parsed = header(pending, self.audio.duration)
                        if parsed:
                            patched, priming = parsed
                            pending[:] = patched
                    elif not self.ready.is_set():
                        pending.extend(block)
                    else:
                        await self._put(block)
                    continue
                remaining = size - 8
                block = prefix
                while remaining:
                    chunk = await process.stdout.readexactly(min(remaining, CHUNK_BYTES))
                    remaining -= len(chunk)
                    block += chunk
                    if not self.ready.is_set():
                        pending.extend(block)
                        if len(pending) > HEADER_LIMIT:
                            raise PreparationError("First audio fragment exceeds buffer limit")
                    else:
                        await self._put(block)
                    block = b""
                if not self.ready.is_set():
                    if priming is None or not first_fragment_end(pending):
                        raise PreparationError("Missing complete first audio fragment")
                    actual = await n._probe(bytes(pending))
                    if not self.target.matches(actual):
                        raise PreparationError("Audio encoding verification failed")
                    await self._put(pending)
                    pending.clear()
                    self.ready.set()
            await process.wait()
            if process.returncode or not self.ready.is_set():
                raise PreparationError("Audio encoder did not finish successfully")
            if (
                abs(last_tick / self.target.rate - (priming or 0) - self.audio.duration)
                > 2048 / self.target.rate + 0.01
            ):
                raise PreparationError("Audio duration verification failed")
        except asyncio.CancelledError:
            self.error = PreparationError("Audio preparation cancelled")
            self.cancelled = True
        except Exception as err:
            self.error = (
                err
                if isinstance(err, PreparationError)
                else PreparationError(f"Audio preparation failed ({type(err).__name__})")
            )
        finally:
            if process and process.returncode is None:
                process.kill()
                await process.communicate()
            n.jobs.pop(self.job, None)
            self.relay.close()
            n.encoder.release()
            self.done.set()
            self.ready.set()
            await self.report_failure()
