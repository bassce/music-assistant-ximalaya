"""
Selective AAC normalization for MA's single-decoder audiobook concatenation.

Compatible chapters use the byte-exact relay. Incompatible chapters use an
encoder pipe with bounded memory and backpressure; no audio is written to disk.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import XimalayaProvider

import asyncio
import json
import math
import secrets
import time
from collections import deque
from dataclasses import dataclass
from weakref import WeakValueDictionary

import aiohttp
from aiohttp import web

from .audio_cache import CacheSession, Prefetched, ReadAhead
from .progressive import GrowingAudio
from .protocol import PermissionDenied, PreparationError
from .resource import RANGE, TIMEOUT, Resource
from .timing import correct_duration

IDLE_SECONDS = 135
PREPARE_SECONDS = 25
HEADER_BYTES = 65536


@dataclass(frozen=True)
class Format:
    """Audio properties verified by the decoder probe."""

    codec: str
    profile: str
    rate: int
    channels: int
    time_base: str
    config: str
    duration: float

    @classmethod
    def read(cls, data: dict[str, Any]) -> Format:
        """Validate and construct a resource from the supplied response."""
        try:
            stream = data["streams"][0]
            result = cls(
                stream["codec_name"],
                stream.get("profile", ""),
                int(stream["sample_rate"]),
                int(stream["channels"]),
                stream["time_base"],
                stream.get("extradata_hash", ""),
                float(data["format"]["duration"]),
            )
            if (
                not math.isfinite(result.duration)
                or result.duration <= 0
                or result.rate <= 0
                or result.channels not in (1, 2)
            ):
                raise ValueError
            if abs(float(stream.get("start_time", 0))) > 0.1:
                raise ValueError
            return result
        except KeyError, IndexError, TypeError, ValueError:
            raise PreparationError("无法确认章节编码和时间轴，已停止兼容处理") from None


@dataclass
class Target:
    """The common decoder format for chapters in one playback."""

    bitrate: int
    profile: str
    rate: int
    channels: int
    native_config: str = ""

    @classmethod
    def create(cls, source: Format, quality: str) -> Target:
        # Use the resource actually authorized/selected, not the requested tier.
        """Construct the requested representation from validated input."""
        bitrate = 96000 if "128" in quality else 48000 if "64" in quality else 24000
        profile = "HE-AACv2" if bitrate >= 48000 and source.channels == 2 else "HE-AAC"
        return cls(bitrate, profile, source.rate, source.channels)

    def matches(self, source: Format) -> bool:
        """Return whether the source matches the target decoder format."""
        return (
            source.codec == "aac"
            and source.profile == self.profile
            and source.rate == self.rate
            and source.channels == self.channels
            and source.time_base == f"1/{self.rate}"
        )

    def passthrough(self, source: Format) -> bool:
        """Return whether this source can share the existing decoder."""
        if not self.matches(source) or not source.config:
            return False
        # ASC bytes can differ between upstream and libfdk even for compatible
        # SBR/PS. Compare native chapters against their native baseline; verify
        # our encoder separately by decoded profile/rate/channels/time base.
        if not self.native_config:
            self.native_config = source.config
        return self.native_config == source.config


@dataclass
class Source:
    """An authorized resource, scoped to one bounded preparation operation."""

    url: str
    duration: float
    quality: str
    resource: Resource
    header: bytes
    content_type: str


@dataclass
class Entry:
    """Source metadata and any currently unconsumed encoder output."""

    source: Resource
    quality: str
    audio: Format
    output_id: str | None = None
    url: str | None = None
    touched: float = 0
    readers: int = 0
    progressive: GrowingAudio | None = None


class Normalizer:
    """Isolated by Provider instance, playback capability and source identity."""

    def __init__(self, provider: XimalayaProvider) -> None:
        """Initialize per-instance state."""
        self.provider = provider
        self.targets: dict[str, Target] = {}
        self.entries: dict[tuple[str, str], Entry] = {}
        self.locks: WeakValueDictionary[tuple[str, str], asyncio.Lock] = WeakValueDictionary()
        self.encoder = asyncio.Semaphore(5)
        self.live: dict[str, GrowingAudio] = {}
        self.worker = asyncio.Semaphore(1)
        self.background: dict[asyncio.Task[tuple[str, float]], tuple[str, str]] = {}
        self.foreground_count = 0
        self.foreground_idle = asyncio.Event()
        self.foreground_idle.set()
        self.tasks: set[asyncio.Task[tuple[str, float]]] = set()
        self.jobs: dict[str, tuple[str, Source, CacheSession]] = {}
        self.outputs: dict[str, tuple[str, str]] = {}
        self.runner: web.AppRunner | None = None
        self.closed = False
        self.encoder_checked = False
        self.timer: asyncio.TimerHandle | None = None

    async def resolve(self, token: str, ident: str, *, prefetch: bool = False) -> tuple[str, float]:
        """Authorize and resolve the requested audio resource."""
        if self.closed:
            raise PreparationError("音频兼容处理已关闭")
        key = (token, ident)
        if not prefetch:
            for live in tuple(self.live.values()):
                if live.key == key:
                    live.prefetch = False
                elif live.prefetch and not live.done.is_set():
                    live.cancel()
            self.foreground_count += 1
            self.foreground_idle.clear()
            for background, background_key in tuple(self.background.items()):
                if background_key != key:
                    background.cancel()
                else:
                    # Let already-running work for this exact chapter finish.
                    self.background.pop(background, None)

        async def prepare() -> tuple[str, float]:
            if prefetch:
                await self.foreground_idle.wait()
            return await self._resolve(token, ident, prefetch=prefetch)

        task = asyncio.create_task(prepare())
        if prefetch:
            self.background[task] = key
        self.tasks.add(task)
        deadline = asyncio.timeout(PREPARE_SECONDS)
        try:
            async with deadline:
                return await task
        except PermissionDenied:
            raise
        except PreparationError:
            raise
        except TimeoutError:
            if not deadline.expired():
                raise  # A CDN timeout stays eligible for the relay's recovery policy.
            raise PreparationError("章节编码检测或转换超时，已保留播放断点") from None
        except OSError as err:
            if isinstance(err, aiohttp.ClientError):
                raise
            raise PreparationError("章节编码检测或转换失败，已保留播放断点") from None
        finally:
            self.tasks.discard(task)
            self.background.pop(task, None)
            if not prefetch:
                self.foreground_count -= 1
                if not self.foreground_count:
                    self.foreground_idle.set()

    async def serve(
        self,
        request: web.Request,
        key: tuple[str, str],
        progress: Callable[[int, float], None] | None = None,
    ) -> web.StreamResponse:
        """Authorize each read and start a new pipe after consumed output is discarded."""
        await self.resolve(*key)
        entry = self.entries[key]
        if not entry.progressive:
            return web.Response(status=404)
        return await entry.progressive.serve(request, entry, progress)

    def prune(self) -> None:
        """Discard expired playback state and unused encoder output."""
        self.timer = None
        if self.closed:
            return
        now = time.monotonic()
        sessions = self.provider._part_sessions
        for key, entry in list(self.entries.items()):
            expired = key[0] not in sessions or sessions[key[0]][0] < now
            if not entry.readers and (expired or now - entry.touched > IDLE_SECONDS):
                self._discard(entry)
                if expired:
                    self.entries.pop(key, None)
                    self.outputs.pop(entry.output_id or "", None)
        # Metadata only; audio is removed above. Avoid retaining entire catalogs.
        for key, entry in sorted(self.entries.items(), key=lambda x: x[1].touched):
            if len(self.entries) <= 512:
                break
            if not entry.readers and not entry.progressive:
                self.entries.pop(key, None)
                self.outputs.pop(entry.output_id or "", None)
        self.targets = {
            k: v for k, v in self.targets.items() if k in sessions and sessions[k][0] > now
        }
        self.live = {k: v for k, v in self.live.items() if not v.done.is_set()}
        self.timer = asyncio.get_running_loop().call_later(30, self.prune)

    async def close(self) -> None:
        """Cancel work and release buffered resources."""
        self.closed = True
        if self.timer:
            self.timer.cancel()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(live.close() for live in self.live.values()))
        self.live.clear()
        if self.runner:
            await self.runner.cleanup()
        self.entries.clear()
        self.targets.clear()
        self.outputs.clear()
        self.jobs.clear()

    async def _source(self, ident: str, *, header: bool = True) -> Source:
        # Retrying belongs to ReadAhead, which knows the remaining buffer and
        # recovery budget. Do not turn transient CDN failures into codec errors.
        url, duration, quality = await self.provider._client.resolve_source(ident)
        wanted = f"bytes=0-{HEADER_BYTES - 1}" if header else "bytes=0-0"
        assert self.provider._session is not None
        async with self.provider._session.get(
            url,
            headers={"Range": wanted, "Accept-Encoding": "identity"},
            timeout=TIMEOUT,
            auto_decompress=False,
            allow_redirects=False,
        ) as response:
            resource = Resource.read(response, url, wanted)
            data = await response.read()
            if len(data) != resource.end + 1:
                raise aiohttp.ClientPayloadError("Incomplete audio header")
            content_type = response.headers.get("Content-Type", "audio/mp4")
        return Source(url, duration, quality, resource, data if header else b"", content_type)

    async def _resolve(  # noqa: PLR0915 -- Publish or clean up the encoder and relay atomically.
        self, token: str, ident: str, *, prefetch: bool = False
    ) -> tuple[str, float]:

        started = time.monotonic()
        phases = {}

        async def measured[T](name: str, operation: Awaitable[T]) -> T:
            before = time.monotonic()
            try:
                return await operation
            finally:
                phases[name] = round((time.monotonic() - before) * 1000)

        key = (token, ident)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            phases["章节等待"] = round((time.monotonic() - started) * 1000)
            # Each external use still checks rights. Internal reads within this
            # preparation reuse the checked address, refreshing only on recovery.
            entry = self.entries.get(key)
            source = await measured("授权及文件头", self._source(ident, header=entry is None))
            url, quality, resource = source.url, source.quality, source.resource
            if entry:
                if quality != entry.quality or not entry.source.same_file(resource):
                    raise PreparationError("章节源音频已变化，请重新播放以建立新的音频会话")
                entry.touched = time.monotonic()
                if entry.output_id is None or (
                    entry.progressive
                    and not entry.progressive.served
                    and not entry.progressive.error
                ):
                    await correct_duration(self.provider, token, ident, entry.audio.duration)
                    self.provider.logger.debug(
                        "章节 %s 复用准备耗时 %s ms（%s）",
                        ident,
                        round((time.monotonic() - started) * 1000),
                        ", ".join(f"{name}={ms} ms" for name, ms in phases.items()),
                    )
                    return entry.url or url, entry.audio.duration
            waiting = time.monotonic()
            async with self.worker:
                phases["处理等待"] = round((time.monotonic() - waiting) * 1000)
                await self._start()
                job = secrets.token_urlsafe(24)
                relay = CacheSession(self.provider)
                refresh_lock = asyncio.Lock()

                async def resolve_input(track_id: str) -> tuple[str, float]:
                    assert track_id == ident
                    return source.url, source.duration

                async def refresh_input(track_id: str) -> None:
                    nonlocal source
                    assert track_id == ident
                    async with refresh_lock:
                        renewed = await self._source(ident, header=False)
                        if renewed.quality != quality or not resource.same_file(renewed.resource):
                            raise PreparationError("章节源音频已变化，停止续传")
                        source.url = renewed.url

                relay.resolve = resolve_input
                relay.refresh_source = refresh_input
                self.jobs[job] = (ident, source, relay)
                source_url = f"{self.base}/input/{job}"
                live = None
                published = False
                try:
                    audio = (
                        entry.audio
                        if entry
                        else await measured("编码检测", self._probe(source_url))
                    )
                    target = self.targets.setdefault(token, Target.create(audio, quality))
                    result = Entry(resource, quality, audio, touched=time.monotonic())
                    if not target.passthrough(audio):
                        await self._check_encoder(audio, target)
                        await self.encoder.acquire()
                        live = GrowingAudio(self, key, job, relay, audio, target, prefetch=prefetch)
                        self.live[job] = live
                        await measured("首段转码", live.start())
                        result.progressive = live
                        result.output_id = (
                            entry.output_id if entry else None
                        ) or secrets.token_urlsafe(24)
                        result.url = f"{self.base}/output/{result.output_id}"
                        self.outputs[result.output_id] = key
                        self.provider.logger.info(
                            "章节 %s 按需转码：%s → %s / %s kbps / %s Hz / %s 声道",
                            ident,
                            audio.profile,
                            target.profile,
                            target.bitrate // 1000,
                            target.rate,
                            target.channels,
                        )
                    else:
                        self.provider.logger.debug("章节 %s 编码兼容，保留原音频", ident)
                    self.entries[key] = result
                    await measured(
                        "时长检查", correct_duration(self.provider, token, ident, audio.duration)
                    )
                    published = True
                    return result.url or source.url, audio.duration
                finally:
                    self.provider.logger.info(
                        "章节 %s %s准备耗时 %s ms（%s）",
                        ident,
                        "预读" if prefetch else "起播",
                        round((time.monotonic() - started) * 1000),
                        ", ".join(f"{name}={ms} ms" for name, ms in phases.items()),
                    )
                    if live:
                        if not published:
                            await live.close()
                    else:
                        self.jobs.pop(job, None)
                        relay.close()

    async def _run(self, *args: str, input_data: bytes | None = None) -> bytes:
        # No signed CDN URLs or account credentials are passed to subprocesses.
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE
                if input_data is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            process = await creation
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise
        try:
            stdout, _ = await process.communicate(input_data)
            if process.returncode:
                raise PreparationError("音频编码工具未能完成兼容处理，已保留播放断点")
            return stdout
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()

    async def _probe(self, url: str | bytes) -> Format:
        raw = await self._run(
            "ffprobe",
            "-v",
            "error",
            "-rw_timeout",
            "15000000",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,profile,sample_rate,channels,time_base,extradata_hash,start_time:format=duration",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            "pipe:0" if isinstance(url, bytes) else str(url),
            input_data=url if isinstance(url, bytes) else None,
        )
        try:
            return Format.read(json.loads(raw))
        except ValueError, TypeError:
            raise PreparationError("无法读取章节编码参数") from None

    async def _check_encoder(self, audio: Format, target: Target) -> None:
        if not self.encoder_checked:
            encoders = await self._run("ffmpeg", "-hide_banner", "-encoders")
            if b"libfdk_aac" not in encoders:
                raise PreparationError("当前 FFmpeg 缺少 HE-AAC 编码器，已保留播放断点")
            self.encoder_checked = True

    @staticmethod
    def _encode_args(url: str, audio: Format, target: Target) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-xerror",
            "-filter_threads",
            "1",
            "-rw_timeout",
            "15000000",
            "-threads",
            "1",
            "-i",
            url,
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-vn",
            "-sn",
            "-dn",
            "-c:a",
            "libfdk_aac",
            "-threads",
            "1",
            "-profile:a",
            "aac_he_v2" if target.profile == "HE-AACv2" else "aac_he",
            "-b:a",
            str(target.bitrate),
            "-ar",
            str(target.rate),
            "-ac",
            str(target.channels),
            "-t",
            str(audio.duration),
        ]

    async def _start(self) -> None:
        if self.runner:
            return
        app = web.Application()
        app.router.add_get("/input/{key}", self._input)
        app.router.add_get("/output/{key}", self._output)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=2)
        try:
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self.base = f"http://127.0.0.1:{runner.addresses[0][1]}"
            self.runner = runner
        finally:
            if self.runner is None:
                await runner.cleanup()
        self.timer = asyncio.get_running_loop().call_later(30, self.prune)

    async def _input(self, request: web.Request) -> web.StreamResponse:
        """Private decoder input: the same resumable relay, pinned to one version."""
        job = self.jobs.get(request.match_info["key"])
        if not job:
            return web.Response(status=404)
        ident, source, session = job
        wanted = request.headers.get("Range")
        if wanted and not RANGE.fullmatch(wanted):
            return web.Response(status=416)
        seed = None
        match = RANGE.fullmatch(wanted or "bytes=0-")
        if source.header and match and match[1] and int(match[1]) < len(source.header):
            await session.budget.reserve(len(source.header))
            seed = Prefetched(
                ident, source.resource, source.duration, source.content_type, deque([source.header])
            )
        buffer = ReadAhead(self.provider, session, ident, wanted, seed, expected=source.resource)
        try:
            await buffer.ready.wait()
            resource = buffer.resource
            if not resource:
                return web.Response(status=buffer.status)
            headers = {
                "Content-Type": buffer.content_type,
                "Accept-Ranges": "bytes",
                "Content-Length": str(resource.end - resource.start + 1),
            }
            if wanted:
                headers["Content-Range"] = f"bytes {resource.start}-{resource.end}/{resource.total}"
            response = web.StreamResponse(status=206 if wanted else 200, headers=headers)
            await response.prepare(request)
            if request.method == "HEAD":
                return response
            while data := await buffer.read():
                await response.write(data)
            if buffer.error:
                response.force_close()
                if request.transport:
                    request.transport.close()
            return response
        except ConnectionError, RuntimeError:
            return web.Response(status=499)
        finally:
            await buffer.close()

    async def _output(self, request: web.Request) -> web.StreamResponse:
        key = self.outputs.get(request.match_info["key"])
        if key is None:
            return web.Response(status=404)
        return await self.serve(request, key)

    def _discard(self, entry: Entry) -> None:
        if entry.progressive and not entry.readers:
            entry.progressive.cancel()
            entry.progressive = None
