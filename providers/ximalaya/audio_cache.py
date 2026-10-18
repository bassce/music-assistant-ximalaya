"""Per-playback RAM read-ahead; never persist audio, credentials or signed URLs."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import XimalayaProvider

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import aiohttp

from .protocol import CarError, PermissionDenied, PreparationError
from .resource import ATTEMPTS, RANGE, RETRY_DELAYS, TIMEOUT, RelayError, Resource

READ_AHEAD = 120
MAX_BYTES = 4 * 1024 * 1024
CHUNK = 65536
IDLE_TTL = READ_AHEAD + 15
RECOVERY_SECONDS = 120
MAX_RECOVERY_ATTEMPTS = 8
RECOVERY_DELAYS = (5, 10, 15, 20)


class CacheBudget:
    """A shared cap covering the current response and the following episode."""

    def __init__(self) -> None:
        """Initialize per-instance state."""
        self.used = 0
        self.peak = 0
        self.changed = asyncio.Event()

    async def reserve(self, size: int) -> None:
        """Wait for enough buffer capacity and reserve it."""
        while self.used + size > MAX_BYTES:
            self.changed.clear()
            await self.changed.wait()
        self.used += size
        self.peak = max(self.peak, self.used)

    def release(self, size: int) -> None:
        """Release previously reserved buffer capacity."""
        self.used -= size
        assert self.used >= 0
        self.changed.set()


@dataclass
class Prefetched:
    """A bounded unread byte range belonging to one playback session."""

    ident: str
    resource: Resource
    duration: float
    content_type: str
    chunks: deque[bytes]
    start: int = 0

    @property
    def size(self) -> int:
        """Return the number of unread bytes."""
        return sum(len(chunk) for chunk in self.chunks)


class CacheSession:
    """One queue (or unbound playback capability), isolated within its provider."""

    def __init__(self, provider: XimalayaProvider) -> None:
        """Initialize per-instance state."""
        self.provider = provider
        self.resolve: Callable[[str], Awaitable[tuple[str, float]]] = provider._client.resolve
        self.resolve_prefetch: Callable[[str], Awaitable[tuple[str, float]]] | None = None
        self.refresh_source: Callable[[str], Awaitable[None]] | None = None
        self.budget = CacheBudget()
        self.seed: Prefetched | None = None
        self.warmer: asyncio.Task[None] | None = None
        self.warming_ident: str | None = None
        self.target: tuple[str, str] | None = None
        self.target_seconds: float = 0
        self.desired: tuple[str, str, float] | None = None
        self.seen_playing = False
        self.active: set[ReadAhead] = set()
        self.expiry: asyncio.TimerHandle | None = None
        self.closed = False

    def is_closed(self) -> bool:
        """Read state that may change while a network operation is awaited."""
        return self.closed

    def clear_prefetch(self) -> None:
        """Cancel speculative work and release its unread bytes."""
        if self.warmer:
            self.warmer.cancel()
            self.warmer = None
        if self.seed:
            self.budget.release(self.seed.size)
            self.seed = None
        self.target = None
        self.warming_ident = None
        self.target_seconds = 0
        self.desired = None

    def take(self, ident: str, requested: str | None) -> Prefetched | None:
        """Consume a matching prefetched range, if available."""
        if self.expiry:
            self.expiry.cancel()
            self.expiry = None
        seed = self.seed
        # A seed is consumable only by a request that actually starts within it.
        # Probe/suffix requests use their own transfer and leave the seed intact.
        if seed and seed.ident == ident:
            match = RANGE.fullmatch(requested or "bytes=0-")
            if match and match[1] and seed.start <= int(match[1]) < seed.start + seed.size:
                if self.warmer:
                    self.warmer.cancel()
                    self.warmer = None
                self.seed = None
                self.target = None
                self.desired = None
                return seed
        elif self.warming_ident != ident:
            self.clear_prefetch()
        return None

    def warm(self, token: str, ident: str, remaining: float) -> None:
        """Schedule bounded read-ahead for the next chapter."""
        if self.closed or not 0 <= remaining < READ_AHEAD:
            return
        seconds = READ_AHEAD - remaining
        self.desired = (token, ident, seconds)
        # Keep one fetch in flight. Once complete, extend in modest increments
        # as the current episode drains, instead of issuing requests per chunk.
        if self.warmer and not self.warmer.done():
            return
        if self.target == (token, ident) and seconds < self.target_seconds + 15:
            return
        self.target = (token, ident)
        self.target_seconds = seconds
        self.warmer = asyncio.create_task(self._warm(token, ident, seconds))
        self.warmer.add_done_callback(self._warm_done)

    def idle(self) -> None:
        """Schedule cleanup when this playback has no readers."""
        if not self.active and not self.closed:
            loop = asyncio.get_running_loop()
            if self.expiry:
                self.expiry.cancel()
            self.expiry = loop.call_later(IDLE_TTL, self.close)

    def close(self) -> None:
        """Cancel work and release buffered resources."""
        self.closed = True
        if self.expiry:
            self.expiry.cancel()
            self.expiry = None
        self.clear_prefetch()
        for buffer in tuple(self.active):
            buffer.cancel()

    def _warm_done(self, task: asyncio.Task[None]) -> None:
        if task is not self.warmer or task.cancelled() or self.closed:
            return
        if self.desired and self.desired[2] >= self.target_seconds + 15:
            token, ident, seconds = self.desired
            self.warm(token, ident, READ_AHEAD - seconds)

    async def _warm(self, token: str, ident: str, seconds: float) -> None:  # noqa: PLR0915 -- Keep reservation ownership and cleanup in one scope.

        reserved = 0
        try:
            target = await self.provider._next_cache_track(token, ident)
            if not target or self.closed:
                return
            next_id, skip_head = target
            self.warming_ident = next_id
            resolve = self.resolve_prefetch or self.resolve
            url, duration = await resolve(next_id)
            if duration <= 0:
                return
            # Use a short header request to obtain the exact resource size first.
            headers = {"Accept-Encoding": "identity", "User-Agent": "Lavf/61.7.100"}
            assert self.provider._session is not None
            async with self.provider._session.get(
                url,
                headers={**headers, "Range": "bytes=0-0"},
                timeout=TIMEOUT,
                auto_decompress=False,
                allow_redirects=False,
            ) as response:
                resource = Resource.read(response, url, "bytes=0-0")
                content_type = response.headers.get("Content-Type", "audio/mp4")
                await response.read()
            rate = resource.total / duration
            # Average-rate positioning is an estimate, so retain a 64 KiB lead-in.
            # FFmpeg still reads its own MP4 index and requests exact seek offsets.
            start = max(0, int(skip_head * rate) - CHUNK)
            end = min(resource.total - 1, int((skip_head + seconds) * rate) + CHUNK - 1)
            end = min(end, start + MAX_BYTES // 2 - 1)
            previous = self.seed
            if previous and previous.ident == next_id:
                if previous.start != start or not previous.resource.same_file(resource):
                    self.budget.release(previous.size)
                    self.seed = previous = None
                else:
                    start += previous.size
            elif previous:
                self.budget.release(previous.size)
                self.seed = previous = None
            available = MAX_BYTES - self.budget.used - CHUNK
            end = min(end, start + available - 1)
            if end < start:
                return
            wanted = f"bytes={start}-{end}"
            size = end - start + 1
            await self.budget.reserve(size)
            reserved = size
            assert self.provider._session is not None
            async with self.provider._session.get(
                url,
                headers={**headers, "Range": wanted},
                timeout=TIMEOUT,
                auto_decompress=False,
                allow_redirects=False,
            ) as response:
                actual = Resource.read(response, url, wanted)
                if not resource.same_file(actual):
                    return
                chunks: deque[bytes] = deque()
                received = 0
                async for chunk in response.content.iter_chunked(CHUNK):
                    received += len(chunk)
                    if received > size:
                        return
                    chunks.append(chunk)
                if received != size:
                    return
            if self.is_closed():
                return
            if previous and self.seed is previous:
                previous.chunks.extend(chunks)
            elif previous:
                return
            else:
                self.seed = Prefetched(next_id, resource, duration, content_type, chunks, start)
            reserved = 0
            self.provider.logger.debug("已预读下一集 %s，缓存 %s 字节", next_id, size)
        except CarError, aiohttp.ClientError, TimeoutError, OSError, asyncio.IncompleteReadError:
            # This is speculative: never mark history or stop the current episode.
            self.provider.logger.debug("下一集预读暂不可用，将在实际播放时重新校验")
        except Exception as err:
            # Resource validation is intentionally sanitized, but not all exceptions are.
            self.provider.logger.debug("下一集预读已停止（%s）", type(err).__name__)
        finally:
            if reserved:
                self.budget.release(reserved)


class AudioCache:
    """No cache is shared between provider instances/accounts or queue owners."""

    def __init__(self, provider: XimalayaProvider) -> None:
        """Initialize per-instance state."""
        self.provider = provider
        self.sessions: dict[str | tuple[str | None, ...], CacheSession] = {}

    def session(self, key: str | tuple[str | None, ...]) -> CacheSession:
        """Return the isolated buffer session for this playback owner."""
        for ident, session in list(self.sessions.items()):
            if session.closed:
                self.sessions.pop(ident)
        if key not in self.sessions:
            # Bound unbound probe sessions as well as the bytes within each queue.
            if len(self.sessions) >= 64:
                raise CarError("同时播放的音频过多")
            self.sessions[key] = CacheSession(self.provider)
        return self.sessions[key]

    async def close(self) -> None:
        """Cancel work and release buffered resources."""
        tasks = []
        for session in self.sessions.values():
            if session.warmer:
                tasks.append(session.warmer)
            tasks.extend(buffer.task for buffer in session.active)
            session.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()

    def prune_queues(self) -> None:
        """Release buffers belonging to paused or stopped queues."""
        for key, session in self.sessions.items():
            if not isinstance(key, tuple) or session.closed:
                continue
            queue = self.provider.mass.player_queues.get(key[0] or "")
            state = str(queue.state) if queue else "idle"
            if state == "playing":
                session.seen_playing = True
            elif session.seen_playing and state in ("idle", "paused"):
                session.close()


class ReadAhead:
    """Producer and consumer run independently, so recovery can fill a live buffer."""

    def __init__(
        self,
        provider: XimalayaProvider,
        session: CacheSession,
        ident: str,
        requested: str | None,
        seed: Prefetched | None = None,
        *,
        expected: Resource | None = None,
    ) -> None:
        """Initialize per-instance state."""
        self.provider, self.session = provider, session
        self.ident, self.requested = ident, requested
        self.seed: Prefetched | None = seed
        self.resolve: Callable[[str], Awaitable[tuple[str, float]]] = session.resolve
        self.expected = expected
        self.ready = asyncio.Event()
        self.changed = asyncio.Condition()
        self.chunks: deque[bytes] = deque()
        self.size = 0
        self.resource: Resource | None = None
        self.content_type = "audio/mp4"
        self.duration: float = 0
        self.error: str | None = None
        self.status = 503
        self.done = False
        self.closed = False
        self.capacity = CHUNK
        self.task = asyncio.create_task(self._produce())
        session.active.add(self)

    def is_closed(self) -> bool:
        """Read cancellation state after an awaited operation."""
        return self.closed

    async def read(self) -> bytes:
        """Validate and construct a resource from the supplied response."""
        async with self.changed:
            await self.changed.wait_for(lambda: self.chunks or self.done or self.closed)
            if self.chunks:
                data = self.chunks.popleft()
                self.size -= len(data)
                self.session.budget.release(len(data))
                self.changed.notify_all()
                return data
            return b""

    def cancel(self) -> None:
        """Cancel ongoing work without waiting for completion."""
        self.closed = True
        self.task.cancel()

    async def close(self) -> None:
        """Cancel work and release buffered resources."""
        self.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.session.budget.release(self.size)
        self.size = 0
        self.chunks.clear()
        self.session.active.discard(self)
        self.session.idle()

    async def _put(self, data: bytes) -> None:
        async with self.changed:
            await self.changed.wait_for(
                lambda: self.size + len(data) <= self.capacity or self.closed
            )
            if self.closed:
                raise asyncio.CancelledError
        await self.session.budget.reserve(len(data))
        self.chunks.append(data)
        self.size += len(data)
        async with self.changed:
            self.changed.notify_all()

    async def _produce(self) -> None:  # noqa: PLR0915 -- Keep byte offsets and retry budget under one producer lifetime.

        offset = 0
        recovery_started = None
        try:
            if self.seed:
                seed = self.seed
                # Fresh account authorization is required even for cached next-episode bytes.
                for attempt in range(ATTEMPTS):
                    try:
                        url, _ = await self.resolve(self.ident)
                        break
                    except PermissionDenied, PreparationError:
                        raise
                    except CarError:
                        if attempt == ATTEMPTS - 1:
                            raise
                        await asyncio.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                if urlsplit(url).path != seed.resource.path:
                    self.session.budget.release(seed.size)
                    self.seed = None
                else:
                    match = RANGE.fullmatch(self.requested or "bytes=0-")
                    if match is None or not match[1]:
                        raise RelayError("Invalid prefetched byte range")
                    start = int(match[1])
                    end = (
                        min(int(match[2]), seed.resource.total - 1)
                        if match[2]
                        else seed.resource.total - 1
                    )
                    if end < start:
                        raise RelayError("CDN 字节范围不匹配，停止续传")
                    self.resource = Resource(
                        start,
                        end,
                        seed.resource.total,
                        seed.resource.path,
                        seed.resource.etag,
                        seed.resource.modified,
                    )
                    self.duration, self.content_type = seed.duration, seed.content_type
                    self._capacity()
                    self.ready.set()
                    offset = start
                    while seed.chunks:
                        data = seed.chunks.popleft()
                        chunk_start = seed.start
                        seed.start += len(data)
                        self.session.budget.release(len(data))
                        left, right = (
                            max(0, start - chunk_start),
                            min(len(data), end - chunk_start + 1),
                        )
                        if left < right:
                            await self._put(data[left:right])
                            offset = chunk_start + right
                    self.seed = None
            for attempt in range(MAX_RECOVERY_ATTEMPTS):
                if self.resource and offset > self.resource.end:
                    return
                if attempt:
                    if attempt >= ATTEMPTS and not self.size:
                        break
                    delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
                    if attempt >= ATTEMPTS:
                        delay = RECOVERY_DELAYS[min(attempt - ATTEMPTS, len(RECOVERY_DELAYS) - 1)]
                    if (
                        recovery_started
                        and time.monotonic() - recovery_started + delay >= RECOVERY_SECONDS
                    ):
                        break
                    await asyncio.sleep(delay)
                try:
                    if attempt and self.session.refresh_source:
                        await self.session.refresh_source(self.ident)
                    url, duration = await self.resolve(self.ident)
                    self.duration = duration
                    wanted = (
                        f"bytes={offset}-{self.resource.end}" if self.resource else self.requested
                    )
                    headers = {"Accept-Encoding": "identity", "User-Agent": "Lavf/61.7.100"}
                    if wanted:
                        headers["Range"] = wanted
                    assert self.provider._session is not None
                    async with self.provider._session.get(
                        url,
                        headers=headers,
                        timeout=TIMEOUT,
                        auto_decompress=False,
                        allow_redirects=False,
                    ) as response:
                        if response.status == 416 and self.resource is None:
                            self.status = 416
                            raise RelayError("CDN 字节范围不匹配，停止续传")
                        resource = Resource.read(response, url, wanted)
                        if self.expected and not self.expected.same_file(resource):
                            raise RelayError("CDN 资源已变化，停止续传以避免音频损坏")
                        if self.resource and not self.resource.same_file(resource):
                            raise RelayError("CDN 资源已变化，停止续传以避免音频损坏")
                        if self.resource is None:
                            self.resource, offset = resource, resource.start
                            self.content_type = response.headers.get("Content-Type", "audio/mp4")
                            self._capacity()
                            self.ready.set()
                        async for data in response.content.iter_chunked(CHUNK):
                            if offset + len(data) > self.resource.end + 1:
                                raise RelayError("CDN 返回超出请求范围的数据")
                            await self._put(data)
                            offset += len(data)
                        if offset != self.resource.end + 1:
                            raise RelayError("CDN 音频提前结束")
                        if attempt:
                            self.provider.logger.info("章节 %s 重连成功，继续补充缓存", self.ident)
                        self.error = None
                        return
                except PreparationError as err:
                    self.error = str(err)
                    return
                except PermissionDenied:
                    self.status, self.error = 403, "当前账号无权播放此章节"
                    # Explicit loss of rights is terminal, not a transient CDN error.
                    self.session.budget.release(self.size)
                    self.chunks.clear()
                    self.size = 0
                    return
                except RelayError as err:
                    self.error = str(err)
                    if any(word in self.error for word in ("范围", "资源已变化", "压缩传输")):
                        return
                except (aiohttp.ClientError, TimeoutError, OSError, CarError) as err:
                    self.error = f"音频连接中断（{type(err).__name__}）"
                recovery_started = recovery_started or time.monotonic()
                self.provider.logger.warning(
                    "章节 %s 下载失败：%s；剩余缓存 %s 字节，尝试 %s/%s",
                    self.ident,
                    self.error,
                    self.size,
                    attempt + 1,
                    MAX_RECOVERY_ATTEMPTS,
                )
        except PreparationError as err:
            self.error = str(err)
        except PermissionDenied:
            self.status, self.error = 403, "当前账号无权播放此章节"
        except asyncio.CancelledError:
            raise
        except Exception as err:
            self.error = f"音频缓存读取失败（{type(err).__name__}）"
        finally:
            if self.seed:
                self.session.budget.release(self.seed.size)
                self.seed = None
            self.done = True
            self.ready.set()
            async with self.changed:
                self.changed.notify_all()

    def _capacity(self) -> None:
        assert self.resource is not None
        rate = self.resource.total / self.duration if self.duration > 0 else 16000
        self.capacity = min(MAX_BYTES - CHUNK, max(CHUNK, math.ceil(rate * READ_AHEAD)))
