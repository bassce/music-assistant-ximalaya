"""Bounded, byte-exact chapter relay; signed CDN URLs never reach the decoder."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import XimalayaProvider
    from .normalization import Normalizer

import asyncio
import contextlib

import aiohttp
from aiohttp import web

from .audio_cache import AudioCache, ReadAhead
from .protocol import CarError, PermissionDenied, PreparationError
from .resource import ATTEMPTS, RANGE, RETRY_DELAYS, RelayError


async def proxy_chapter(  # noqa: PLR0915 -- Keep response, prefetch seed and reader cleanup under one lifetime.
    provider: XimalayaProvider, request: web.Request, ident: str, token: str
) -> web.StreamResponse:
    """Serve bytes immediately while a separate task maintains the read-ahead window."""
    requested = request.headers.get("Range")
    if requested and (not (m := RANGE.fullmatch(requested)) or not any(m.groups())):
        return web.Response(status=416)
    if request.method not in ("GET", "HEAD"):
        return web.Response(status=405)
    if not hasattr(provider, "_audio_cache"):
        provider._audio_cache = AudioCache(provider)
    key: str | tuple[str | None, ...] = (
        provider._cache_owner(token) if hasattr(provider, "_cache_owner") else token
    )
    normalizer: Normalizer | None = getattr(provider, "_normalizer", None)
    normalize = normalizer is not None and token in provider._part_books
    if normalize and normalizer is not None:
        # A new playback can choose a different target after quality fallback.
        # Do not carry encoded bytes across those decoder sessions.
        key = (*key, token) if isinstance(key, tuple) else (key, None, token)
    cache = provider._audio_cache.session(key)
    if normalize and normalizer is not None:

        async def resolve(ident: str) -> tuple[str, float]:
            assert normalizer is not None
            """Authorize and resolve the requested audio resource."""
            return await normalizer.resolve(token, ident)

        cache.resolve = resolve

        async def resolve_prefetch(ident: str) -> tuple[str, float]:
            assert normalizer is not None
            url, duration = await normalizer.resolve(token, ident, prefetch=True)
            entry = normalizer.entries.get((token, ident))
            if entry and entry.progressive:
                # The encoder prepares only a bounded in-memory prefix.
                return url, 0
            return url, duration

        cache.resolve_prefetch = resolve_prefetch
    seed = cache.take(ident, requested)
    if normalize and normalizer is not None:

        async def prepare() -> tuple[str, float]:
            assert normalizer is not None
            for attempt in range(ATTEMPTS):
                try:
                    return await normalizer.resolve(token, ident)
                except PermissionDenied, PreparationError:
                    raise
                except CarError, RelayError, aiohttp.ClientError, TimeoutError, OSError:
                    if attempt == ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])

            raise RuntimeError("Audio preparation retry loop exhausted")

        preparation = asyncio.create_task(prepare())
        prepared = None
        try:
            while not preparation.done():
                if request.transport is None or request.transport.is_closing():
                    return web.Response(status=499)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(preparation), 0.5)
            prepared = await preparation
        except (CarError, RelayError, aiohttp.ClientError, TimeoutError, OSError) as err:
            await provider._chapter_failed(
                token,
                ident,
                str(err)
                if isinstance(err, (PreparationError, PermissionDenied))
                else "音频准备失败",
            )
            return web.Response(
                status=403 if isinstance(err, PermissionDenied) else 503,
                text="喜马拉雅音频暂不可用，已保留播放断点",
            )
        finally:
            if not preparation.done():
                preparation.cancel()
            await asyncio.gather(preparation, return_exceptions=True)
            if seed and prepared is None:
                cache.budget.release(seed.size)
                seed = None
        entry = normalizer.entries.get((token, ident))
        if entry and entry.progressive:
            if seed:
                cache.budget.release(seed.size)
                seed = None

            def progress(sent: int, total: float) -> None:
                if total:
                    cache.warm(token, ident, max(0, (total - sent) * prepared[1] / total))

            try:
                return await entry.progressive.serve(request, entry, progress)
            finally:
                cache.idle()

        initial: tuple[str, float] | None = prepared

        async def resolve(ident: str) -> tuple[str, float]:
            assert normalizer is not None
            """Authorize and resolve the requested audio resource."""
            nonlocal initial
            if initial is not None:
                value, initial = initial, None
                return value
            return await normalizer.resolve(token, ident)

        cache.resolve = resolve
    buffer = ReadAhead(provider, cache, ident, requested, seed)
    response = None
    try:
        while not buffer.ready.is_set():
            if request.transport is None or request.transport.is_closing():
                return web.Response(status=499)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(buffer.ready.wait(), 0.5)
        if buffer.closed or request.transport is None or request.transport.is_closing():
            return web.Response(status=499)
        resource = buffer.resource
        if resource is None:
            if buffer.status != 416:
                await provider._chapter_failed(token, ident, buffer.error or "音频读取失败")
            return web.Response(status=buffer.status, text="喜马拉雅音频暂不可用，已保留播放断点")
        headers = {
            "Content-Type": buffer.content_type,
            "Content-Length": str(resource.end - resource.start + 1),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        }
        if requested:
            headers["Content-Range"] = f"bytes {resource.start}-{resource.end}/{resource.total}"
        response = web.StreamResponse(status=206 if requested else 200, headers=headers)
        await response.prepare(request)
        if request.method == "HEAD":
            return response
        offset = resource.start
        while not buffer.closed:
            if request.transport is None or request.transport.is_closing():
                return response
            try:
                data = await asyncio.wait_for(buffer.read(), 0.5)
            except TimeoutError:
                continue
            if not data:
                break
            try:
                await response.write(data)
            except ConnectionError, RuntimeError:
                return response
            offset += len(data)
            # Only a transfer toward EOF predicts a chapter transition; MP4 header
            # probes and short byte ranges must not warm the next episode.
            if resource.end == resource.total - 1 and buffer.duration > 0:
                remaining = (resource.total - offset) * buffer.duration / resource.total
                if hasattr(provider, "_cache_remaining"):
                    remaining = provider._cache_remaining(token, ident, remaining)
                cache.warm(token, ident, remaining)
        if buffer.is_closed():
            return response
        if buffer.error or offset != resource.end + 1:
            await provider._chapter_failed(token, ident, buffer.error or "音频缓存已耗尽")
            response.force_close()
            if request.transport:
                request.transport.close()
            return response
        with contextlib.suppress(ConnectionError):
            await response.write_eof()
        return response
    finally:
        await buffer.close()
