# Copyright (c) 2026 ha_ximalaya contributors
"""Bound relay capabilities by reclaiming unused playback history, never live readers."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

from music_assistant_models.errors import ResourceTemporarilyUnavailable

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

    from . import XimalayaProvider

SESSION_LIMIT = 64
SESSION_TTL = 86400


def stream_token(stream: StreamDetails | None) -> str | None:
    """Read our capability from either a whole book or an individual episode."""
    if stream is None:
        return None
    if isinstance(stream.data, dict) and isinstance(stream.data.get("relay_session"), str):
        return stream.data["relay_session"]
    path = stream.path
    if isinstance(path, list):
        path = path[0].path if path else None
    if isinstance(path, str):
        return parse_qs(urlsplit(path).query).get("session", [None])[0]
    return None


@contextmanager
def hold_session(provider: XimalayaProvider, token: str | None) -> Iterator[None]:
    """Keep a capability alive across HTTP reads and gaps between chapter decoders."""
    if token is None or token not in getattr(provider, "_part_sessions", {}):
        yield
        return
    uses = provider._part_uses
    uses[token] = uses.get(token, 0) + 1
    try:
        yield
    finally:
        remaining = uses.get(token, 1) - 1
        if remaining:
            uses[token] = remaining
        else:
            uses.pop(token, None)


def protected_sessions(provider: XimalayaProvider) -> set[str]:
    """Include queued streams and preloaded decoders still using an older stream."""
    protected = set(provider._part_uses)
    for queue in provider.mass.player_queues.all():
        for item in (queue.current_item, getattr(queue, "next_item", None)):
            stream = item.streamdetails if item else None
            if stream and stream.provider == provider.instance_id and (token := stream_token(stream)):
                protected.add(token)
    # A native resume seek replaces StreamDetails while the adopted decoder keeps
    # the old URLs. Include preparations even before their task starts running.
    for prepared in provider._resume_preloader.live:
        if not prepared.closed and (token := stream_token(prepared.origin)):
            protected.add(token)
    return protected


def discard_session(provider: XimalayaProvider, token: str) -> None:
    """Retire unused authorization and its private buffers without touching bookmarks."""
    provider._part_sessions.pop(token, None)
    provider._part_users.pop(token, None)
    provider._part_books.pop(token, None)
    provider._failed_parts.discard(token)
    for key, cache in list(provider._audio_cache.sessions.items()):
        # Queue-scoped podcast buffers can be shared with a later episode; only
        # token-scoped raw/normalized buffers belong exclusively to this session.
        if key == token or (isinstance(key, tuple) and len(key) >= 3 and key[-1] == token):
            cache.close()
            provider._audio_cache.sessions.pop(key, None)


def new_session(provider: XimalayaProvider, tracks: set[str], user_id: str | None) -> str:
    """Allocate atomically after async preparation, reclaiming the oldest unused slot."""
    now = time.monotonic()
    protected = protected_sessions(provider)
    for token, (expires, _) in list(provider._part_sessions.items()):
        if expires <= now and token not in protected:
            discard_session(provider, token)
    if len(provider._part_sessions) >= SESSION_LIMIT:
        unused = [token for token in provider._part_sessions if token not in protected]
        if not unused:
            raise ResourceTemporarilyUnavailable("同时播放的音频过多，请稍后重试")
        oldest = min(unused, key=lambda token: provider._part_sessions[token][0])
        discard_session(provider, oldest)
        provider.logger.debug("已回收闲置播放会话，保留 %s 个受保护会话", len(protected))
    token = secrets.token_urlsafe(32)
    provider._part_sessions[token] = (now + SESSION_TTL, tracks)
    provider._part_users[token] = user_id
    return token
