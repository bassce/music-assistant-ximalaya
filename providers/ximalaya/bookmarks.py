"""Backend bookmarks through public provider callbacks, without MA database access."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MusicAssistantError

from .protocol import CarError

if TYPE_CHECKING:
    from music_assistant_models.auth import User

    from . import XimalayaProvider
    from .longform import Chapters


async def progress_user(provider: XimalayaProvider) -> User | None:
    """Resolve only unambiguous ownership; MA's callback has no user argument."""
    if user := get_current_user():
        return user
    users = await provider.mass.webserver.auth.list_users()
    eligible = [
        user
        for user in users
        if not user.provider_filter or provider.instance_id in user.provider_filter
    ]
    return eligible[0] if len(eligible) == 1 else None


async def read_bookmark(
    provider: XimalayaProvider, album: str, chapters: Chapters | None
) -> tuple[bool, int, float] | None:
    candidates = []
    recovery = await provider.mass.cache.get(
        key=f"stream-recovery-account.{album}", provider=provider.instance_id
    )
    if (
        isinstance(recovery, dict)
        and type(recovery.get("position")) is int
        and recovery["position"] >= 0
        and (chapters is None or recovery["position"] <= chapters.duration)
        and type(recovery.get("timestamp")) in (int, float)
        and math.isfinite(recovery["timestamp"])
    ):
        candidates.append((False, recovery["position"] * 1000, recovery["timestamp"]))
    if chapters is None:
        return max(candidates, key=lambda row: row[2]) if candidates else None
    try:
        saved = await provider._client.local_progress(album)
    except CarError:
        provider.logger.warning("API bookmark unavailable; retaining MA's own progress")
        return max(candidates, key=lambda row: row[2]) if candidates else None
    record = saved.get("record")
    if record:
        absolute = chapters.absolute(record["chapter_id"], record["position_ms"] / 1000)
        if absolute is not None:
            candidates.append(
                (
                    bool(record["fully_played"]) or absolute >= chapters.duration,
                    int(absolute * 1000),
                    record["timestamp"],
                )
            )
    return max(candidates, key=lambda row: row[2]) if candidates else None


async def save_bookmark(
    provider: XimalayaProvider,
    media_type: MediaType,
    ident: str,
    position: float,
    fully_played: bool,
) -> None:
    """Persist actual progress only; prefetch and metadata reads never call this."""
    if not math.isfinite(position) or position < 0:
        return
    try:
        if media_type == MediaType.AUDIOBOOK:
            chapters = await provider._chapters(ident)
            assert chapters is not None
            row, offset = chapters.locate(min(position, chapters.duration))
            album, chapter = ident, str(row["id"])
        else:
            row = await provider._client.track(ident)
            album, chapter = str(row["album_id"]), ident
            offset = min(position, row["duration"])
            # Completing an episode does not complete its entire album. Episode
            # callers infer completion from the offset; books know their final chapter.
            fully_played = False
        await provider._client.save_local_progress(
            album,
            chapter_id=chapter,
            position_ms=round(offset * 1000),
            fully_played=fully_played,
        )
    except CarError, MusicAssistantError:
        # A failed report is not replayed: the next callback supplies current progress.
        provider.logger.warning("API bookmark write failed; retaining MA's own progress")


async def import_journal(
    provider: XimalayaProvider, album: str, records: list[dict[str, Any]]
) -> None:
    """Recover older provider migration journals without opening MA's database."""
    if not records:
        return
    chapters = await provider._chapters(album)
    assert chapters is not None
    # A Ximalaya account has one bookmark per book, shared by all MA users.
    for record in sorted(records, key=lambda row: row["timestamp"], reverse=True):
        position = record.get("seconds_played") or 0
        if not 0 <= position <= chapters.duration:
            continue
        chapter, offset = chapters.locate(position)
        await provider._client.save_local_progress(
            album,
            chapter_id=str(chapter["id"]),
            position_ms=round(offset * 1000),
            fully_played=bool(record.get("fully_played")),
            timestamp=record["timestamp"],
            import_only=True,
        )
