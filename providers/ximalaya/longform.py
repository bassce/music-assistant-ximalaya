# Copyright (c) 2026 ha_ximalaya contributors
"""Native chapter playback and account-scoped progress for MA 2.10.6."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.media_items import MediaItemType, Podcast

from music_assistant.controllers.music.media.audiobooks import AudiobooksController
from music_assistant.controllers.music.media.podcasts import PodcastsController

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.queue_item import QueueItem

    from . import XimalayaProvider

import asyncio
import json
import math
import secrets
import time
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit
from weakref import WeakValueDictionary

from aiohttp import web
from music_assistant_models.enums import (
    ContentType,
    EventType,
    MediaType,
    PlaybackState,
    StreamType,
)
from music_assistant_models.errors import (
    MediaNotFoundError,
    MusicAssistantError,
    ResourceTemporarilyUnavailable,
)
from music_assistant_models.media_items import (
    Audiobook,
    AudioFormat,
    MediaItemChapter,
    UniqueList,
)
from music_assistant_models.streamdetails import MultiPartPath, StreamDetails

from .audio_cache import AudioCache
from .book_pcm import pcm_format
from .bookmarks import import_journal, progress_user, read_bookmark, save_bookmark
from .normalization import Normalizer
from .protocol import CarError, item_id
from .resume_preload import ResumePreloader
from .streaming import proxy_chapter


@dataclass(frozen=True)
class Chapters:
    """An immutable timeline. Authorization is always rechecked at playback time."""

    rows: list[dict[str, Any]]
    starts: list[int]
    duration: int

    @classmethod
    def create(cls, rows: list[dict[str, Any]]) -> Chapters:
        """Construct the requested representation from validated input."""
        starts, total = [], 0
        if not rows:
            raise ResourceTemporarilyUnavailable("这本有声书没有可用章节")
        for row in rows:
            if type(row.get("duration")) is not int or row["duration"] <= 0:
                raise ResourceTemporarilyUnavailable("章节时长不完整，无法准确定位")
            starts.append(total)
            total += row["duration"]
        return cls(rows, starts, total)

    def locate(self, position: float) -> tuple[dict[str, Any], float]:
        """Return the chapter and offset at an absolute position."""
        if not 0 <= position <= self.duration:
            raise MediaNotFoundError("定位超出有声书时长")
        index = min(bisect_right(self.starts, position) - 1, len(self.rows) - 1)
        return self.rows[index], position - self.starts[index]

    def absolute(self, ident: str, position: float) -> float | None:
        """Return the absolute position for a chapter and offset."""
        for row, start in zip(self.rows, self.starts, strict=True):
            if str(row["id"]) == ident and 0 <= position <= row["duration"]:
                return start + position
        return None


class LongformMixin:
    """Optional cloud writes; legacy episode URIs continue to work after migration."""

    reload_on_streams_network_change = True

    async def get_audiobook(self, prov_audiobook_id: str) -> Audiobook:
        """Return audiobook metadata and chapters when already cached."""
        row = await self._provider._checked(self._provider._client.album(prov_audiobook_id))
        chapters = await self._provider._chapters(prov_audiobook_id, row=row)
        assert chapters is not None
        book = self._provider._book(row, chapters)
        try:
            # MA consumes these fields when a manually favorited book enters the library.
            finished, position, _ = await self._provider.get_resume_position(
                prov_audiobook_id, MediaType.AUDIOBOOK
            )
            book.fully_played, book.resume_position_ms = finished, position
        except NotImplementedError:
            pass
        return book

    async def get_library_audiobooks(self) -> AsyncGenerator[Audiobook]:
        """Yield subscribed audiobooks."""
        rows = await self._provider._subscription_rows(MediaType.AUDIOBOOK)
        books = [self._provider._book(row) for row in rows]
        for book in books:
            yield book

    async def on_streamed(self, streamdetails: StreamDetails) -> None:
        """Release playback capabilities after the stream ends."""
        if not streamdetails.stream_error:
            return
        # Also protect against a decoder error after a valid HTTP transfer. Normal
        # stop/seek cancellation has no stream_error and must never stop new playback.
        paths = (
            [part.path for part in streamdetails.path[:1]]
            if isinstance(streamdetails.path, list)
            else [streamdetails.path]
        )
        for path in paths:
            if not isinstance(path, str):
                continue
            token = parse_qs(urlsplit(path).query).get("session", [None])[0]
            if token is not None and token in self._provider._part_sessions:
                await self._provider._chapter_failed(token, "当前", "音频解码或读取失败")

    async def get_resume_position(
        self, item_id: str, media_type: MediaType
    ) -> tuple[bool, int, datetime]:
        """Reconcile local and account progress for this item."""
        if media_type == MediaType.PODCAST_EPISODE:
            row = await self._provider._checked(self._provider._client.track(item_id))
            album_id = str(row["album_id"])
            chapters = await self._provider._chapters(album_id, cached_only=True)
            candidates = []
            local = await self._provider._local_progress(album_id, chapters) if chapters else None
            if local and chapters:
                chapter, offset = chapters.locate(local[1] / 1000)
                if str(chapter["id"]) == item_id:
                    candidates.append((offset >= chapter["duration"], int(offset * 1000), local[2]))
            # Podcast resume does not need to fetch the entire album catalog.
            if not chapters:
                try:
                    saved = await self._provider._client.local_progress(album_id)
                    record = saved.get("record")
                    if (
                        record
                        and record["chapter_id"] == item_id
                        and (0 <= record["position_ms"] <= row["duration"] * 1000)
                    ):
                        candidates.append(
                            (
                                record["position_ms"] >= row["duration"] * 1000,
                                record["position_ms"],
                                record["timestamp"],
                            )
                        )
                except CarError:
                    self._provider.logger.warning("API bookmark unavailable; using MA history")
            recovery = await self._provider.mass.cache.get(
                key=f"stream-recovery-account.episode.{item_id}",
                provider=self._provider.instance_id,
            )
            if (
                isinstance(recovery, dict)
                and type(recovery.get("position")) is int
                and 0 <= recovery["position"] <= row["duration"]
                and type(recovery.get("timestamp")) in (int, float)
                and math.isfinite(recovery["timestamp"])
            ):
                candidates.append((False, recovery["position"] * 1000, recovery["timestamp"]))
            if self._provider.config.get_value("cloud_progress_write", False):
                try:
                    remote = await self._provider._client.cloud_progress(album_id)
                except CarError:
                    self._provider.logger.warning("云端进度暂不可用，使用本地断点")
                else:
                    if (
                        remote
                        and remote["track_id"] == item_id
                        and remote["position"] <= row["duration"]
                    ):
                        candidates.append(
                            (
                                remote["position"] >= row["duration"],
                                int(remote["position"] * 1000),
                                remote["timestamp"],
                            )
                        )
            if candidates:
                finished, position, timestamp = max(candidates, key=lambda x: x[2])
                return finished, position, datetime.fromtimestamp(timestamp, UTC)
            raise NotImplementedError
        if media_type != MediaType.AUDIOBOOK:
            raise NotImplementedError
        chapters = await self._provider._chapters(item_id)
        assert chapters is not None
        local = await self._provider._local_progress(item_id, chapters)
        remote = None
        if self._provider.config.get_value("cloud_progress_write", False):
            try:
                remote = await self._provider._client.cloud_progress(item_id)
            except CarError:
                self._provider.logger.warning("云端进度暂不可用，使用本地断点")
        if remote:
            absolute = chapters.absolute(remote["track_id"], remote["position"])
            if absolute is not None and (not local or remote["timestamp"] > local[2]):
                return (
                    absolute >= chapters.duration,
                    int(absolute * 1000),
                    datetime.fromtimestamp(remote["timestamp"], UTC),
                )
        if local:
            return local[0], local[1], datetime.fromtimestamp(local[2], UTC)
        raise NotImplementedError

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: float,
        media_item: MediaItemType | None,
        is_playing: bool = False,
    ) -> None:
        """Persist listening progress when account synchronization is enabled."""
        if media_type not in (
            MediaType.AUDIOBOOK,
            MediaType.PODCAST_EPISODE,
        ):
            return
        await save_bookmark(self._provider, media_type, prov_item_id, position, fully_played)
        if not self._provider.config.get_value("cloud_progress_write", False):
            return
        key = f"episode:{prov_item_id}" if media_type == MediaType.PODCAST_EPISODE else prov_item_id
        async with self._provider._progress_lock:
            state = self._provider._progress_sessions.get(key)
            if not state or state["blocked"]:
                return
            album_id = state["album_id"]
            try:
                current = await self._provider._client.cloud_progress(album_id)
                if current != state["baseline"]:
                    state["blocked"] = True
                    self._provider.logger.warning(
                        "其他设备已更新云端进度，本次播放停止写回；本地进度保留"
                    )
                    return
                if media_type == MediaType.AUDIOBOOK:
                    chapters = await self._provider._chapters(prov_item_id)
                    assert chapters is not None
                    row, offset = chapters.locate(min(position, chapters.duration))
                    track_id = str(row["id"])
                    track = await self._provider._client.track(track_id)
                else:
                    track_id = prov_item_id
                    track = await self._provider._client.track(track_id)
                    if str(track["album_id"]) != album_id:
                        state["blocked"] = True
                        return
                    offset = min(position, track["duration"])
                # Entitlement remains authoritative even for progress writes.
                if track.get("is_authorized") is not True:
                    return
                now = time.monotonic()
                elapsed = max(0, min(30, now - state["time"]))
                if state["last_position"] is not None:
                    elapsed = min(elapsed, max(0, position - state["last_position"]))
                await self._provider._client.save_progress(
                    album_id,
                    track_id,
                    offset,
                    elapsed,
                    state["wall"],
                    int(time.time() * 1000),
                )
                confirmed = await self._provider._client.cloud_progress(album_id)
                if (
                    not confirmed
                    or confirmed["track_id"] != track_id
                    or abs(confirmed["position"] - offset) > 2
                ):
                    state["blocked"] = True
                    self._provider.logger.warning("云端进度写回未确认，停止重试；本地进度保留")
                    return
                state.update(
                    baseline=confirmed,
                    time=now,
                    wall=int(time.time() * 1000),
                    last_position=position,
                )
            except CarError:
                state["blocked"] = True
                self._provider.logger.warning("云端进度写回失败，停止重试；本地进度保留")

    async def library_add(self, item: MediaItemType) -> bool:
        """Subscribe to the selected album when synchronization is enabled."""
        if self._provider.config.get_value("subscription_write", False) and item.media_type in (
            MediaType.AUDIOBOOK,
            MediaType.PODCAST,
        ):
            mapping = next(
                (
                    x
                    for x in item.provider_mappings
                    if x.provider_instance == self._provider.instance_id
                ),
                None,
            )
            if mapping:
                async with self._provider._subscription_lock():
                    await self._provider._checked(
                        self._provider._client.set_subscription(mapping.item_id, True)
                    )
                    await self._provider._remember_subscription_write(
                        mapping.item_id, item.media_type, True
                    )
        return True

    async def library_remove(self, prov_item_id: str, media_type: MediaType) -> bool:
        """Unsubscribe from the selected album when synchronization is enabled."""
        if self._provider.config.get_value("subscription_write", False) and media_type in (
            MediaType.AUDIOBOOK,
            MediaType.PODCAST,
        ):
            async with self._provider._subscription_lock():
                await self._provider._checked(
                    self._provider._client.set_subscription(prov_item_id, False)
                )
                await self._provider._remember_subscription_write(prov_item_id, media_type, False)
        return True

    def _init_longform(self) -> None:
        self._provider._catalogs = {}
        self._provider._catalog_locks = WeakValueDictionary()
        self._provider._part_sessions = {}
        self._provider._part_users = {}
        self._provider._part_books = {}
        self._provider._audio_cache = AudioCache(self._provider)
        self._provider._resume_preloader = ResumePreloader(self._provider)
        self._provider._normalizer = Normalizer(self._provider)
        self._provider._failed_parts = set()
        self._provider._recovery_tasks = set()
        self._provider._queue_chapter_tasks = {}
        self._provider._queue_chapter_pending = set()
        self._provider._progress_sessions = {}
        self._provider._progress_lock = asyncio.Lock()
        self._provider._skip_unloaded = False
        self._provider._skip_configs = {}
        self._provider._skip_config_locks = WeakValueDictionary()
        self._provider._skip_streams = {}
        self._provider._skip_queue_locks = WeakValueDictionary()
        self._provider._skip_attempts = {}
        self._provider._remove_part_route = self._provider.mass.streams.register_dynamic_route(
            f"/{self._provider.instance_id}_chapter", self._provider._chapter_request
        )

    async def _chapters(
        self, album_id: str, *, row: dict[str, Any] | None = None, cached_only: bool = False
    ) -> Chapters | None:
        """Reuse complete timelines, fetching missing chapters unless cache-only is requested."""
        album_id = item_id(album_id)
        version = self._provider._catalog_version(row) if row else None
        cached = self._provider._catalogs.get(album_id)
        if cached and cached[0] > time.monotonic() and (version is None or cached[2] == version):
            return cached[1]
        # A detail read can observe an update before playback asks for the cached timeline.
        # Do not let that later request reuse the superseded in-memory version.
        self._provider._catalogs.pop(album_id, None)
        if row is None:
            row = await self._provider._checked(self._provider._client.album(album_id))
            version = self._provider._catalog_version(row)
        uid = getattr(self._provider._client, "token", {}).get("uid", "")
        key = f"chapters-catalog-v2.{uid}.{album_id}"
        saved = await self._provider.mass.cache.get(key=key, provider=self._provider.instance_id)
        if isinstance(saved, dict) and version and saved.get("version") == version:
            try:
                rows = saved["rows"]
                if not isinstance(rows, list) or len(rows) != row.get("include_track_count"):
                    raise ValueError
                seen = set()
                for index, entry in enumerate(rows):
                    self._provider._validate_metadata(entry, episode=True)
                    if (
                        str(entry["album_id"]) != album_id
                        or entry["order_num"] != index
                        or entry["id"] in seen
                    ):
                        raise ValueError
                    seen.add(entry["id"])
                timeline = Chapters.create(rows)
            except KeyError, TypeError, ValueError, ResourceTemporarilyUnavailable:
                pass
            else:
                self._provider._remember_catalog(album_id, timeline, version)
                return timeline
        if cached_only:
            return None
        lock = self._provider._catalog_locks.setdefault(album_id, asyncio.Lock())
        async with lock:
            cached = self._provider._catalogs.get(album_id)
            if cached and cached[0] > time.monotonic() and cached[2] == version:
                return cached[1]
            rows = await self._provider._checked(self._provider._client.episodes(album_id))
            for entry in rows:
                self._provider._validate_metadata(entry, episode=True)
            unknown = [entry for entry in rows if entry["duration"] == 0]
            if len(unknown) > 5:
                raise ResourceTemporarilyUnavailable("过多章节缺少时长，保留原播客入口")
            for entry in unknown:
                entry["duration"] = await self._provider._probe_chapter_duration(str(entry["id"]))
            timeline = Chapters.create(rows)
            # Only retain fields needed for timeline/progress. No URLs or account tokens.
            fields = (
                "id",
                "title",
                "album_id",
                "duration",
                "order_num",
                "is_authorized",
                "is_paid",
            )
            if version and len(rows) == row.get("include_track_count"):
                await self._provider.mass.cache.set(
                    key=key,
                    provider=self._provider.instance_id,
                    expiration=30 * 86400,
                    data={"version": version, "rows": [{k: r[k] for k in fields} for r in rows]},
                )
            self._provider._remember_catalog(album_id, timeline, version)
            return timeline

    @staticmethod
    def _catalog_version(row: dict[str, Any]) -> str | None:
        if (
            type(row.get("updated_at")) is not int
            or type(row.get("include_track_count")) is not int
        ):
            return None
        return f"{row['updated_at']}:{row['include_track_count']}"

    def _remember_catalog(self, album_id: str, timeline: Chapters, version: str | None) -> None:
        if hasattr(self._provider._client, "_duration_hints"):
            for row in timeline.rows:
                self._provider._client._duration_hints[str(row["id"])] = row["duration"]
        if len(self._provider._catalogs) >= 32:
            self._provider._catalogs.pop(next(iter(self._provider._catalogs)))
        self._provider._catalogs[album_id] = (time.monotonic() + 300, timeline, version)

    async def _probe_chapter_duration(self, track_id: str) -> int:
        """Use an authorized audio header for rare missing durations; never estimate bitrate."""
        url, _ = await self._provider._checked(
            self._provider._client.resolve(track_id, probe_unknown_duration=True)
        )
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "error",
                "-rw_timeout",
                "15000000",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async with asyncio.timeout(20):
                output, _ = await process.communicate()
            data = json.loads(output)
            duration = float(data["format"]["duration"])
            if (
                process.returncode
                or not data.get("streams")
                or not math.isfinite(duration)
                or not 0 < duration <= 604800
            ):
                raise ValueError
        except TimeoutError, OSError, ValueError, KeyError:
            raise ResourceTemporarilyUnavailable("无法从音频确认章节时长，保留原播客入口") from None
        finally:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
        result = math.ceil(duration)
        self._provider._client._duration_hints[track_id] = result
        return result

    def _book(self, row: dict[str, Any], chapters: Chapters | None = None) -> Audiobook:
        self._provider._validate_metadata(row)
        self._provider._remember_album(row)
        publisher = (row.get("announcer") or {}).get("nickname")
        book = Audiobook(
            item_id=str(row["id"]),
            provider=self._provider.instance_id,
            name=row["title"],
            publisher=publisher,
            narrators=UniqueList(
                [self._provider._announcer(row["announcer"])]
                if publisher and row["announcer"].get("id") is not None
                else ([publisher] if publisher else [])
            ),
            provider_mappings=self._provider._mapping(row),
            metadata=self._provider._metadata(row),
            duration=chapters.duration if chapters else 0,
        )
        for mapping in book.provider_mappings:
            mapping.details = f"catalog-v2:{row.get('updated_at')}:{row.get('include_track_count')}"
        if chapters:
            self._provider._apply_chapters(book, chapters)
        return book

    @staticmethod
    def _apply_chapters(book: Audiobook, chapters: Chapters) -> None:
        book.duration = chapters.duration
        book.metadata.chapters = [
            MediaItemChapter(
                position=index + 1,
                name=chapter["title"],
                start=start,
                end=start + chapter["duration"],
            )
            for index, (chapter, start) in enumerate(
                zip(chapters.rows, chapters.starts, strict=True)
            )
        ]
        for mapping in book.provider_mappings:
            mapping.available = any(x.get("is_authorized") is True for x in chapters.rows)

    async def _sync_library_audiobooks(self) -> set[int]:
        books = [book async for book in self._provider.get_library_audiobooks()]
        controller = self._provider.mass.music.audiobooks
        ids = await self._provider._retained_classification_ids(MediaType.AUDIOBOOK, controller)
        for book in books:
            self._provider._note_listed_sync_item(MediaType.AUDIOBOOK, book.item_id)
            self._provider._update_sync_task_item_status(MediaType.AUDIOBOOK, len(ids) + 1)
            for mapping in book.provider_mappings:
                mapping.in_library = True
            existing = await controller.get_library_item_by_prov_id(
                book.item_id, self._provider.instance_id
            )
            needs_update = (
                existing is None
                or await self._provider._book_changed(existing, book)
                or not existing.metadata.chapters
            )
            if needs_update:
                old_version = (
                    next(
                        (
                            m.details
                            for m in existing.provider_mappings
                            if m.provider_instance == self._provider.instance_id
                        ),
                        None,
                    )
                    if existing
                    else None
                )
                new_version = next(iter(book.provider_mappings)).details
                if existing and existing.metadata.chapters and old_version == new_version:
                    # Cover/title edits must not crawl thousands of unchanged chapters.
                    book.duration = existing.duration
                    book.metadata.chapters = existing.metadata.chapters
                    for mapping in book.provider_mappings:
                        mapping.available = existing.available
                else:
                    self._provider._catalogs.pop(book.item_id, None)
                    try:
                        chapters = await self._provider._chapters(book.item_id)
                        assert chapters is not None
                        self._provider._apply_chapters(book, chapters)
                    except (ResourceTemporarilyUnavailable, MediaNotFoundError) as err:
                        self._provider.report_skipped_sync_item(
                            MediaType.AUDIOBOOK, book.item_id, err
                        )
                        await self._provider._refresh_legacy_metadata(book.item_id)
                        continue
            if existing is None:
                await self._provider._journal_book_progress(book.item_id)
                existing = await controller.add_item_to_library(book)
            elif needs_update:
                await self._provider._journal_book_progress(book.item_id)
                book.provider_mappings.update(
                    x
                    for x in existing.provider_mappings
                    if x.provider_instance != self._provider.instance_id
                )
                existing = await controller.update_item_in_library(
                    existing.item_id, book, overwrite=True
                )
            await self._provider._migrate_podcast(book.item_id, existing)
            ids.add(int(existing.item_id))
        return ids

    async def _refresh_legacy_metadata(self, album_id: str) -> None:
        """Keep a retained legacy entry's cover current even if its catalog is incomplete."""
        controller = self._provider.mass.music.podcasts
        old = await controller.get_library_item_by_prov_id(album_id, self._provider.instance_id)
        if old is None:
            return
        try:
            incoming = await self._provider.get_podcast(album_id)
        except ResourceTemporarilyUnavailable, MediaNotFoundError:
            return
        if incoming.metadata.images and incoming.metadata.images != old.metadata.images:
            await controller.update_item_in_library(old.item_id, incoming, overwrite=False)

    async def _journal_book_progress(self, album_id: str) -> None:
        """Import journals from the previous release before library synchronization."""
        key = f"book-progress-migration.{album_id}"
        pending = await self._provider.mass.cache.get(key=key, provider=self._provider.instance_id)
        if pending:
            await import_journal(self._provider, album_id, pending)
            await self._provider.mass.cache.set(
                key=key, provider=self._provider.instance_id, data=[]
            )

    async def _book_changed(self, old: Audiobook, new: Audiobook) -> bool:
        def fields(book: Audiobook) -> tuple[Any, ...]:
            return (
                book.name,
                book.publisher,
                book.metadata.description,
                sorted(x.path for x in book.metadata.images or []),
                next(
                    (
                        x.details
                        for x in book.provider_mappings
                        if x.provider_instance == self._provider.instance_id
                    ),
                    None,
                ),
            )

        if fields(old) != fields(new) or not self._provider._check_provider_mappings(
            old, new, True
        ):
            return True
        if len(old.narrators) != len(new.narrators):
            return True
        for previous, incoming in zip(old.narrators, new.narrators, strict=True):
            if isinstance(previous, str) or isinstance(incoming, str):
                if previous != incoming:
                    return True
                continue
            if previous.name != incoming.name:
                return True
            if previous.provider == "library":
                linked = await self._provider.mass.music.artists.get_library_item_by_prov_id(
                    incoming.item_id, self._provider.instance_id
                )
                if linked is None or linked.item_id != previous.item_id:
                    return True
            elif previous.provider != incoming.provider or previous.item_id != incoming.item_id:
                return True
        return False

    async def _migrate_podcast(self, album_id: str, book: Audiobook) -> None:
        """Hide a migrated podcast only after its book exists; keep all episode playlogs."""
        await self._provider._migrate_local(
            album_id, book, self._provider.mass.music.podcasts, self._provider.mass.music.audiobooks
        )

    async def _migrate_local(
        self,
        album_id: str,
        target: Audiobook | Podcast,
        controller: AudiobooksController | PodcastsController,
        target_controller: AudiobooksController | PodcastsController,
    ) -> None:
        old = await controller.get_library_item_by_prov_id(album_id, self._provider.instance_id)
        if not old:
            return
        mappings = [
            x for x in old.provider_mappings if x.provider_instance == self._provider.instance_id
        ]
        key = f"migration.{target.media_type.value}.{album_id}"
        if not any(x.in_library for x in mappings):
            journal = await self._provider.mass.cache.get(
                key=key, provider=self._provider.instance_id
            )
            if journal and journal.get("pending"):
                await self._provider._clear_search_results()
                if old.favorite and not any(x.in_library for x in old.provider_mappings):
                    await controller.set_favorite(old.item_id, False)
                await self._provider.mass.cache.set(
                    key=key, provider=self._provider.instance_id, data={**journal, "pending": False}
                )
            return
        if old.favorite and not target.favorite:
            await target_controller.set_favorite(target.item_id, True)
        # Save a migration journal before changes, to make interrupted migrations repeatable.
        await self._provider.mass.cache.set(
            key=key,
            provider=self._provider.instance_id,
            data={
                "source_id": old.item_id,
                "target_id": target.item_id,
                "favorite": old.favorite,
                "pending": True,
            },
        )
        for mapping in mappings:
            mapping.in_library = False
        await controller.set_provider_mappings(old.item_id, old.provider_mappings)
        # Searches made while the migration was running can still contain the old type.
        await self._provider._clear_search_results()
        if old.favorite and not any(x.in_library for x in old.provider_mappings):
            await controller.set_favorite(old.item_id, False)
        await self._provider.mass.cache.set(
            key=key,
            provider=self._provider.instance_id,
            data={"source_id": old.item_id, "target_id": target.item_id, "pending": False},
        )

    def _start_queue_chapters(self) -> None:
        self._provider._remove_queue_listener = self._provider.mass.subscribe(
            self._provider._queue_chapters_changed,
            (EventType.QUEUE_ADDED, EventType.QUEUE_ITEMS_UPDATED),
        )
        for queue in self._provider.mass.player_queues.all():
            self._provider._schedule_queue_chapters(queue.queue_id)

    def _queue_chapters_changed(self, event: MassEvent) -> None:
        if event.object_id:
            self._provider._schedule_queue_chapters(event.object_id)

    def _schedule_queue_chapters(self, queue_id: str) -> None:
        if self._provider._skip_unloaded:
            return
        self._provider._queue_chapter_pending.add(queue_id)
        if queue_id not in self._provider._queue_chapter_tasks:
            self._provider._queue_chapter_tasks[queue_id] = asyncio.create_task(
                self._provider._hydrate_queue_chapters(queue_id)
            )

    def _queued_books(self, queue_id: str) -> list[tuple[QueueItem, Audiobook, str]]:
        controller = self._provider.mass.player_queues
        queue = controller.get(queue_id)
        if queue is None:
            return []
        result = []
        for item in controller.items(queue_id, limit=queue.items):
            book = item.media_item
            if not isinstance(book, Audiobook) or book.metadata.chapters:
                continue
            album_id = next(
                (
                    mapping.item_id
                    for mapping in book.provider_mappings
                    if mapping.provider_instance == self._provider.instance_id
                ),
                None,
            )
            if album_id is not None:
                result.append((item, book, album_id))
        return result

    def _fill_queue_chapters(self, queue_id: str, album_id: str, chapters: Chapters | None) -> None:
        if chapters is None:
            return
        for item, book, ident in self._provider._queued_books(queue_id):
            if ident == album_id:
                updated = deepcopy(book)
                self._provider._apply_chapters(updated, chapters)
                # Stable 2.10.6 exposes live QueueItem models through items().
                # Only hydrate missing metadata; never change playback clocks.
                item.media_item = updated
                item.duration = updated.duration
                self._provider.mass.player_queues.signal_update(queue_id, items_changed=True)

    async def _hydrate_queue_chapters(self, queue_id: str) -> None:
        try:
            while (
                queue_id in self._provider._queue_chapter_pending
                and not self._provider._skip_unloaded
            ):
                self._provider._queue_chapter_pending.discard(queue_id)
                albums = dict.fromkeys(
                    ident for _, _, ident in self._provider._queued_books(queue_id)
                )
                for album_id in albums:
                    try:
                        chapters = await self._provider._chapters(album_id)
                        assert chapters is not None
                        if not self._provider._skip_unloaded:
                            self._provider._fill_queue_chapters(queue_id, album_id, chapters)
                    except MusicAssistantError, CarError:
                        self._provider.logger.warning(
                            "队列有声书 %s 章节暂时读取失败，播放时将重试", album_id
                        )
        finally:
            self._provider._queue_chapter_tasks.pop(queue_id, None)

    async def _book_stream(self, album_id: str) -> StreamDetails:
        started = time.monotonic()
        phases = {}

        async def measured[T](name: str, operation: Awaitable[T]) -> T:
            before = time.monotonic()
            try:
                return await operation
            finally:
                phases[name] = round((time.monotonic() - before) * 1000)

        chapter_task = asyncio.create_task(measured("章节目录", self._provider._chapters(album_id)))
        progress_task = asyncio.create_task(
            measured("云端进度", self._provider._begin_progress(album_id, defer=True))
        )
        tasks: list[asyncio.Task[Any]] = [
            chapter_task,
            progress_task,
        ]
        if self._provider.config.get_value("skip_sync", True):
            tasks.append(
                asyncio.create_task(
                    measured("片头片尾", self._provider._skip_settings(album_id, refresh=True))
                )
            )
        try:
            await asyncio.gather(*tasks)
            chapters, progress = chapter_task.result(), progress_task.result()
            assert chapters is not None
        finally:
            # A failed/cancelled preparation must not leave cloud requests running.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # The recent/search shelf stays lightweight. Once playback needs the timeline,
        # expose that same timeline on the queued book so MA can display its chapters.
        for queue in self._provider.mass.player_queues.all():
            self._provider._fill_queue_chapters(queue.queue_id, album_id, chapters)
        await self._provider._prepare_skip(
            MediaType.AUDIOBOOK,
            album_id,
            album_id,
            chapters.duration,
            chapters,
            settings_ready=True,
        )
        now = time.monotonic()
        self._provider._part_sessions = {
            k: v for k, v in self._provider._part_sessions.items() if v[0] > now
        }
        self._provider._part_users = {
            k: v
            for k, v in self._provider._part_users.items()
            if k in self._provider._part_sessions
        }
        self._provider._part_books = {
            k: v
            for k, v in self._provider._part_books.items()
            if k in self._provider._part_sessions
        }
        self._provider._failed_parts.intersection_update(self._provider._part_sessions)
        if len(self._provider._part_sessions) >= 64:
            raise ResourceTemporarilyUnavailable("同时播放的有声书过多，请稍后重试")
        token = secrets.token_urlsafe(32)
        self._provider._part_sessions[token] = (now + 86400, {str(x["id"]) for x in chapters.rows})
        self._provider._part_books[token] = (album_id, chapters)
        user = await progress_user(self._provider)
        self._provider._part_users[token] = user.user_id if user else None
        parts = [
            MultiPartPath(
                path=f"{self._provider.mass.streams.base_url}/{self._provider.instance_id}_chapter?session={token}&id={row['id']}",
                duration=row["duration"],
            )
            for row in chapters.rows
        ]
        if self._provider.config.get_value("cloud_progress_write", False):
            if progress is None:
                self._provider._progress_sessions.pop(album_id, None)
            else:
                progress.update(time=time.monotonic(), wall=int(time.time() * 1000))
                self._provider._progress_sessions[album_id] = progress
        self._provider.logger.info(
            "有声书 %s 起播资料准备 %s ms（%s，并行）",
            album_id,
            round((time.monotonic() - started) * 1000),
            ", ".join(f"{name}={ms} ms" for name, ms in phases.items()),
        )
        return StreamDetails(
            provider=self._provider.instance_id,
            item_id=album_id,
            media_type=MediaType.AUDIOBOOK,
            stream_type=StreamType.CUSTOM,
            audio_format=AudioFormat(content_type=ContentType.M4A),
            decoded_audio_format=pcm_format(),
            path=parts,
            data={"chapters": chapters, "relay_session": token},
            duration=chapters.duration,
            can_seek=True,
            allow_seek=True,
            expiration=0,
        )

    async def _skip_settings(self, album_id: str, *, refresh: bool = False) -> tuple[int, int]:
        if not self._provider.config.get_value("skip_sync", True):
            return (0, 0)
        lock = self._provider._skip_config_locks.setdefault(album_id, asyncio.Lock())
        async with lock:
            cached = self._provider._skip_configs.get(album_id)
            if cached and cached[0] > time.monotonic() and not refresh:
                return cached[1]
            try:
                settings = await self._provider._client.skip_config(album_id)
            except CarError, MusicAssistantError:
                self._provider.logger.warning("云端片头片尾设置暂不可用，本次不自动跳过")
                settings = (0, 0)
            if len(self._provider._skip_configs) >= 64:
                self._provider._skip_configs.pop(next(iter(self._provider._skip_configs)))
            self._provider._skip_configs[album_id] = (time.monotonic() + 30, settings)
            return settings

    async def _prepare_skip(
        self,
        media_type: MediaType,
        ident: str,
        album_id: str,
        duration: float,
        chapters: Chapters | None = None,
        *,
        settings_ready: bool = False,
    ) -> None:
        if not self._provider.config.get_value("skip_sync", True):
            return
        if not settings_ready:
            await self._provider._skip_settings(album_id, refresh=True)
        if len(self._provider._skip_streams) >= 128:
            self._provider._skip_streams.pop(next(iter(self._provider._skip_streams)))
        self._provider._skip_streams[(media_type, ident)] = (album_id, duration, chapters)

    @staticmethod
    def _skip_target(
        position: float, duration: float, chapters: Chapters | None, settings: tuple[int, int]
    ) -> int | None:
        """Keep original media time: jump over configured ranges, never shorten the book."""
        head, tail = settings
        if not (head or tail) or not 0 <= position < duration:
            return None
        if chapters:
            index = min(bisect_right(chapters.starts, position) - 1, len(chapters.rows) - 1)
            start, length = chapters.starts[index], chapters.rows[index]["duration"]
        else:
            index, start, length = 0, 0, duration
        # A short trailer/chapter must not be skipped in its entirety by album-wide settings.
        if head + tail >= length:
            return None
        if head and position < start + head:
            return math.ceil(start + head)
        if tail and position >= start + length - tail:
            if chapters and index + 1 < len(chapters.rows):
                next_length = chapters.rows[index + 1]["duration"]
                return math.ceil(
                    chapters.starts[index + 1] + (head if head + tail < next_length else 0)
                )
            return math.ceil(duration)
        return None

    async def _skip_monitor(self) -> None:
        # MA suppresses time events during ordinary playback; inspect its clock locally.
        # No network polling per tick: account settings have their own 30-second cache.
        last_warning = -math.inf
        while not self._provider._skip_unloaded:
            results = await asyncio.gather(
                *(
                    self._provider._skip_queue(queue.queue_id)
                    for queue in self._provider.mass.player_queues.all()
                ),
                return_exceptions=True,
            )
            # An API or queue failure must not permanently disable automatic skipping.
            # Do not include exception text: transport errors can contain private URLs.
            for result in results:
                if isinstance(result, Exception) and time.monotonic() - last_warning >= 30:
                    self._provider.logger.warning(
                        "片头片尾检查暂时失败，将自动重试（%s）", type(result).__name__
                    )
                    last_warning = time.monotonic()
            await asyncio.sleep(0.5)

    def _skip_transitioning(self, queue_id: str) -> bool:
        queue = self._provider.mass.player_queues.get(queue_id)
        monitor = getattr(self._provider, "_local_chapter_resume", None)
        return (
            not queue
            or queue.state != PlaybackState.PLAYING
            or getattr(queue, "extra_attributes", {}).get("play_action_in_progress", False)
            or bool(monitor and monitor.transitioning(queue_id))
        )

    async def _skip_queue(self, queue_id: str) -> None:
        if (
            not self._provider.config.get_value("skip_sync", True)
            or self._provider._skip_unloaded
            or self._provider._skip_transitioning(queue_id)
        ):
            return
        lock = self._provider._skip_queue_locks.setdefault(queue_id, asyncio.Lock())
        if lock.locked():
            return
        async with lock:
            queue = self._provider.mass.player_queues.get(queue_id)
            if not queue or not queue.active or queue.state != PlaybackState.PLAYING:
                return
            item = queue.current_item
            stream = item.streamdetails if item else None
            if (
                item is None
                or not stream
                or stream.provider != self._provider.instance_id
                or stream.stream_error
            ):
                return
            spec = self._provider._skip_streams.get((stream.media_type, stream.item_id))
            if spec is None:
                return
            album_id, duration, chapters = spec
            settings = await self._provider._skip_settings(album_id)
            # The user can switch media, pause, or unload the provider during the request.
            current = self._provider.mass.player_queues.get(queue_id)
            if (
                self._provider._skip_unloaded
                or self._provider._skip_transitioning(queue_id)
                or not current
                or not current.active
                or current.state != PlaybackState.PLAYING
                or not current.current_item
                or current.current_item.queue_item_id != item.queue_item_id
                or current.current_item.streamdetails is not stream
            ):
                return
            target = self._provider._skip_target(
                current.corrected_elapsed_time, duration, chapters, settings
            )
            if target is None:
                return
            signature = (item.queue_item_id, target)
            previous = self._provider._skip_attempts.get(queue_id)
            if previous and previous[0] == signature and time.monotonic() - previous[1] < 5:
                return
            self._provider._skip_attempts[queue_id] = (signature, time.monotonic())
            try:
                if target < duration:
                    await self._provider.mass.player_queues.seek(queue_id, target)
                elif current.next_item:
                    if monitor := getattr(self._provider, "_local_chapter_resume", None):
                        await monitor.mark_tail_completed(queue_id)
                    await self._provider.mass.player_queues.next(queue_id)
                else:
                    if monitor := getattr(self._provider, "_local_chapter_resume", None):
                        await monitor.mark_tail_completed(queue_id)
                    await self._provider.mass.player_queues.stop(queue_id)
            except MusicAssistantError:
                self._provider.logger.warning("播放器暂时无法执行片头片尾跳转")

    async def _chapter_request(self, request: web.Request) -> web.StreamResponse:
        token, ident = request.query.get("session", ""), request.query.get("id", "")
        session = self._provider._part_sessions.get(token)
        if not session or session[0] < time.monotonic() or ident not in session[1]:
            return web.Response(status=404)
        self._provider._part_sessions[token] = (time.monotonic() + 86400, session[1])
        return await proxy_chapter(
            self._provider, request, ident, token, raw=request.query.get("raw") == "1"
        )

    @staticmethod
    def _stream_has_token(details: StreamDetails | None, token: str) -> bool:
        if not details:
            return False
        paths = [x.path for x in details.path] if isinstance(details.path, list) else [details.path]
        return any(isinstance(path, str) and f"session={token}&" in path for path in paths)

    def _cache_owner(self, token: str) -> str | tuple[str, str | None]:
        for queue in self._provider.mass.player_queues.all():
            for item in (queue.current_item, getattr(queue, "next_item", None)):
                if item and self._provider._stream_has_token(item.streamdetails, token):
                    return (queue.queue_id, self._provider._part_users.get(token))
        return token

    def _cache_remaining(self, token: str, ident: str, estimated: float) -> float:
        # Account for skipped tails without treating decoder read-ahead as actual
        # listening progress. Skip values have already been loaded for playback.
        if book := self._provider._part_books.get(token):
            album_id, _ = book
            settings = self._provider._skip_configs.get(album_id)
            if settings and self._provider.config.get_value("skip_sync", True):
                estimated = max(0, estimated - settings[1][1])
        return estimated

    async def _next_cache_track(self, token: str, ident: str) -> tuple[str, int] | None:
        if book := self._provider._part_books.get(token):
            album_id, chapters = book
            for index, row in enumerate(chapters.rows[:-1]):
                if str(row["id"]) == ident:
                    head, _ = await self._provider._skip_settings(album_id)
                    return str(chapters.rows[index + 1]["id"]), head
            return None
        # A podcast's successor is the actual queue item, never album ID + 1 or
        # a guessed chronological order. Do not prefetch other providers.
        for queue in self._provider.mass.player_queues.all():
            current, next_item = queue.current_item, getattr(queue, "next_item", None)
            if not current or not self._provider._stream_has_token(current.streamdetails, token):
                continue
            media = next_item.media_item if next_item else None
            if not media or media.media_type != MediaType.PODCAST_EPISODE:
                return None
            for mapping in media.provider_mappings:
                if mapping.provider_instance == self._provider.instance_id:
                    track = await self._provider._checked(
                        self._provider._client.track(mapping.item_id)
                    )
                    head, _ = await self._provider._skip_settings(str(track["album_id"]))
                    return str(mapping.item_id), head
        return None

    async def _episode_cache_path(self, ident: str) -> str:
        now = time.monotonic()
        self._provider._part_sessions = {
            k: v for k, v in self._provider._part_sessions.items() if v[0] > now
        }
        self._provider._part_users = {
            k: v
            for k, v in self._provider._part_users.items()
            if k in self._provider._part_sessions
        }
        self._provider._part_books = {
            k: v
            for k, v in self._provider._part_books.items()
            if k in self._provider._part_sessions
        }
        self._provider._failed_parts.intersection_update(self._provider._part_sessions)
        if len(self._provider._part_sessions) >= 64:
            raise ResourceTemporarilyUnavailable("同时播放的音频过多，请稍后重试")
        token = secrets.token_urlsafe(32)
        self._provider._part_sessions[token] = (now + 86400, {ident})
        user = await progress_user(self._provider)
        self._provider._part_users[token] = user.user_id if user else None
        return (
            f"{self._provider.mass.streams.base_url}/{self._provider.instance_id}_chapter"
            f"?session={token}&id={ident}"
        )

    async def _chapter_failed(self, token: str, ident: str, reason: str) -> None:
        if token in self._provider._failed_parts or self._provider._skip_unloaded:
            return
        self._provider._failed_parts.add(token)
        # Stopping the player closes this HTTP handler. Finish saving its position
        # in a separate task so that cancellation cannot discard the checkpoint.
        task = asyncio.create_task(self._provider._protect_failed_playback(token, ident, reason))
        self._provider._recovery_tasks.add(task)
        task.add_done_callback(self._provider._recovery_tasks.discard)
        await asyncio.shield(task)

    async def _recovery_monitor(self) -> None:
        # MA does not invoke on_streamed for errors before 90 seconds of audio.
        # Observe only our current stream; no network or playback writes on healthy ticks.
        while not self._provider._skip_unloaded:
            self._provider._audio_cache.prune_queues()
            for queue in self._provider.mass.player_queues.all():
                current = queue.current_item
                details = current.streamdetails if current else None
                if (
                    details
                    and details.provider == self._provider.instance_id
                    and details.stream_error
                ):
                    await self._provider.on_streamed(details)
            await asyncio.sleep(0.5)

    async def _protect_failed_playback(self, token: str, ident: str, reason: str) -> None:
        for queue in self._provider.mass.player_queues.all():
            current = queue.current_item
            details = current.streamdetails if current else None
            if (
                current is None
                or not queue.active
                or not details
                or details.provider != self._provider.instance_id
                or not self._provider._stream_has_token(details, token)
            ):
                continue
            position = int(max(0, queue.corrected_elapsed_time))
            if self._provider._skip_transitioning(queue.queue_id):
                position = int(details.seek_position or 0)
            details.stream_error = True
            user_id = self._provider._part_users.get(token)
            recovery_id = (
                f"episode.{details.item_id}"
                if details.media_type == MediaType.PODCAST_EPISODE
                else details.item_id
            )
            try:
                await self._provider.mass.cache.set(
                    key=f"stream-recovery-account.{recovery_id}",
                    provider=self._provider.instance_id,
                    data={"position": position, "timestamp": time.time()},
                    persistent=True,
                )
            except Exception:
                self._provider.logger.error(
                    "断流恢复点缓存写入失败，继续保留队列位置并写入播放记录"
                )
            if queue.current_item is not current or current.streamdetails is not details:
                continue
            # Explicit stop tears down the failed flow before MA interprets its EOF
            # as the end of the entire book. It retains the queue and resume position.
            try:
                async with asyncio.timeout(10):
                    await self._provider.mass.player_queues.stop(queue.queue_id)
            except MusicAssistantError, TimeoutError:
                self._provider.logger.warning("播放器未确认停止，已保存断流恢复点")
            # A user may have changed tracks while stop/save was awaiting I/O.
            if queue.current_item is not current or current.streamdetails is not details:
                continue
            # MA 2.10.6 stop retains its queue position. Keep our durable checkpoint
            # independently; do not rewrite the controller's clock or ended flags.
            await save_bookmark(
                self._provider, details.media_type, details.item_id, position, False
            )
            if user_id and current.media_item:
                try:
                    await self._provider.mass.music.mark_item_played(
                        current.media_item,
                        fully_played=False,
                        seconds_played=position,
                        userid=user_id,
                        queue_id=queue.queue_id,
                        user_initiated=False,
                    )
                except Exception:
                    self._provider.logger.error(
                        "断流播放记录写入失败，恢复点保留在队列与 Provider 缓存中"
                    )
            message = (
                f"{current.name}：章节 {ident} 读取失败，已停止并保留断点；请稍后继续播放。{reason}"
            )
            self._provider.logger.error(message)

            async def report_failure(message: str = message) -> None:
                """Preserve progress once when a consumed stream fails."""
                raise ResourceTemporarilyUnavailable(message)

            self._provider.mass.tasks.run_background_task(
                name="喜马拉雅播放中断：已保留断点",
                handler=report_failure,
                task_id=f"ximalaya-stream-error-{self._provider.instance_id}-{queue.queue_id}",
                user_id=user_id,
                allow_retry=False,
                priority=True,
            )

    async def _local_progress(
        self, album_id: str, chapters: Chapters | None
    ) -> tuple[bool, int, float] | None:
        return await read_bookmark(self._provider, album_id, chapters)

    async def _begin_progress(
        self, album_id: str, *, session_key: str | None = None, defer: bool = False
    ) -> dict[str, Any] | None:
        if not self._provider.config.get_value("cloud_progress_write", False):
            return None
        key = session_key or album_id
        try:
            baseline = await self._provider._client.cloud_progress(album_id)
        except CarError:
            self._provider.logger.warning("无法确认云端进度，本次播放不写回")
            if not defer:
                self._provider._progress_sessions.pop(key, None)
            return None
        state = {
            "album_id": album_id,
            "baseline": baseline,
            "time": time.monotonic(),
            "wall": int(time.time() * 1000),
            "last_position": None,
            "blocked": False,
        }
        if not defer:
            self._provider._progress_sessions[key] = state
        return state

    @property
    def _provider(self) -> XimalayaProvider:
        """Return the concrete provider hosting these operations."""
        return cast("XimalayaProvider", self)
