"""China-only Ximalaya provider for Music Assistant."""

from __future__ import annotations

import asyncio
import html
import json
import re
from collections.abc import AsyncGenerator, Awaitable, Callable

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

import aiohttp
from music_assistant.controllers.music.constants import CACHE_CATEGORY_SEARCH_RESULTS
from music_assistant.controllers.music.media.audiobooks import AudiobooksController
from music_assistant.controllers.music.media.podcasts import PodcastsController
from music_assistant.models.music_provider import CACHE_CATEGORY_PREV_LIBRARY_IDS, MusicProvider
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import (
    ArtistType,
    ConfigEntryType,
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    ResourceTemporarilyUnavailable,
)
from music_assistant_models.media_items import (
    Artist,
    Audiobook,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    MediaItemMetadata,
    MediaItemType,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
    RecommendationFolder,
    SearchResults,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from .audio_cache import AudioCache
from .client import CarClient, cdn_url
from .discovery import DiscoveryMixin
from .labels import config_entry
from .longform import Chapters, LongformMixin
from .normalization import Normalizer
from .protocol import AuthenticationError, CarError, NotFound, PermissionDenied, item_id

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

SUPPORTED_FEATURES = {
    ProviderFeature.SEARCH,
    ProviderFeature.RECOMMENDATIONS,
    ProviderFeature.NARRATOR_AUDIOBOOKS,
    ProviderFeature.BROWSE,
    ProviderFeature.LIBRARY_PODCASTS,
    ProviderFeature.LIBRARY_AUDIOBOOKS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> XimalayaProvider:
    """Construct the provider; MA invokes handle_async_init afterwards."""
    features = set(SUPPORTED_FEATURES)
    if config.get_value("subscription_write", False):
        features.add(ProviderFeature.LIBRARY_AUDIOBOOKS_EDIT)
        features.add(ProviderFeature.LIBRARY_PODCASTS_EDIT)
    return XimalayaProvider(mass, manifest, config, features)


async def checked[T](operation: Awaitable[T]) -> T:
    """Translate only sanitized protocol errors to Music Assistant errors."""
    try:
        return await operation
    except AuthenticationError as err:
        raise LoginFailed(str(err)) from None
    except (PermissionDenied, NotFound) as err:
        raise MediaNotFoundError(str(err)) from None
    except CarError as err:
        raise ResourceTemporarilyUnavailable(str(err)) from None


class XimalayaProvider(DiscoveryMixin, LongformMixin, MusicProvider):
    """Explicit podcast tags select podcasts; other albums use native audiobooks."""

    _catalogs: dict[str, tuple[float, Chapters, str | None]]
    _catalog_locks: WeakValueDictionary[str, asyncio.Lock]
    _part_sessions: dict[str, tuple[float, set[str]]]
    _part_users: dict[str, str | None]
    _part_books: dict[str, tuple[str, Chapters]]
    _audio_cache: AudioCache
    _normalizer: Normalizer
    _failed_parts: set[str]
    _recovery_tasks: set[asyncio.Task[None]]
    _queue_chapter_tasks: dict[str, asyncio.Task[None]]
    _queue_chapter_pending: set[str]
    _progress_sessions: dict[str, dict[str, Any]]
    _progress_lock: asyncio.Lock
    _skip_unloaded: bool
    _skip_configs: dict[str, tuple[float, tuple[int, int]]]
    _skip_config_locks: WeakValueDictionary[str, asyncio.Lock]
    _skip_streams: dict[tuple[MediaType, str], tuple[str, float, Chapters | None]]
    _skip_queue_locks: WeakValueDictionary[str, asyncio.Lock]
    _skip_attempts: dict[str, tuple[tuple[Any, ...], float]]
    _remove_part_route: Callable[[], None] | None
    _remove_queue_listener: Callable[[], None] | None
    _recovery_task: asyncio.Task[None] | None
    _skip_task: asyncio.Task[None] | None
    _subscription_task: asyncio.Task[None] | None
    _recent_albums: dict[str, dict[str, Any]]

    _session: aiohttp.ClientSession | None = None
    _checked = staticmethod(checked)

    @property
    def instance_name_postfix(self) -> str | None:
        """Identify accounts using MA's supported instance-name extension."""
        nickname = getattr(getattr(self, "_client", None), "nickname", None)
        return nickname if isinstance(nickname, str) and nickname else None

    @property
    def is_streaming_provider(self) -> bool:
        # Entitlements and subscriptions differ across accounts. Avoid cross-account lookups.
        """Keep account-specific catalogs and entitlements isolated."""
        return False

    @property
    def max_concurrent_streams(self) -> int:
        """Retain MA's default streaming guard while keeping account lookup isolated."""
        return 5

    @property
    def supported_media_types(self) -> set[MediaType]:
        """Return the media kinds exposed by this provider."""
        return {
            MediaType.ARTIST,
            MediaType.AUDIOBOOK,
            MediaType.PODCAST,
            MediaType.PODCAST_EPISODE,
            MediaType.RADIO,
        }

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return settings localized through the provider string catalog."""
        return (
            config_entry(
                key="account_identity",
                type=ConfigEntryType.LABEL,
                translation_params=[self.instance_name_postfix or ""],
            ),
            config_entry(
                key="audio_quality",
                type=ConfigEntryType.STRING,
                default_value="highest",
                options=[
                    ConfigValueOption(value="highest"),
                    ConfigValueOption(value="high"),
                    ConfigValueOption(value="standard"),
                ],
            ),
            config_entry(
                key="search_cache_minutes",
                type=ConfigEntryType.INTEGER,
                default_value=1,
                validate=lambda value: type(value) is int and value >= 1,
            ),
            config_entry(
                key="subscription_write",
                type=ConfigEntryType.BOOLEAN,
                default_value=False,
            ),
            config_entry(
                key="subscription_check_interval",
                type=ConfigEntryType.INTEGER,
                default_value=5,
                options=[ConfigValueOption(value=minutes) for minutes in (1, 5, 15, 30, 60)],
            ),
            config_entry(
                key="cloud_progress_write",
                type=ConfigEntryType.BOOLEAN,
                default_value=False,
            ),
            config_entry(
                key="skip_sync",
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
            ),
            # Override MA's generated controls without changing MA itself. Their
            # persisted values are derived from the single subscription switch.
            *(
                config_entry(
                    key=key,
                    type=ConfigEntryType.BOOLEAN,
                    default_value=True,
                    hidden=True,
                )
                for key in ("library_sync_audiobooks", "library_sync_podcasts", "library_sync_back")
            ),
            # Preserve an already configured sound-search entry without a fifth control.
            config_entry(
                key="sound_query",
                type=ConfigEntryType.STRING,
                required=False,
                hidden=True,
            ),
        )

    async def update_config(self, config: ProviderConfig, changed_keys: set[str]) -> None:
        """Apply user settings and align the library synchronization controls."""
        self.config = config
        self._align_subscription_options()
        await super().update_config(config, changed_keys)

    async def loaded_in_mass(self) -> None:
        """Start queue metadata hydration and subscription monitoring."""
        self._start_queue_chapters()
        # The built-in scheduler only supports whole hours. Keep its initial/manual
        # sync and trigger the same managed tasks on the provider's minute interval.
        self._subscription_task = (
            asyncio.create_task(self._subscription_monitor())
            if self.config.get_value("subscription_write", False)
            else None
        )

    async def sync_library(self, media_type: MediaType) -> None:
        """Synchronize subscriptions under the account write lock."""
        if not self.config.get_value("subscription_write", False):
            return
        # Serialize cloud snapshots and MA-initiated writes within this account.
        async with self._subscription_lock():
            await self._seed_manual_subscriptions(media_type)
            await super().sync_library(media_type)

    async def handle_async_init(self) -> None:
        """Connect the backend and initialize playback services."""
        try:
            credentials = json.loads(str(self.get_setup_value("credentials") or ""))
            if not isinstance(credentials, dict):
                raise TypeError
        except ValueError, TypeError:
            raise LoginFailed("请先配置并登录中国版喜马拉雅") from None
        self._session = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar(), trust_env=False)
        try:
            try:
                self._client = CarClient(
                    self._session,
                    credentials,
                    self._save_credentials,
                    audio_quality=str(self.config.get_value("audio_quality", "highest")),
                    search_cache_minutes=int(str(self.config.get_value("search_cache_minutes", 1))),
                )
            except CarError as err:
                raise LoginFailed(str(err)) from None
            await checked(self._client.start())
            self._align_subscription_options()
            self._init_longform()
            # Every settings reload must expire both provider and combined results.
            # A one-time migration marker leaves later config changes using stale types.
            await self._clear_search_results()
            self._skip_task = (
                asyncio.create_task(self._skip_monitor())
                if self.config.get_value("skip_sync", True)
                else None
            )
            self._recovery_task = asyncio.create_task(self._recovery_monitor())
        except BaseException:
            await self._session.close()
            self._session = None
            raise

    async def unload(self, is_removed: bool = False) -> None:
        """Stop provider tasks and release playback resources."""
        if task := getattr(self, "_subscription_task", None):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._subscription_task = None
        self._skip_unloaded = True
        if cache := getattr(self, "_audio_cache", None):
            await cache.close()
        if normalizer := getattr(self, "_normalizer", None):
            await normalizer.close()
        if remove_listener := getattr(self, "_remove_queue_listener", None):
            remove_listener()
            self._remove_queue_listener = None
        queue_tasks = list(getattr(self, "_queue_chapter_tasks", {}).values())
        for task in queue_tasks:
            task.cancel()
        if queue_tasks:
            await asyncio.gather(*queue_tasks, return_exceptions=True)
        if hasattr(self, "_queue_chapter_pending"):
            self._queue_chapter_pending.clear()
        if recovery_task := getattr(self, "_recovery_task", None):
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)
            self._recovery_task = None
        if skip_task := getattr(self, "_skip_task", None):
            skip_task.cancel()
            await asyncio.gather(skip_task, return_exceptions=True)
            self._skip_task = None
        if remove_route := getattr(self, "_remove_part_route", None):
            remove_route()
        if recovery_tasks := getattr(self, "_recovery_tasks", None):
            await asyncio.gather(*recovery_tasks, return_exceptions=True)
            self._remove_part_route = None
        if hasattr(self, "_part_sessions"):
            self._part_sessions.clear()
            self._part_users.clear()
            self._part_books.clear()
            self._progress_sessions.clear()
        if self._session:
            await self._session.close()
            self._session = None

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Search the requested media kinds within the current account."""
        results = SearchResults()
        if not search_query.strip() or limit <= 0:
            return results
        requested = (
            set(media_types)
            if media_types
            else {MediaType.ARTIST, MediaType.AUDIOBOOK, MediaType.PODCAST}
        )
        if MediaType.ARTIST in requested:
            rows = await checked(self._client.search(search_query, "users", limit))
            seen = set()
            for row in rows:
                artist = self._announcer(row)
                if artist.item_id not in seen:
                    results.artists = [*results.artists, artist]
                    seen.add(artist.item_id)
        if requested.intersection({MediaType.AUDIOBOOK, MediaType.PODCAST}):
            rows = await checked(self._client.search(search_query, "albums", limit))
            for row in rows:
                self._remember_album(row)
                kind = await self._album_media_type(row)
                if kind == MediaType.AUDIOBOOK and kind in requested:
                    results.audiobooks = [*results.audiobooks, self._book(row)]
                elif kind == MediaType.PODCAST and kind in requested:
                    results.podcasts = [*results.podcasts, self._podcast(row)]
        return results

    async def get_recommendations(self) -> list[RecommendationFolder]:
        # Static descriptors keep Discover fast; account reads happen only for the row.
        """Return localized recommendation descriptors without fetching their contents."""
        return [
            RecommendationFolder(
                item_id="recent_listening",
                translation_key="recent_listening",
                provider=self.instance_id,
                name="喜马拉雅 · 最近收听",
                subtitle="账号云端历史 · 最近 20 条",
                icon="mdi-history",
            ),
            RecommendationFolder(
                item_id="discovery",
                translation_key="discovery",
                provider=self.instance_id,
                name="喜马拉雅",
                subtitle="电台广播 · 热播榜单 · 猜你喜欢 · 详细分类",
                icon="mdi-compass-outline",
            ),
        ]

    async def get_recommendation_items(
        self, item_id: str
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """Load the selected recommendation row."""
        if item_id == "discovery":
            return UniqueList(self._discovery_entries())
        if item_id != "recent_listening":
            raise MediaNotFoundError("喜马拉雅推荐模块不存在")
        rows = await checked(self._client.recent_history())
        items: UniqueList[MediaItemType | ItemMapping | BrowseFolder] = UniqueList()
        for row in rows:
            album = row["album"]
            if await self._album_media_type(album) == MediaType.AUDIOBOOK:
                # Never crawl full chapter timelines while rendering a history shelf.
                items.append(self._book(album))
            else:
                # For podcasts/legacy albums, open the actual last-heard episode.
                items.append(self._episode(row["track"]))
        return items

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Return an announcer as a native MA artist."""
        return self._announcer(await checked(self._client.announcer(prov_artist_id)))

    async def get_narrator_audiobooks(self, prov_artist_id: str) -> list[Audiobook]:
        """Return albums narrated by the requested announcer."""
        rows = await checked(self._client.announcer_albums(prov_artist_id))
        books = []
        for row in rows:
            if await self._album_media_type(row) == MediaType.AUDIOBOOK:
                books.append(self._book(row))
        return books

    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """Return podcast metadata."""
        return self._podcast(await checked(self._client.album(prov_podcast_id)))

    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """Return episode metadata."""
        return self._episode(await checked(self._client.track(prov_episode_id)))

    async def get_podcast_episodes(self, prov_podcast_id: str) -> AsyncGenerator[PodcastEpisode]:
        """Yield the podcast episodes in catalog order."""
        rows = await checked(self._client.episodes(prov_podcast_id))
        items = [self._episode(row) for row in rows]
        for item in items:
            yield item

    async def get_library_podcasts(self) -> AsyncGenerator[Podcast]:
        """Yield the subscribed podcasts."""
        rows = await self._subscription_rows(MediaType.PODCAST)
        items = [self._podcast(row) for row in rows]
        for item in items:
            yield item

    async def browse(self, path: str) -> list[MediaItemType | BrowseFolder | ItemMapping]:
        """Return the requested navigation level."""
        root = f"{self.instance_id}://"
        if path == f"{root}discovery" or path.startswith(f"{root}discovery/"):
            return await self._browse_discovery(path[len(root) :])
        if path == root:
            return [
                self._discovery_folder("subscriptions", "喜马拉雅订阅", "subscriptions"),
                self._discovery_folder("recent_listening", "最近收听", "recent_listening"),
                self._discovery_folder("discovery", "分类显示", "discovery"),
            ]
        if path == f"{root}recent_listening":
            return list(await self.get_recommendation_items("recent_listening"))
        if path == f"{root}chapters":
            return list(await self._chapter_folders())
        if path.startswith(f"{root}chapters/"):
            # Keep the whole page address in one path segment: MA adds its own
            # parent folder by stripping the last slash. Also accept old links.
            parts = path[len(root + "chapters/") :].replace("/", "_").split("_")
            if (
                len(parts) not in (1, 2, 3)
                or any(not s.isascii() or not s.isdigit() for s in parts)
                or int(parts[0]) <= 0
            ):
                raise MediaNotFoundError("章节路径无效")
            album_id = item_id(parts[0])
            offset = int(parts[1]) if len(parts) >= 2 else 0
            expected = int(parts[2]) if len(parts) == 3 else None
            page = await checked(self._client.episode_page(album_id, offset, expected))
            result: list[MediaItemType | BrowseFolder | ItemMapping] = [
                self._episode(row) for row in page.items
            ]
            if offset:
                result.insert(
                    0,
                    self._chapter_page_folder(album_id, max(0, offset - 50), page.total, "上一页"),
                )
            if offset + len(page.items) < page.total:
                result.append(
                    self._chapter_page_folder(album_id, offset + 50, page.total, "下一页")
                )
            return result
        if path == f"{root}subscriptions":
            rows = await checked(self._client.subscription_preview())
            result = []
            for row in rows:
                self._remember_album(row)
                kind = await self._album_media_type(row)
                result.append(
                    self._book(row) if kind == MediaType.AUDIOBOOK else self._podcast(row)
                )
            return result
        if path == f"{root}sounds":
            query = str(self.get_config_value("sound_query") or "").strip()
            if not query:
                return []
            rows = await checked(self._client.search(query, "tracks", 20))
            return [self._episode(row) for row in rows]
        raise MediaNotFoundError("喜马拉雅浏览路径不存在")

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Resolve an authorized stream for server-side playback."""
        if media_type == MediaType.RADIO:
            return await self._radio_stream(item_id)
        if media_type == MediaType.AUDIOBOOK:
            return await self._book_stream(item_id)
        if media_type != MediaType.PODCAST_EPISODE:
            raise MediaNotFoundError("请选择具体声音播放")
        _, duration = await checked(self._client.resolve(item_id))
        track = await checked(self._client.track(item_id))
        await self._prepare_skip(
            MediaType.PODCAST_EPISODE, item_id, str(track["album_id"]), duration
        )
        if self.config.get_value("cloud_progress_write", False):
            await self._begin_progress(str(track["album_id"]), session_key=f"episode:{item_id}")
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            media_type=MediaType.PODCAST_EPISODE,
            stream_type=StreamType.HTTP,
            audio_format=AudioFormat(content_type=ContentType.M4A),
            path=await self._episode_cache_path(item_id),
            duration=round(duration),
            can_seek=True,
            allow_seek=True,
            expiration=0,
        )

    def _align_subscription_options(self) -> None:
        """Keep MA's scheduler and write-back gates tied to the visible master switch."""
        enabled = bool(self.config.get_value("subscription_write", False))
        for key in ("library_sync_audiobooks", "library_sync_podcasts", "library_sync_back"):
            self._update_config_value(key, enabled)
            if key in self.config.values:
                self.config.values[key].value = enabled

    def _subscription_interval_seconds(self) -> int:
        minutes = self.config.get_value("subscription_check_interval", 5)
        if type(minutes) is not int or minutes not in (1, 5, 15, 30, 60):
            minutes = 5
        return minutes * 60

    async def _subscription_monitor(self) -> None:
        while True:
            await asyncio.sleep(self._subscription_interval_seconds())
            if not self.config.get_value("subscription_write", False):
                continue
            try:
                await self.mass.music.start_sync(
                    media_types=[MediaType.AUDIOBOOK, MediaType.PODCAST],
                    providers=[self.instance_id],
                )
            except Exception:
                # Keep the timer alive after a temporary scheduling failure; no deletion
                # or account writes happen here. MA reports actual sync errors itself.
                self.logger.warning("订阅检查暂时无法启动，将在下个周期重试")

    def _subscription_lock(self) -> asyncio.Lock:
        if not hasattr(self, "_subscription_operation_lock"):
            self._subscription_operation_lock = asyncio.Lock()
        return self._subscription_operation_lock

    async def _remember_subscription_write(
        self, ident: str, media_type: MediaType, subscribed: bool
    ) -> None:
        # A cloud cancellation may precede the first scheduled import. Remember MA's
        # confirmed additions until their local row can enter the sync checkpoint.
        key = f"subscription-pending-{media_type.value}"
        pending = set(await self.mass.cache.get(key=key, provider=self.instance_id) or [])
        if subscribed:
            pending.add(ident)
        else:
            pending.discard(ident)
        await self.mass.cache.set(
            key=key, provider=self.instance_id, data=sorted(pending), persistent=True
        )

    async def _seed_manual_subscriptions(self, media_type: MediaType) -> None:
        key = f"subscription-pending-{media_type.value}"
        pending = await self.mass.cache.get(key=key, provider=self.instance_id) or []
        if not pending:
            return
        previous = set(
            await self.mass.cache.get(
                key=media_type.value,
                provider=self.instance_id,
                category=CACHE_CATEGORY_PREV_LIBRARY_IDS,
            )
            or []
        )
        remaining = []
        controller = self.mass.music.get_controller(media_type)
        for ident in pending:
            item = await controller.get_library_item_by_prov_id(ident, self.instance_id)
            if item:
                previous.add(int(item.item_id))
            else:
                remaining.append(ident)
        await self.mass.cache.set(
            key=media_type.value,
            provider=self.instance_id,
            category=CACHE_CATEGORY_PREV_LIBRARY_IDS,
            data=list(previous),
        )
        await self.mass.cache.set(
            key=key, provider=self.instance_id, data=remaining, persistent=True
        )

    async def _clear_search_results(self) -> None:
        """Invalidate this instance's searches without touching chapters or progress."""
        await self.mass.cache.clear(
            provider_filter=self.instance_id, category_filter=CACHE_CATEGORY_SEARCH_RESULTS
        )
        await self.mass.cache.clear(
            provider_filter="music",
            key_filter=self.instance_id,
            category_filter=CACHE_CATEGORY_SEARCH_RESULTS,
        )

    def _save_credentials(self, credentials: dict[str, Any]) -> None:
        # MA encrypts the entire JSON string and persists rotated tokens immediately.
        self._update_setup_data("credentials", json.dumps(credentials), immediate=True)

    def _announcer(self, row: dict[str, Any]) -> Artist:
        ident = row.get("id", row.get("uid"))
        name = row.get("nickname")
        if type(ident) is not int or ident <= 0 or not isinstance(name, str) or not name.strip():
            raise ResourceTemporarilyUnavailable("喜马拉雅主播信息格式已变化")
        metadata = MediaItemMetadata(description=row.get("person_describe") or row.get("ptitle"))
        avatar = row.get("avatar_url") or row.get("small_pic")
        if avatar:
            try:
                url = cdn_url(avatar)
            except CarError:
                pass
            else:
                metadata.images = UniqueList(
                    [
                        MediaItemImage(
                            type=ImageType.THUMB,
                            path=url,
                            provider=self.instance_id,
                            remotely_accessible=False,
                        )
                    ]
                )
        return Artist(
            item_id=str(ident),
            provider=self.instance_id,
            name=name,
            artist_type=ArtistType.NARRATOR,
            metadata=metadata,
            provider_mappings={
                ProviderMapping(
                    item_id=str(ident),
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"https://www.ximalaya.com/zhubo/{ident}/",
                )
            },
        )

    async def _album_media_type(self, row: dict[str, Any]) -> MediaType:
        """Classify from explicit upstream podcast evidence, never from legacy settings or rows."""
        tags = row.get("tags") or ""
        if isinstance(tags, str):
            tags = re.split(r"[,，;；、|]", tags)
        if isinstance(tags, list) and any(
            isinstance(tag, str) and tag.strip().casefold() in {"播客", "podcast"} for tag in tags
        ):
            return MediaType.PODCAST
        # Some actual podcasts (e.g. 文化有限) have no tags. Accept an explicit
        # self-description at the start of the official introduction, not a title,
        # genre, ranking membership or an incidental mention of another podcast.
        intro = row.get("intro")
        if isinstance(intro, str):
            intro = html.unescape(re.sub(r"<[^>]*>", "", intro)).strip()[:160]
            if re.match(
                r"^(?:(?!.*(?:不是|并非))[^。！？!?\n]{0,50}是)?一档"
                r"[^。！？!?\n，,；;]{0,24}播客(?:[，,。.!！?？；;：:\s]|$)",
                intro,
            ):
                return MediaType.PODCAST
        return MediaType.AUDIOBOOK

    def _remember_album(self, row: dict[str, Any]) -> None:
        """Offer recently viewed/search albums in the chapter browser without favoriting them."""
        recent = getattr(self, "_recent_albums", None)
        if recent is None:
            self._recent_albums = recent = {}
        ident = str(row["id"])
        recent.pop(ident, None)
        if len(recent) >= 64:
            recent.pop(next(iter(recent)))
        recent[ident] = {k: row[k] for k in ("id", "title", "cover", "image") if k in row}

    async def _subscription_rows(self, media_type: MediaType) -> list[dict[str, Any]]:
        snapshot = await checked(self._client.subscription_snapshot())
        selected: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []
        for row in snapshot.items:
            kind = await self._album_media_type(row)
            (selected if kind == media_type else other).append(row)
        if not hasattr(self, "_other_subscription_ids"):
            self._other_subscription_ids = {}
        self._other_subscription_ids[media_type] = [str(row["id"]) for row in other]
        if not snapshot.complete:
            try:
                # Invisible/delisted subscriptions make totals permanently incomplete.
                # Only explicit per-album membership can authorize a missing item's removal.
                retained = await self._verify_missing_subscriptions(media_type, snapshot.items)
            except CarError:
                self.report_skipped_sync_item(
                    media_type,
                    None,
                    ResourceTemporarilyUnavailable(
                        f"订阅返回不完整：声明 {snapshot.total} 项，"
                        f"仅返回 {len(snapshot.items)} 项；"
                        "未能逐项确认订阅，本轮禁止删除"
                    ),
                )
            else:
                self._other_subscription_ids[media_type].extend(retained)
                self.logger.debug("订阅列表含不可见条目，已逐项核对缺失的库内订阅")
        return selected

    async def _verify_missing_subscriptions(
        self, media_type: MediaType, rows: list[dict[str, Any]]
    ) -> list[str]:
        previous = (
            await self.mass.cache.get(
                key=media_type.value,
                provider=self.instance_id,
                category=CACHE_CATEGORY_PREV_LIBRARY_IDS,
            )
            or []
        )
        visible = {str(row["id"]) for row in rows}
        retained = []
        controller = self.mass.music.get_controller(media_type)
        for db_id in previous:
            try:
                item = await controller.get_library_item(db_id)
            except MediaNotFoundError:
                continue
            for mapping in item.provider_mappings:
                if mapping.provider_instance != self.instance_id or mapping.item_id in visible:
                    continue
                subscribed = await self._client.subscription_status(mapping.item_id)
                if type(subscribed) is not bool:
                    raise CarError("无法确认缺失专辑的订阅状态")
                if subscribed:
                    retained.append(mapping.item_id)
        return retained

    async def _retained_classification_ids(
        self, media_type: MediaType, controller: AudiobooksController | PodcastsController
    ) -> set[int]:
        # Preserve old records/playlogs while subscribed. Migration hides their
        # mappings; MA must not delete them when a format changes.
        ids = set()
        for ident in getattr(self, "_other_subscription_ids", {}).get(media_type, []):
            if old := await controller.get_library_item_by_prov_id(ident, self.instance_id):
                self._note_listed_sync_item(media_type, ident)
                ids.add(int(old.item_id))
        return ids

    async def _sync_library_podcasts(self) -> set[int]:
        """Use MA's deletion/checkpoint engine, updating only changed album metadata."""
        # Resolve and validate the entire snapshot before the first library write.
        items = [item async for item in self.get_library_podcasts()]
        controller = self.mass.music.podcasts
        ids = await self._retained_classification_ids(MediaType.PODCAST, controller)
        added = updated = unchanged = 0
        for item in items:
            self._note_listed_sync_item(MediaType.PODCAST, item.item_id)
            self._update_sync_task_item_status(MediaType.PODCAST, len(ids) + 1)
            for mapping in item.provider_mappings:
                mapping.in_library = True
            existing = await controller.get_library_item_by_prov_id(item.item_id, self.instance_id)
            if existing is None:
                existing = await controller.add_item_to_library(item)
                added += 1
            elif self._podcast_changed(existing, item):
                # Keep mappings from other instances; sync never changes local favorites.
                item.provider_mappings.update(
                    x for x in existing.provider_mappings if x.provider_instance != self.instance_id
                )
                existing = await controller.update_item_in_library(
                    existing.item_id, item, overwrite=True
                )
                updated += 1
            else:
                unchanged += 1
            if books := getattr(self.mass.music, "audiobooks", None):
                await self._migrate_local(item.item_id, existing, books, controller)
            ids.add(int(existing.item_id))
        self.logger.info("订阅导入：新增 %s，更新 %s，未变化 %s", added, updated, unchanged)
        return ids

    def _podcast_changed(self, existing: Podcast, incoming: Podcast) -> bool:
        """Compare persisted metadata, so retries/restarts need no separate delta cache."""

        def fields(item: Podcast) -> tuple[Any, ...]:
            images = sorted((x.path, x.type.value) for x in item.metadata.images or [])
            return (
                item.name,
                item.publisher,
                item.total_episodes or 0,
                item.metadata.description,
                images,
            )

        return fields(existing) != fields(incoming) or not self._check_provider_mappings(
            existing, incoming, True
        )

    def _chapter_page_folder(
        self, album_id: str, offset: int, total: int, direction: str
    ) -> BrowseFolder:
        return BrowseFolder(
            item_id=f"chapters/{album_id}_{offset}_{total}",
            provider=self.instance_id,
            name=f"{direction}（第 {offset + 1}–{min(offset + 50, total)} 章，共 {total} 章）",
            translation_key="previous_chapters" if direction == "上一页" else "next_chapters",
            translation_params=[str(offset + 1), str(min(offset + 50, total)), str(total)],
            path=f"{self.instance_id}://chapters/{album_id}_{offset}_{total}",
        )

    async def _chapter_folders(self) -> list[BrowseFolder]:
        """Expose subscription/library/recent albums as folders, never duplicate library media."""
        folders = {}
        rows = await checked(self._client.subscription_preview())
        for row in rows:
            ident = str(row["id"])
            images = self._metadata(row).images
            folders[ident] = BrowseFolder(
                item_id=f"chapters/{ident}_0",
                provider=self.instance_id,
                name=row["title"],
                path=f"{self.instance_id}://chapters/{ident}_0",
                image=images[0] if images else None,
            )
        if music := getattr(self.mass, "music", None):
            for controller in (music.audiobooks, music.podcasts):
                for item in await controller.library_items(
                    provider=self.instance_id, limit=500, summary=True
                ):
                    for mapping in item.provider_mappings:
                        if mapping.provider_instance != self.instance_id or not mapping.in_library:
                            continue
                        ident = mapping.item_id
                        folders[ident] = BrowseFolder(
                            item_id=f"chapters/{ident}_0",
                            provider=self.instance_id,
                            name=item.name,
                            path=f"{self.instance_id}://chapters/{ident}_0",
                            image=item.image,
                        )
        for ident, row in getattr(self, "_recent_albums", {}).items():
            images = self._metadata(row).images
            folders[ident] = BrowseFolder(
                item_id=f"chapters/{ident}_0",
                provider=self.instance_id,
                name=row["title"],
                path=f"{self.instance_id}://chapters/{ident}_0",
                image=images[0] if images else None,
            )
        return list(folders.values())

    def _mapping(self, data: dict[str, Any], *, episode: bool = False) -> set[ProviderMapping]:
        return {
            ProviderMapping(
                item_id=str(data["id"]),
                provider_domain=self.domain,
                provider_instance=self.instance_id,
                available=data.get("is_authorized") is True if episode else True,
                url=f"https://www.ximalaya.com/{'sound' if episode else 'album'}/{data['id']}",
            )
        }

    def _metadata(self, data: dict[str, Any]) -> MediaItemMetadata:
        metadata = MediaItemMetadata(description=data.get("intro") or data.get("description"))
        cover = data.get("cover") or {}
        image = data.get("image") or {}
        for candidate in (cover.get("large"), cover.get("middle"), cover.get("small"), image):
            # Car API sizes are image objects; older responses may contain a plain URL.
            url = candidate.get("url") if isinstance(candidate, dict) else candidate
            if not url:
                continue
            try:
                url = cdn_url(url)
            except CarError:
                continue
            metadata.images = UniqueList(
                [
                    MediaItemImage(
                        type=ImageType.THUMB,
                        path=url,
                        provider=self.instance_id,
                        remotely_accessible=False,
                    )
                ]
            )
            break
        return metadata

    def _podcast(self, data: dict[str, Any]) -> Podcast:
        self._validate_metadata(data)
        self._remember_album(data)
        return Podcast(
            item_id=str(data["id"]),
            provider=self.instance_id,
            name=data["title"],
            publisher=(data.get("announcer") or {}).get("nickname"),
            total_episodes=data.get("include_track_count"),
            provider_mappings=self._mapping(data),
            metadata=self._metadata(data),
        )

    def _episode(self, data: dict[str, Any]) -> PodcastEpisode:
        self._validate_metadata(data, episode=True)
        return PodcastEpisode(
            item_id=str(data["id"]),
            provider=self.instance_id,
            name=data["title"],
            duration=data["duration"],
            position=data["order_num"] + 1,
            podcast=ItemMapping(
                media_type=MediaType.PODCAST,
                item_id=str(data["album_id"]),
                provider=self.instance_id,
                name=data.get("album_title") or "喜马拉雅专辑",
            ),
            provider_mappings=self._mapping(data, episode=True),
            metadata=self._metadata(data),
        )

    @staticmethod
    def _validate_metadata(data: dict[str, Any], *, episode: bool = False) -> None:
        if (
            type(data.get("id")) is not int
            or data["id"] <= 0
            or not isinstance(data.get("title"), str)
            or not data["title"]
        ):
            raise ResourceTemporarilyUnavailable("喜马拉雅媒体信息格式已变化")
        if any(
            data.get(key) is not None and not isinstance(data[key], dict)
            for key in ("announcer", "cover", "image")
        ):
            raise ResourceTemporarilyUnavailable("喜马拉雅媒体信息格式已变化")
        if episode and (
            any(type(data.get(k)) is not int for k in ("album_id", "duration", "order_num"))
            or data["album_id"] <= 0
            or data["duration"] < 0
            or data["order_num"] < 0
            or any(type(data.get(k)) is not bool for k in ("is_authorized", "is_paid"))
        ):
            raise ResourceTemporarilyUnavailable("喜马拉雅分集信息格式已变化")
