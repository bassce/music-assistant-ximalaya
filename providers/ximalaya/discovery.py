"""Four approved Discover entries using native folders and media types only."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.media_items import ItemMapping, MediaItemType

if TYPE_CHECKING:
    from . import XimalayaProvider

import re
from urllib.parse import urlsplit

from music_assistant_models.enums import ContentType, ImageType, MediaType, StreamType
from music_assistant_models.errors import MediaNotFoundError, ResourceTemporarilyUnavailable
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    MediaItemImage,
    ProviderMapping,
    Radio,
)
from music_assistant_models.streamdetails import StreamDetails

from .artwork import PREFIX, folder_icon, local_image
from .protocol import DiscoveryPage


class DiscoveryMixin:
    """Navigation stays shallow and reads only the page the user opens."""

    async def resolve_image(self, path: str) -> str | bytes:
        """Resolve a packaged navigation icon or retain the supplied image path."""
        try:
            return local_image(path) or path
        except ValueError:
            raise MediaNotFoundError("目录图标不存在") from None

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Return a station from the backend catalog."""
        return self._provider._radio(
            await self._provider._checked(self._provider._client.radio(prov_radio_id))
        )

    def _discovery_folder(
        self, path: str, name: str, key: str | None = None, params: list[str] | None = None
    ) -> BrowseFolder:
        return BrowseFolder(
            item_id=path,
            provider=self._provider.instance_id,
            name=name,
            translation_key=key,
            translation_params=params,
            path=f"{self._provider.instance_id}://{path}",
            image=MediaItemImage(
                type=ImageType.THUMB,
                path=f"{PREFIX}{folder_icon(path, name)}.png",
                provider=self._provider.instance_id,
                remotely_accessible=False,
            ),
        )

    def _discovery_entries(self) -> list[BrowseFolder]:
        return [
            self._provider._discovery_folder(f"discovery/{path}", name, path)
            for path, name in (
                ("radio", "电台广播"),
                ("hot", "热播榜单"),
                ("recommendations", "猜你喜欢"),
                ("groups", "详细分类"),
            )
        ]

    async def _discovery_albums(
        self, rows: list[dict[str, Any]]
    ) -> list[MediaItemType | BrowseFolder | ItemMapping]:
        result: list[MediaItemType | BrowseFolder | ItemMapping] = []
        for row in rows:
            kind = await self._provider._album_media_type(row)
            result.append(
                self._provider._book(row)
                if kind == MediaType.AUDIOBOOK
                else self._provider._podcast(row)
            )
        return result

    def _discovery_pages(
        self,
        items: list[MediaItemType | BrowseFolder | ItemMapping],
        page: DiscoveryPage,
        prefix: str,
    ) -> list[MediaItemType | BrowseFolder | ItemMapping]:
        # The API filters hidden entries after pagination. len(items) is not an end marker.
        if page.page > 1:
            items.insert(
                0,
                self._provider._discovery_folder(
                    f"{prefix}_{page.page - 1}", "上一页", "previous_page"
                ),
            )
        if page.page < page.pages:
            items.append(
                self._provider._discovery_folder(
                    f"{prefix}_{page.page + 1}",
                    f"下一页（{page.page + 1}/{page.pages}）",
                    "next_page",
                    [str(page.page + 1), str(page.pages)],
                )
            )
        return items

    async def _browse_discovery(
        self, relative: str
    ) -> list[MediaItemType | BrowseFolder | ItemMapping]:
        if relative == "discovery":
            return list(self._provider._discovery_entries())
        if relative == "discovery/recommendations":
            return await self._provider._discovery_albums(
                await self._provider._checked(self._provider._client.recommended_albums())
            )
        if relative == "discovery/radio":
            categories = await self._provider._checked(
                self._provider._client.discovery_catalog("radio")
            )
            return [
                self._provider._discovery_folder(
                    f"discovery/radio/{row['id']}_1", row["category_name"]
                )
                for row in categories
            ]
        if match := re.fullmatch(r"discovery/radio/([1-9][0-9]{0,18})_([1-9][0-9]{0,2})", relative):
            ident, number = match.groups()
            categories = await self._provider._checked(
                self._provider._client.discovery_catalog("radio")
            )
            if not any(str(row["id"]) == ident for row in categories):
                raise MediaNotFoundError("喜马拉雅广播分类不存在")
            page = await self._provider._checked(
                self._provider._client.radio_page(ident, int(number))
            )
            return self._provider._discovery_pages(
                [self._provider._radio(row) for row in page.items], page, f"discovery/radio/{ident}"
            )
        if relative not in ("discovery/groups", "discovery/hot") and not re.fullmatch(
            r"discovery/(?:hot/[1-9][0-9]{0,18}_[1-9][0-9]{0,2}|groups/[1-9][0-9]{0,18}(?:/[1-9][0-9]{0,18}_[1-9][0-9]{0,2})?)",
            relative,
        ):
            raise MediaNotFoundError("喜马拉雅发现目录不存在")
        groups = await self._provider._checked(self._provider._client.discovery_catalog("groups"))
        if relative == "discovery/groups":
            return [
                self._provider._discovery_folder(
                    f"discovery/groups/{row['group_type']}", row["group_name"]
                )
                for row in groups
            ]
        parts = relative.split("/")
        group_id = "1" if parts[1] == "hot" else parts[2]
        group = next((row for row in groups if str(row["group_type"]) == group_id), None)
        if group is None:
            raise MediaNotFoundError("喜马拉雅榜单分组不存在")
        prefix = "discovery/hot" if parts[1] == "hot" else f"discovery/groups/{group_id}"
        if relative == prefix:
            return [
                self._provider._discovery_folder(f"{prefix}/{row['rank_id']}_1", row["rank_name"])
                for row in group["ranks"]
            ]
        ident, number = parts[-1].split("_")
        rank = next((row for row in group["ranks"] if str(row["rank_id"]) == ident), None)
        if rank is None:
            raise MediaNotFoundError("喜马拉雅榜单不存在")
        page = await self._provider._checked(self._provider._client.rank_page(ident, int(number)))
        items: list[MediaItemType | BrowseFolder | ItemMapping]
        if rank["content_type"] == 1:
            items = [self._provider._episode(row) for row in page.items]
        elif rank["content_type"] == 2:
            items = await self._provider._discovery_albums(page.items)
        else:
            items = [self._provider._announcer(row) for row in page.items]
        return self._provider._discovery_pages(items, page, f"{prefix}/{ident}")

    def _radio(self, row: dict[str, Any]) -> Radio:
        ident, name = row.get("id"), row.get("name")
        if type(ident) is not int or ident <= 0 or not isinstance(name, str) or not name.strip():
            raise ResourceTemporarilyUnavailable("喜马拉雅电台信息格式已变化")
        return Radio(
            item_id=str(ident),
            provider=self._provider.instance_id,
            name=name,
            metadata=self._provider._metadata(
                {
                    "intro": row.get("intro"),
                    "image": row.get("large_cover_url") or row.get("small_cover_url"),
                }
            ),
            provider_mappings={
                ProviderMapping(
                    item_id=str(ident),
                    provider_domain=self._provider.domain,
                    provider_instance=self._provider.instance_id,
                    available=row.get("online_status") is True,
                )
            },
        )

    async def _radio_stream(self, radio_id: str) -> StreamDetails:
        url = await self._provider._checked(self._provider._client.resolve_radio(radio_id))
        return StreamDetails(
            provider=self._provider.instance_id,
            item_id=radio_id,
            media_type=MediaType.RADIO,
            stream_type=StreamType.HLS if urlsplit(url).path.endswith(".m3u8") else StreamType.HTTP,
            audio_format=AudioFormat(content_type=ContentType.AAC),
            path=url,
            can_seek=False,
            allow_seek=False,
            expiration=0,
        )

    @property
    def _provider(self) -> XimalayaProvider:
        """Return the concrete provider hosting these operations."""
        return cast("XimalayaProvider", self)
