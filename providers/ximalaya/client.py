"""MA client for the stable local API; no Ximalaya endpoints or signing keys."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from music_assistant.helpers import throttle_retry
from music_assistant_models.errors import RateLimited

from .protocol import (  # re-export for existing provider modules
    API_VERSION,
    AuthenticationError,
    CarError,
    DiscoveryPage,
    EpisodePage,
    NotFound,
    PermissionDenied,
    SubscriptionSnapshot,
    item_id,
)


class QrExpired(CarError):
    """The backend QR deadline elapsed before MA's progress step expired."""


DEFAULT_API_URL = "http://127.0.0.1:17863"
ERROR_TYPES = {
    "authentication_required": AuthenticationError,
    "access_denied": AuthenticationError,
    "permission_denied": PermissionDenied,
    "not_found": NotFound,
    "qr_expired": QrExpired,
}
ERROR_MESSAGES = {
    "authentication_required": "喜马拉雅登录已失效，请在 MA 中重新扫码",
    "access_denied": "外置 API 访问凭据无效，请重新配置连接",
    "permission_denied": "当前账号无权完整播放此声音",
    "not_found": "内容不存在或已下架",
    "client_config_required": "请在 MA 中导入中国版客户端配置",
    "already_paired": "外置 API 已配对，请填写已有服务访问密钥",
    "local_pairing_only": "首次连接请使用本机 API 地址，或填写已有服务访问密钥",
    "qr_expired": "二维码已过期，请重试",
}


def http_url(value: Any) -> str:
    """Validate transport URLs returned by our own service (including local media tickets)."""
    if not isinstance(value, str) or any(ord(char) < 33 for char in value):
        raise CarError("媒体地址格式无效")
    try:
        parts = urlsplit(value)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError
        _ = parts.port
    except ValueError:
        raise CarError("媒体地址格式无效") from None
    return value


def cdn_url(value: Any) -> str:
    """Accept backend media tickets and covers already stored by the legacy provider."""
    validated = http_url(value)
    parts = urlsplit(validated)
    if parts.path.startswith("/v1/media/") and parts.query.startswith("ticket="):
        return validated
    # Legacy library covers can remain cached across the migration.
    host = parts.hostname or ""
    if host == "xmcdn.com" or host.endswith(".xmcdn.com"):
        return urlunsplit(parts._replace(scheme="https"))
    raise CarError("媒体地址不属于外置 API")


def api_url(value: Any) -> str:
    """Validate a backend service root URL."""
    parts = urlsplit(http_url(value.strip()))
    if parts.query or parts.path not in ("", "/"):
        raise CarError("请输入外置 API 的服务根地址，不含路径或查询参数")
    return urlunsplit(parts._replace(path=""))


class GatewayClient:
    """Authenticated client for the versioned local backend protocol."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        credentials: dict[str, Any],
        save: Callable[[dict[str, Any]], None] | None = None,
        *,
        audio_quality: str = "highest",
        search_cache_minutes: int = 1,
    ) -> None:
        """Initialize per-instance state."""
        if not isinstance(credentials, dict) or not credentials.get("api_url"):
            raise AuthenticationError("请在 MA 中重新配置喜马拉雅，连接外置 API 并迁移账号")
        self.session = session
        self.credentials = dict(credentials)
        self.base_url = api_url(credentials["api_url"])
        self.audio_quality = audio_quality
        self.search_cache_minutes = search_cache_minutes
        self.nickname = credentials.get("nickname", "")
        self.token = {"uid": credentials.get("uid")}  # identity only; no upstream token in MA
        self.verified = False
        self.save = save
        self._duration_hints: dict[str, int] = {}
        self.pending_session: str | None = None
        self.throttler = throttle_retry.Throttler(4, 1)

    def export(self) -> dict[str, Any]:
        """Return backend connection credentials for MA secure storage."""
        return {
            key: self.credentials[key]
            for key in ("api_url", "service_key", "account_id", "account_key", "uid", "nickname")
            if key in self.credentials
        }

    async def request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        *,
        owner: bool = False,
        anonymous: bool = False,
        request_timeout: float = 30,
    ) -> Any:
        """Call the backend with bounded replies and shared request throttling."""
        key = self.credentials.get("service_key" if owner else "account_key")
        if not key and not anonymous:
            raise AuthenticationError("外置 API 连接凭据缺失，请重新配置")
        try:
            async with (
                self.throttler,
                self.session.request(
                    method,
                    self.base_url + path,
                    json=data if data is not None else None,
                    headers={
                        **({"Authorization": "Bearer " + key} if key else {}),
                        "X-MA-Request-Priority": str(int(throttle_retry.current_priority()))
                        if hasattr(throttle_retry, "current_priority")
                        else "1",
                    },
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                ) as reply,
            ):
                raw = bytearray()
                async for chunk in reply.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if len(raw) > 16 * 1024 * 1024:
                        raise CarError("外置 API 响应过大")
                try:
                    result = json.loads(raw)
                except ValueError, UnicodeError:
                    raise CarError("外置 API 返回格式无效") from None
                if not isinstance(result, dict) or result.get("api_version") != API_VERSION:
                    raise CarError("外置 API 协议版本不兼容，请检查插件版本")
                if reply.status != 200 or "error" in result:
                    code = (result.get("error") or {}).get("code", "upstream_unavailable")
                    if code == "rate_limited":
                        delay = (result.get("error") or {}).get("retry_after")
                        raise RateLimited(
                            "Ximalaya rate limit reached; try again later",
                            backoff_time=min(86400, max(1, int(delay)))
                            if type(delay) is int
                            else 5,
                        )
                    raise ERROR_TYPES.get(code, CarError)(
                        ERROR_MESSAGES.get(code, "外置 API 或喜马拉雅暂时不可用，请稍后重试")
                    )
                if "data" not in result:
                    raise CarError("外置 API 返回格式无效")
                return self._media_urls(result["data"])
        except aiohttp.ClientError, TimeoutError, OSError:
            raise CarError("外置 API 连接失败，请检查后端运行状态") from None

    async def local_progress(self, album: str) -> dict[str, Any]:
        """Read the encrypted backend bookmark, independently of cloud history."""
        return await self.request("GET", f"/v1/progress/{item_id(album)}")

    async def save_local_progress(
        self,
        album: str,
        *,
        chapter_id: str,
        position_ms: int,
        fully_played: bool,
        timestamp: float | None = None,
        import_only: bool = False,
    ) -> dict[str, Any]:
        """Replace the account bookmark, or fill a missing bookmark during import."""
        return await self.request(
            "PUT",
            f"/v1/progress/{item_id(album)}",
            {
                "chapter_id": chapter_id,
                "position_ms": position_ms,
                "fully_played": fully_played,
                "timestamp": timestamp,
                "import_only": import_only,
            },
        )

    async def start(self) -> None:
        """Prepare the client or stream for use."""
        data = await self.request("GET", "/v1/account")
        if str(data.get("uid")) != str(self.credentials.get("uid")) or data.get(
            "account_id"
        ) != self.credentials.get("account_id"):
            raise AuthenticationError("外置 API 返回的账号与当前实例不一致")
        if data.get("bookmark_schema") != 1:
            raise CarError("请先将喜马拉雅 API 更新至 0.1.5 或更新的兼容版本，以支持账号断点")
        self.nickname = data.get("nickname") or ""
        self.verified = True

    async def rpc(self, operation: str, **arguments: Any) -> Any:
        """Invoke a versioned backend operation."""
        return await self.request(
            "POST",
            "/v1/rpc",
            {
                "operation": operation,
                "arguments": arguments,
                "options": {
                    "audio_quality": self.audio_quality,
                    "search_cache_minutes": self.search_cache_minutes,
                },
            },
            request_timeout=600
            if operation in ("episodes", "subscription_snapshot", "announcer_albums")
            else 60,
        )

    async def create_qr(self) -> tuple[str, None, str]:
        """Create a pending login and return its QR image."""
        data = await self.request(
            "POST", "/v1/auth/qr", {"account_id": self.credentials.get("account_id")}, owner=True
        )
        self.pending_session = data["session_id"]
        return data["session_id"], None, data["image"]

    async def poll_qr(
        self, qr_id: str, state: str | None = None, expected_uid: str | int | None = None
    ) -> None:
        """Wait for login confirmation from the expected account."""
        while True:
            result = await self.request("GET", f"/v1/auth/{qr_id}", owner=True)
            if result.get("state") == "authorized":
                if expected_uid and str(result["uid"]) != str(expected_uid):
                    raise AuthenticationError("重新登录必须使用原来的喜马拉雅账号")
                self.token = {"uid": result["uid"]}
                self.nickname = result.get("nickname", "")
                return
            if result.get("state") != "pending":
                raise AuthenticationError("扫码状态无效，请重试")
            await asyncio.sleep(1.5)

    async def commit_login(self) -> None:
        """Commit a confirmed login after duplicate-account checks."""
        if self.pending_session:
            result = await self.request(
                "POST", f"/v1/auth/{self.pending_session}/commit", {}, owner=True
            )
            self.credentials.update(result)
            self.token = {"uid": result["uid"]}
            self.pending_session = None

    async def cancel_login(self) -> None:
        """Discard an uncommitted login session."""
        if self.pending_session:
            ident, self.pending_session = self.pending_session, None
            with contextlib.suppress(CarError):
                await self.request("DELETE", f"/v1/auth/{ident}", owner=True, request_timeout=5)

    async def resolve_source(
        self, track_id: str, *, probe_unknown_duration: bool = False
    ) -> tuple[str, float, str]:
        """Return an authorized media ticket and its audio properties."""
        hint = self._duration_hints.get(str(track_id), 0)
        result = await self.rpc(
            "resolve_source",
            track_id=track_id,
            probe_unknown_duration=probe_unknown_duration or bool(hint),
        )
        return result["path"], result["duration"] or hint, result["quality"]

    async def resolve(
        self, track_id: str, *, probe_unknown_duration: bool = False
    ) -> tuple[str, float]:
        """Authorize and resolve the requested audio resource."""
        url, duration, _ = await self.resolve_source(
            track_id, probe_unknown_duration=probe_unknown_duration
        )
        return url, duration

    async def resolve_radio(self, radio_id: str) -> str:
        """Return a server-side station media ticket."""
        return cast("str", (await self.rpc("resolve_radio", radio_id=radio_id))["path"])

    async def transcript(
        self, track_id: str, *, position_ms: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        """Read the reserved timed-transcript endpoint."""
        return cast(
            "dict[str, Any]",
            await self.request(
                "GET",
                f"/v1/tracks/{item_id(track_id)}/transcript?position_ms={position_ms}&limit={limit}",
            ),
        )

    async def album(self, album_id: str) -> dict[str, Any]:
        """Read album metadata."""
        return cast("dict[str, Any]", await self.rpc("album", album_id=album_id))

    async def track(self, track_id: str) -> dict[str, Any]:
        """Read episode metadata."""
        return cast("dict[str, Any]", await self.rpc("track", track_id=track_id))

    async def search(self, query: str, kind: str, limit: int) -> list[dict[str, Any]]:
        """Search the requested media kinds within the current account."""
        return cast(
            "list[dict[str, Any]]", await self.rpc("search", query=query, kind=kind, limit=limit)
        )

    async def announcer(self, announcer_id: str) -> dict[str, Any]:
        """Read announcer metadata."""
        return cast("dict[str, Any]", await self.rpc("announcer", announcer_id=announcer_id))

    async def announcer_albums(self, announcer_id: str) -> list[dict[str, Any]]:
        """Read albums published by an announcer."""
        return cast(
            "list[dict[str, Any]]", await self.rpc("announcer_albums", announcer_id=announcer_id)
        )

    async def episodes(self, album_id: str) -> list[dict[str, Any]]:
        """Read the complete ordered episode catalog."""
        return cast("list[dict[str, Any]]", await self.rpc("episodes", album_id=album_id))

    async def episode_page(
        self, album_id: str, offset: int, expected_total: int | None = None
    ) -> EpisodePage:
        """Read a single chapter page."""
        data = await self.rpc(
            "episode_page", album_id=album_id, offset=offset, expected_total=expected_total
        )
        return EpisodePage(**data)

    async def subscription_preview(self) -> list[dict[str, Any]]:
        """Read a bounded subscription preview."""
        return cast("list[dict[str, Any]]", await self.rpc("subscription_preview"))

    async def subscription_snapshot(self) -> SubscriptionSnapshot:
        """Read subscriptions with explicit completeness information."""
        data = await self.rpc("subscription_snapshot")
        return SubscriptionSnapshot(**data)

    async def subscription_status(self, album_id: str) -> bool:
        """Read the current subscription state."""
        return cast("bool", await self.rpc("subscription_status", album_id=album_id))

    async def set_subscription(self, album_id: str, subscribed: bool) -> None:
        """Change the subscription state without replaying the write."""
        await self.rpc("set_subscription", album_id=album_id, subscribed=subscribed)

    async def recent_history(self) -> list[dict[str, Any]]:
        """Read recent account listening history."""
        return cast("list[dict[str, Any]]", await self.rpc("recent_history"))

    async def cloud_progress(self, album_id: str) -> dict[str, Any] | None:
        """Read account progress for an album."""
        return cast("dict[str, Any] | None", await self.rpc("cloud_progress", album_id=album_id))

    async def skip_config(self, album_id: str) -> tuple[int, int]:
        """Read the album intro and outro settings."""
        data = await self.rpc("skip_config", album_id=album_id)
        return tuple(data)

    async def save_progress(
        self,
        album_id: str,
        track_id: str,
        position: float,
        listened: float,
        started: int,
        ended: int,
    ) -> None:
        """Write a listening event without automatic replay."""
        await self.rpc(
            "save_progress",
            album_id=album_id,
            track_id=track_id,
            position=position,
            listened=listened,
            started=started,
            ended=ended,
        )

    async def discovery_catalog(self, kind: str) -> list[dict[str, Any]]:
        """Read the requested discovery categories."""
        return cast("list[dict[str, Any]]", await self.rpc("discovery_catalog", kind=kind))

    async def recommended_albums(self) -> list[dict[str, Any]]:
        """Read account album recommendations."""
        return cast("list[dict[str, Any]]", await self.rpc("recommended_albums"))

    async def radio(self, radio_id: str) -> dict[str, Any]:
        """Read station metadata."""
        return cast("dict[str, Any]", await self.rpc("radio", radio_id=radio_id))

    async def radio_page(self, category_id: str, page: int) -> DiscoveryPage:
        """Read a page of stations."""
        data = await self.rpc("radio_page", category_id=category_id, page=page)
        return DiscoveryPage(**data)

    async def rank_page(self, rank_id: str, page: int) -> DiscoveryPage:
        """Read a page of ranked content."""
        data = await self.rpc("rank_page", rank_id=rank_id, page=page)
        return DiscoveryPage(**data)

    def _media_urls(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._media_urls(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._media_urls(item) for item in value]
        if isinstance(value, str) and value.startswith("/v1/media/"):
            return self.base_url + value
        return value


CarClient = GatewayClient
