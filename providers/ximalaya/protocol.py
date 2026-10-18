"""Version 1 wire types owned by this self-contained MA provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

API_VERSION = 1


@dataclass(frozen=True)
class SubscriptionSnapshot:
    """Visible subscriptions with explicit completeness; gaps never imply unsubscription."""

    items: list[dict[str, Any]]
    total: int
    complete: bool


@dataclass(frozen=True)
class EpisodePage:
    """One visible chapter page, not a complete album snapshot."""

    items: list[dict[str, Any]]
    total: int
    offset: int


@dataclass(frozen=True)
class DiscoveryPage:
    """Server pagination can contain fewer visible items than its page size."""

    items: list[dict[str, Any]]
    page: int
    pages: int


class CarError(Exception):
    """An error containing only a fixed, credential-free message."""


class AuthenticationError(CarError):
    """The Chinese account must be authenticated again."""


class PermissionDenied(CarError):
    """The account cannot play this episode in full."""


class NotFound(CarError):
    """The requested item no longer exists."""


def item_id(value: Any) -> str:
    """Accept positive decimal IDs only; URLs and international IDs are not API paths."""
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit():
        raise CarError("声音或专辑 ID 无效")
    if not 0 < int(value) < 2**63:
        raise CarError("声音或专辑 ID 无效")
    return str(int(value))


class PreparationError(CarError):
    """Stop playback when an audio encoding or timeline cannot be validated."""
