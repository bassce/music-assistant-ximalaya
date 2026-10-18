"""Validate byte ranges and the identity of an authorized HTTP audio resource."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from urllib.parse import urlsplit

import aiohttp

ATTEMPTS = 4
RETRY_DELAYS = (0.5, 1, 2)
TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=10)
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)$")


class RelayError(Exception):
    """An intentionally sanitized upstream failure."""


@dataclass(frozen=True)
class Resource:
    """The exact byte interval and validators for one HTTP resource."""

    start: int
    end: int
    total: int
    path: str
    etag: str | None
    modified: str | None

    @classmethod
    def read(cls, response: aiohttp.ClientResponse, url: str, requested: str | None) -> Resource:
        """Validate and construct a resource from the supplied response."""
        if response.status not in (200, 206):
            raise RelayError(f"CDN HTTP {response.status}")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise RelayError("CDN 返回了压缩传输，无法安全续传")
        try:
            length = int(response.headers["Content-Length"])
            if response.status == 206:
                match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise ValueError
                start, end, total = map(int, match.groups())
            else:
                start, end, total = 0, length - 1, length
            if length <= 0 or not 0 <= start <= end < total or end - start + 1 != length:
                raise ValueError
            if requested:
                match = RANGE.fullmatch(requested)
                if match is None:
                    raise ValueError
                left, right = match.groups()
                want_start = int(left) if left else max(0, total - int(right))
                want_end = min(total - 1, int(right)) if left and right else total - 1
                if response.status != 206 or (start, end) != (want_start, want_end):
                    raise ValueError
        except KeyError, ValueError:
            raise RelayError("CDN 字节范围不匹配，停止续传") from None
        return cls(
            start,
            end,
            total,
            urlsplit(url).path,
            response.headers.get("ETag"),
            response.headers.get("Last-Modified"),
        )

    def same_file(self, other: Resource) -> bool:
        # Refuse to splice different qualities/versions even if their duration matches.
        """Compare resource identity before resuming a byte stream."""
        if (self.total, self.path) != (other.total, other.path):
            return False
        if self.etag and not self.etag.startswith("W/"):
            return self.etag == other.etag
        return bool(self.modified and self.modified == other.modified)
