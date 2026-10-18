"""Observe source durations without changing the catalog's playback timeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import XimalayaProvider


async def correct_duration(
    provider: XimalayaProvider, token: str, ident: str, seconds: float
) -> None:
    """Validate a format probe, retaining catalog times for chapters and skipping.

    The historic name remains for the normalizer's call sites. Probing still chooses
    a compatible audio format; it must not rewrite MA progress or queue clocks.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid audio duration")
    book = provider._part_books.get(token)
    if not book:
        return
    _, timeline = book
    row = next((row for row in timeline.rows if str(row["id"]) == ident), None)
    if row and abs(row["duration"] - seconds) > 1.1:
        provider.logger.debug(
            "Chapter %s catalog duration %s s, source %.3f s; keeping catalog timeline",
            ident,
            row["duration"],
            seconds,
        )
