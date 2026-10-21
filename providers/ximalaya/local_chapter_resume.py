# Copyright (c) 2026 ha_ximalaya contributors
"""Local chapter bookmarks using MA queue snapshots and public playback controls."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
import asyncio
import math
import time
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

from music_assistant_models.enums import MediaType, PlaybackState

from .resume_audio import gate_for

if TYPE_CHECKING:
    from . import XimalayaProvider

SAVE_INTERVAL = 5
RESUME_WINDOW = 8
SEEK_TIMEOUT = 60
IO_TIMEOUT = 5
MAX_PENDING_CHAPTERS = 512


@dataclass(frozen=True)
class Sample:
    """A listening observation, never a decoder/prefetch position."""

    queue: str
    item: str
    album: str
    track: str
    index: int
    start: float
    duration: float
    offset: float
    clock: float
    observed_at: float
    playing: bool
    generation: int
    last: bool
    head: float = 0
    tail: float = 0
    finished: bool = False

    @property
    def identity(self) -> tuple[str, str]:
        """Identify an individual chapter within the current queue item."""
        return self.item, self.track


@dataclass(frozen=True)
class Settlement:
    """Track an expected native seek while the player clock catches up."""

    identity: tuple[str, str]
    offset: float
    expires: float


class LocalChapterResume:
    """Keep slow API work independent of queue observation and other players."""

    def __init__(self, provider: XimalayaProvider) -> None:
        """Initialize per-queue observation and independently scheduled API work."""
        self.provider = provider
        self.samples: dict[str, Sample] = {}
        self.saved_at: dict[str, float] = {}
        self.pending: set[str] = set()
        self.pending_since: dict[str, float] = {}
        self.settling: dict[str, Settlement] = {}
        self.resuming: dict[str, asyncio.Task] = {}
        self.retry_at: dict[str, float] = {}
        self.outbox: dict[str, Sample] = {}
        self.write_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self.buffers: dict[str, object] = {}
        self.cleanup_tasks: set[asyncio.Task] = set()
        self.task: asyncio.Task | None = None
        self.writer: asyncio.Task | None = None
        self.closed = False
        self.last_warning = -math.inf
        self.write_slots = asyncio.Semaphore(4)

    def start(self) -> None:
        """Start queue observation and the local-progress writer."""
        self.task = asyncio.create_task(self.run())
        self.writer = asyncio.create_task(self.write_loop())

    def warn(self, error: Exception) -> None:
        """Rate-limit sanitized progress-service warnings."""
        if time.monotonic() - self.last_warning >= 30:
            self.provider.logger.warning("分集本地进度暂不可用，将重试（%s）", type(error).__name__)
            self.last_warning = time.monotonic()

    async def close(self) -> None:
        """Stop workers and flush the last observed positions."""
        self.closed = True
        tasks = [t for t in (self.task, self.writer, *self.resuming.values()) if t]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sample in self.samples.values():
            self.queue_save(sample)
        try:
            async with asyncio.timeout(IO_TIMEOUT):
                await self.flush()
        except Exception as error:
            self.warn(error)
        for task in self.cleanup_tasks:
            task.cancel()
        await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
        self.samples.clear()
        self.buffers.clear()

    @staticmethod
    def busy(queue) -> bool:
        """Check whether MA is applying a queue command."""
        return bool(getattr(queue, "extra_attributes", {}).get("play_action_in_progress"))

    def transitioning(self, queue_id: str) -> bool:
        """Keep the head skipper from racing a pending local resume."""
        settlement = self.settling.get(queue_id)
        queue = self.provider.mass.player_queues.get(queue_id)
        stream = queue.current_item.streamdetails if queue and queue.current_item else None
        # The source can be ready before the first observer tick. A head-skip in
        # that window would replace the guarded stream and cancel its local seek.
        if (
            bool(stream and gate_for(stream))
            or queue_id in self.pending
            or bool(settlement and settlement.expires > time.monotonic())
        ):
            return True
        current = self.snapshot(queue) if queue else None
        previous = self.samples.get(queue_id)
        return bool(
            current
            and current.offset <= current.head + 5
            and (previous is None or previous.identity != current.identity)
            and not (previous and self.natural_transition(previous, current))
        )

    def snapshot(self, queue) -> Sample | None:
        """Map a queue position onto the original whole-book timeline."""
        item = queue.current_item
        stream = item.streamdetails if item else None
        if not queue.active or not stream or stream.provider != self.provider.instance_id:
            return None
        if stream.media_type != MediaType.AUDIOBOOK or stream.stream_error:
            return None
        if queue.state not in (PlaybackState.PLAYING, PlaybackState.PAUSED):
            return None
        catalog = self.provider._catalogs.get(stream.item_id)
        stream_spec = self.provider._skip_streams.get((MediaType.AUDIOBOOK, stream.item_id))
        chapters = catalog[1] if catalog else stream_spec[2] if stream_spec else None
        if chapters is None:
            return None
        position = queue.corrected_elapsed_time
        if not math.isfinite(position) or not 0 <= position <= chapters.duration:
            return None
        chapter, offset = chapters.locate(position)
        index = max(0, bisect_right(chapters.starts, position) - 1)
        seek = getattr(stream, "seek_position", -2)
        # MA subtracts 500 ms and floors a selected chapter's start to seconds.
        if (
            index + 1 < len(chapters.rows)
            and seek is not None
            and 0 < chapters.starts[index + 1] - seek <= 1
            and seek <= position < chapters.starts[index + 1]
        ):
            index += 1
            chapter = chapters.rows[index]
            offset = position - chapters.starts[index]
        settings = self.provider._skip_configs.get(stream.item_id)
        head, tail = (
            settings[1]
            if settings and self.provider.config.get_value("skip_sync", True)
            else (0, 0)
        )
        if head + tail >= chapter["duration"]:
            head, tail = 0, 0
        gate = gate_for(stream)
        if gate:
            # Silence while MA is starting is not actual listening progress.
            index = chapters.starts.index(gate.start)
            chapter = chapters.rows[index]
            offset = head
        return Sample(
            queue.queue_id,
            item.queue_item_id,
            stream.item_id,
            str(chapter["id"]),
            index,
            chapters.starts[index],
            chapter["duration"],
            max(0, offset),
            time.monotonic(),
            time.time(),
            queue.state == PlaybackState.PLAYING,
            id(getattr(stream, "buffer", None) or stream),
            index == len(chapters.rows) - 1,
            head,
            tail,
        )

    def queue_save(self, sample: Sample, *, finished: bool = False) -> None:
        """Coalesce the latest observation for an individual chapter."""
        if not sample.playing or sample.offset < 2:
            return
        sample = replace(sample, finished=sample.finished or finished)
        old = self.outbox.get(sample.track)
        if old is None and len(self.outbox) >= MAX_PENDING_CHAPTERS:
            oldest = min(self.outbox, key=lambda track: self.outbox[track].observed_at)
            self.outbox.pop(oldest)
            self.provider.logger.warning("本地进度服务长期不可用，未保存记录超过 512 集")
        if old is None or old.observed_at <= sample.observed_at:
            self.outbox[sample.track] = sample
        self.saved_at[sample.queue] = sample.clock

    async def flush_track(self, track: str) -> None:
        """Serialize pending writes before reading the same chapter's bookmark."""
        lock = self.write_locks.setdefault(track, asyncio.Lock())
        async with lock, self.write_slots:
            if sample := self.outbox.get(track):
                async with asyncio.timeout(IO_TIMEOUT):
                    await self.provider._client.save_chapter_progress(
                        track,
                        round((sample.duration if sample.finished else sample.offset) * 1000),
                        sample.finished,
                        observed_at=sample.observed_at,
                    )
                if self.outbox.get(track) is sample:
                    self.outbox.pop(track, None)

    async def flush(self) -> None:
        """Flush pending chapter writes under the shared I/O limit."""
        results = await asyncio.gather(
            *(self.flush_track(track) for track in list(self.outbox)), return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                raise result

    async def write_loop(self) -> None:
        """Retry local progress writes without delaying queue observation."""
        while True:
            try:
                await self.flush()
            except Exception as error:
                self.warn(error)
                await asyncio.sleep(2)
            await asyncio.sleep(0.25)

    def release_superseded_buffer(self, queue) -> None:
        """Release a replaced buffer only when no active queue still references it."""
        item = queue.current_item
        stream = item.streamdetails if item else None
        buffer = (
            getattr(stream, "buffer", None)
            if (stream and stream.provider == self.provider.instance_id)
            else None
        )
        previous = self.buffers.pop(queue.queue_id, None)
        if buffer is not None:
            self.buffers[queue.queue_id] = buffer
        if previous is None or previous is buffer:
            return
        # Retain buffers referenced by another current/prefetched item.
        for other in self.provider.mass.player_queues.all():
            for candidate in (other.current_item, getattr(other, "next_item", None)):
                details = candidate.streamdetails if candidate else None
                if details and getattr(details, "buffer", None) is previous:
                    return
        task = asyncio.create_task(self.clear_buffer(previous))
        self.cleanup_tasks.add(task)
        task.add_done_callback(self.cleanup_tasks.discard)

    async def clear_buffer(self, buffer) -> None:
        """Close a superseded MA buffer within the cleanup deadline."""
        try:
            async with asyncio.timeout(IO_TIMEOUT):
                await buffer.clear()
        except Exception as error:
            self.warn(error)

    def natural_transition(self, previous: Sample, current: Sample) -> bool:
        """Distinguish natural chapter completion from explicit selection."""
        if (
            previous.item != current.item
            or previous.album != current.album
            or current.index != previous.index + 1
            or not previous.playing
            or not current.playing
            or previous.offset < previous.duration - previous.tail - 2
        ):
            return False
        skip = self.provider._skip_attempts.get(current.queue)
        skipped_tail = bool(
            skip
            and skip[0] == (current.item, math.ceil(current.start + current.head))
            and current.clock - skip[1] < 5
        )
        if previous.generation != current.generation and not skipped_tail:
            return False
        delta = current.start + current.offset - previous.start - previous.offset
        return (
            0
            <= delta
            <= max(3, (current.clock - previous.clock) * 2 + 1)
            + (previous.tail + current.head if skipped_tail else 0)
        )

    async def tick(self, queue) -> None:
        """Observe one queue and schedule progress writes or local resume."""
        if preloader := getattr(self.provider, "_resume_preloader", None):
            preloader.prune(queue)
        self.release_superseded_buffer(queue)
        previous = self.samples.get(queue.queue_id)
        if self.busy(queue):
            if (
                previous
                and queue.queue_id not in self.pending
                and queue.queue_id not in self.settling
            ):
                self.queue_save(previous)
            return
        current = self.snapshot(queue)
        if current is None:
            if previous and queue.queue_id not in self.pending:
                self.queue_save(previous)
            if previous:
                self.samples.pop(queue.queue_id, None)
            self.pending.discard(queue.queue_id)
            self.pending_since.pop(queue.queue_id, None)
            self.settling.pop(queue.queue_id, None)
            self.saved_at.pop(queue.queue_id, None)
            return
        settlement = self.settling.get(current.queue)
        if settlement:
            # A different chapter immediately ends the wait: rapid clicks must
            # still save the outgoing chapter and inspect the new one.
            if settlement.identity != current.identity or settlement.expires <= current.clock:
                self.settling.pop(current.queue, None)
            elif abs(current.offset - settlement.offset) > 3:
                return
        changed = previous is None or previous.identity != current.identity
        natural = bool(previous and changed and self.natural_transition(previous, current))
        if previous and not changed:
            delta = current.offset - previous.offset
            continuous = (
                previous.playing
                and current.playing
                and previous.generation == current.generation
                and 0 < delta <= max(2, (current.clock - previous.clock) * 2 + 1)
            )
            # Keep completion through pause/stop; a real rewind starts a new pass.
            finished = (previous.finished and delta >= -1) or (
                continuous and current.offset >= current.duration - max(current.tail, 1)
            )
            current = replace(current, finished=finished)
        self.samples[current.queue] = current
        if changed:
            was_pending = current.queue in self.pending
            self.pending.discard(current.queue)
            if not natural and current.offset <= current.head + 5:
                self.pending.add(current.queue)
                self.pending_since[current.queue] = current.clock
            if previous and not was_pending:
                self.queue_save(
                    replace(previous, observed_at=current.observed_at) if natural else previous,
                    finished=natural,
                )
        elif previous and previous.playing and not current.playing:
            # A paused clock is an actual final playback position.
            self.queue_save(replace(current, playing=True))
        if current.queue in self.pending and current.playing:
            if (
                current.offset > current.head + 5
                or current.clock - self.pending_since.get(current.queue, current.clock)
                > RESUME_WINDOW
            ):
                self.pending.discard(current.queue)
            elif current.queue not in self.resuming and current.clock >= self.retry_at.get(
                current.queue, 0
            ):
                self.resuming[current.queue] = asyncio.create_task(self.resume(current))
        elif current.playing and (
            current.finished
            or current.clock - self.saved_at.get(current.queue, -math.inf) >= SAVE_INTERVAL
        ):
            # Do not issue a completed write on every tick.
            if (
                not previous
                or not previous.finished
                or current.clock - self.saved_at.get(current.queue, -math.inf) >= SAVE_INTERVAL
            ):
                self.queue_save(current)

    async def resume(self, sample: Sample) -> None:
        """Apply a checked local bookmark through MA's native seek command."""
        stream = None
        try:
            async with asyncio.timeout(RESUME_WINDOW) as deadline:
                # A quick return must see its outgoing write before reading.
                await self.flush_track(sample.track)
                queue = self.provider.mass.player_queues.get(sample.queue)
                stream = queue.current_item.streamdetails if queue and queue.current_item else None
                gate = gate_for(stream) if stream else None
                if gate and gate.track == sample.track:
                    record = {"chapter_id": sample.track, "position_ms": round(gate.target * 1000)}
                else:
                    response = await self.provider._client.resolve_playback(sample.track)
                    record = (response.get("local_progress") or {}).get("record")
                if not record or record.get("chapter_id") != sample.track:
                    self.finish_resume(sample)
                    return
                position = record.get("position_ms")
                if (
                    record.get("fully_played")
                    or type(position) is not int
                    or not 2000 < position < sample.duration * 1000
                ):
                    self.finish_resume(sample)
                    return
                queue = self.provider.mass.player_queues.get(sample.queue)
                latest = self.snapshot(queue) if queue else None
                if (
                    not latest
                    or latest.identity != sample.identity
                    or not latest.playing
                    or self.busy(queue)
                    or latest.generation != sample.generation
                ):
                    return
                if latest.offset > latest.head + 5 or sample.queue not in self.pending:
                    self.finish_resume(sample)
                    return
                target_offset = max(latest.head, position / 1000)
                self.settling[sample.queue] = Settlement(
                    sample.identity, target_offset, time.monotonic() + 3
                )
                # The API-read budget must not cancel MA's playback preparation.
                # A deep seek through a transcoded chapter can take >8 seconds;
                # MA also has its own source-buffer and player-start deadlines.
                deadline.reschedule(asyncio.get_running_loop().time() + SEEK_TIMEOUT)
                if gate and (preloader := getattr(self.provider, "_resume_preloader", None)):
                    preloader.allow_handoff(stream)
                if gate and stream and stream.buffer:
                    # Otherwise MA could satisfy a short seek from cached silence.
                    await stream.buffer.clear()
                    # Clearing an encoder is asynchronous. Respect a user command
                    # that selected another item or paused while it was closing.
                    queue = self.provider.mass.player_queues.get(sample.queue)
                    current = self.snapshot(queue) if queue else None
                    if (
                        not current
                        or current.identity != sample.identity
                        or current.generation != sample.generation
                        or not current.playing
                        or self.busy(queue)
                    ):
                        self.settling.pop(sample.queue, None)
                        return
                await self.provider.mass.player_queues.seek(
                    sample.queue, int(sample.start + target_offset)
                )
                now = self.samples.get(sample.queue)
                if now and now.identity == sample.identity:
                    self.samples[sample.queue] = replace(
                        latest,
                        offset=target_offset,
                        clock=time.monotonic(),
                        observed_at=time.time(),
                    )
                self.finish_resume(sample)
        except Exception as error:
            self.warn(error)
            self.settling.pop(sample.queue, None)
            self.retry_at[sample.queue] = time.monotonic() + 1
        finally:
            if stream and (preloader := getattr(self.provider, "_resume_preloader", None)):
                preloader.discard(sample.queue, stream)
            self.resuming.pop(sample.queue, None)

    def finish_resume(self, sample: Sample) -> None:
        """Clear pending flags only for the chapter that completed its resume."""
        current = self.samples.get(sample.queue)
        if current and current.identity == sample.identity:
            self.pending.discard(sample.queue)
            self.pending_since.pop(sample.queue, None)
            self.retry_at.pop(sample.queue, None)

    async def mark_tail_completed(self, queue_id: str) -> None:
        """Mark completion immediately before our tail skipper's final stop or next."""
        sample = self.samples.get(queue_id)
        queue = self.provider.mass.player_queues.get(queue_id)
        latest = self.snapshot(queue) if queue else None
        if (
            sample
            and latest
            and sample.identity == latest.identity
            and latest.playing
            and latest.offset >= latest.duration - latest.tail
        ):
            latest = replace(latest, finished=True)
            self.samples[queue_id] = latest
            self.queue_save(latest)

    async def run(self) -> None:
        """Poll queue snapshots without putting network I/O on the observer path."""
        while True:
            # tick performs no network I/O. API reads/writes have separate tasks.
            for queue in self.provider.mass.player_queues.all():
                try:
                    await self.tick(queue)
                except Exception as error:
                    self.warn(error)
            await asyncio.sleep(0.25)
