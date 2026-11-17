"""Records every frame escalated to the verifier together with its verdict (day-zero tuning data).

Each actual verifier call becomes one JPEG plus one line in ``escalations.jsonl``; escalations skipped because of
the hourly call cap become a JSONL line without an image. All disk I/O and JPEG encoding run in worker threads,
the caller never waits and never sees an exception: failed or overflowing writes are logged (rate-limited) and
dropped. The directory is kept under a size limit by deleting the oldest JPEGs (the index lines are kept).
"""

import asyncio
from collections import deque
import contextlib
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from is_goat_burning.config import EscalationLogSettings
from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger

logger = get_logger("EscalationLog")

INDEX_NAME = "escalations.jsonl"
ROTATED_NAME = "escalations.jsonl.1"
#: Rotate the index when it grows beyond this many bytes (further capped to a quarter of the size limit).
MAX_INDEX_BYTES = 50 * 1024 * 1024
#: Maximum number of queued/running writes; further records are dropped.
MAX_IN_FLIGHT = 4
#: Minimum seconds between two logged write failures / drops.
WARN_INTERVAL_SECONDS = 60.0
#: Minimum seconds between two ``capped`` index lines (at the cap every CV-positive frame would write one).
CAPPED_INTERVAL_SECONDS = 60.0


class EscalationLog:
    """Non-blocking, size-limited recorder of verifier escalations."""

    def __init__(self, directory: str | Path, max_mb: int, jpeg_quality: int = 85) -> None:
        """Initializes the log; the directory is created and scanned lazily on the first write.

        Args:
            directory: Target directory for the JPEG frames and the JSONL index.
            max_mb: Size limit of the directory in MB; the oldest JPEGs are deleted beyond it.
            jpeg_quality: JPEG quality (1-100).
        """
        self._dir = Path(directory)
        self._max_bytes = max_mb * 1024 * 1024
        self._quality = jpeg_quality
        self._lock = threading.Lock()
        self._scanned = False
        self._total = 0
        self._jpegs: deque[tuple[Path, int]] = deque()
        self._seq = 0
        self._in_flight = 0
        self._pending: set[asyncio.Task[None]] = set()
        self._last_warning = float("-inf")
        self._last_capped = float("-inf")
        self._suppressed_capped = 0
        self._write_order = asyncio.Lock()  # keeps the index in submission order (threads would race)

    @classmethod
    def from_settings(cls, cfg: EscalationLogSettings) -> EscalationLog:
        """Builds a log from its settings model.

        Args:
            cfg: The escalation log settings.

        Returns:
            A new ``EscalationLog``.
        """
        return cls(cfg.directory, cfg.max_mb, cfg.jpeg_quality)

    def record(
        self,
        frame: np.ndarray,
        verifier: str,
        verdict: str,
        probability: float | None,
        latency_ms: float,
        error_type: str | None = None,
        policy: str | None = None,
    ) -> None:
        """Queues the frame and its verdict for writing; never blocks and never raises.

        Args:
            frame: The BGR frame sent to the verifier (copied, so the caller may reuse it).
            verifier: Verifier class name.
            verdict: ``fire``, ``no_fire`` or ``error``.
            probability: ``P(fire)`` when the verifier exposes it.
            latency_ms: Verifier call duration in milliseconds.
            error_type: Exception class name for ``error`` (never the message, it may contain secrets).
            policy: Policy outcome applied to this call, if any (``inconclusive``, ``alert_unverified``, ``suppress``).
        """
        self._submit(frame, self._entry(verifier, verdict, probability, latency_ms, error_type, policy))

    def record_capped(self, verifier: str, policy: str | None) -> None:
        """Queues an index line (no image) for an escalation skipped due to the hourly call cap.

        Args:
            verifier: Verifier class name.
            policy: The failure policy applied.
        """
        now = time.monotonic()
        if now - self._last_capped < CAPPED_INTERVAL_SECONDS:
            self._suppressed_capped += 1
            return
        self._last_capped = now
        entry = self._entry(verifier, "capped", None, None, None, policy)
        entry["suppressed"] = self._suppressed_capped  # capped skips not written since the previous line
        self._suppressed_capped = 0
        self._submit(None, entry)

    @staticmethod
    def _entry(
        verifier: str,
        verdict: str,
        probability: float | None,
        latency_ms: float | None,
        error_type: str | None,
        policy: str | None,
    ) -> dict[str, object]:
        return {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "file": None,
            "verifier": verifier,
            "verdict": verdict,
            "probability": None if probability is None else round(float(probability), 4),
            "latency_ms": None if latency_ms is None else round(float(latency_ms), 1),
            "error_type": error_type,
            "policy": policy,
        }

    def _warn(self, message: str, *args: object, **kwargs: object) -> None:
        now = time.monotonic()
        if now - self._last_warning >= WARN_INTERVAL_SECONDS:
            self._last_warning = now
            logger.warning(message, *args, **kwargs)

    def _submit(self, frame: np.ndarray | None, entry: dict[str, object]) -> None:
        try:
            if self._in_flight >= MAX_IN_FLIGHT:
                self._warn("Escalation log backlog is full; dropping a record.")
                return
            payload = None if frame is None else frame.copy()
            self._in_flight += 1
            task = asyncio.get_running_loop().create_task(self._run(payload, entry))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        except Exception:  # noqa: BLE001 - the log must never break detection
            self._warn("Escalation log could not queue a record.", exc_info=True)

    async def _run(self, frame: np.ndarray | None, entry: dict[str, object]) -> None:
        try:
            async with self._write_order:
                await asyncio.to_thread(self._write, frame, entry)
        except Exception as exc:  # noqa: BLE001 - drop and report, never propagate
            self._warn("Escalation log write failed (%s); record dropped.", type(exc).__name__)
        finally:
            self._in_flight -= 1

    async def aclose(self) -> None:
        """Waits for the queued writes to finish (call on shutdown)."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    # --- worker-thread part ---

    def _scan(self) -> None:
        """Creates the directory and totals its current content (once)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        found: list[tuple[float, Path, int]] = []
        total = 0
        for path in self._dir.iterdir():
            try:
                st = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            total += st.st_size
            if path.suffix == ".jpg":
                found.append((st.st_mtime, path, st.st_size))
        found.sort(key=lambda item: (item[0], item[1].name))
        self._jpegs = deque((path, size) for _, path, size in found)
        self._total = total
        self._scanned = True

    def _write(self, frame: np.ndarray | None, entry: dict[str, object]) -> None:
        encoded: bytes | None = None
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                raise ValueError("JPEG encoding failed")
            encoded = buf.tobytes()
        with self._lock:
            if not self._scanned:
                self._scan()
            if encoded is not None:
                self._seq += 1
                name = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}Z_{self._seq:04d}_{entry['verdict']}.jpg"
                path = self._dir / name
                path.write_bytes(encoded)
                self._jpegs.append((path, len(encoded)))
                self._total += len(encoded)
                entry["file"] = name
            line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
            index = self._dir / INDEX_NAME
            self._rotate_index(index)  # before appending, so the index file always exists after a write
            with index.open("ab") as handle:
                handle.write(line)
            self._total += len(line)
            self._prune()

    def _rotate_index(self, index: Path) -> None:
        limit = min(MAX_INDEX_BYTES, self._max_bytes // 4)
        if not index.exists() or index.stat().st_size <= limit:
            return
        rotated = self._dir / ROTATED_NAME
        if rotated.exists():
            self._total -= rotated.stat().st_size
        index.replace(rotated)  # the previous rotated index is overwritten

    def _prune(self) -> None:
        """Deletes the oldest JPEGs, then the rotated index, until the total fits the limit."""
        while self._total > self._max_bytes and self._jpegs:
            path, size = self._jpegs.popleft()
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            self._total -= size
        if self._total > self._max_bytes:
            rotated = self._dir / ROTATED_NAME
            if rotated.exists():
                self._total -= rotated.stat().st_size
                rotated.unlink()


_instance: EscalationLog | None = None


def get_escalation_log() -> EscalationLog | None:
    """Returns the process-wide escalation log, or None when ``ESCALATION_LOG__ENABLED`` is false.

    Returns:
        The shared ``EscalationLog`` (created on first use) or None.
    """
    global _instance  # noqa: PLW0603
    if not settings.escalation_log.enabled:
        return None
    if _instance is None:
        _instance = EscalationLog.from_settings(settings.escalation_log)
        logger.info(
            "Escalation log enabled: directory=%s limit=%d MB",
            settings.escalation_log.directory,
            settings.escalation_log.max_mb,
        )
    return _instance


async def close_escalation_log() -> None:
    """Waits for pending escalation writes (no-op when disabled); call on application shutdown."""
    if _instance is not None:
        await _instance.aclose()
