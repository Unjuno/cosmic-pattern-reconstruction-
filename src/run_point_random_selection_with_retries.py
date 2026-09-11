#!/usr/bin/env python3
"""Run the fixed point-random selection experiment with transport-only retries.

This wrapper changes no field set, sampling fraction, features, model, null, or
decision rule. It retries only failed HTTP Range transactions so a transient
NERSC connection timeout is not misclassified as an experimental result.
"""
from __future__ import annotations

import time

import point_random_selection_residual_validate as experiment

_ORIGINAL_FETCH_RANGE = experiment.fetch_range


def retry_fetch_range(url: str, start: int, end: int, timeout: int = 360):
    delays = (0, 5, 15, 30)
    last = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        try:
            return _ORIGINAL_FETCH_RANGE(url, start, end, timeout)
        except Exception as exc:
            last = exc
            print(
                f"[point-selection-transport] Range attempt {attempt}/{len(delays)} "
                f"failed for bytes={start}-{end} url={url}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    raise RuntimeError(
        f"HTTP Range failed after {len(delays)} attempts for bytes={start}-{end} url={url}: {last}"
    ) from last


experiment.fetch_range = retry_fetch_range

if __name__ == "__main__":
    raise SystemExit(experiment.main())
