"""Shared output gate and XR freshness helpers for hardware control."""

import time


def shared_flag_is_set(shared_flag) -> bool:
    """Read a multiprocessing-style shared boolean.

    ``None`` keeps controller classes backward compatible with existing callers.
    The teleoperation entry point always supplies an explicit, initially-false gate.
    """
    if shared_flag is None:
        return True
    with shared_flag.get_lock():
        return bool(shared_flag.value)


def set_shared_flag(shared_flag, enabled: bool) -> None:
    if shared_flag is None:
        return
    with shared_flag.get_lock():
        shared_flag.value = bool(enabled)


def motion_input_is_live(ready: bool, timestamp: float, timeout: float, now: float | None = None) -> bool:
    """Return whether XR motion has been seen recently enough to control hardware."""
    if not ready or timestamp <= 0 or timeout <= 0:
        return False
    if now is None:
        now = time.monotonic()
    age = now - timestamp
    return 0 <= age <= timeout + 1e-9
