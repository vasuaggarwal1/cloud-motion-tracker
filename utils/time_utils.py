"""
utils/time_utils.py — MOSDAC time-slot helpers.

Rules (from manual.py / recent.py):
  • INSAT-3DR CMK files exist ONLY at HH:15 and HH:45 (UTC).
  • Snap any datetime to the nearest such mark.
  • Build 4 consecutive 30-min slots ending at a given slot.
  • Match files within an 8-minute tolerance window.
"""

from datetime import datetime, timedelta, timezone


# ─── Slot snapping ────────────────────────────────────────────────────────────

def snap_to_nearest_slot(dt: datetime) -> datetime:
    """
    Snap any UTC datetime to the nearest :15 or :45 mark.

    Examples (from manual.py docstring):
      14:35  →  14:45   (10 min away vs 20 min to 14:15)
      14:20  →  14:15   (5  min away vs 25 min to 14:45)
      14:30  →  14:45   (tie breaks toward :45)

    Identical logic to manual.py / recent.py  snap_to_nearest_slot().
    """
    dt = dt.replace(second=0, microsecond=0)
    h  = dt.replace(minute=0)
    candidates = [
        h - timedelta(minutes=15),   # :45 of previous hour
        h + timedelta(minutes=15),   # :15 of this hour
        h + timedelta(minutes=45),   # :45 of this hour
        h + timedelta(minutes=75),   # :15 of next hour
    ]
    return min(candidates, key=lambda c: abs((c - dt).total_seconds()))


def four_slots_ending_at(end_slot: datetime) -> list[datetime]:
    """
    Return 4 consecutive 30-min slots (oldest first) ending at end_slot.
    Mirrors four_slots_ending_at() in both scripts.
    """
    return [end_slot - timedelta(minutes=30 * i) for i in range(3, -1, -1)]


def prediction_slots_after(last_input_slot: datetime, n: int = 4) -> list[datetime]:
    """
    Return n prediction slots after the last input slot, each 30 min apart.
    """
    return [last_input_slot + timedelta(minutes=30 * (i + 1)) for i in range(n)]


# ─── Formatting helpers ───────────────────────────────────────────────────────

def fmt_slot(dt: datetime) -> str:
    """Human-readable UTC label: '19 Feb 14:45 UTC'"""
    return dt.strftime("%d %b %H:%M UTC")


def fmt_iso(dt: datetime) -> str:
    """ISO-8601 UTC string for JSON."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def slot_key(dt: datetime) -> str:
    """Filesystem-safe key for a slot, e.g. '20240219_1445'."""
    return dt.strftime("%Y%m%d_%H%M")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime_param(raw: str) -> datetime:
    """
    Parse a datetime string supplied by the frontend.
    Handles all ISO formats including JS toISOString() with milliseconds:
      2026-01-20T20:40:00.000Z  ← JS Date.toISOString()
      2026-01-20T20:40:00Z
      2026-01-20T20:40
      2026-01-20 20:40
    Always returns a UTC-aware datetime.
    """
    raw = raw.strip().rstrip("Z")
    # Strip milliseconds if present (e.g. "2026-01-20T20:40:00.000")
    if "." in raw:
        raw = raw.split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {raw!r}")