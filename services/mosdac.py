"""
services/mosdac.py — MOSDAC API: authentication, file search, HDF5 download.

Adapted directly from recent.py and manual.py:
  • get_token / refresh_token / logout
  • search_one_day (NO boundingBox — MOSDAC filter is broken)
  • find_latest_available_slot
  • find_files_for_slots (8-min tolerance matching)
  • download_file (with HDF5 validation + retry)
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import h5py

from config import (
    TOKEN_URL, SEARCH_URL, DOWNLOAD_URL, REFRESH_URL, LOGOUT_URL,
    DATASET_ID, MAX_DAYS_BACK, DOWNLOAD_MAX_ATTEMPTS, DOWNLOAD_CHUNK_SIZE,
)
from utils.time_utils import snap_to_nearest_slot

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═════════════════════════════════════════════════════════════════════════════

def get_token(username: str, password: str, max_attempts: int = 5) -> dict:
    """Authenticate with MOSDAC and return token dict {access_token, refresh_token}."""
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(
                TOKEN_URL,
                json={"username": username, "password": password},
                timeout=30,
            )
            r.raise_for_status()
            d = r.json()
            return {
                "access_token":  d["access_token"],
                "refresh_token": d["refresh_token"],
            }
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"[MOSDAC auth] Network error (attempt {attempt}/{max_attempts}): {e}")
            if attempt == max_attempts:
                raise
            time.sleep(5 * attempt)


def do_refresh_token(refresh_tok: str) -> dict:
    """Refresh an expired access token."""
    r = requests.post(REFRESH_URL, json={"refresh_token": refresh_tok}, timeout=30)
    r.raise_for_status()
    return r.json()


def logout(username: str) -> None:
    """Logout from MOSDAC — best-effort, never raises."""
    try:
        requests.post(LOGOUT_URL, json={"username": username}, timeout=10)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# SEARCH  — boundingBox intentionally omitted (MOSDAC filter is broken)
# ═════════════════════════════════════════════════════════════════════════════

def search_one_day(date_str: str, count: int = 96) -> list[dict]:
    """
    Query MOSDAC for all CMK files on a single calendar day (YYYY-MM-DD).
    Returns list of dicts sorted oldest → newest, or [] if no data.

    CRITICAL: No 'boundingBox' parameter — the MOSDAC API silently drops all
    results when boundingBox is provided.  India is cropped manually after
    download using IndiaCloudExtractor.  (See comment in both scripts.)
    """
    params = {
        "datasetId": DATASET_ID,
        "startTime": date_str,
        "endTime":   date_str,
        "count":     str(count),
        # boundingBox intentionally omitted
    }
    try:
        r = requests.get(SEARCH_URL, params=params, timeout=30)
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning(f"[MOSDAC search] Network error for {date_str}: {e}")
        return []

    if r.status_code in (400, 500):
        return []

    try:
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"[MOSDAC search] HTTP error {r.status_code} for {date_str}: {e}")
        return []

    entries = r.json().get("entries", [])
    results = []
    for e in entries:
        try:
            updated = datetime.strptime(
                e["updated"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            updated = None
        results.append({
            "id":         e["id"],
            "identifier": e["identifier"],
            "updated":    updated,
        })

    results.sort(
        key=lambda x: x["updated"] or datetime.min.replace(tzinfo=timezone.utc)
    )
    return results


def find_latest_available_slot(max_days_back: int = MAX_DAYS_BACK) -> tuple[datetime, list]:
    """
    Walk backwards day by day until MOSDAC has data.
    Returns (latest_snapped_slot, all_entries_that_day).
    Mirrors find_latest_available_slot() in recent.py.
    """
    today = datetime.now(timezone.utc).date()

    for days_back in range(1, max_days_back + 1):
        candidate = today - timedelta(days=days_back)
        date_str  = candidate.strftime("%Y-%m-%d")
        entries   = search_one_day(date_str)

        if entries:
            latest_dt = entries[-1]["updated"]
            snapped   = snap_to_nearest_slot(latest_dt)
            logger.info(
                f"[MOSDAC] Latest data: {date_str} | "
                f"last file {latest_dt.strftime('%H:%M')} UTC → snapped {snapped.strftime('%H:%M')} UTC"
            )
            return snapped, entries

        logger.debug(f"[MOSDAC] {date_str} — no data, going further back…")

    raise RuntimeError(
        f"No MOSDAC data found in the last {max_days_back} days. "
        "Check dataset ID or internet connection."
    )


def find_files_for_slots(slots: list[datetime]) -> list[dict]:
    """
    Match each slot to the closest MOSDAC file within an 8-minute window.
    Mirrors find_files_for_slots() in both scripts.
    """
    dates   = sorted({s.strftime("%Y-%m-%d") for s in slots})
    entries = []
    for d in dates:
        entries += search_one_day(d, count=96)

    if not entries:
        raise ValueError(f"No MOSDAC files found for dates: {dates}.")

    matched = []
    for slot in slots:
        best, best_diff = None, timedelta(minutes=8)
        for e in entries:
            if e["updated"] is None:
                continue
            diff = abs(e["updated"] - slot)
            if diff < best_diff:
                best_diff, best = diff, e

        if best is None:
            available = [
                e["updated"].strftime("%H:%M") for e in entries if e["updated"]
            ]
            raise ValueError(
                f"No file within 8 min of {slot.strftime('%Y-%m-%d %H:%M')} UTC.\n"
                f"Available times that day: {available}"
            )
        matched.append(best)

    return matched


# ═════════════════════════════════════════════════════════════════════════════
# HDF5 VALIDATION  (from recent.py — catches truncated downloads)
# ═════════════════════════════════════════════════════════════════════════════

def _is_valid_h5(path: str | Path) -> bool:
    """
    Return True if h5py can open the file and minimal keys exist.
    Adapted from _is_valid_h5() in recent.py.
    """
    try:
        with h5py.File(str(path), "r") as f:
            keys = list(f.keys())
            if ("Latitude" in keys and "Longitude" in keys) or ("CMK" in keys):
                return True
            if keys:
                ds = f[keys[0]]
                _ = ds.shape
                return True
            return False
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD  (with HDF5 validation + retry — from recent.py)
# ═════════════════════════════════════════════════════════════════════════════

def download_file(
    record_id: str,
    identifier: str,
    access_token: str,
    dest_dir: str | Path,
    refresh_tok: str | None = None,
    max_attempts: int = DOWNLOAD_MAX_ATTEMPTS,
) -> str:
    """
    Download an INSAT-3DR HDF5 file from MOSDAC.

    • Uses a .part temp file to avoid partial writes.
    • Validates downloaded HDF5; deletes and retries if corrupt.
    • Refreshes access token on 401.

    Adapted from download_file() in recent.py (includes HDF5 validation).
    Returns the path to the final valid file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / identifier
    tmp  = dest_dir / (identifier + ".part")

    # Check cache first
    if dest.exists():
        if _is_valid_h5(dest):
            logger.info(f"[cache] {identifier} — valid, skipping download")
            return str(dest)
        else:
            logger.warning(f"[cache] {identifier} — corrupted, re-downloading")
            dest.unlink(missing_ok=True)

    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {"id": record_id}

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(
                DOWNLOAD_URL, headers=headers, params=params,
                stream=True, timeout=120,
            )

            # Refresh token if expired
            if r.status_code == 401 and refresh_tok:
                logger.info("[token] Refreshing expired MOSDAC token…")
                new_tokens   = do_refresh_token(refresh_tok)
                access_token = new_tokens["access_token"]
                headers["Authorization"] = f"Bearer {access_token}"
                r = requests.get(
                    DOWNLOAD_URL, headers=headers, params=params,
                    stream=True, timeout=120,
                )

            r.raise_for_status()

            total      = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            sys.stdout.write(f"\r  ↓ {identifier}  {pct:5.1f}%")
                            sys.stdout.flush()
            print()

            # Rename temp → final
            try:
                os.replace(tmp, dest)
            except Exception:
                os.rename(tmp, dest)

            # Validate the downloaded file
            if _is_valid_h5(dest):
                logger.info(f"[download] {identifier} — OK")
                return str(dest)
            else:
                logger.warning(
                    f"[download] {identifier} invalid after download "
                    f"(attempt {attempt}/{max_attempts}), retrying…"
                )
                dest.unlink(missing_ok=True)
                time.sleep(3 * attempt)
                continue

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(
                f"[download] Network error (attempt {attempt}/{max_attempts}): {e}"
            )
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(5 * attempt)
            continue

        except requests.HTTPError as e:
            logger.error(f"[download] HTTP error downloading {identifier}: {e}")
            raise

    raise RuntimeError(
        f"Failed to download a valid copy of {identifier} "
        f"after {max_attempts} attempts."
    )
