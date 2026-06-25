"""
routes/manual.py — Manual historical DATA BROWSER (no model inference).

Endpoints:
  GET /api/historical?datetime=ISO   
      Snap to :15/:45 slot, download 8 consecutive real frames from that slot,
      return as RGBA PNG overlays.  Results cached in data/archive/<key>/.

  GET /api/historical/status?jobId=X
      Poll for background download progress.

Navigation:
  The frontend moves forward/backward in 4-hour steps (8 x 30-min slots).
  It just calls /api/historical with a new datetime — no server-side state needed.
"""

import json
import logging
import threading
import traceback
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse

from config import ARCHIVE_DIR, TEMP_DIR, MANUAL_BATCH_SIZE, load_credentials
from services.mosdac import get_token, logout, find_files_for_slots, download_file
from services.preprocessing import IndiaCloudExtractor
from services.overlay import save_real_frames_only
from utils.time_utils import (
    snap_to_nearest_slot, fmt_slot, fmt_iso, slot_key, parse_datetime_param,
)
from utils.file_utils import write_json, read_json, static_url

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── In-memory job registry ──────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _batch_key(snapped_slot) -> str:
    """Unique dir-safe key for a batch starting at snapped_slot."""
    return f"browse_{slot_key(snapped_slot)}"


def _batch_dir(snapped_slot) -> Path:
    return ARCHIVE_DIR / _batch_key(snapped_slot)


def _eight_slots(start_slot) -> list:
    """8 consecutive 30-min slots starting at start_slot (oldest → newest)."""
    return [start_slot + timedelta(minutes=30 * i) for i in range(MANUAL_BATCH_SIZE)]


def _load_cached_frames(batch_dir: Path) -> list:
    meta = read_json(batch_dir / "metadata.json")
    if not meta or "frames" not in meta:
        return []
    return [
        {
            "url":       static_url(f"archive/{batch_dir.name}/{f['filename']}"),
            "label":     f.get("label", ""),
            "timestamp": f.get("timestamp", ""),
            "type":      "real",
        }
        for f in meta["frames"]
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Background pipeline — download 8 real frames, no inference
# ═════════════════════════════════════════════════════════════════════════════

def _run_browse_pipeline(
    job_id: str, start_slot, username: str, password: str
):
    def _upd(status, **kw):
        with _lock:
            _jobs[job_id].update({"status": status, **kw})

    batch_dir = _batch_dir(start_slot)
    slots     = _eight_slots(start_slot)
    temp_dir  = TEMP_DIR / job_id

    try:
        _upd("processing", message="Authenticating with MOSDAC…", progress=5)
        tokens = get_token(username, password)
        tok, rtok = tokens["access_token"], tokens["refresh_token"]

        _upd("processing", message="Locating 8 MOSDAC files…", progress=12)
        entries = find_files_for_slots(slots)

        h5_paths = []
        for i, entry in enumerate(entries):
            _upd("processing",
                 message=f"Downloading file {i+1}/{MANUAL_BATCH_SIZE}…",
                 progress=12 + i * 9)
            path = download_file(
                entry["id"], entry["identifier"], tok, temp_dir, rtok,
            )
            h5_paths.append(path)

        _upd("processing", message="Extracting India cloud masks…", progress=82)
        extractor = IndiaCloudExtractor(h5_paths[0])
        masks, labels = [], []
        for path, slot in zip(h5_paths, slots):
            binary, _ = extractor.extract(path)
            masks.append(binary)
            labels.append(fmt_slot(slot))

        _upd("processing", message="Generating overlay images…", progress=92)
        frame_meta = save_real_frames_only(
            masks=masks, labels=labels,
            out_dir=batch_dir, prefix="",
        )

        # Persist metadata
        full = [
            {**fm, "timestamp": fmt_iso(slots[i])}
            for i, fm in enumerate(frame_meta)
        ]
        write_json(batch_dir / "metadata.json", {"frames": full})

        # Build response frames
        response_frames = [
            {
                "url":       static_url(f"archive/{batch_dir.name}/{fm['filename']}"),
                "label":     fm["label"],
                "timestamp": fmt_iso(slots[i]),
                "type":      "real",
            }
            for i, fm in enumerate(frame_meta)
        ]

        logout(username)
        _upd("ready", message="Data ready", progress=100,
             frames=response_frames,
             startSlot=fmt_slot(start_slot),
             endSlot=fmt_slot(slots[-1]),
             startIso=fmt_iso(start_slot))
        logger.info(f"[manual] Job {job_id} done — {len(response_frames)} real frames.")

    except Exception as e:
        logger.error(f"[manual] Job {job_id} failed: {e}\n{traceback.format_exc()}")
        _upd("error", message=str(e))
        try:
            logout(username)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/api/historical")
async def get_historical(
    datetime: str = Query(..., description="ISO datetime string (UTC)"),
    background_tasks: BackgroundTasks = None,
):
    """
    Return 8 consecutive real INSAT-3DR cloud frames starting at the snapped slot.
    Results are cached. No model inference runs.
    """
    try:
        user_dt = parse_datetime_param(datetime)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    start_slot = snap_to_nearest_slot(user_dt)
    jid        = _batch_key(start_slot)
    batch_dir  = _batch_dir(start_slot)

    # ── Cache hit ─────────────────────────────────────────────────────────
    frames = _load_cached_frames(batch_dir)
    if frames:
        slots = _eight_slots(start_slot)
        return JSONResponse({
            "status":     "ready",
            "jobId":      jid,
            "frames":     frames,
            "startSlot":  fmt_slot(start_slot),
            "endSlot":    fmt_slot(slots[-1]),
            "startIso":   fmt_iso(start_slot),
        })

    # ── Already running ────────────────────────────────────────────────────
    with _lock:
        if jid in _jobs:
            j = _jobs[jid]
            resp = {
                "status":   j["status"],
                "jobId":    jid,
                "message":  j.get("message", ""),
                "progress": j.get("progress", 0),
                "startSlot": j.get("startSlot", fmt_slot(start_slot)),
            }
            if j["status"] == "ready":
                resp["frames"]    = j["frames"]
                resp["endSlot"]   = j.get("endSlot", "")
                resp["startIso"]  = j.get("startIso", "")
            return JSONResponse(resp)

        _jobs[jid] = {
            "status":   "processing",
            "message":  "Searching MOSDAC…",
            "progress": 0,
            "startSlot": fmt_slot(start_slot),
        }

    username, password = load_credentials()
    if not username:
        with _lock:
            del _jobs[jid]
        raise HTTPException(
            status_code=500,
            detail="MOSDAC credentials not configured.",
        )

    background_tasks.add_task(
        _run_browse_pipeline, jid, start_slot, username, password,
    )

    return JSONResponse({
        "status":    "processing",
        "jobId":     jid,
        "message":   "Searching MOSDAC…",
        "progress":  0,
        "startSlot": fmt_slot(start_slot),
    })


@router.get("/api/historical/status")
async def get_historical_status(
    jobId: str = Query(...),
):
    with _lock:
        job = _jobs.get(jobId)

    if job:
        resp = {
            "status":   job["status"],
            "jobId":    jobId,
            "message":  job.get("message", ""),
            "progress": job.get("progress", 0),
            "startSlot": job.get("startSlot", ""),
        }
        if job["status"] == "ready":
            resp["frames"]   = job["frames"]
            resp["endSlot"]  = job.get("endSlot", "")
            resp["startIso"] = job.get("startIso", "")
        return JSONResponse(resp)

    # Fallback: check archive directory (server restart after completion)
    batch_dir = ARCHIVE_DIR / jobId
    frames    = _load_cached_frames(batch_dir)
    if frames:
        return JSONResponse({
            "status": "ready", "jobId": jobId,
            "message": "Data ready", "frames": frames,
        })

    raise HTTPException(
        status_code=404,
        detail=f"Job '{jobId}' not found.",
    )