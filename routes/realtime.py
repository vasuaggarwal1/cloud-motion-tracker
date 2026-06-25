"""
routes/realtime.py — GET /api/realtime/metadata

Returns pre-generated frame metadata from the scheduler.
Does NOT run inference on request — that is the scheduler's job.
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config import REALTIME_DIR

logger   = logging.getLogger(__name__)
router   = APIRouter()

METADATA_FILE = REALTIME_DIR / "metadata.json"


@router.get("/api/realtime/metadata")
async def get_realtime_metadata():
    """
    Return the latest scheduler-generated frames.

    Response shape (consumed by realtime.html):
    {
      "frames": [
        { "url": "/static/realtime/input_00.png",
          "timestamp": "2024-02-19T14:15:00Z",
          "label":     "19 Feb 14:15 UTC",
          "type":      "input" },
        ...
        { "url": "/static/realtime/pred_00.png",
          "timestamp": "2024-02-19T15:15:00Z",
          "label":     "19 Feb 15:15 UTC",
          "type":      "pred"  },
        ...
      ],
      "lastUpdated": "2024-02-19T14:50:00Z"
    }
    """
    if not METADATA_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Real-time data not yet available. "
                "The scheduler has not completed its first run. "
                "Please try again in a few minutes."
            ),
        )

    try:
        meta = json.loads(METADATA_FILE.read_text())
    except Exception as e:
        logger.error(f"[realtime] Failed to read metadata.json: {e}")
        raise HTTPException(status_code=500, detail="Corrupt metadata file.")

    # Validate minimal structure
    if "frames" not in meta:
        raise HTTPException(status_code=500, detail="Corrupt metadata file.")

    return JSONResponse(content=meta)
