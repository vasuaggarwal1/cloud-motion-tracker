"""
utils/file_utils.py — Filesystem utility helpers.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def remove_if_exists(path: Path | str) -> None:
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except Exception as e:
            logger.warning(f"Could not remove {p}: {e}")


def write_json(path: Path | str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def read_json(path: Path | str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning(f"Could not read JSON {p}: {e}")
        return None


def archive_key_from_slot(snapped_slot: datetime) -> str:
    """Return a directory-safe key for a snapped slot, e.g. '20240219_1445'."""
    return snapped_slot.strftime("%Y%m%d_%H%M")


def archive_dir_for_slot(archive_root: Path, snapped_slot: datetime) -> Path:
    return archive_root / archive_key_from_slot(snapped_slot)


def static_url(relative_path: str) -> str:
    """Build a /static/... URL from a relative path using forward slashes."""
    return "/static/" + relative_path.replace("\\", "/")
