"""
config.py — Central configuration for Cloud Motion Tracker backend.
"""

import os
import json
import re
import math
from pathlib import Path

# ─── Directory roots ──────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
DATA_DIR      = BASE_DIR / "data"
REALTIME_DIR  = DATA_DIR / "realtime"
ARCHIVE_DIR   = DATA_DIR / "archive"
TEMP_DIR      = DATA_DIR / "temp"

for _d in (REALTIME_DIR, ARCHIVE_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── ONNX Model ───────────────────────────────────────────────────────────────
# Supports .onnx or .keras or .h5 — inference.py auto-detects by extension
ONNX_MODEL_PATH: str = os.environ.get(
    "ONNX_MODEL_PATH",
    str(BASE_DIR / "checkpoints" / "best_model.onnx"),   # also checks .keras/.h5 fallback
)

# ─── MOSDAC API endpoints ─────────────────────────────────────────────────────
TOKEN_URL    = "https://mosdac.gov.in/download_api/gettoken"
SEARCH_URL   = "https://mosdac.gov.in/apios/datasets.json"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
REFRESH_URL  = "https://mosdac.gov.in/download_api/refresh-token"
LOGOUT_URL   = "https://mosdac.gov.in/download_api/logout"

DATASET_ID = "3RIMG_L2B_CMK"

# ─── India geographic bounding box ───────────────────────────────────────────
INDIA_BBOX = {
    "LAT_MIN": 5.0,
    "LAT_MAX": 35.0,
    "LON_MIN": 65.0,
    "LON_MAX": 100.0,
}

# ─── Model native resolution (from predict_and_show.py) ──────────────────────
# Model outputs at native India-crop resolution 758x929 — NO resizing needed.
NATIVE_H = 758
NATIVE_W = 929
# Input must be padded to next multiple-of-8 for the model
PAD_H    = math.ceil(NATIVE_H / 8) * 8   # 760
PAD_W    = math.ceil(NATIVE_W / 8) * 8   # 936

# ─── Scheduler ───────────────────────────────────────────────────────────────
SCHEDULER_INTERVAL_MINUTES = 30
MAX_DAYS_BACK              = 15

# ─── Manual mode ─────────────────────────────────────────────────────────────
MANUAL_BATCH_SIZE = 8   # real images shown per page

# ─── Download settings ───────────────────────────────────────────────────────
DOWNLOAD_MAX_ATTEMPTS = 3
DOWNLOAD_CHUNK_SIZE   = 1 << 20   # 1 MiB


def _preprocess_json(raw: str) -> str:
    raw = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', raw)
    raw = re.sub(r'(?<!\\)\\(?=\s*")', r'\\\\', raw)
    return raw


def load_credentials() -> tuple[str, str]:
    env_user = os.environ.get("MOSDAC_USERNAME", "")
    env_pass = os.environ.get("MOSDAC_PASSWORD", "")
    if env_user and env_pass:
        return env_user, env_pass
    cfg_path = BASE_DIR / "config.json"
    if cfg_path.exists():
        raw = cfg_path.read_text()
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            cfg = json.loads(_preprocess_json(raw))
        creds = cfg.get("user_credentials", {})
        return (creds.get("username/email", ""), creds.get("password", ""))
    return "", ""