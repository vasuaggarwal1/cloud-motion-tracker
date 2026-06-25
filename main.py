"""
main.py — Cloud Motion Tracker FastAPI entrypoint.
uvicorn main:app --host 127.0.0.1 --port 8000uvicorn main:app --host 127.0.0.1 --port 8000
Startup sequence:
  1. Load ONNX model (singleton — used by all requests)
  2. Mount /static/ for serving overlay PNGs
  3. Register API routers
  4. Start APScheduler — runs real-time pipeline every 30 min

Run with:
  python main.py
  -- or --
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import (
    BASE_DIR, REALTIME_DIR, TEMP_DIR, DATA_DIR, SCHEDULER_INTERVAL_MINUTES,
    load_credentials,
)
from services.inference import load_model, get_predictor
from services.mosdac import (
    get_token, logout,
    find_latest_available_slot, find_files_for_slots, download_file,
)
from services.preprocessing import IndiaCloudExtractor, preprocess, postprocess
from services.overlay import save_all_overlays
from utils.time_utils import (
    four_slots_ending_at, prediction_slots_after,
    fmt_slot, fmt_iso, slot_key, now_utc,
)
from utils.file_utils import static_url

import routes.realtime as rt_routes
import routes.manual     as man_routes    # paginated real-image browser (no inference)
import routes.assistant  as ai_routes     # RAG AI assistant endpoint
import routes.summary    as summary_routes

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cloud Motion Tracker — India",
    description="INSAT-3DR cloud motion prediction API (CREvNet / MOSDAC)",
    version="1.0.0",
)

# ─── CORS (allow frontend served on any origin during dev) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─── Static file serving ──────────────────────────────────────────────────────
# /static/realtime/* → data/realtime/
# /static/archive/*  → data/archive/
app.mount("/static/realtime", StaticFiles(directory=str(REALTIME_DIR)), name="realtime_static")
app.mount("/static/archive",  StaticFiles(directory=str(DATA_DIR / "archive")),  name="archive_static")

# ─── Frontend HTML page routes ────────────────────────────────────────────────
FRONTEND_DIR = BASE_DIR / "static_frontend"

NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

@app.get("/", include_in_schema=False)
async def serve_root():
    return FileResponse(str(FRONTEND_DIR / "index.html"), headers=NO_CACHE)

@app.get("/index.html", include_in_schema=False)
async def serve_index_html():
    return FileResponse(str(FRONTEND_DIR / "index.html"), headers=NO_CACHE)

@app.get("/realtime.html", include_in_schema=False)
async def serve_realtime():
    return FileResponse(str(FRONTEND_DIR / "realtime.html"), headers=NO_CACHE)

@app.get("/manual.html", include_in_schema=False)
async def serve_manual():
    return FileResponse(str(FRONTEND_DIR / "manual.html"), headers=NO_CACHE)

@app.get("/summary.html", include_in_schema=False)
async def serve_summary_page():
    return FileResponse(str(FRONTEND_DIR / "summary.html"), headers=NO_CACHE)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(rt_routes.router)
app.include_router(man_routes.router)
app.include_router(ai_routes.router)
app.include_router(summary_routes.router)


# ═════════════════════════════════════════════════════════════════════════════
# REAL-TIME SCHEDULER PIPELINE
# Logic adapted from recent.py  main()
# ═════════════════════════════════════════════════════════════════════════════

def run_realtime_pipeline() -> None:
    """
    Scheduled task: fetch latest MOSDAC data, run inference, save overlays.
    Updates data/realtime/metadata.json consumed by GET /api/realtime/metadata.
    """
    logger.info("═══ Scheduler: real-time pipeline starting ═══")

    username, password = load_credentials()
    if not username or not password:
        logger.error(
            "[scheduler] MOSDAC credentials not set. "
            "Provide MOSDAC_USERNAME/MOSDAC_PASSWORD env vars or config.json."
        )
        return

    try:
        predictor = get_predictor()
    except RuntimeError as e:
        logger.error(f"[scheduler] Model not loaded: {e}")
        return

    try:
        # ── 1. Find latest available slot (mirrors recent.py) ──────────────
        latest_slot, _ = find_latest_available_slot()
        input_slots    = four_slots_ending_at(latest_slot)
        logger.info(
            f"[scheduler] Input slots: "
            f"{[s.strftime('%d %b %H:%M') for s in input_slots]} UTC"
        )

        # ── 2. Authenticate ────────────────────────────────────────────────
        tokens        = get_token(username, password)
        access_token  = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        # ── 3. Match slots → MOSDAC files ─────────────────────────────────
        entries = find_files_for_slots(input_slots)

        # ── 4. Download HDF5 files to temp/realtime/ ──────────────────────
        temp_rt = TEMP_DIR / "realtime"
        temp_rt.mkdir(parents=True, exist_ok=True)

        h5_paths = []
        for entry in entries:
            path = download_file(
                entry["id"], entry["identifier"],
                access_token, temp_rt, refresh_token,
            )
            h5_paths.append(path)

        # ── 5. Extract India cloud masks ───────────────────────────────────
        extractor = IndiaCloudExtractor(h5_paths[0])
        binary_images = []
        for path in h5_paths:
            binary, meta = extractor.extract(path)
            binary_images.append(binary)
            logger.info(
                f"[scheduler] {Path(path).name}: "
                f"cloud={meta['cloud_frac']:.2f} shape={meta['shape']}"
            )

        # ── 6. ONNX inference ──────────────────────────────────────────────
        model_input  = preprocess(binary_images)
        raw_preds    = predictor.predict(model_input)
        # Pass input frames for optical-flow-guided postprocessing
        binary_preds = postprocess(raw_preds, input_frames=binary_images)
        logger.info(
            f"[scheduler] Inference done. Shape {raw_preds.shape}. "
            f"Cloud cover {binary_preds.mean()*100:.1f}%"
        )

        # ── 7. Save overlay PNGs ───────────────────────────────────────────
        pred_slots   = prediction_slots_after(latest_slot, n=raw_preds.shape[0])
        input_labels = [fmt_slot(s) for s in input_slots]
        pred_labels  = [fmt_slot(s) for s in pred_slots]

        frame_meta = save_all_overlays(
            input_masks  = binary_images,
            pred_masks   = binary_preds,
            input_labels = input_labels,
            pred_labels  = pred_labels,
            out_dir      = REALTIME_DIR,
            prefix       = "rt_",
        )

        # ── 8. Write metadata.json ─────────────────────────────────────────
        all_slots = input_slots + pred_slots
        frames    = []
        for i, fm in enumerate(frame_meta):
            ts_slot = all_slots[i]
            url     = static_url(f"realtime/{fm['filename']}")
            frames.append({
                "url":       url,
                "timestamp": fmt_iso(ts_slot),
                "label":     fm["label"],
                "type":      fm["type"],
            })

        metadata = {
            "frames":      frames,
            "lastUpdated": fmt_iso(now_utc()),
            "latestSlot":  fmt_iso(latest_slot),
        }
        (REALTIME_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info(
            f"[scheduler] Metadata written. "
            f"{len(frames)} frames ({len(input_slots)} input + {len(pred_slots)} pred)."
        )

        logout(username)
        logger.info("═══ Scheduler: pipeline complete ═══")

    except Exception as e:
        logger.error(
            f"[scheduler] Pipeline failed: {e}\n{traceback.format_exc()}"
        )
        try:
            logout(username)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ═════════════════════════════════════════════════════════════════════════════

scheduler = BackgroundScheduler(timezone="UTC")


@app.on_event("startup")
async def startup_event():
    # ── Load model (once, at startup) ─────────────────────────────────────
    try:
        load_model()
        logger.info("[startup] ONNX model loaded successfully.")
            # ── Seed empty metadata so frontend doesn't get 503 ───────────────────
        meta_path = REALTIME_DIR / "metadata.json"
        if not meta_path.exists():
            REALTIME_DIR.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({
                "frames": [],
                "lastUpdated": None,
                "latestSlot": None,
                "status": "pending"
            }))
            logger.info("[startup] Seeded empty metadata.json")
    except FileNotFoundError as e:
        logger.warning(
            f"[startup] ONNX model NOT loaded: {e}\n"
            "API will start but inference endpoints will fail until model is placed at the configured path."
        )

    # ── Start real-time scheduler ──────────────────────────────────────────
    scheduler.add_job(
        run_realtime_pipeline,
        trigger=IntervalTrigger(minutes=SCHEDULER_INTERVAL_MINUTES),
        id="realtime_pipeline",
        name="Real-time cloud prediction",
        replace_existing=True,
        max_instances=1,           # don't overlap runs
        coalesce=True,
    )
    scheduler.start()
    logger.info(
        f"[startup] Scheduler started — "
        f"real-time pipeline runs every {SCHEDULER_INTERVAL_MINUTES} min."
    )

    # Run once immediately on startup (in background thread)
    import threading
    threading.Thread(target=run_realtime_pipeline, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown(wait=False)
    logger.info("[shutdown] Scheduler stopped.")


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    try:
        get_predictor()
        model_ok = True
    except RuntimeError:
        model_ok = False

    rt_ready = (REALTIME_DIR / "metadata.json").exists()

    return JSONResponse({
        "status":     "ok",
        "model":      "loaded" if model_ok else "not loaded",
        "realtime":   "ready"  if rt_ready else "pending",
        "utcTime":    fmt_iso(now_utc()),
    })


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",   # localhost — shows a clickable link in the terminal
        port=8000,
        reload=True,
    )