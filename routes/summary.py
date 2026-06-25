"""
routes/summary.py — GET /api/summary
LLM-generated weather narrative from latest realtime metadata.
Uses Groq (llama3-8b-8192).
"""

import os
import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from groq import Groq

from config import REALTIME_DIR

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        config_path = Path(__file__).parent.parent / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            api_key = cfg.get("groq_api_key")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in env or config.json")
    return Groq(api_key=api_key)


def _build_prompt(metadata: dict) -> str:
    frames       = metadata.get("frames", [])
    last_updated = metadata.get("lastUpdated", "unknown")
    latest_slot  = metadata.get("latestSlot", "unknown")

    input_frames = [f for f in frames if f.get("type") == "input"]
    pred_frames  = [f for f in frames if f.get("type") == "pred"]

    def fmt_frames(flist):
        return "\n".join(
            f"  - {f['label']} ({f['timestamp']})"
            for f in flist
        ) or "  (none)"

    return f"""You are a meteorological assistant analysing INSAT-3DR satellite cloud imagery over India.

The following data was produced by a deep learning cloud motion prediction model (CREvNet) at {last_updated} UTC.
Latest observed satellite slot: {latest_slot} UTC.

OBSERVED INPUT FRAMES (real satellite data):
{fmt_frames(input_frames)}

PREDICTED FUTURE FRAMES (model output):
{fmt_frames(pred_frames)}

Based on this temporal sequence of cloud cover observations and predictions, write a concise,
informative weather narrative (3-5 sentences) suitable for a general audience in India.
Mention the time window covered, whether cloud cover is increasing or decreasing,
and any notable implications (e.g. possible rain, clearing skies, monsoon activity if relevant).
Do not speculate beyond what the data implies. Be factual and clear.
"""


@router.get("/api/summary")
async def get_summary():
    meta_path = REALTIME_DIR / "metadata.json"

    if not meta_path.exists():
        return JSONResponse(
            {"error": "No realtime metadata available yet."},
            status_code=503,
        )

    metadata = json.loads(meta_path.read_text())

    if not metadata.get("frames"):
        return JSONResponse(
            {"error": "Metadata exists but contains no frames yet."},
            status_code=503,
        )

    try:
        client = _get_groq_client()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    prompt = _build_prompt(metadata)
    logger.info("[summary] Sending prompt to Groq...")

    try:
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise, accurate meteorological assistant. "
                        "You interpret satellite cloud imagery data and produce "
                        "clear weather summaries for the Indian subcontinent."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        summary = chat.choices[0].message.content.strip()
        logger.info("[summary] Summary generated successfully.")
    except Exception as e:
        logger.error(f"[summary] Groq error: {e}")
        return JSONResponse({"error": f"Groq API error: {e}"}, status_code=502)

    return JSONResponse({
        "summary":     summary,
        "model":       "llama3-8b-8192",
        "generatedAt": metadata.get("lastUpdated"),
        "latestSlot":  metadata.get("latestSlot"),
    })