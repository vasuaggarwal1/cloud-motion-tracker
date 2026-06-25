"""
services/overlay.py — Convert binary cloud masks to RGBA PNG overlays.

Model now outputs at native 758x929 — same as extracted input masks.
No resizing needed. Both input and prediction PNGs are the same pixel shape
and both overlay on the same INDIA_BOUNDS lat/lon rectangle.
"""

import io
import logging
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

CLOUD_R, CLOUD_G, CLOUD_B = 220, 240, 255
CLOUD_ALPHA                = 195   # ~76% opacity


def mask_to_rgba_png(binary_mask: np.ndarray) -> bytes:
    """
    Convert 2-D binary mask (uint8, 0=clear, 1=cloud) -> RGBA PNG bytes.
    Cloud -> soft blue-white semi-transparent. Clear -> fully transparent.
    """
    h, w  = binary_mask.shape
    rgba  = np.zeros((h, w, 4), dtype=np.uint8)
    cloud = binary_mask == 1
    rgba[cloud, 0] = CLOUD_R
    rgba[cloud, 1] = CLOUD_G
    rgba[cloud, 2] = CLOUD_B
    rgba[cloud, 3] = CLOUD_ALPHA
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def save_mask_as_overlay(binary_mask: np.ndarray, out_path) -> str:
    """Save a binary mask directly as RGBA PNG — no resizing."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(mask_to_rgba_png(binary_mask))
    logger.debug(f"[overlay] Saved {out_path}  shape={binary_mask.shape}")
    return str(out_path)


def save_all_overlays(
    input_masks:  list,
    pred_masks:   np.ndarray,
    input_labels: list,
    pred_labels:  list,
    out_dir,
    prefix: str = "",
) -> list:
    """
    Save input + prediction frames as RGBA PNGs at native resolution.
    Both are 758x929 — no resizing required.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for i, (mask, label) in enumerate(zip(input_masks, input_labels)):
        fname = f"{prefix}input_{i:02d}.png"
        save_mask_as_overlay(mask, out_dir / fname)
        frames.append({"filename": fname, "label": label, "type": "input"})

    for t in range(pred_masks.shape[0]):
        fname = f"{prefix}pred_{t:02d}.png"
        save_mask_as_overlay(pred_masks[t], out_dir / fname)
        frames.append({"filename": fname, "label": pred_labels[t], "type": "pred"})

    return frames


def save_real_frames_only(
    masks:  list,
    labels: list,
    out_dir,
    prefix: str = "",
) -> list:
    """Save a batch of real observed frames — used by manual browse (no inference)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        fname = f"{prefix}real_{i:02d}.png"
        save_mask_as_overlay(mask, out_dir / fname)
        frames.append({"filename": fname, "label": label, "type": "real"})
    return frames