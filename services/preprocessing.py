"""
services/preprocessing.py — HDF5 → binary cloud mask pipeline.

Preprocessing matches predict_and_show.py exactly:
  - load_frame():  clip to NATIVE_H×NATIVE_W, reflect-pad to PAD_H×PAD_W
  - preprocess():  stack → (1, T, PAD_H, PAD_W, 1)
  - postprocess(): model output is already (T, 758, 929) — NO crop needed.
                   Weather-grade pipeline from predict_and_show.py:
                     1. Gaussian smoothing (σ=1.2)
                     2. Adaptive threshold (65th percentile)
                     3. Morphological closing + opening
                     4. Remove small blobs (<40px)
                     5. Optical-flow temporal warp
"""

import logging
import numpy as np
import h5py
import cv2
from scipy.ndimage import (
    gaussian_filter, binary_opening, binary_closing, label as nd_label,
)

from config import NATIVE_H, NATIVE_W, PAD_H, PAD_W, INDIA_BBOX

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# IndiaCloudExtractor
# ═════════════════════════════════════════════════════════════════════════════

class IndiaCloudExtractor:
    """Crops India cloud masks from full INSAT-3DR HDF5 files by lat/lon."""

    def __init__(self, sample_h5_path: str):
        try:
            with h5py.File(sample_h5_path, "r") as f:
                lat, lon = self._read_latlon(f)
                self.ll_shape = lat.shape
                self.rmin, self.rmax, self.cmin, self.cmax = \
                    self._compute_bbox(lat, lon)
            logger.info(
                f"[IndiaCloudExtractor] bbox: "
                f"rows [{self.rmin}:{self.rmax}] cols [{self.cmin}:{self.cmax}]"
            )
        except OSError as e:
            raise OSError(f"Cannot open HDF5 '{sample_h5_path}': {e}")

    def _read_latlon(self, f):
        lat = f["Latitude"][:].astype(np.float32)
        lon = f["Longitude"][:].astype(np.float32)
        lat[lat == 32767] = np.nan
        lon[lon == 32767] = np.nan
        return lat / 100.0, lon / 100.0

    def _compute_bbox(self, lat, lon):
        lat_mask = (np.isfinite(lat) &
                    (lat >= INDIA_BBOX["LAT_MIN"]) &
                    (lat <= INDIA_BBOX["LAT_MAX"]))
        lon_mask = (np.isfinite(lon) &
                    (lon >= INDIA_BBOX["LON_MIN"]) &
                    (lon <= INDIA_BBOX["LON_MAX"]))
        combined   = lat_mask & lon_mask
        lat_rows   = np.where(np.any(lat_mask, axis=1))[0]
        india_cols = np.where(np.any(combined,  axis=0))[0]
        if lat_rows.size == 0 or india_cols.size == 0:
            raise ValueError("Could not locate India in the lat/lon grid.")
        return (int(lat_rows.min()), int(lat_rows.max()),
                int(india_cols.min()), int(india_cols.max()))

    @staticmethod
    def _get_spatial_dims(ds):
        if ds.ndim == 2:   return 0, 1
        if ds.ndim == 3:
            sd = sorted(range(3), key=lambda d: ds.shape[d], reverse=True)
            return tuple(sorted(sd[:2]))
        raise ValueError(f"Unexpected CMK shape: {ds.shape}")

    def extract(self, h5_path: str) -> tuple[np.ndarray, dict]:
        with h5py.File(h5_path, "r") as f:
            if "CMK" not in f:
                raise KeyError(f"'CMK' not found. Keys: {list(f.keys())}")
            ds = f["CMK"]
            row_dim, col_dim = self._get_spatial_dims(ds)
            sp_rows = ds.shape[row_dim];  sp_cols = ds.shape[col_dim]
            ll_rows, ll_cols = self.ll_shape
            rs = sp_rows / ll_rows;       cs = sp_cols / ll_cols
            r0 = max(0,       int(np.floor(self.rmin * rs)))
            r1 = min(sp_rows, int(np.ceil (self.rmax * rs)) + 1)
            c0 = max(0,       int(np.floor(self.cmin * cs)))
            c1 = min(sp_cols, int(np.ceil (self.cmax * cs)) + 1)
            if ds.ndim == 2:
                raw = ds[r0:r1, c0:c1]
            else:
                idx = [slice(None)] * ds.ndim
                idx[row_dim] = slice(r0, r1)
                idx[col_dim] = slice(c0, c1)
                raw = np.squeeze(ds[tuple(idx)])

        cmk = raw.filled(0).astype(np.uint8) \
              if isinstance(raw, np.ma.MaskedArray) \
              else np.asarray(raw, dtype=np.uint8)
        cmk[cmk == 255] = 0
        binary = np.zeros_like(cmk, dtype=np.uint8)
        binary[(cmk == 2) | (cmk == 3)] = 1
        valid      = (cmk == 1) | (cmk == 2) | (cmk == 3)
        cloud_frac = float(binary[valid].sum() / max(valid.sum(), 1))
        return binary, {"shape": binary.shape, "cloud_frac": cloud_frac}


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESS — matches load_frame() + build_input() in predict_and_show.py
# ═════════════════════════════════════════════════════════════════════════════

def _prepare_frame(img: np.ndarray) -> np.ndarray:
    """
    Clip to NATIVE_H×NATIVE_W then reflect-pad to PAD_H×PAD_W.
    Mirrors load_frame() in predict_and_show.py exactly.
    """
    img = img.astype(np.float32)
    img = img[:NATIVE_H, :NATIVE_W]                     # clip
    ph  = PAD_H - img.shape[0]
    pw  = PAD_W - img.shape[1]
    img = np.pad(img, ((0, ph), (0, pw)), mode='reflect')  # pad
    return img   # (PAD_H, PAD_W)


def preprocess(binary_images: list[np.ndarray]) -> np.ndarray:
    """
    Stack frames → (1, T, PAD_H, PAD_W, 1) float32 ready for ONNX/Keras.
    Mirrors build_input() in predict_and_show.py.
    """
    frames = np.stack([_prepare_frame(img) for img in binary_images])  # (T,H,W)
    return frames[np.newaxis, ..., np.newaxis].astype(np.float32)       # (1,T,H,W,1)


# ═════════════════════════════════════════════════════════════════════════════
# OPTICAL FLOW — from predict_and_show.py verbatim
# ═════════════════════════════════════════════════════════════════════════════

def _compute_flow(prev_img: np.ndarray, next_img: np.ndarray) -> np.ndarray:
    prev = (prev_img * 255).astype(np.uint8)
    nxt  = (next_img * 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(
        prev, nxt, None,
        pyr_scale=0.5, levels=3, winsize=25,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        mask.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return (warped > 0.5).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# POSTPROCESS — matches visualise_and_save() in predict_and_show.py exactly
#
# IMPORTANT: model output is already at (T, NATIVE_H, NATIVE_W) = (T,758,929)
# DO NOT crop or resize — operate directly at native resolution.
# ═════════════════════════════════════════════════════════════════════════════

def postprocess(
    predictions: np.ndarray,
    input_frames: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Convert raw model probabilities → binary cloud masks at native 758×929.

    predictions:  (T, H, W) float32  — already at native resolution from model
    input_frames: list of T binary uint8 arrays — used for optical flow

    Pipeline (from predict_and_show.py visualise_and_save):
      1. Gaussian smoothing σ=1.2
      2. Adaptive threshold at 65th percentile  (NOT fixed 0.5)
      3. Morphological closing (3×3) then opening (2×2)
      4. Remove blobs < 40 pixels
      5. Optical-flow warp of previous mask + blend (60/40)

    Returns: (T, NATIVE_H, NATIVE_W) uint8
    """
    T = predictions.shape[0]

    # Compute optical flow from last two real input frames (once per sequence)
    flow = None
    if input_frames is not None and len(input_frames) >= 2:
        # Use native-cropped portion of the input frames for flow
        prev_f = input_frames[-2].astype(np.float32)[:NATIVE_H, :NATIVE_W]
        last_f = input_frames[-1].astype(np.float32)[:NATIVE_H, :NATIVE_W]
        try:
            flow = _compute_flow(prev_f, last_f)
            logger.debug("[postprocess] Optical flow computed OK")
        except Exception as e:
            logger.warning(f"[postprocess] Optical flow failed: {e} — skipping warp")

    out       = []
    prev_mask = None

    for t in range(T):
        prob = predictions[t]   # (H, W)

        # 1. Normalise to native resolution:
        #    • New model (PAD_H×PAD_W → 758×929): slice off padding
        #    • Old model (512×512): resize UP to 758×929 so overlay aligns with input
        h, w = prob.shape
        if h == NATIVE_H and w == NATIVE_W:
            pass                                            # already native
        elif h >= NATIVE_H and w >= NATIVE_W:
            prob = prob[:NATIVE_H, :NATIVE_W]              # padded → crop
        else:
            # Old model output (e.g. 512×512) → resize up to 758×929
            prob = cv2.resize(
                prob.astype(np.float32),
                (NATIVE_W, NATIVE_H),                       # cv2: (width, height)
                interpolation=cv2.INTER_LINEAR,
            )

        # 1. Gaussian smoothing
        prob = gaussian_filter(prob, sigma=1.2)

        # 2. Adaptive threshold (65th percentile)
        thr  = np.percentile(prob, 65)
        mask = (prob >= thr).astype(np.uint8)

        # 3. Morphological closing then opening
        mask = binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)
        mask = binary_opening(mask, structure=np.ones((2, 2))).astype(np.uint8)

        # 4. Remove tiny blobs < 40 pixels
        lbl, n = nd_label(mask)
        clean  = np.zeros_like(mask)
        for i in range(1, n + 1):
            region = (lbl == i)
            if region.sum() > 40:
                clean[region] = 1
        mask = clean

        # 5. Optical-flow temporal consistency
        if flow is not None and prev_mask is not None:
            warped = _warp_mask(prev_mask, flow)
            mask   = ((0.6 * mask + 0.4 * warped) > 0.5).astype(np.uint8)

        prev_mask = mask.copy()
        out.append(mask)

    return np.array(out)   # (T, NATIVE_H, NATIVE_W) uint8