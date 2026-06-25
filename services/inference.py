"""
services/inference.py — Model loading + inference singleton.

Supports both ONNX (.onnx) and Keras (.keras / .h5) models.
Loaded ONCE at startup, reused for all requests.

Model output shape: (1, T, 758, 929, 1)
  → pred[0, t, :, :, 0] = (758, 929) float32 probability map
    at native India resolution — NO resizing needed.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from config import ONNX_MODEL_PATH

logger = logging.getLogger(__name__)

_predictor: Optional["CloudPredictor"] = None


class CloudPredictor:
    """
    Unified wrapper for ONNX or Keras cloud-motion model.
    Detects model type by file extension.

    predict() input:  (1, T, PAD_H, PAD_W, 1)  float32
    predict() output: (T, NATIVE_H, NATIVE_W)   float32  probabilities
                      → already at native 758×929, NO postprocess crop needed
    """

    def __init__(self, model_path: str):
        ext = Path(model_path).suffix.lower()

        if ext == ".onnx":
            self._load_onnx(model_path)
        elif ext in (".keras", ".h5"):
            self._load_keras(model_path)
        else:
            raise ValueError(
                f"Unsupported model format: {ext}. Use .onnx or .keras/.h5"
            )

    # ── ONNX ─────────────────────────────────────────────────────────────────
    def _load_onnx(self, model_path: str):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._sess  = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self._iname = self._sess.get_inputs()[0].name
        self._backend = "onnx"

        logger.info(
            f"[inference] ONNX loaded | input: {self._iname} "
            f"{self._sess.get_inputs()[0].shape} | "
            f"provider: {self._sess.get_providers()[0]}"
        )

    def _predict_onnx(self, model_input: np.ndarray) -> np.ndarray:
        out = self._sess.run(None, {self._iname: model_input.astype(np.float32)})
        raw = out[0]   # (1, T, H, W, 1)
        return raw[0, :, :, :, 0]   # (T, H, W)

    # ── Keras ────────────────────────────────────────────────────────────────
    def _load_keras(self, model_path: str):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required to load .keras/.h5 models. "
                "pip install tensorflow"
            )
        self._model   = tf.keras.models.load_model(model_path, compile=False)
        self._backend = "keras"
        logger.info(
            f"[inference] Keras model loaded from {model_path}"
        )

    def _predict_keras(self, model_input: np.ndarray) -> np.ndarray:
        raw = self._model.predict(model_input.astype(np.float32), verbose=0)
        # Expected output: (1, T, H, W, 1)
        if raw.ndim == 5:
            return raw[0, :, :, :, 0]   # (T, H, W)
        elif raw.ndim == 4:
            return raw[0]               # (T, H, W) if no channel dim
        else:
            raise ValueError(f"Unexpected Keras output shape: {raw.shape}")

    # ── Public ───────────────────────────────────────────────────────────────
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
        Run inference.
        Input:  (1, T, PAD_H, PAD_W, 1)  float32
        Output: (T, NATIVE_H, NATIVE_W)   float32  [0–1 probabilities]
        """
        if self._backend == "onnx":
            return self._predict_onnx(model_input)
        else:
            return self._predict_keras(model_input)


# ── Public helpers ────────────────────────────────────────────────────────────

def load_model() -> CloudPredictor:
    """
    Load the model into the module-level singleton.
    Auto-detects .onnx or .keras/.h5 from ONNX_MODEL_PATH.
    Also checks for a .keras sidecar if .onnx is not found.
    """
    global _predictor

    path = ONNX_MODEL_PATH
    # If configured path doesn't exist, look for .keras sidecar
    if not os.path.isfile(path):
        keras_path = str(Path(path).with_suffix(".keras"))
        h5_path    = str(Path(path).with_suffix(".h5"))
        if os.path.isfile(keras_path):
            path = keras_path
            logger.info(f"[inference] .onnx not found, falling back to {keras_path}")
        elif os.path.isfile(h5_path):
            path = h5_path
            logger.info(f"[inference] .onnx not found, falling back to {h5_path}")
        else:
            raise FileNotFoundError(
                f"Model not found at {ONNX_MODEL_PATH} "
                f"(also checked .keras and .h5 variants).\n"
                "Set ONNX_MODEL_PATH env variable or update config.py."
            )

    _predictor = CloudPredictor(path)
    logger.info(f"[inference] Model ready ({Path(path).suffix}): {path}")
    return _predictor


def get_predictor() -> CloudPredictor:
    if _predictor is None:
        raise RuntimeError(
            "Model is not loaded. Call load_model() at app startup."
        )
    return _predictor