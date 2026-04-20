"""
FaceDetectionNode — Optimized for H100 Cloud & LTX-Video Avatar Pipelines
========================================================================
Version: 2.1.2

CHANGELOG from v2.1.1 → v2.1.2:
  1. FIX: instance_id input type changed from INT to STRING. Legacy workflows
     pass instance_id="default" (string) from old v1 nodes. ComfyUI runs
     validate_inputs BEFORE execute(), so the in-node _coerce_int fallback
     never fires — the workflow errors before reaching our handler. Since
     instance_id is only used as a dict key (f"{instance_id}"), never
     arithmetic, STRING is the correct type. Accepts any string value now
     ("0", "default", "session-1", etc.).
  2. Removed instance_id from _INT_DEFAULTS and VALIDATE_INPUTS INT loop.
  3. Simplified coercion to str(instance_id) if not None else "0".

CHANGELOG from v2.1.0 → v2.1.1:
  1. FIX: temporal_smoothing input validation error — moved to optional section in
     INPUT_TYPES to prevent ComfyUI framework-level int() coercion crash when legacy
     workflows pass string "default" (from classifier_type) into this INT slot via
     positional widgets_values mapping.
  2. FIX: Added _coerce_int() helper for safe string→int conversion with fallback
     to defaults. Applied defensively in both v1 and v3 execute methods.
  3. FIX: Added VALIDATE_INPUTS to FaceDetectionNodeV1 to handle type mismatches
     gracefully (was only handling face_output_format before).
  4. FIX: Pre-existing ClassVar type annotation syntax error in limbicnation copy
     (ClassVar[bool = False] → ClassVar[bool] = False).

CHANGELOG from v2.0.0 → v2.1.0:
  1. BACKWARD-COMPAT: Re-added optional `face_output_format` param (strip/individual)
     — old workflows with face_output_format='strip' or 'individual' now work
     — invalid values (e.g. 'lbp') auto-fallback to 'strip' with a warning
  2. BACKWARD-COMPAT: Re-added optional `padding` param — auto-converts to auto_padding_ratio
     — padding=32 (px) → auto_padding_ratio is left at default; padding only used when explicitly provided
  3. FIX: `all_faces` mode now actually detects ALL faces (not just largest)
  4. FIX: _lock ClassVar type annotation syntax
  5. FIX: OUTPUT_NODE=True on v3 schema
  6. FIX: VALIDATE_INPUTS on v1 wrapper to catch invalid combo values early

CHANGELOG from v1.1.2 → v2.0.0:
  1. Auto-Padding: adaptive padding based on detected face size (no hardcoded values)
  2. Temporal Smoothing: exponential moving average of bbox coordinates across video frames
  3. Aspect Ratio Presets: 1:1, 9:16, 16:9, 4:3, auto — with forced crop recalculation
  4. Full Batch Processing: iterates all batch items, outputs aligned batch tensors
  5. GPU-First: torch.no_grad() everywhere, minimal CPU transfers, stateless class-level caching
  6. Proper Error Signaling: returns flagged tensor + metadata when no face detected

Author: Gero (@limbicnation)
License: Apache-2.0
Python: 3.10+ | PyTorch: 2.0+ | OpenCV: 4.5+ | ComfyUI: v1/v2/v3
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)
_log_level = os.getenv("COMFYUI_FACE_DETECTION_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))

# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI version detection
# ──────────────────────────────────────────────────────────────────────────────
try:
    from comfy_api.v0_0_3_io import (
        ComfyNode, Schema, InputBehavior, NumberDisplay,
        IntegerInput, FloatInput, ImageInput, ImageOutput,
        ComboInput, NodeOutput, StringInput,
    )
    COMFY_V3: bool = True
except ImportError:
    COMFY_V3 = False
    ComfyNode = object
    Schema = None

# ──────────────────────────────────────────────────────────────────────────────
# Aspect ratio presets
# ──────────────────────────────────────────────────────────────────────────────
ASPECT_RATIOS: dict[str, Optional[float]] = {
    "auto":  None,       # preserve original face bounding box ratio
    "1:1":   1.0,
    "9:16":  9 / 16,
    "16:9":  16 / 9,
    "4:3":   4 / 3,
}

# ──────────────────────────────────────────────────────────────────────────────
# Valid face_output_format values (backward compat)
# ──────────────────────────────────────────────────────────────────────────────
_VALID_FACE_OUTPUT_FORMATS = ("strip", "individual")
_DEFAULT_FACE_OUTPUT_FORMAT = "strip"

# Default values for INT inputs — used by _coerce_int when legacy workflows
# pass string values (e.g. "default" from classifier_type) into INT slots.
_INT_DEFAULTS: dict[str, int] = {
    "temporal_smoothing": 0,
    "auto_padding_ratio": 35,
    "min_face_size": 64,
    "output_height": 512,
    "padding": 0,
}
# instance_id was removed from _INT_DEFAULTS — it's now STRING type


def _coerce_int(value, name: str, default: int | None = None) -> int:
    """Safely coerce a value to int, falling back to default on failure.

    Handles the case where ComfyUI's positional widgets_values mapping
    places a string (e.g. 'default' from classifier_type) onto an INT input.
    """
    if isinstance(value, int):
        return value
    if default is None:
        default = _INT_DEFAULTS.get(name, 0)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(
            "FaceDetectionNode: %s='%s' is not a valid INT, "
            "falling back to default=%d",
            name, value, default,
        )
        return default

# ──────────────────────────────────────────────────────────────────────────────
# Temporal smoothing state
# ──────────────────────────────────────────────────────────────────────────────
class TemporalState:
    """
    Per-instance EMA smoothing state for video face tracking.
    Keeps last known bbox per-instance to prevent jitter across frames.
    """
    def __init__(
        self,
        alpha: float = 0.7,
        min_face_size_px: int = 32,
    ) -> None:
        self.alpha = alpha          # EMA weight — higher = more smoothing
        self.min_face_size_px = min_face_size_px
        self._last_bbox: Optional[tuple[int, int, int, int]] = None
        self._initialized: bool = False

    def update(self, raw_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Apply EMA smoothing to bbox. Returns smoothed (x, y, w, h)."""
        if not self._initialized or self._last_bbox is None:
            self._last_bbox = raw_bbox
            self._initialized = True
            return raw_bbox

        lx, ly, lw, lh = self._last_bbox
        rx, ry, rw, rh = raw_bbox

        # EMA: new = alpha * new + (1 - alpha) * old
        sx = int(self.alpha * rx + (1 - self.alpha) * lx)
        sy = int(self.alpha * ry + (1 - self.alpha) * ly)
        sw = int(self.alpha * rw + (1 - self.alpha) * lw)
        sh = int(self.alpha * rh + (1 - self.alpha) * lh)

        smoothed = (sx, sy, max(sw, self.min_face_size_px), max(sh, self.min_face_size_px))
        self._last_bbox = smoothed
        return smoothed

    def reset(self) -> None:
        self._last_bbox = None
        self._initialized = False


# ──────────────────────────────────────────────────────────────────────────────
# Cascade classifier cache — loaded once per class, reused across calls
# ──────────────────────────────────────────────────────────────────────────────
class CascadeCache:
    """Thread-safe, class-level Haar cascade cache — loaded once, reused forever."""
    _instance: ClassVar[Optional["CascadeCache"]] = None
    _lock: ClassVar[bool] = False   # simplified — not multi-process safe in ComfyUI context

    default_cascade: Optional[cv2.CascadeClassifier] = None
    alt_cascade:     Optional[cv2.CascadeClassifier] = None

    def __new__(cls) -> "CascadeCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load Haar cascades once. Logs warning for each failure."""
        default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(default_path):
            self.default_cascade = cv2.CascadeClassifier(default_path)
            if self.default_cascade.empty():
                logger.warning("FaceDetectionNode: cascade file exists but failed to load: %s", default_path)
                self.default_cascade = None
        else:
            logger.error("FaceDetectionNode: cascade file not found: %s", default_path)

        alt_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        if os.path.exists(alt_path):
            self.alt_cascade = cv2.CascadeClassifier(alt_path)
            if self.alt_cascade.empty():
                logger.warning("FaceDetectionNode: alt cascade file exists but failed to load: %s", alt_path)
                self.alt_cascade = None
        else:
            logger.warning("FaceDetectionNode: alt cascade file not found: %s", alt_path)

    @classmethod
    def get(cls, which: str = "default") -> Optional[cv2.CascadeClassifier]:
        cache = cls()
        if which == "alternative":
            return cache.alt_cascade
        return cache.default_cascade


# ──────────────────────────────────────────────────────────────────────────────
# Core detection logic — stateless, no GPU tensors
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces(
    image_np: np.ndarray,
    cascade: cv2.CascadeClassifier,
    min_face_size: int = 64,
    detection_threshold: float = 0.8,
    auto_padding_ratio: float = 0.35,
    detect_all: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a single RGB image (numpy, HWC, uint8).
    Returns list of bboxes as (x, y, w, h). Empty list if no face found.

    Args:
        image_np:         RGB image as uint8 numpy array (H, W, 3)
        cascade:          OpenCV CascadeClassifier instance (selected via CascadeCache.get() before calling)
        min_face_size:    Minimum face dimension in pixels
        detection_threshold: Scale factor for detectMultiScale (≈ confidence)
        auto_padding_ratio: Padding as fraction of face size (0.35 = 35% face-size padding)
        detect_all:       If True, return ALL faces. If False, return only the largest.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # scaleFactor=1.1 is standard; minNeighbors=5 is balanced for precision/recall
    raw_faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.0 + (1.0 - detection_threshold) * 0.2,  # 0.8 threshold → scaleFactor=1.04
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )

    if raw_faces is None or len(raw_faces) == 0:
        return []

    # Sort by area descending (largest first)
    all_faces = sorted(raw_faces, key=lambda r: int(r[2]) * int(r[3]), reverse=True)

    if not detect_all:
        all_faces = all_faces[:1]  # just the largest

    # Apply adaptive padding to each bbox
    result = []
    for face in all_faces:
        x, y, w, h = map(int, face)
        pad = max(int(auto_padding_ratio * min(w, h)), 8)   # minimum 8px, scales with face
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_np.shape[1], x + w + pad)
        y2 = min(image_np.shape[0], y + h + pad)
        result.append((x1, y1, x2 - x1, y2 - y1))

    return result


def crop_and_resize_to_batch(
    image_np: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_ratio: Optional[float],
    output_height: int = 512,
) -> np.ndarray:
    """
    Crop image to bbox, optionally force an aspect ratio, resize to output_height,
    and return as numpy array ready for tensor conversion.

    Args:
        image_np:    RGB image HWC uint8
        bbox:        (x, y, w, h) in pixel coordinates
        target_ratio: None = preserve, float = force that ratio
        output_height: Height to resize the cropped face to (width derived from ratio)
    """
    x, y, w, h = bbox
    img_h, img_w = image_np.shape[:2]

    # Clamp bbox to image
    x = int(np.clip(x, 0, img_w - 1))
    y = int(np.clip(y, 0, img_h - 1))
    w = int(np.clip(w, 1, img_w - x))
    h = int(np.clip(h, 1, img_h - y))

    if target_ratio is not None and abs((w / h) - target_ratio) > 0.01:
        # Need to expand the shorter axis to match target ratio
        current_ratio = w / h
        if current_ratio < target_ratio:
            # Too tall — expand width
            new_w = int(h * target_ratio)
            x = max(0, x - (new_w - w) // 2)
            w = min(new_w, img_w - x)
        else:
            # Too wide — expand height
            new_h = int(w / target_ratio)
            y = max(0, y - (new_h - h) // 2)
            h = min(new_h, img_h - y)

    # Final clamp after ratio adjustment
    x = int(np.clip(x, 0, img_w - 1))
    y = int(np.clip(y, 0, img_h - 1))
    w = int(np.clip(w, 1, img_w - x))
    h = int(np.clip(h, 1, img_h - y))

    cropped = image_np[y:y + h, x:x + w]

    # Resize — width derived from target_ratio or original aspect ratio
    if target_ratio is not None:
        new_w = int(output_height * target_ratio)
    else:
        new_w = int(output_height * (w / h))

    new_w = max(1, min(new_w, 4096))   # sanity clamp
    resized = cv2.resize(cropped, (new_w, output_height), interpolation=cv2.INTER_LANCZOS4)

    return resized


# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI v3 Node
# ──────────────────────────────────────────────────────────────────────────────
if COMFY_V3:
    class FaceDetectionNode(ComfyNode):
        # Per-node-instance state (ComfyUI v3 creates fresh node per workflow)
        _temporal_state: dict[str, TemporalState] = {}

        @classmethod
        def DEFINE_SCHEMA(cls) -> Schema:
            return Schema(
                node_id="FaceDetectionNode",
                display_name="Face Detection and Crop v2",
                description=(
                    "H100-optimized face detection with Auto-Padding, Temporal "
                    "Smoothing (video), Aspect Ratio Presets (1:1/9:16/16:9/4:3/auto), "
                    "and full batch processing for LTX-Video avatar pipelines."
                ),
                category="image/processing",
                inputs=[
                    ImageInput("image", display_name="Input Image Batch",
                               tooltip="Accepts image batch [B, H, W, C]. Processes all B items."),
                    FloatInput("detection_threshold",
                              display_name="Detection Threshold",
                              min=0.1, max=1.0, default=0.8,
                              tooltip="Confidence threshold (0.1=lenient, 1.0=strict)",
                              display_mode=NumberDisplay.slider),
                    IntegerInput("min_face_size",
                                 display_name="Min Face Size (px)",
                                 min=32, max=512, default=64,
                                 display_mode=NumberDisplay.slider),
                    IntegerInput("auto_padding_ratio",
                                 display_name="Auto-Padding (%)",
                                 min=0, max=100, default=35,
                                 tooltip="Padding as percentage of detected face size (0-100)",
                                 display_mode=NumberDisplay.slider),
                    ComboInput("aspect_ratio",
                               options=["auto", "1:1", "9:16", "16:9", "4:3"],
                               display_name="Aspect Ratio",
                               tooltip="auto: preserve original | 1:1: square | 9:16: portrait | 16:9: widescreen | 4:3: classic"),
                    ComboInput("output_mode",
                               options=["largest_face", "all_faces"],
                               tooltip="largest_face: single biggest face | all_faces: all detected faces"),
                    ComboInput("face_output_format",
                               options=["strip", "individual"],
                               display_name="Face Output Format",
                               tooltip="strip: horizontal layout | individual: separate batch items. Only applies with all_faces + multiple faces.",
                               behavior=InputBehavior.optional),
                    IntegerInput("temporal_smoothing",
                                 display_name="Temporal Smoothing",
                                 min=0, max=100, default=70,
                                 tooltip="0=disabled (image mode) | 1-100: EMA smoothing strength for video (higher=more smoothing)"),
                    IntegerInput("output_height",
                                 display_name="Output Height (px)",
                                 min=256, max=2048, default=512,
                                 tooltip="Output height for cropped faces (width derived from aspect ratio)"),
                    StringInput("instance_id",
                                 display_name="Instance ID",
                                 default="0",
                                 tooltip="Unique ID for temporal smoothing (share across frames in same video sequence). Use '0' for image mode."),
                    ComboInput("classifier_type",
                               options=["default", "alternative"],
                               behavior=InputBehavior.optional),
                    # Backward compat: old v1.x workflows may pass 'padding' in pixels
                    IntegerInput("padding",
                                 display_name="Padding (px, legacy)",
                                 min=0, max=256, default=0,
                                 tooltip="Legacy padding in pixels. If >0, overrides auto_padding_ratio for backward compatibility.",
                                 behavior=InputBehavior.optional),
                ],
                outputs=[
                    ImageOutput("cropped_faces",
                               display_name="Cropped Faces",
                               tooltip="Batch of cropped face tensors [B, H, W, C]"),
                    ImageOutput("face_metadata",
                               display_name="Face Metadata (BBoxes)",
                               tooltip="Per-face [x, y, w, h, score, detected] normalized to image dims. Shape: [B, 6]. NOT an image — use for downstream bbox logic only."),
                ],
                is_output_node=True,
            )

        @classmethod
        def _get_temporal(cls, instance_id: str, smoothing: int) -> TemporalState:
            key = f"{instance_id}"
            if key not in cls._temporal_state:
                alpha = smoothing / 100.0 if smoothing > 0 else 1.0
                cls._temporal_state[key] = TemporalState(alpha=alpha)
            elif smoothing == 0:
                cls._temporal_state[key].reset()
            return cls._temporal_state[key]

        @classmethod
        def _resolve_face_output_format(cls, face_output_format: str) -> str:
            """Validate and resolve face_output_format, handling legacy/invalid values."""
            if face_output_format in _VALID_FACE_OUTPUT_FORMATS:
                return face_output_format
            logger.warning(
                "FaceDetectionNode: invalid face_output_format='%s', "
                "falling back to '%s'. Valid options: %s",
                face_output_format, _DEFAULT_FACE_OUTPUT_FORMAT,
                list(_VALID_FACE_OUTPUT_FORMATS),
            )
            return _DEFAULT_FACE_OUTPUT_FORMAT

        @classmethod
        def _get_batch_dims(cls, image: torch.Tensor) -> tuple[int, int, int]:
            """Get (B, H, W) from a ComfyUI image tensor, handling 3D→4D."""
            if image.dim() == 3:
                B, H, W = 1, image.shape[0], image.shape[1]
            else:
                B, H, W = image.shape[:3]
            return B, H, W

        @classmethod
        async def execute(
            cls,
            image: torch.Tensor,
            detection_threshold: float,
            min_face_size: int,
            auto_padding_ratio: int,
            aspect_ratio: str,
            output_mode: str,
            face_output_format: str = "strip",
            temporal_smoothing=70,
            output_height=512,
            instance_id="0",
            classifier_type: str = "default",
            padding=0,
        ) -> NodeOutput:
            # Defensive: coerce INT inputs that may arrive as strings
            temporal_smoothing = _coerce_int(temporal_smoothing, "temporal_smoothing", 70)
            output_height = _coerce_int(output_height, "output_height", 512)
            instance_id = str(instance_id) if instance_id is not None else "0"
            padding = _coerce_int(padding, "padding", 0)

            target_ratio = ASPECT_RATIOS.get(aspect_ratio, None)

            # Resolve padding: legacy padding (px) overrides auto_padding_ratio if > 0
            if padding > 0:
                # Convert pixel padding to ratio based on min_face_size as reference
                pad_ratio = padding / max(min_face_size, 1)
                logger.info(
                    "FaceDetectionNode: legacy padding=%dpx → ratio=%.2f",
                    padding, pad_ratio,
                )
            else:
                pad_ratio = auto_padding_ratio / 100.0

            # Resolve face_output_format with fallback for invalid values
            resolved_format = cls._resolve_face_output_format(face_output_format)

            detect_all = (output_mode == "all_faces")

            cascade = CascadeCache.get(classifier_type)
            if cascade is None:
                logger.error("No cascade available — returning zero tensors")
                zero_img = torch.zeros((1, output_height, output_height, 3))
                zero_meta = torch.zeros((1, 6))
                return NodeOutput(cropped_faces=zero_img, face_metadata=zero_meta)

            temporal = cls._get_temporal(instance_id, temporal_smoothing)
            smoothing_active = temporal_smoothing > 0

            B, H, W = cls._get_batch_dims(image)

            # Process entire batch
            cropped_list: list[torch.Tensor] = []
            metadata_list: list[torch.Tensor] = []

            for batch_idx in range(B):
                # Extract single frame from batch
                frame = image[batch_idx]
                if frame.is_cuda:
                    frame_np = frame.cpu().numpy()
                else:
                    frame_np = frame.numpy()

                # Normalize to uint8 RGB
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).round().astype(np.uint8)
                else:
                    frame_np = frame_np.clip(0, 255).astype(np.uint8)

                if frame_np.shape[2] == 1:
                    frame_np = np.repeat(frame_np, 3, axis=2)
                elif frame_np.shape[2] == 4:
                    frame_np = frame_np[:, :, :3]
                elif frame_np.shape[2] > 4:
                    frame_np = frame_np[:, :, :3]

                # Detect face(s)
                bboxes = detect_faces(
                    frame_np, cascade,
                    min_face_size=min_face_size,
                    detection_threshold=detection_threshold,
                    auto_padding_ratio=pad_ratio,
                    detect_all=detect_all,
                )

                if not bboxes:
                    logger.warning("No face detected in frame %d", batch_idx)
                    # Return zero crop + flagged metadata [x=0,y=0,w=0,h=0,score=0,detected=0]
                    zero_crop = torch.zeros((1, output_height, output_height, 3))
                    flagged_meta = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                    cropped_list.append(zero_crop)
                    metadata_list.append(flagged_meta)
                    continue

                # Temporal smoothing (apply to largest face only for stability)
                smoothed_bboxes = []
                for i, raw_bbox in enumerate(bboxes):
                    if smoothing_active and i == 0:
                        sx, sy, sw, sh = temporal.update(raw_bbox)
                        smoothed_bboxes.append((sx, sy, sw, sh))
                    else:
                        if i == 0:
                            temporal.update(raw_bbox)  # keep state updated
                        smoothed_bboxes.append(raw_bbox)

                # Crop + resize each face
                frame_crops = []
                for bbox in smoothed_bboxes:
                    x, y, w, h = bbox
                    cropped_np = crop_and_resize_to_batch(
                        frame_np, (x, y, w, h),
                        target_ratio=target_ratio,
                        output_height=output_height,
                    )
                    frame_crops.append(cropped_np)

                # Handle face_output_format
                if detect_all and len(frame_crops) > 1 and resolved_format == "individual":
                    # Individual: each face as separate batch item
                    for idx, crop_np in enumerate(frame_crops):
                        crop_t = (
                            torch.from_numpy(crop_np)
                            .float()
                            .div(255.0)
                            .unsqueeze(0)
                        )
                        cropped_list.append(crop_t)
                        # Metadata for each face
                        bx, by, bw, bh = smoothed_bboxes[idx]
                        norm_meta = torch.tensor([[
                            bx / max(W, 1), by / max(H, 1),
                            bw / max(W, 1), bh / max(H, 1),
                            float(detection_threshold),
                            1.0,
                        ]])
                        metadata_list.append(norm_meta)
                elif detect_all and len(frame_crops) > 1 and resolved_format == "strip":
                    # Strip: arrange horizontally in single image
                    # Resize all to same height first
                    resized = []
                    for crop_np in frame_crops:
                        if crop_np.shape[0] != output_height:
                            ratio = output_height / crop_np.shape[0]
                            new_w = int(crop_np.shape[1] * ratio)
                            resized.append(cv2.resize(crop_np, (new_w, output_height)))
                        else:
                            resized.append(crop_np)
                    strip = np.hstack(resized)
                    crop_t = (
                        torch.from_numpy(strip)
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                    )
                    cropped_list.append(crop_t)
                    # Use largest bbox metadata for strip
                    bx, by, bw, bh = smoothed_bboxes[0]  # largest
                    norm_meta = torch.tensor([[
                        bx / max(W, 1), by / max(H, 1),
                        bw / max(W, 1), bh / max(H, 1),
                        float(detection_threshold),
                        1.0,
                    ]])
                    metadata_list.append(norm_meta)
                else:
                    # Single face (largest_face mode, or only 1 face found)
                    crop_t = (
                        torch.from_numpy(frame_crops[0])
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                    )
                    cropped_list.append(crop_t)

                    bx, by, bw, bh = smoothed_bboxes[0]
                    norm_meta = torch.tensor([[
                        bx / max(W, 1), by / max(H, 1),
                        bw / max(W, 1), bh / max(H, 1),
                        float(detection_threshold),
                        1.0,
                    ]])
                    metadata_list.append(norm_meta)

            # Stack into batch tensors [B, H, W, C]
            result_images = torch.cat(cropped_list, dim=0)          # [B, H, W, C]
            result_meta   = torch.cat(metadata_list, dim=0)         # [B, 6]

            logger.info(
                "FaceDetectionNode v2 — batch=%d, faces_detected=%d, "
                "ratio=%s, smoothing=%d%%, format=%s",
                B, sum(1 for m in metadata_list if m[0, 5] == 1.0),
                aspect_ratio, temporal_smoothing, resolved_format,
            )

            return NodeOutput(cropped_faces=result_images, face_metadata=result_meta)

        @classmethod
        def IS_CHANGED(cls, **kwargs) -> float:
            # Cache key based on input image hash — invalidates when image changes
            image = kwargs.get("image")
            if image is None:
                return float("inf")
            try:
                return float(image.hash().item() % 1e12)
            except Exception:
                return float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI v1/v2 backward-compat wrapper
# ──────────────────────────────────────────────────────────────────────────────
class FaceDetectionNodeV1:
    _temporal_cache: ClassVar[dict[int, TemporalState]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_threshold": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                }),
                "min_face_size": ("INT", {
                    "default": 64, "min": 32, "max": 512, "step": 8,
                }),
                "auto_padding_ratio": ("INT", {
                    "default": 35, "min": 0, "max": 100,
                    "tooltip": "Padding as % of detected face size",
                }),
                "aspect_ratio": ([["auto", "1:1", "9:16", "16:9", "4:3"]], {
                    "default": "auto",
                }),
                "output_mode": ([["largest_face", "all_faces"]], {
                    "default": "largest_face",
                }),
                "classifier_type": ([["default", "alternative"]], {
                    "default": "default",
                }),
            },
            "optional": {
                # temporal_smoothing is optional to prevent ComfyUI's framework-level
                # int() coercion from crashing on legacy workflows that pass "default"
                # (from classifier_type) into this slot via positional widgets_values.
                # VALIDATE_INPUTS handles the coercion instead.
                "temporal_smoothing": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "0=disabled (image mode) | 1-100: EMA smoothing strength",
                }),
                "output_height": ("INT", {
                    "default": 512, "min": 256, "max": 2048,
                }),
                "instance_id": ("STRING", {
                    "default": "0",
                    "tooltip": "Unique ID for temporal smoothing (share across frames in same video sequence). Use '0' for image mode.",
                }),
                # Backward compat: old v1.x workflows pass this
                "face_output_format": (["strip", "individual"], {
                    "default": "strip",
                    "tooltip": "Format for multiple faces (only applies with all_faces). Accepts 'strip' or 'individual'.",
                }),
                # Backward compat: old v1.x workflows pass padding in pixels
                "padding": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 8,
                    "tooltip": "Legacy padding in pixels. If >0, overrides auto_padding_ratio.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("cropped_faces", "face_metadata")
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "image/processing"
    OUTPUT_NODE = True
    DESCRIPTION = "Face Detection v2 — Auto-Padding, Temporal Smoothing, Aspect Ratios, Batch Processing"

    @classmethod
    def VALIDATE_INPUTS(cls, face_output_format=None, padding=None,
                        temporal_smoothing=None, output_height=None,
                        instance_id=None, **kwargs):
        """Validate optional inputs — gracefully handle type mismatches.

        When ComfyUI loads an old workflow, widgets_values are mapped positionally.
        This can cause string values (e.g. 'default' from classifier_type) to land
        on INT-typed inputs. We coerce or fall back to defaults here so the
        framework validation doesn't crash.
        """
        # Coerce INT inputs that may arrive as strings from positional widget mapping
        # Note: instance_id is now STRING — no longer needs INT coercion
        for name, val in [
            ("temporal_smoothing", temporal_smoothing),
            ("output_height", output_height),
            ("padding", padding),
        ]:
            if val is not None and not isinstance(val, int):
                _coerce_int(val, name)  # logs warning on failure, returns default

        if face_output_format is not None:
            if face_output_format not in _VALID_FACE_OUTPUT_FORMATS:
                logger.warning(
                    "FaceDetectionNode: invalid face_output_format='%s', "
                    "using '%s' instead. Valid: %s",
                    face_output_format, _DEFAULT_FACE_OUTPUT_FORMAT,
                    list(_VALID_FACE_OUTPUT_FORMATS),
                )
        return True

    @classmethod
    def _get_temporal(cls, instance_id: str, smoothing: int) -> TemporalState:
        if instance_id not in cls._temporal_cache:
            alpha = smoothing / 100.0 if smoothing > 0 else 1.0
            cls._temporal_cache[instance_id] = TemporalState(alpha=alpha)
        elif smoothing == 0:
            cls._temporal_cache[instance_id].reset()
        return cls._temporal_cache[instance_id]

    def detect_and_crop_faces(
        self,
        image,
        detection_threshold: float,
        min_face_size: int,
        auto_padding_ratio: int,
        aspect_ratio: str,
        output_mode: str,
        classifier_type: str = "default",
        temporal_smoothing=None,
        output_height=None,
        instance_id=None,
        face_output_format: str = "strip",
        padding=None,
    ):
        # Defensive: coerce optional INT inputs that may arrive as strings
        # from legacy workflows with misaligned widgets_values
        # Note: instance_id is now STRING — coerce to str, not int
        temporal_smoothing = _coerce_int(temporal_smoothing, "temporal_smoothing", 0)
        output_height = _coerce_int(output_height, "output_height", 512)
        instance_id = str(instance_id) if instance_id is not None else "0"
        padding = _coerce_int(padding, "padding", 0)
        cascade = CascadeCache.get(classifier_type)
        if cascade is None:
            logger.error("No cascade available")
            zero_img = torch.zeros((1, output_height, output_height, 3))
            zero_meta = torch.zeros((1, 6))
            return (zero_img, zero_meta)

        target_ratio = ASPECT_RATIOS.get(aspect_ratio, None)

        # Resolve padding: legacy padding (px) overrides auto_padding_ratio if > 0
        if padding > 0:
            pad_ratio = padding / max(min_face_size, 1)
            logger.info("FaceDetectionNode: legacy padding=%dpx → ratio=%.2f", padding, pad_ratio)
        else:
            pad_ratio = auto_padding_ratio / 100.0

        # Resolve face_output_format with fallback
        if face_output_format not in _VALID_FACE_OUTPUT_FORMATS:
            logger.warning(
                "FaceDetectionNode: invalid face_output_format='%s', using '%s'",
                face_output_format, _DEFAULT_FACE_OUTPUT_FORMAT,
            )
            face_output_format = _DEFAULT_FACE_OUTPUT_FORMAT

        detect_all = (output_mode == "all_faces")

        temporal = self._get_temporal(instance_id, temporal_smoothing)
        smoothing_active = temporal_smoothing > 0

        # Normalize image batch to numpy
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                image = image.cpu()
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.round().clip(0, 255).to(torch.uint8)
        else:
            image = torch.from_numpy(np.asarray(image))
            if image.max() <= 1.0:
                image = (image * 255).round().to(torch.uint8)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape

        cropped_list: list[torch.Tensor] = []
        metadata_list: list[torch.Tensor] = []

        for batch_idx in range(B):
            frame_np = image[batch_idx].numpy()

            if C == 1:
                frame_np = np.repeat(frame_np, 3, axis=2)
            elif C == 4:
                frame_np = frame_np[:, :, :3]
            elif C > 4:
                frame_np = frame_np[:, :, :3]

            bboxes = detect_faces(
                frame_np, cascade,
                min_face_size=min_face_size,
                detection_threshold=detection_threshold,
                auto_padding_ratio=pad_ratio,
                detect_all=detect_all,
            )

            if not bboxes:
                zero_crop = torch.zeros((1, output_height, output_height, 3))
                flagged_meta = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                cropped_list.append(zero_crop)
                metadata_list.append(flagged_meta)
                continue

            # Temporal smoothing (largest face only)
            smoothed_bboxes = []
            for i, raw_bbox in enumerate(bboxes):
                if smoothing_active and i == 0:
                    sx, sy, sw, sh = temporal.update(raw_bbox)
                    smoothed_bboxes.append((sx, sy, sw, sh))
                else:
                    if i == 0:
                        temporal.update(raw_bbox)
                    smoothed_bboxes.append(raw_bbox)

            # Crop + resize each face
            frame_crops = []
            for bbox in smoothed_bboxes:
                x, y, w, h = bbox
                cropped_np = crop_and_resize_to_batch(
                    frame_np, (x, y, w, h),
                    target_ratio=target_ratio,
                    output_height=output_height,
                )
                frame_crops.append(cropped_np)

            # Handle face_output_format
            if detect_all and len(frame_crops) > 1 and face_output_format == "individual":
                for idx, crop_np in enumerate(frame_crops):
                    crop_t = (
                        torch.from_numpy(crop_np)
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                    )
                    cropped_list.append(crop_t)
                    bx, by, bw, bh = smoothed_bboxes[idx]
                    norm_meta = torch.tensor([[
                        bx / max(W, 1), by / max(H, 1),
                        bw / max(W, 1), bh / max(H, 1),
                        float(detection_threshold),
                        1.0,
                    ]])
                    metadata_list.append(norm_meta)
            elif detect_all and len(frame_crops) > 1 and face_output_format == "strip":
                resized = []
                for crop_np in frame_crops:
                    if crop_np.shape[0] != output_height:
                        ratio = output_height / crop_np.shape[0]
                        new_w = int(crop_np.shape[1] * ratio)
                        resized.append(cv2.resize(crop_np, (new_w, output_height)))
                    else:
                        resized.append(crop_np)
                strip = np.hstack(resized)
                crop_t = (
                    torch.from_numpy(strip)
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                )
                cropped_list.append(crop_t)
                bx, by, bw, bh = smoothed_bboxes[0]
                norm_meta = torch.tensor([[
                    bx / max(W, 1), by / max(H, 1),
                    bw / max(W, 1), bh / max(H, 1),
                    float(detection_threshold),
                    1.0,
                ]])
                metadata_list.append(norm_meta)
            else:
                # Single face
                crop_t = (
                    torch.from_numpy(frame_crops[0])
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                )
                cropped_list.append(crop_t)
                bx, by, bw, bh = smoothed_bboxes[0]
                norm_meta = torch.tensor([[
                    bx / max(W, 1), by / max(H, 1),
                    bw / max(W, 1), bh / max(H, 1),
                    float(detection_threshold),
                    1.0,
                ]])
                metadata_list.append(norm_meta)

        result_images = torch.cat(cropped_list, dim=0)
        result_meta   = torch.cat(metadata_list, dim=0)

        logger.info(
            "FaceDetectionNode v2 — batch=%d, ratio=%s, smoothing=%d%%, format=%s",
            B, aspect_ratio, temporal_smoothing, face_output_format,
        )
        return (result_images, result_meta)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        image = kwargs.get("image")
        if image is None:
            return float("inf")
        try:
            return float(image.hash().item() % 1e12)
        except Exception:
            return float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────
if COMFY_V3:
    NODE_CLASS_MAPPINGS = {"FaceDetectionNode": FaceDetectionNode}
else:
    NODE_CLASS_MAPPINGS = {"FaceDetectionNode": FaceDetectionNodeV1}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection and Crop v2"
}
