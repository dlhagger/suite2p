"""
Utilities for drawing and extracting manually defined ROIs.

The functions in this module deliberately avoid Qt so mask cleanup, Suite2p
conversion, and prompt-model adapters can be tested independently of the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage


def normalize_image(image, lower=1.0, upper=99.0):
    """Robustly normalize a 2-D image to the interval [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if image.ndim != 2:
        raise ValueError("manual ROI summary images must be two-dimensional")
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    lo, hi = np.percentile(image[finite], [lower, upper])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (np.where(finite, image, lo) - lo) / (hi - lo)
    return np.clip(normalized, 0, 1).astype(np.float32)


def _full_frame_image(image, shape, yrange=None, xrange=None):
    """Place a cropped image in a full-frame canvas when necessary."""
    image = np.asarray(image)
    if image.shape == tuple(shape):
        return image
    if yrange is None or xrange is None:
        raise ValueError(
            f"image shape {image.shape} does not match full frame {tuple(shape)}"
        )
    canvas = np.zeros(shape, dtype=image.dtype)
    expected = (int(yrange[1] - yrange[0]), int(xrange[1] - xrange[0]))
    if image.shape != expected:
        raise ValueError(
            f"cropped image shape {image.shape} does not match valid range {expected}"
        )
    canvas[slice(*yrange), slice(*xrange)] = image
    return canvas


def build_prompt_image(ops):
    """Create an RGB prompt image from registered Suite2p image products.

    Enhanced mean intensity, functional correlation, and maximum projection
    occupy the red, green, and blue channels respectively. Missing products
    fall back to the registered mean image.
    """
    shape = (int(ops["Ly"]), int(ops["Lx"]))
    yrange = ops.get("yrange", [0, shape[0]])
    xrange = ops.get("xrange", [0, shape[1]])
    fallback = ops.get("meanImgE", ops.get("meanImg"))
    if fallback is None:
        raise ValueError("manual ROI prompting requires meanImg or meanImgE")

    channels = []
    for key in ("meanImgE", "Vcorr", "max_proj"):
        source = ops.get(key, fallback)
        source = _full_frame_image(source, shape, yrange, xrange)
        channels.append(normalize_image(source))
    return np.round(255 * np.stack(channels, axis=-1)).astype(np.uint8)


def prepare_manual_images(f_reg, reg_outputs, tau, fs, detection_settings):
    """Compute lightweight functional image products without detecting ROIs."""
    from .detection import bin_movie
    from .detection import utils as detection_utils

    n_frames, Ly, Lx = f_reg.shape
    yrange = reg_outputs.get("yrange", [0, Ly])
    xrange = reg_outputs.get("xrange", [0, Lx])
    badframes = np.asarray(
        reg_outputs.get("badframes", np.zeros(n_frames, dtype=bool)), dtype=bool
    )
    nbins = min(int(detection_settings.get("nbins", 5000)), 500)
    bin_size = detection_settings.get("bin_size")
    if not bin_size:
        bin_size = int(max(1, n_frames // max(nbins, 1), np.round(tau * fs)))
    mov = bin_movie(
        f_reg,
        bin_size=bin_size,
        yrange=yrange,
        xrange=xrange,
        badframes=badframes,
        nbins=nbins,
    )
    if mov.size == 0:
        return {}

    mean_img = mov.mean(axis=0)
    filtered = detection_utils.temporal_high_pass_filter(
        mov=mov, width=detection_settings.get("highpass_time", 100)
    )
    return {
        "meanImg_crop": mean_img,
        "max_proj": filtered.max(axis=0),
        "manual_prompt_channels": ["meanImgE", "Vcorr_or_meanImgE", "max_proj"],
    }


def clean_mask(mask, positive_point=None, yrange=None, xrange=None, min_pixels=3):
    """Clean a proposed ROI while preserving the component that was clicked."""
    cleaned = np.asarray(mask, dtype=bool).copy()
    if cleaned.ndim != 2:
        raise ValueError("ROI masks must be two-dimensional")

    if yrange is not None:
        cleaned[:int(yrange[0])] = False
        cleaned[int(yrange[1]):] = False
    if xrange is not None:
        cleaned[:, :int(xrange[0])] = False
        cleaned[:, int(xrange[1]):] = False

    labels, count = ndimage.label(cleaned)
    if count == 0:
        return cleaned

    chosen = 0
    if positive_point is not None:
        x, y = np.rint(positive_point).astype(int)
        if 0 <= y < cleaned.shape[0] and 0 <= x < cleaned.shape[1]:
            chosen = int(labels[y, x])
    if chosen == 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        chosen = int(np.argmax(sizes))

    cleaned = labels == chosen
    cleaned = ndimage.binary_fill_holes(cleaned)
    if int(cleaned.sum()) < int(min_pixels):
        cleaned[:] = False
    return cleaned


def mask_to_stat(mask, weights=None, source="manual", provenance=None):
    """Convert a full-frame binary mask to a Suite2p ``stat`` dictionary."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("ROI masks must be two-dimensional")
    ypix, xpix = np.nonzero(mask)
    if ypix.size == 0:
        raise ValueError("cannot create an ROI from an empty mask")

    if weights is None:
        lam = np.ones(ypix.size, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape == mask.shape:
            lam = weights[ypix, xpix]
        elif weights.size == ypix.size:
            lam = weights.reshape(-1)
        else:
            raise ValueError("weights must match either the mask or its selected pixels")
        lam = np.maximum(np.nan_to_num(lam, nan=0.0), 0)
        if not np.any(lam):
            lam = np.ones(ypix.size, dtype=np.float32)

    stat = {
        "ypix": ypix.astype(np.int32),
        "xpix": xpix.astype(np.int32),
        "lam": lam.astype(np.float32),
        "npix": int(ypix.size),
        "med": [float(np.median(ypix)), float(np.median(xpix))],
        "manual": 1,
        "roi_source": str(source),
    }
    if provenance:
        stat["manual_provenance"] = dict(provenance)
    return stat


def stat_to_mask(stat, shape):
    """Return a full-frame mask for a Suite2p ``stat`` dictionary."""
    mask = np.zeros(shape, dtype=bool)
    mask[np.asarray(stat["ypix"], dtype=int), np.asarray(stat["xpix"], dtype=int)] = True
    return mask


def default_sam3_checkpoint_path():
    """Return the conventional local SAM3 checkpoint path when it exists."""
    candidates = [
        Path(__file__).resolve().parents[1] / "models" / "sam3.pt",
        Path.cwd() / "models" / "sam3.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_sam3_checkpoint_path(checkpoint_path=None):
    """Expand an explicit checkpoint or fall back to ``models/sam3.pt``."""
    if checkpoint_path is None:
        return default_sam3_checkpoint_path()
    checkpoint_path = str(checkpoint_path).strip()
    if not checkpoint_path:
        return default_sam3_checkpoint_path()
    return str(Path(checkpoint_path).expanduser())


def _array_from_tensor_like(value):
    """Convert numpy, torch, or array-like objects to a CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resize_mask_nearest(mask, shape):
    """Resize a boolean mask to ``shape`` using nearest-neighbor sampling."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == tuple(shape):
        return mask
    from PIL import Image

    pil_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:  # pragma: no cover - Pillow < 9 compatibility
        resample = Image.NEAREST
    pil_mask = pil_mask.resize((int(shape[1]), int(shape[0])), resample=resample)
    return np.asarray(pil_mask) > 0


@dataclass
class Sam3Proposal:
    """A single SAM3 candidate and the metadata needed to reproduce it."""

    mask: np.ndarray
    score: float
    logits: np.ndarray | None
    points: np.ndarray
    labels: np.ndarray


class UltralyticsSam3MaskProvider:
    """Lazy adapter around Ultralytics' SAM3 point-prompt API."""

    def __init__(self, checkpoint_path=None, device=None, imgsz=1024, conf=0.0):
        self.checkpoint_path = resolve_sam3_checkpoint_path(checkpoint_path)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.model = None
        self.image = None
        self.image_shape = None

    @staticmethod
    def installed():
        try:
            version("ultralytics")
            return True
        except PackageNotFoundError:
            return importlib_util.find_spec("ultralytics") is not None

    def _load(self):
        if self.model is not None:
            return
        if not self.checkpoint_path or not Path(self.checkpoint_path).is_file():
            raise RuntimeError(
                "SAM3 checkpoint not found. Put sam3.pt at models/sam3.pt."
            )
        try:
            from ultralytics import SAM
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed in this environment. Install "
                "ultralytics to use SAM3 point-prompted ROIs."
            ) from exc
        self.model = SAM(self.checkpoint_path)

    def set_image(self, image):
        """Cache the RGB uint8 prompt image used for point-prompt proposals."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("SAM3 prompt images must have RGB shape (Ly, Lx, 3)")
        self.image = np.clip(image, 0, 255).astype(np.uint8)
        self.image_shape = self.image.shape[:2]
        self._load()

    def propose(self, points: Iterable, labels: Iterable, refine=True):
        """Return the best Ultralytics SAM3 mask for point prompts."""
        if self.image is None:
            raise RuntimeError("call set_image before requesting a SAM3 proposal")
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if points.shape[0] == 0 or points.shape[0] != labels.size:
            raise ValueError("points and labels must contain the same non-zero count")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("SAM3 point labels must be 1 (include) or 0 (exclude)")
        if not np.any(labels == 1):
            raise ValueError("SAM3 needs at least one positive include point")

        from PIL import Image

        results = self.model.predict(
            source=Image.fromarray(self.image, mode="RGB"),
            points=points.tolist(),
            labels=labels.tolist(),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        mask, score = self._best_mask_from_results(results)
        return Sam3Proposal(
            mask=mask,
            score=score,
            logits=None,
            points=points.copy(),
            labels=labels.copy(),
        )

    def _best_mask_from_results(self, results):
        if not results:
            raise RuntimeError("SAM3 returned no results")
        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            raise RuntimeError("SAM3 returned no masks for these prompts")

        mask_data = _array_from_tensor_like(masks.data).astype(bool)
        if mask_data.ndim == 2:
            mask_data = mask_data[None, :, :]
        if mask_data.shape[0] == 0:
            raise RuntimeError("SAM3 returned only empty masks for these prompts")

        scores = None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            scores = _array_from_tensor_like(boxes.conf).astype(np.float32).reshape(-1)
        if scores is None or scores.size != mask_data.shape[0]:
            scores = np.full(mask_data.shape[0], np.nan, dtype=np.float32)

        if np.isfinite(scores).any():
            best = int(np.nanargmax(scores))
            score = float(scores[best])
        else:
            areas = mask_data.reshape(mask_data.shape[0], -1).sum(axis=1)
            best = int(np.argmax(areas))
            score = 0.0

        mask = _resize_mask_nearest(mask_data[best], self.image_shape)
        if not mask.any():
            raise RuntimeError("SAM3 returned an empty mask for these prompts")
        return mask, score

    def reset_prompts(self):
        pass

    def provenance(self):
        try:
            package_version = version("ultralytics")
        except PackageNotFoundError:
            package_version = "unknown"
        return {
            "backend": "ultralytics_sam3",
            "ultralytics_version": package_version,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
        }
