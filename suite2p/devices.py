"""Shared device selection for the pipeline and prompt models."""

import logging

import torch

logger = logging.getLogger(__name__)


def assign_torch_device(requested="auto"):
    """Select CUDA, MPS, then CPU in auto mode; validate explicit devices.

    A successful allocation only checks device availability. Unsupported model
    operations may still require the user to select CPU explicitly.
    """
    requested = "auto" if requested is None else str(requested).strip().lower()
    if requested == "auto":
        candidates = []
        if torch.cuda.is_available():
            candidates.append("cuda")
        if torch.backends.mps.is_available():
            candidates.append("mps")
        candidates.append("cpu")
    else:
        candidates = [requested] if requested == "cpu" else [requested, "cpu"]

    for candidate in candidates:
        try:
            device = torch.device(candidate)
            if device.type != "cpu":
                torch.zeros(1, device=device)
            logger.info("Using torch device: %s (requested %s)", device, requested)
            return device
        except Exception as exc:
            logger.warning("Torch device %r unavailable: %s", candidate, exc)
