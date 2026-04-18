"""SAM 2 wrapper — image segmentation and video mask propagation.

Loads SAM 2.1 via build_sam2 / build_sam2_video_predictor from the local
sam2 repo (already installed as an editable package).

Public API
----------
segment_image(image, boxes)
    Generate masks for one frame given bounding boxes from Grounding DINO.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3).
    boxes : list[list[float]]
        Bounding boxes [[x_min, y_min, x_max, y_max], ...] in pixel coords.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) — uint8, 0 or 255.

propagate_masks(frames_dir, initial_mask, frame_idx)
    Propagate a mask from a single annotated frame across a video's frames
    using SAM 2's video predictor.

    Parameters
    ----------
    frames_dir : str | Path
        Directory containing sequentially-named frame images
        (e.g. frame_000000.jpg, frame_000001.jpg, …).
    initial_mask : np.ndarray
        Binary mask (H, W) for the seed frame.
    frame_idx : int
        Index of the seed frame within the frame sequence.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from frame index → binary mask (H, W), uint8 0/255.
"""

from __future__ import annotations

import numpy as np


def segment_image(image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    """Generate a binary mask for *image* given bounding *boxes*."""
    raise NotImplementedError


def propagate_masks(
    frames_dir,
    initial_mask: np.ndarray,
    frame_idx: int,
) -> dict[int, np.ndarray]:
    """Propagate *initial_mask* from *frame_idx* across all frames."""
    raise NotImplementedError
