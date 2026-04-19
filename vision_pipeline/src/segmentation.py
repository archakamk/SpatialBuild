"""SAM 2 wrapper — single-frame segmentation and video mask propagation.

Loads SAM 2.1 via build_sam2 / build_sam2_video_predictor from the local
sam2 repo (already pip-installed as an editable package).
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from . import config


def _prepare_video_dir(frames_dir: str) -> tuple[str, str | None]:
    """Ensure the frame directory uses bare-integer filenames for SAM 2.

    SAM 2's ``load_video_frames`` expects filenames like ``000000.jpg``.
    If the directory contains ``frame_``-prefixed files, we create a
    temporary directory with symlinks that strip the prefix.

    Returns ``(effective_dir, temp_dir_to_cleanup)``.  *temp_dir* is
    ``None`` when no remapping was needed.
    """
    fdir = Path(frames_dir)
    jpgs = sorted(fdir.glob("*.jpg"))
    if not jpgs:
        return frames_dir, None

    has_prefix = any(p.name.startswith("frame_") for p in jpgs)
    if not has_prefix:
        return frames_dir, None

    tmp = tempfile.mkdtemp(prefix="sam2_frames_")
    for p in jpgs:
        name = p.name
        if name.startswith("frame_"):
            name = name[len("frame_"):]
        os.symlink(str(p.resolve()), os.path.join(tmp, name))

    return tmp, tmp


class FrameSegmenter:
    """Box-prompted mask prediction (single frame) and video propagation."""

    def __init__(self, device: str | None = None):
        self.device = device or config.DEVICE

        sam2_model = build_sam2(
            config.SAM2_CONFIG,
            config.SAM2_CHECKPOINT,
            device=self.device,
        )
        self.predictor = SAM2ImagePredictor(sam2_model)

    # ── single-frame segmentation ───────────────────────────────────────

    def segment_frame(
        self,
        image: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray:
        """Segment a region in one image given a bounding box.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), dtype uint8.
        bbox : list[int]
            ``[x1, y1, x2, y2]`` in absolute pixel coordinates (from
            :pyclass:`ObjectGrounder`).

        Returns
        -------
        np.ndarray
            Binary mask (H, W), uint8 with values 0 or 255.
        """
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox, dtype=np.float32),
            multimask_output=True,
        )
        # masks: (C, H, W) bool — pick the one with the highest IoU score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]  # (H, W) bool
        return (mask.astype(np.uint8) * 255)

    # ── video mask propagation ──────────────────────────────────────────

    def propagate_mask(
        self,
        frames_dir: str | Path,
        anchor_frame_idx: int,
        anchor_mask: np.ndarray,
        frame_range: tuple[int, int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Propagate a mask from one frame across a sequence of JPEG frames.

        Parameters
        ----------
        frames_dir : str | Path
            Directory of sequentially-named frames (``frame_000000.jpg`` …).
        anchor_frame_idx : int
            Index of the frame that *anchor_mask* belongs to.
        anchor_mask : np.ndarray
            Binary mask (H, W), uint8 0/255, for the anchor frame.
        frame_range : tuple[int, int] | None
            ``(start, end)`` inclusive frame indices to propagate over.
            *None* → propagate across all frames in the directory.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from frame index → binary mask (H, W), uint8 0/255.
        """
        effective_dir, tmp_dir = _prepare_video_dir(str(frames_dir))

        try:
            video_predictor = build_sam2_video_predictor(
                config.SAM2_CONFIG,
                config.SAM2_CHECKPOINT,
                device=self.device,
            )

            inference_state = video_predictor.init_state(video_path=effective_dir)

            # Register the anchor mask (convert 0/255 uint8 → bool for SAM 2)
            bool_mask = torch.from_numpy((anchor_mask > 0).astype(np.float32))
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=anchor_frame_idx,
                obj_id=0,
                mask=bool_mask,
            )

            # Determine total frame count and optional range limits
            num_frames = inference_state["num_frames"]
            start = 0 if frame_range is None else frame_range[0]
            end = num_frames - 1 if frame_range is None else frame_range[1]

            masks_out: dict[int, np.ndarray] = {}

            # Forward pass (anchor → end)
            if anchor_frame_idx <= end:
                for frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=anchor_frame_idx,
                    max_frame_num_to_track=end - anchor_frame_idx + 1,
                    reverse=False,
                ):
                    if start <= frame_idx <= end:
                        mask = (video_res_masks[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                        masks_out[frame_idx] = mask

            # Backward pass (anchor → start)
            if anchor_frame_idx > start:
                video_predictor.reset_state(inference_state)
                inference_state = video_predictor.init_state(video_path=effective_dir)
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=anchor_frame_idx,
                    obj_id=0,
                    mask=bool_mask,
                )
                for frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=anchor_frame_idx,
                    max_frame_num_to_track=anchor_frame_idx - start + 1,
                    reverse=True,
                ):
                    if start <= frame_idx <= end:
                        mask = (video_res_masks[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                        masks_out[frame_idx] = mask

        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return masks_out


if __name__ == "__main__":
    import cv2

    from .grounding import ObjectGrounder

    # ── 1. Load first frame ─────────────────────────────────────────────
    frame_paths = sorted(glob.glob(str(config.FRAMES_DIR / "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames in {config.FRAMES_DIR}")

    bgr = cv2.imread(frame_paths[0])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded {frame_paths[0]}  shape={rgb.shape}")

    # ── 2. Detect "wall" ────────────────────────────────────────────────
    grounder = ObjectGrounder()
    detections = grounder.ground(rgb, "wall")
    if not detections:
        raise RuntimeError("Grounding DINO found no 'wall' detections")

    best = max(detections, key=lambda d: d["score"])
    bbox = [int(v) for v in best["bbox"]]
    print(f"Best detection: {best['label']}  score={best['score']:.4f}  bbox={bbox}")

    # ── 3. Segment ──────────────────────────────────────────────────────
    segmenter = FrameSegmenter()
    mask = segmenter.segment_frame(rgb, bbox)
    print(f"Mask shape: {mask.shape}  dtype={mask.dtype}")

    total_pixels = mask.shape[0] * mask.shape[1]
    covered = int(np.count_nonzero(mask))
    print(f"Coverage: {covered}/{total_pixels} pixels ({100 * covered / total_pixels:.1f}%)")

    # ── 4. Visualize: semi-transparent red overlay ──────────────────────
    overlay = bgr.copy()
    overlay[mask == 255] = (0, 0, 255)  # red in BGR
    vis = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    config.TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.TEST_OUTPUTS_DIR / "segmentation_test.jpg")
    cv2.imwrite(out_path, vis)
    print(f"Saved overlay to {out_path}")
