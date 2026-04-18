"""Surface editing — recolor and retexture masked regions.

All images are RGB uint8 (H, W, 3).  Masks are uint8 (H, W) with 0/255.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config

FEATHER_RADIUS = 5  # pixels of gaussian blur on mask edges


def _feather_mask(mask: np.ndarray) -> np.ndarray:
    """Return a float32 [0, 1] mask with soft edges."""
    ksize = FEATHER_RADIUS * 2 + 1
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    return soft


class SurfaceEditor:
    """Pixel-level surface edits (recolor / retexture) driven by a binary mask."""

    # ── recolor ─────────────────────────────────────────────────────────

    def recolor(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        target_color: list[int],
    ) -> np.ndarray:
        """Recolor the masked region while preserving luminance detail.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), uint8.
        mask : np.ndarray
            Binary mask (H, W), uint8 0/255.
        target_color : list[int]
            Desired colour as ``[R, G, B]`` in 0–255.

        Returns
        -------
        np.ndarray
            Edited RGB image (H, W, 3), uint8.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Target hue / saturation from the desired colour
        target_patch = np.uint8([[target_color]])
        target_hsv = cv2.cvtColor(target_patch, cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)

        recolored_hsv = hsv.copy()
        recolored_hsv[:, :, 0] = target_hsv[0]  # hue
        recolored_hsv[:, :, 1] = target_hsv[1]  # saturation
        # V channel kept from original → preserves shadows, highlights, texture

        recolored_hsv = np.clip(recolored_hsv, 0, 255).astype(np.uint8)
        recolored_rgb = cv2.cvtColor(recolored_hsv, cv2.COLOR_HSV2RGB)

        alpha = _feather_mask(mask)[:, :, np.newaxis]  # (H, W, 1)
        blended = (
            image.astype(np.float32) * (1.0 - alpha)
            + recolored_rgb.astype(np.float32) * alpha
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ── retexture ───────────────────────────────────────────────────────

    def retexture(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        texture_path: str | Path,
    ) -> np.ndarray:
        """Replace the masked region with a tiled texture, matching luminance.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), uint8.
        mask : np.ndarray
            Binary mask (H, W), uint8 0/255.
        texture_path : str | Path
            Path to the texture image (resolved against
            :pydata:`config.TEXTURES_DIR` if not absolute).

        Returns
        -------
        np.ndarray
            Edited RGB image (H, W, 3), uint8.
        """
        tex_path = Path(texture_path)
        if not tex_path.is_absolute():
            tex_path = config.TEXTURES_DIR / tex_path

        tex_bgr = cv2.imread(str(tex_path))
        if tex_bgr is None:
            raise FileNotFoundError(f"Cannot load texture: {tex_path}")
        tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        tiled = self._tile_texture(tex_rgb, h, w)

        # Scale tiled texture to the mask's bounding box so the tile density
        # roughly matches the surface area, then paste back into a full-size
        # canvas.
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            region = cv2.resize(tiled[0:bh, 0:bw], (bw, bh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros_like(image)
            canvas[y : y + bh, x : x + bw] = region
            tiled = canvas

        # Luminance matching: modulate texture V with original V
        orig_v = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
        tex_hsv = cv2.cvtColor(tiled, cv2.COLOR_RGB2HSV).astype(np.float32)
        tex_v = tex_hsv[:, :, 2]
        # Avoid division by zero; scale texture brightness to match original
        mean_tex_v = tex_v[mask == 255].mean() if np.any(mask == 255) else 1.0
        if mean_tex_v < 1.0:
            mean_tex_v = 1.0
        tex_hsv[:, :, 2] = np.clip(tex_v * (orig_v / mean_tex_v), 0, 255)
        tiled_matched = cv2.cvtColor(tex_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        alpha = _feather_mask(mask)[:, :, np.newaxis]
        blended = (
            image.astype(np.float32) * (1.0 - alpha)
            + tiled_matched.astype(np.float32) * alpha
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _tile_texture(tex: np.ndarray, h: int, w: int) -> np.ndarray:
        """Tile *tex* to at least (h, w) then crop."""
        th, tw = tex.shape[:2]
        reps_y = (h + th - 1) // th
        reps_x = (w + tw - 1) // tw
        tiled = np.tile(tex, (reps_y, reps_x, 1))
        return tiled[:h, :w]


if __name__ == "__main__":
    import glob

    from .grounding import ObjectGrounder
    from .segmentation import FrameSegmenter

    # ── 1. Load first frame ─────────────────────────────────────────────
    frame_paths = sorted(glob.glob(str(config.FRAMES_DIR / "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames in {config.FRAMES_DIR}")

    bgr = cv2.imread(frame_paths[0])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded {frame_paths[0]}  shape={rgb.shape}")

    # ── 2. Detect + segment "wall" ──────────────────────────────────────
    grounder = ObjectGrounder()
    detections = grounder.ground(rgb, "wall")
    if not detections:
        raise RuntimeError("No 'wall' detections")
    best = max(detections, key=lambda d: d["score"])
    bbox = [int(v) for v in best["bbox"]]
    print(f"Detection: {best['label']}  score={best['score']:.4f}  bbox={bbox}")

    segmenter = FrameSegmenter()
    mask = segmenter.segment_frame(rgb, bbox)
    print(f"Mask coverage: {100 * np.count_nonzero(mask) / mask.size:.1f}%")

    # ── 3. Recolor to steel blue ────────────────────────────────────────
    editor = SurfaceEditor()
    recolored = editor.recolor(rgb, mask, [70, 130, 180])

    # ── 4. Save before/after side-by-side ───────────────────────────────
    side_by_side = np.concatenate([rgb, recolored], axis=1)
    side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

    config.TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.TEST_OUTPUTS_DIR / "recolor_test.jpg")
    cv2.imwrite(out_path, side_by_side_bgr)
    print(f"Saved side-by-side to {out_path}")
