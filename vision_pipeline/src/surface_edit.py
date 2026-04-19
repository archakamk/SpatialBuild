"""Surface editing — recolor and retexture masked regions.

All images are RGB uint8 (H, W, 3).  Masks are uint8 (H, W) with 0/255.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config

FEATHER_RADIUS = 7  # pixels of gaussian blur on mask edges
RECOLOR_BLEND = 0.5  # how much of the target color to mix in (0 = none, 1 = full)


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

        Works well even on near-white or very light surfaces where a
        pure HSV hue/saturation swap would produce almost no visible
        change.  Instead we blend a solid colour fill at controlled
        strength, then re-inject the original lighting (V channel) so
        shadow gradients and surface texture are preserved.

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
        h, w = image.shape[:2]
        img_f = image.astype(np.float32)

        # Solid fill of the target colour at full image size
        colored = np.full_like(img_f, target_color, dtype=np.float32)

        # Feathered alpha from the mask
        alpha = _feather_mask(mask)[:, :, np.newaxis]  # (H, W, 1)
        effective = alpha * RECOLOR_BLEND  # blend strength inside the mask

        # Blend: keep some original texture, push towards target colour
        blended = img_f * (1.0 - effective) + colored * effective

        # Preserve original lighting: mix V channels so shadows stay
        orig_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        orig_v = orig_hsv[:, :, 2]

        blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
        result_hsv = cv2.cvtColor(blended_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        result_v = result_hsv[:, :, 2]

        result_hsv[:, :, 2] = np.clip(result_v * 0.3 + orig_v * 0.7, 0, 255)
        final = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Outside the mask, keep the original image untouched
        mask_3ch = np.broadcast_to(alpha > 0, final.shape)
        out = image.copy()
        out[mask_3ch] = final[mask_3ch]
        return out

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

    # ── place_image ──────────────────────────────────────────────────────

    def place_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        overlay_path: str | Path,
        scale: float = 0.4,
    ) -> np.ndarray:
        """Place an image (e.g. a painting) onto a masked surface region.

        The overlay is aspect-ratio-scaled, centred within the mask's
        bounding rect, clipped to the mask, brightness-matched, and
        alpha-blended with feathered edges.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), uint8.
        mask : np.ndarray
            Binary mask (H, W), uint8 0/255.
        overlay_path : str | Path
            Path to the overlay image (resolved against
            :pydata:`config.DATA_DIR` if not absolute).
        scale : float
            Fraction of ``min(mask_width, mask_height)`` to use as the
            overlay's largest dimension.

        Returns
        -------
        np.ndarray
            Composited RGB image (H, W, 3), uint8.
        """
        ov_path = Path(overlay_path)
        if not ov_path.is_absolute():
            ov_path = config.DATA_DIR / ov_path

        ov_bgr = cv2.imread(str(ov_path), cv2.IMREAD_UNCHANGED)
        if ov_bgr is None:
            raise FileNotFoundError(f"Cannot load overlay: {ov_path}")

        # Handle BGRA (with alpha channel) or plain BGR
        if ov_bgr.shape[2] == 4:
            ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGRA2RGBA)
            ov_alpha = ov_rgb[:, :, 3].astype(np.float32) / 255.0
            ov_rgb = ov_rgb[:, :, :3]
        else:
            ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)
            ov_alpha = np.ones(ov_rgb.shape[:2], dtype=np.float32)

        # Bounding rect of the mask
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image.copy()
        bx, by, bw, bh = cv2.boundingRect(coords)

        # Scale overlay to fit, preserving aspect ratio
        target_dim = int(scale * min(bw, bh))
        if target_dim < 1:
            return image.copy()

        oh, ow = ov_rgb.shape[:2]
        ratio = target_dim / max(oh, ow)
        new_w, new_h = max(1, int(ow * ratio)), max(1, int(oh * ratio))
        ov_rgb = cv2.resize(ov_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ov_alpha = cv2.resize(ov_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Centre within the bounding rect
        cx = bx + bw // 2 - new_w // 2
        cy = by + bh // 2 - new_h // 2

        # Clip to image bounds
        img_h, img_w = image.shape[:2]
        sx = max(0, -cx)
        sy = max(0, -cy)
        ex = min(new_w, img_w - cx)
        ey = min(new_h, img_h - cy)
        if sx >= ex or sy >= ey:
            return image.copy()

        dst_y1, dst_y2 = cy + sy, cy + ey
        dst_x1, dst_x2 = cx + sx, cx + ex
        ov_crop = ov_rgb[sy:ey, sx:ex].astype(np.float32)
        al_crop = ov_alpha[sy:ey, sx:ex]

        # Only place where the wall mask is active
        wall_mask_region = mask[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32) / 255.0
        effective_alpha = al_crop * wall_mask_region

        # Feather edges of the placement
        ksize = 5
        effective_alpha = cv2.GaussianBlur(effective_alpha, (ksize, ksize), 0)
        effective_alpha = effective_alpha[:, :, np.newaxis]

        # Brightness matching: shift overlay V toward original V
        orig_region = image[dst_y1:dst_y2, dst_x1:dst_x2]
        orig_v = cv2.cvtColor(orig_region, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
        ov_hsv = cv2.cvtColor(ov_crop.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        ov_hsv[:, :, 2] = np.clip(ov_hsv[:, :, 2] * 0.7 + orig_v * 0.3, 0, 255)
        ov_matched = cv2.cvtColor(ov_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Alpha-blend
        dst_region = orig_region.astype(np.float32)
        blended = dst_region * (1.0 - effective_alpha) + ov_matched * effective_alpha

        out = image.copy()
        out[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

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
