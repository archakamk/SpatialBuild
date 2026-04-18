"""Object removal via LaMa inpainting.

Uses the simple-lama-inpainting package (already installed in the venv).
All images are RGB uint8 (H, W, 3).  Masks are uint8 (H, W) with 0/255.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from . import config

DILATE_PX = 15  # expand mask to cover edge artefacts / shadows


class ObjectRemover:
    """Remove objects from a scene by inpainting the masked region with LaMa."""

    def __init__(self, device: str | None = None):
        from simple_lama_inpainting import SimpleLama

        self.lama = SimpleLama(device=device or config.DEVICE)

    def remove_object(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint the masked region to erase an object.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), uint8.
        mask : np.ndarray
            Binary mask (H, W), uint8 0/255.  White = region to remove.

        Returns
        -------
        np.ndarray
            Inpainted RGB image (H, W, 3), uint8.
        """
        h, w = image.shape[:2]

        # Dilate the mask so LaMa covers shadow/edge pixels around the object
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1, DILATE_PX * 2 + 1)
        )
        dilated = cv2.dilate(mask, kernel, iterations=1)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(dilated).convert("L")

        result_pil = self.lama(image_pil, mask_pil)

        # simple-lama may pad the output; crop back to original size
        result = np.array(result_pil)[:h, :w]
        return result


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

    # ── 2. Detect + segment ─────────────────────────────────────────────
    grounder = ObjectGrounder()

    target = None
    for prompt in ("door", "furniture", "table", "chair"):
        detections = grounder.ground(rgb, prompt)
        if detections:
            target = prompt
            break

    if not detections:
        raise RuntimeError("No objects detected with any of the test prompts")

    best = max(detections, key=lambda d: d["score"])
    bbox = [int(v) for v in best["bbox"]]
    print(f"Detected '{best['label']}' (prompt='{target}')  "
          f"score={best['score']:.4f}  bbox={bbox}")

    segmenter = FrameSegmenter()
    mask = segmenter.segment_frame(rgb, bbox)
    print(f"Mask coverage: {100 * np.count_nonzero(mask) / mask.size:.1f}%")

    # ── 3. Remove ───────────────────────────────────────────────────────
    remover = ObjectRemover()
    result = remover.remove_object(rgb, mask)
    print(f"Inpainting complete  result shape={result.shape}")

    # ── 4. Save before/after side-by-side ───────────────────────────────
    side_by_side = np.concatenate([rgb, result], axis=1)
    side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

    config.TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.TEST_OUTPUTS_DIR / "removal_test.jpg")
    cv2.imwrite(out_path, side_by_side_bgr)
    print(f"Saved side-by-side to {out_path}")
