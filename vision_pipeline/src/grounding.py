"""Grounding DINO wrapper — open-vocabulary object detection via HuggingFace transformers.

Uses AutoProcessor + AutoModelForZeroShotObjectDetection loaded from
config.GROUNDING_DINO_MODEL ("IDEA-Research/grounding-dino-base").

Does NOT depend on the original GroundingDINO repo (which requires CUDA
compilation).  Everything runs through the pure-Python HuggingFace pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from . import config


class ObjectGrounder:
    """Text-prompted bounding-box detector backed by Grounding DINO."""

    def __init__(self, device: str | None = None):
        self.device = device or config.DEVICE
        self.processor = AutoProcessor.from_pretrained(config.GROUNDING_DINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.GROUNDING_DINO_MODEL,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def ground(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """Detect regions matching *text_prompt* in an RGB numpy image.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), dtype uint8.
        text_prompt : str
            What to detect, e.g. ``"wall"`` or ``"bed"``.
        box_threshold : float
            Minimum confidence to keep a bounding box.
        text_threshold : float
            Minimum confidence for text–box association.

        Returns
        -------
        list[dict]
            Each dict has keys ``"bbox"`` ([x1, y1, x2, y2] in pixels),
            ``"score"`` (float), and ``"label"`` (str).
        """
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)

        # Grounding DINO expects the prompt to end with a period.
        prompt = text_prompt.strip()
        if not prompt.endswith("."):
            prompt += "."

        inputs = self.processor(
            images=pil_image,
            text=[[prompt]],
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

        detections: list[dict] = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            detections.append(
                {
                    "bbox": box.cpu().tolist(),
                    "score": round(score.item(), 4),
                    "label": label,
                }
            )

        return detections

    def ground_centered(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> dict | None:
        """Pick the detection whose centre is closest to the image centre.

        Ray-Ban Meta cameras point where the user is looking, so the
        intended target is almost always near the frame centre.  This
        method uses a moderate ``box_threshold`` (default 0.25) to gather
        candidates, then ranks them by a weighted combination of detection
        confidence and centre proximity.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H, W, 3), dtype uint8.
        text_prompt : str
            What to detect, e.g. ``"wall"`` or ``"bed"``.
        box_threshold : float
            Minimum confidence for bounding-box proposals (kept low to
            maximise candidates).
        text_threshold : float
            Minimum confidence for text–box association.

        Returns
        -------
        dict | None
            The best detection (with extra key ``"combined_score"``), or
            *None* if nothing was detected.
        """
        detections = self.ground(
            image, text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not detections:
            return None

        h, w = image.shape[:2]
        cx_img, cy_img = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)

        best, best_combined = None, -1.0
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx_box = (x1 + x2) / 2.0
            cy_box = (y1 + y2) / 2.0
            dist = np.sqrt((cx_box - cx_img) ** 2 + (cy_box - cy_img) ** 2)
            proximity = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0

            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratio = bbox_area / (h * w) if (h * w) > 0 else 0.0
            if area_ratio > 0.7:
                size_penalty = 0.1
            elif area_ratio > 0.5:
                size_penalty = 0.5
            else:
                size_penalty = 1.0

            combined = (d["score"] * 0.4 + proximity * 0.6) * size_penalty
            if combined > best_combined:
                best_combined = combined
                best = d

        best["combined_score"] = round(best_combined, 4)
        return best


if __name__ == "__main__":
    import cv2
    import glob

    frame_pattern = str(config.FRAMES_DIR / "*.jpg")
    frame_paths = sorted(glob.glob(frame_pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in {config.FRAMES_DIR}")

    bgr = cv2.imread(frame_paths[0])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    grounder = ObjectGrounder()

    # ── ground() — all detections ───────────────────────────────────────
    detections = grounder.ground(rgb, "wall")
    print(f"Image: {frame_paths[0]}")
    print(f"\n=== ground() — {len(detections)} detections ===")

    vis_all = bgr.copy()
    for d in detections:
        print(f"  {d['label']:>12s}  score={d['score']:.4f}  bbox={d['bbox']}")
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(vis_all, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_all,
            f"{d['label']} {d['score']:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ── ground_centered() — single best pick ────────────────────────────
    centered = grounder.ground_centered(rgb, "wall")
    print(f"\n=== ground_centered() ===")

    vis_centered = bgr.copy()
    if centered:
        print(
            f"  {centered['label']:>12s}  score={centered['score']:.4f}  "
            f"combined={centered['combined_score']:.4f}  bbox={centered['bbox']}"
        )
        x1, y1, x2, y2 = [int(v) for v in centered["bbox"]]
        cv2.rectangle(vis_centered, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            vis_centered,
            f"{centered['label']} comb={centered['combined_score']:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    else:
        print("  No detections")

    # ── Save both ───────────────────────────────────────────────────────
    config.TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_all = str(config.TEST_OUTPUTS_DIR / "grounding_test.jpg")
    out_ctr = str(config.TEST_OUTPUTS_DIR / "grounding_centered_test.jpg")
    cv2.imwrite(out_all, vis_all)
    cv2.imwrite(out_ctr, vis_centered)
    print(f"\nSaved all-detections  → {out_all}")
    print(f"Saved center-picked   → {out_ctr}")
