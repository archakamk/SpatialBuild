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
    detections = grounder.ground(rgb, "wall")

    print(f"Image: {frame_paths[0]}")
    print(f"Detections ({len(detections)}):")
    for d in detections:
        print(f"  {d['label']:>12s}  score={d['score']:.4f}  bbox={d['bbox']}")
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"{d['label']} {d['score']:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    config.TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.TEST_OUTPUTS_DIR / "grounding_test.jpg")
    cv2.imwrite(out_path, bgr)
    print(f"Saved annotated image to {out_path}")
