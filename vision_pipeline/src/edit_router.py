"""Edit router — loads commands, dispatches to the correct editor module.

Handles the full per-command flow: detect → segment → propagate → edit.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import cv2
import numpy as np

from . import config

VALID_ACTIONS = {"recolor", "retexture", "remove"}


def load_commands(path: str | Path) -> list[dict]:
    """Load and validate a commands JSON file.

    Returns
    -------
    list[dict]
        Each dict has keys: t, frame_idx, action, target, params.
    """
    with open(path) as f:
        commands = json.load(f)

    for i, cmd in enumerate(commands):
        for key in ("frame_idx", "action", "target"):
            if key not in cmd:
                raise ValueError(f"Command {i} missing required key '{key}'")
        if cmd["action"] not in VALID_ACTIONS:
            raise ValueError(
                f"Command {i}: unknown action '{cmd['action']}' "
                f"(expected one of {VALID_ACTIONS})"
            )
        cmd.setdefault("params", {})
        cmd.setdefault("t", 0.0)

    return commands


class EditRouter:
    """Orchestrates detect → segment → (propagate) → edit for one command."""

    def __init__(self):
        from .grounding import ObjectGrounder
        from .object_edit import ObjectRemover
        from .segmentation import FrameSegmenter
        from .surface_edit import SurfaceEditor

        self.grounder = ObjectGrounder()
        self.segmenter = FrameSegmenter()
        self.surface_editor = SurfaceEditor()
        self.object_remover = ObjectRemover()

    def execute_command(
        self,
        frames_dir: str | Path,
        command: dict,
    ) -> dict[int, np.ndarray]:
        """Run a single edit command across the frame sequence.

        Parameters
        ----------
        frames_dir : str | Path
            Directory of sequential JPEG frames (``frame_000000.jpg`` …).
        command : dict
            One entry from :func:`load_commands`.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping frame_idx → edited RGB image (H, W, 3) uint8.
        """
        frames_dir = Path(frames_dir)
        frame_paths = sorted(glob.glob(str(frames_dir / "*.jpg")))
        if not frame_paths:
            raise FileNotFoundError(f"No .jpg frames in {frames_dir}")

        anchor_idx = command["frame_idx"]
        action = command["action"]
        target = command["target"]
        params = command.get("params", {})

        # ── 1. Load anchor frame ────────────────────────────────────────
        anchor_path = frames_dir / f"frame_{anchor_idx:06d}.jpg"
        if not anchor_path.exists():
            raise FileNotFoundError(f"Anchor frame not found: {anchor_path}")

        anchor_bgr = cv2.imread(str(anchor_path))
        anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)

        # ── 2. Detect (centre-biased for Ray-Ban Meta gaze prior) ────────
        best = self.grounder.ground_centered(anchor_rgb, target)
        if best is None:
            print(f"  [warn] No detections for '{target}' — skipping command")
            return {}

        bbox = [int(v) for v in best["bbox"]]
        print(
            f"  Detected '{best['label']}' score={best['score']:.3f} "
            f"combined={best.get('combined_score', 0):.3f} bbox={bbox}"
        )

        # ── 3. Segment anchor frame ─────────────────────────────────────
        anchor_mask = self.segmenter.segment_frame(anchor_rgb, bbox)

        # ── 4. Propagate (or single-frame shortcut) ─────────────────────
        if len(frame_paths) <= 1:
            masks = {anchor_idx: anchor_mask}
        else:
            masks = self.segmenter.propagate_mask(
                str(frames_dir), anchor_idx, anchor_mask
            )
            if anchor_idx not in masks:
                masks[anchor_idx] = anchor_mask

        # ── 5. Apply edit to each frame ─────────────────────────────────
        edited: dict[int, np.ndarray] = {}
        for fidx, mask in sorted(masks.items()):
            fpath = frames_dir / f"frame_{fidx:06d}.jpg"
            if not fpath.exists():
                continue
            bgr = cv2.imread(str(fpath))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            edited[fidx] = self._apply_edit(action, rgb, mask, params)

        return edited

    # ── private ─────────────────────────────────────────────────────────

    def _apply_edit(
        self,
        action: str,
        image: np.ndarray,
        mask: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Dispatch to the right editor and return the edited RGB image."""
        if action == "recolor":
            color = params.get("color", [128, 128, 128])
            return self.surface_editor.recolor(image, mask, color)

        if action == "retexture":
            texture_file = params.get("texture_file", "")
            return self.surface_editor.retexture(image, mask, texture_file)

        if action == "remove":
            return self.object_remover.remove_object(image, mask)

        raise ValueError(f"Unknown action: {action}")
