"""Main pipeline entry point.

Orchestrates the full edit flow:
  1. Load commands from commands.json (via edit_router).
  2. For each command, detect the target with Grounding DINO.
  3. Segment the target with SAM 2 on the seed frame.
  4. Propagate the mask across the video frames.
  5. Apply the edit (recolor / retexture / remove) to every affected frame.
  6. Write edited frames to data/edited_frames/.
  7. Reassemble into an output video via video_io.frames_to_video().

Usage
-----
    python -m src.run_pipeline --frames data/frames --commands commands.json

Public API
----------
run(frames_dir, commands_path, output_dir)
    Execute the full pipeline.

    Parameters
    ----------
    frames_dir : str | Path
        Directory of input frame images (sequential naming, e.g. frame_000000.jpg).
    commands_path : str | Path
        Path to the commands JSON file.
    output_dir : str | Path
        Directory to write edited frames into.

    Returns
    -------
    None
        Edited frames are written to *output_dir*.
"""

from __future__ import annotations


def run(frames_dir, commands_path, output_dir) -> None:
    """Execute the full room-editing pipeline."""
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    from . import config

    parser = argparse.ArgumentParser(description="Room remodeling pipeline")
    parser.add_argument("--frames", default=str(config.FRAMES_DIR))
    parser.add_argument("--commands", default=str(config.PROJECT_ROOT / "commands.json"))
    parser.add_argument("--output", default=str(config.EDITED_FRAMES_DIR))
    args = parser.parse_args()

    run(args.frames, args.commands, args.output)
