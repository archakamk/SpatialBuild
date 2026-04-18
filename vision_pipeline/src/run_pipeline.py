"""Main pipeline entry point.

Usage
-----
    python -m src.run_pipeline --frames data/frames --commands commands.json
    python -m src.run_pipeline --dry-run          # preview without loading models
"""

from __future__ import annotations

import argparse
import glob
import shutil
import time
from pathlib import Path

import cv2

from . import config
from .edit_router import load_commands


def run(
    frames_dir: str | Path,
    commands_path: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
) -> None:
    """Execute the full room-editing pipeline.

    Parameters
    ----------
    frames_dir : str | Path
        Directory of input frames (``frame_000000.jpg`` …).
    commands_path : str | Path
        Path to the commands JSON file.
    output_dir : str | Path
        Directory to write edited frames into.
    dry_run : bool
        If *True*, print the plan without loading models or editing.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # ── Load commands ───────────────────────────────────────────────────
    commands = load_commands(commands_path)
    commands.sort(key=lambda c: c["frame_idx"])

    frame_paths = sorted(glob.glob(str(frames_dir / "*.jpg")))
    total_frames = len(frame_paths)
    print(f"Frames directory : {frames_dir}  ({total_frames} frames)")
    print(f"Commands file    : {commands_path}  ({len(commands)} commands)")
    print(f"Output directory : {output_dir}")
    print()

    # ── Dry-run: just print what would happen ───────────────────────────
    if dry_run:
        for i, cmd in enumerate(commands, 1):
            print(
                f"  [{i}/{len(commands)}] frame={cmd['frame_idx']:>6d}  "
                f"action={cmd['action']:<10s}  target={cmd['target']:<15s}  "
                f"params={cmd.get('params', {})}"
            )
        print("\n(dry run — no models loaded, no edits applied)")
        return

    # ── Copy originals into output dir ──────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() != frames_dir.resolve():
        print("Copying original frames to output dir …")
        for fp in frame_paths:
            shutil.copy2(fp, output_dir / Path(fp).name)
    print()

    # ── Initialise router (loads all models once) ───────────────────────
    from .edit_router import EditRouter

    t0 = time.time()
    router = EditRouter()
    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    # ── Execute each command ────────────────────────────────────────────
    frames_edited: set[int] = set()

    for i, cmd in enumerate(commands, 1):
        print(
            f"[{i}/{len(commands)}] action={cmd['action']}  "
            f"target='{cmd['target']}'  frame_idx={cmd['frame_idx']}"
        )
        t1 = time.time()

        # Edit reads from *output_dir* so sequential commands compose
        edited_frames = router.execute_command(str(output_dir), cmd)

        for fidx, rgb in edited_frames.items():
            out_path = output_dir / f"frame_{fidx:06d}.jpg"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames_edited.add(fidx)

        print(
            f"  → {len(edited_frames)} frames edited  "
            f"({time.time() - t1:.1f}s)\n"
        )

    # ── Reassemble video ────────────────────────────────────────────────
    output_video = str(config.DATA_DIR / "output" / "output.mp4")
    edited_paths = sorted(glob.glob(str(output_dir / "*.jpg")))

    if edited_paths:
        from .video_io import frames_to_video

        print(f"Reassembling {len(edited_paths)} frames → {output_video}")
        success = frames_to_video(edited_paths, output_video, config.FRAME_RATE)
        print(f"Video write {'succeeded' if success else 'FAILED'}: {output_video}")
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(
        f"Done.  {len(commands)} commands processed, "
        f"{len(frames_edited)}/{total_frames} frames edited."
    )
    print(f"Edited frames : {output_dir}")
    print(f"Output video  : {output_video}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Room remodeling pipeline — apply edit commands to video frames"
    )
    parser.add_argument(
        "--frames",
        default=str(config.FRAMES_DIR),
        help="Input frames directory (default: data/frames/)",
    )
    parser.add_argument(
        "--commands",
        default=str(config.PROJECT_ROOT / "commands.json"),
        help="Commands JSON file (default: commands.json)",
    )
    parser.add_argument(
        "--output",
        default=str(config.EDITED_FRAMES_DIR),
        help="Output directory for edited frames (default: data/edited_frames/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the edit plan without loading models or editing",
    )
    args = parser.parse_args()

    run(args.frames, args.commands, args.output, dry_run=args.dry_run)
