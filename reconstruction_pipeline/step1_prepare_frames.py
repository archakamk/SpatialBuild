#!/usr/bin/env python3
"""
step1_prepare_frames.py — Subsample video frames and organise for COLMAP.

Reads a directory of extracted video frames, subsamples to a manageable count,
verifies uniform resolution, and copies them into a COLMAP-ready layout.
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from PIL import Image


def find_frames(frames_dir: str) -> list[str]:
    """Glob for image files and return sorted paths."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    paths.sort()
    if not paths:
        print(f"[ERROR] No image files found in {frames_dir}")
        sys.exit(1)
    return paths


def subsample(paths: list[str], target_min: int = 200, target_max: int = 300) -> list[str]:
    """Evenly subsample frames to stay within [target_min, target_max]."""
    n = len(paths)
    if n <= target_max:
        return paths
    step = max(1, n // target_max)
    selected = paths[::step]
    # Trim if we overshot
    if len(selected) > target_max:
        selected = selected[:target_max]
    return selected


def get_majority_resolution(paths: list[str], sample_size: int = 20) -> tuple[int, int]:
    """Sample a few images and return the most common resolution."""
    from collections import Counter
    sample = paths[:: max(1, len(paths) // sample_size)][:sample_size]
    sizes = Counter()
    for p in sample:
        with Image.open(p) as img:
            sizes[img.size] += 1
    majority_size, _ = sizes.most_common(1)[0]
    return majority_size  # (width, height)


def prepare_frames(
    frames_dir: str,
    workspace: str,
    target_max: int = 300,
) -> dict:
    """
    Main entry point.  Returns a dict with summary info:
      { "total", "selected", "resolution", "output_dir" }
    """
    images_dir = os.path.join(workspace, "images")
    os.makedirs(images_dir, exist_ok=True)

    all_paths = find_frames(frames_dir)
    selected = subsample(all_paths, target_max=target_max)
    print(f"  Frames found: {len(all_paths)}  →  selected: {len(selected)}")

    # Determine target resolution
    target_w, target_h = get_majority_resolution(selected)
    print(f"  Target resolution: {target_w}×{target_h}")

    copied = 0
    for idx, src in enumerate(selected):
        dst = os.path.join(images_dir, f"frame_{idx:05d}.jpg")
        with Image.open(src) as img:
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)
                img.save(dst, "JPEG", quality=95)
            else:
                shutil.copy2(src, dst)
        copied += 1

    summary = {
        "total": len(all_paths),
        "selected": copied,
        "resolution": f"{target_w}x{target_h}",
        "output_dir": images_dir,
    }
    print(f"  Copied {copied} frames to {images_dir}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    default_frames = "/workspace/SpatialBuild/vision_pipeline/data/edited_frames/"
    default_ws = "/workspace/SpatialBuild/reconstruction_pipeline/colmap_workspace/"

    parser = argparse.ArgumentParser(description="Prepare frames for COLMAP")
    parser.add_argument("--frames-dir", default=default_frames,
                        help=f"Directory with extracted video frames (default: {default_frames})")
    parser.add_argument("--workspace", default=default_ws,
                        help=f"COLMAP workspace directory (default: {default_ws})")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to keep (default: 300)")
    args = parser.parse_args()

    summary = prepare_frames(args.frames_dir, args.workspace, target_max=args.max_frames)
    print(f"\n  Summary: {summary}")


if __name__ == "__main__":
    main()
