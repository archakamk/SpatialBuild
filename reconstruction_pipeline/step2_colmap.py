#!/usr/bin/env python3
"""
step2_colmap.py — Run COLMAP sparse reconstruction on prepared frames.

Stages: feature extraction → sequential matching → sparse mapping.
Automatically falls back from GPU to CPU SIFT if GPU fails (common on AMD).
"""

import argparse
import os
import subprocess
import sys
import time


def _run(cmd: list[str], label: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a subprocess with full logging on failure."""
    print(f"    ▸ {label}")
    print(f"      $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ✗ {label} failed ({elapsed:.1f}s)")
        if result.stdout.strip():
            print("      STDOUT (last 20 lines):")
            for line in result.stdout.strip().splitlines()[-20:]:
                print(f"        {line}")
        if result.stderr.strip():
            print("      STDERR (last 30 lines):")
            for line in result.stderr.strip().splitlines()[-30:]:
                print(f"        {line}")
    else:
        print(f"    ✓ {label} done ({elapsed:.1f}s)")
    return result


def feature_extraction(workspace: str, use_gpu: bool = True) -> bool:
    db = os.path.join(workspace, "database.db")
    images = os.path.join(workspace, "images")
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", db,
        "--image_path", images,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        "--SiftExtraction.max_num_features", "8192",
    ]
    result = _run(cmd, "Feature extraction" + (" (GPU)" if use_gpu else " (CPU)"))
    if result.returncode != 0 and use_gpu:
        print("    ↻ GPU SIFT failed — retrying with CPU …")
        return feature_extraction(workspace, use_gpu=False)
    return result.returncode == 0


def sequential_matching(workspace: str, use_gpu: bool = True) -> bool:
    db = os.path.join(workspace, "database.db")
    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", db,
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SequentialMatching.overlap", "10",
        "--SequentialMatching.loop_detection", "1",
    ]
    result = _run(cmd, "Sequential matching" + (" (GPU)" if use_gpu else " (CPU)"))
    if result.returncode != 0 and use_gpu:
        print("    ↻ GPU matching failed — retrying with CPU …")
        return sequential_matching(workspace, use_gpu=False)
    return result.returncode == 0


def sparse_mapping(workspace: str) -> bool:
    sparse_dir = os.path.join(workspace, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    images = os.path.join(workspace, "images")
    db = os.path.join(workspace, "database.db")
    cmd = [
        "colmap", "mapper",
        "--database_path", db,
        "--image_path", images,
        "--output_path", sparse_dir,
    ]
    result = _run(cmd, "Sparse mapper", timeout=3600)
    if result.returncode != 0:
        return False

    model_dir = os.path.join(sparse_dir, "0")
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    for f in required:
        if not os.path.isfile(os.path.join(model_dir, f)):
            print(f"    ✗ Missing {f} in {model_dir}")
            return False
    return True


def analyze_model(workspace: str) -> dict:
    """Run model_analyzer and parse output."""
    model_dir = os.path.join(workspace, "sparse", "0")
    result = _run(
        ["colmap", "model_analyzer", "--path", model_dir],
        "Model analysis",
    )
    stats = {"registered_images": 0, "points": 0, "mean_reprojection_error": 0.0}
    for line in (result.stdout + result.stderr).splitlines():
        low = line.lower()
        if "registered images" in low or "cameras" in low:
            nums = [int(s) for s in line.split() if s.isdigit()]
            if nums:
                stats["registered_images"] = nums[0]
        if "points" in low:
            nums = [int(s) for s in line.split() if s.isdigit()]
            if nums:
                stats["points"] = max(nums)
        if "mean reprojection" in low:
            for p in line.split():
                try:
                    stats["mean_reprojection_error"] = float(p)
                except ValueError:
                    pass
    return stats


def run_colmap(workspace: str) -> dict:
    """
    Full COLMAP pipeline.  Returns summary dict:
      { "success", "sparse_dir", "registered_images", "points", "mean_reproj_error" }
    """
    images_path = os.path.join(workspace, "images")
    if not os.path.isdir(images_path):
        print(f"[ERROR] No images/ directory in {workspace}")
        sys.exit(1)

    num_input = len([
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"  Input images: {num_input}")

    if not feature_extraction(workspace):
        return {"success": False, "error": "Feature extraction failed"}

    if not sequential_matching(workspace):
        return {"success": False, "error": "Feature matching failed"}

    if not sparse_mapping(workspace):
        return {
            "success": False,
            "error": (
                "Sparse mapping failed — no images registered.  "
                "Try more overlap between frames or a different camera model."
            ),
        }

    stats = analyze_model(workspace)
    print(f"  Registered: {stats['registered_images']} / {num_input} images")
    print(f"  3D points:  {stats['points']}")
    print(f"  Reproj err: {stats['mean_reprojection_error']:.4f} px")

    if num_input > 0 and stats["registered_images"] < num_input * 0.5:
        print("  ⚠  WARNING: Less than 50% of images registered — reconstruction may be poor.")
        print("     Suggestions: use more overlapping frames, ensure consistent lighting.")

    sparse_dir = os.path.join(workspace, "sparse", "0")
    return {
        "success": True,
        "sparse_dir": sparse_dir,
        "registered_images": stats["registered_images"],
        "total_input_images": num_input,
        "points": stats["points"],
        "mean_reproj_error": stats["mean_reprojection_error"],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run COLMAP reconstruction")
    parser.add_argument("--workspace", required=True,
                        help="COLMAP workspace with images/ subdirectory")
    args = parser.parse_args()
    summary = run_colmap(args.workspace)
    print(f"\n  Result: {summary}")


if __name__ == "__main__":
    main()
