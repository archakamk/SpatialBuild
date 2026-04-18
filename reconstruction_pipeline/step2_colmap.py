#!/usr/bin/env python3
"""
step2_colmap.py — Run COLMAP sparse reconstruction on prepared frames.

Stages: feature extraction → sequential matching → sparse mapping → undistortion.
Automatically falls back from GPU to CPU if SIFT GPU fails (common on AMD).
"""

import argparse
import os
import subprocess
import sys
import time


def _run(cmd: list[str], label: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a subprocess with logging.  Returns CompletedProcess or exits."""
    print(f"    ▸ {label}")
    print(f"      $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"    ✗ {label} failed ({elapsed:.1f}s)")
        print(f"      STDERR (last 30 lines):\n")
        for line in result.stderr.strip().splitlines()[-30:]:
            print(f"        {line}")
        return result
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
        print("    ↻ GPU SIFT failed, retrying with CPU …")
        return feature_extraction(workspace, use_gpu=False)
    return result.returncode == 0


def sequential_matching(workspace: str, use_gpu: bool = True) -> bool:
    db = os.path.join(workspace, "database.db")
    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", db,
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SequentialMatching.overlap", "10",
        "--SequentialMatching.loop_detection", "0",
    ]
    result = _run(cmd, "Sequential matching" + (" (GPU)" if use_gpu else " (CPU)"))
    if result.returncode != 0 and use_gpu:
        print("    ↻ GPU matching failed, retrying with CPU …")
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
    """Run model_analyzer and parse output.  Returns stats dict."""
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
            parts = line.split()
            for p in parts:
                try:
                    stats["mean_reprojection_error"] = float(p)
                except ValueError:
                    pass
    return stats


def undistort(workspace: str) -> bool:
    dense_dir = os.path.join(workspace, "dense")
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(workspace, "images"),
        "--input_path", os.path.join(workspace, "sparse", "0"),
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ]
    result = _run(cmd, "Image undistortion")
    return result.returncode == 0


def run_colmap(workspace: str, skip_undistort: bool = False) -> dict:
    """
    Full COLMAP pipeline.  Returns summary dict:
      { "success", "sparse_dir", "registered_images", "points", "mean_reproj_error" }
    """
    if not os.path.isdir(os.path.join(workspace, "images")):
        print(f"[ERROR] No images/ directory in {workspace}")
        sys.exit(1)

    num_input = len([
        f for f in os.listdir(os.path.join(workspace, "images"))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"  Input images: {num_input}")

    # Stage 1: Feature extraction
    if not feature_extraction(workspace):
        return {"success": False, "error": "Feature extraction failed"}

    # Stage 2: Matching
    if not sequential_matching(workspace):
        return {"success": False, "error": "Feature matching failed"}

    # Stage 3: Sparse mapping
    if not sparse_mapping(workspace):
        return {"success": False, "error": "Sparse mapping failed. Try more overlap between frames or a different camera model."}

    # Stage 4: Analyze
    stats = analyze_model(workspace)
    print(f"  Registered: {stats['registered_images']} / {num_input} images")
    print(f"  3D points:  {stats['points']}")
    print(f"  Reproj err: {stats['mean_reprojection_error']:.4f} px")

    if stats["registered_images"] < num_input * 0.5:
        print("  ⚠  WARNING: Less than 50% of images registered — reconstruction may be poor.")
        print("     Suggestions: use more overlapping frames, ensure consistent lighting.")

    # Stage 5: Undistort (optional)
    if not skip_undistort:
        undistort(workspace)

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
    parser.add_argument("--workspace", required=True, help="COLMAP workspace with images/ subdirectory")
    parser.add_argument("--skip-undistort", action="store_true")
    args = parser.parse_args()
    summary = run_colmap(args.workspace, skip_undistort=args.skip_undistort)
    print(f"\n  Result: {summary}")


if __name__ == "__main__":
    main()
