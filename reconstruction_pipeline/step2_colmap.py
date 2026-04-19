#!/usr/bin/env python3
"""
step2_colmap.py — Run COLMAP sparse reconstruction on prepared frames.

Stages: feature extraction → sequential matching → sparse mapping.
Uses xvfb-run to provide a virtual display for COLMAP GPU SIFT (OpenGL).
Falls back to CPU if Xvfb or GPU fails.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


def _has_xvfb() -> bool:
    return shutil.which("xvfb-run") is not None


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


def feature_extraction(workspace: str) -> bool:
    db = os.path.join(workspace, "database.db")
    images = os.path.join(workspace, "images")

    base_cmd = [
        "colmap", "feature_extractor",
        "--database_path", db,
        "--image_path", images,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "8192",
    ]

    # Try GPU via xvfb-run first
    if _has_xvfb():
        gpu_cmd = ["xvfb-run", "--auto-servernum", "-s", "-screen 0 1920x1080x24"] + base_cmd + [
            "--SiftExtraction.use_gpu", "1",
        ]
        result = _run(gpu_cmd, "Feature extraction (GPU via Xvfb)")
        if result.returncode == 0:
            return True
        print("    ↻ GPU extraction failed — falling back to CPU …")

    cpu_cmd = base_cmd + ["--SiftExtraction.use_gpu", "0"]
    result = _run(cpu_cmd, "Feature extraction (CPU)")
    return result.returncode == 0


def sequential_matching(workspace: str) -> bool:
    db = os.path.join(workspace, "database.db")
    # Loop detection disabled — requires a vocab tree file that most installs
    # don't ship.  Sequential overlap of 10 is sufficient for video frames.
    base_cmd = [
        "colmap", "sequential_matcher",
        "--database_path", db,
        "--SequentialMatching.overlap", "10",
        "--SequentialMatching.loop_detection", "0",
    ]

    # Try GPU via xvfb-run first
    if _has_xvfb():
        gpu_cmd = ["xvfb-run", "--auto-servernum", "-s", "-screen 0 1920x1080x24"] + base_cmd + [
            "--SiftMatching.use_gpu", "1",
        ]
        result = _run(gpu_cmd, "Sequential matching (GPU via Xvfb)")
        if result.returncode == 0:
            return True
        print("    ↻ GPU matching failed — falling back to CPU …")

    cpu_cmd = base_cmd + ["--SiftMatching.use_gpu", "0"]
    result = _run(cpu_cmd, "Sequential matching (CPU)")
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

    if _has_xvfb():
        print("  Xvfb:        available (GPU SIFT enabled)")
    else:
        print("  Xvfb:        not found (CPU-only mode)")
        print("               Install with: apt-get install -y xvfb")

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