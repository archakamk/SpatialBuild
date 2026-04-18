#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end orchestrator for the 3D reconstruction pipeline.

  frames → COLMAP → gaussian splat → web viewer (+ optional furniture)

Usage:
    python run_pipeline.py \\
        --frames /workspace/SpatialBuild/audio_pipeline/outputs/frames/ \\
        --output /workspace/SpatialBuild/reconstruction_pipeline/output/ \\
        --iterations 2000 \\
        --serve

    python run_pipeline.py --test          # download banana dataset & run full pipeline
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Make sibling modules importable when run as a script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from step1_prepare_frames import prepare_frames
from step2_colmap import run_colmap
from step3_splat import train_splat
from step4_viewer import generate_viewer, serve, serve_background
from step5_add_furniture import process_add_commands


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(step_num: int, total: int, title: str):
    print(f"\n{'═' * 60}")
    print(f"  STEP {step_num}/{total}: {title}")
    print(f"{'═' * 60}\n")


def _timed(label: str):
    """Context manager that prints elapsed time for a block."""
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self._t0 = time.time()
            return self
        def __exit__(self, *_):
            self.elapsed = time.time() - self._t0
            print(f"\n  ⏱  {label}: {self.elapsed:.1f}s\n")
    return Timer()


def detect_system():
    """Print GPU / ROCm info at startup."""
    print("═" * 60)
    print("  System Info")
    print("═" * 60)

    # ROCm
    rocm_path = "/opt/rocm"
    if os.path.isdir(rocm_path):
        ver_file = os.path.join(rocm_path, ".info", "version")
        ver = "unknown"
        if os.path.isfile(ver_file):
            with open(ver_file) as f:
                ver = f.read().strip()
        print(f"  ROCm:        {ver}  ({rocm_path})")

        try:
            info = subprocess.check_output(
                ["/opt/rocm/bin/rocminfo"], text=True, timeout=10
            )
            for line in info.splitlines():
                if "gfx" in line.lower():
                    arch = line.strip().split()[-1]
                    print(f"  GPU arch:    {arch}")
                    break
            for line in info.splitlines():
                if "marketing name" in line.lower():
                    name = line.split(":")[-1].strip()
                    if name:
                        print(f"  GPU name:    {name}")
                    break
        except Exception:
            print("  GPU arch:    (rocminfo failed)")
    else:
        print("  ROCm:        not found")

    # COLMAP
    colmap_ok = shutil.which("colmap") is not None
    print(f"  COLMAP:      {'installed' if colmap_ok else 'NOT FOUND — run setup.sh'}")

    # OpenSplat
    config_file = SCRIPT_DIR / ".pipeline_config"
    osplat = "(not configured — run setup.sh)"
    if config_file.is_file():
        for line in config_file.read_text().splitlines():
            if line.startswith("OPENSPLAT_BIN="):
                val = line.split("=", 1)[1]
                if val and os.path.isfile(val):
                    osplat = val
    print(f"  OpenSplat:   {osplat}")

    # Python
    print(f"  Python:      {sys.version.split()[0]}")
    print("═" * 60)
    return colmap_ok


# ══════════════════════════════════════════════════════════════════════════════
# Test mode — download small dataset
# ══════════════════════════════════════════════════════════════════════════════

def download_test_dataset(dest: str) -> str:
    """
    Download the OpenSplat 'banana' test dataset.
    Returns path to the extracted images directory.
    """
    url = "https://github.com/pierotofy/OpenSplat/releases/download/v1.0/banana.zip"
    zip_path = os.path.join(dest, "banana.zip")
    extract_dir = os.path.join(dest, "banana")

    if os.path.isdir(extract_dir) and len(os.listdir(extract_dir)) > 5:
        print(f"  Test dataset already at {extract_dir}")
        return extract_dir

    os.makedirs(dest, exist_ok=True)
    print(f"  Downloading test dataset from {url} …")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    os.remove(zip_path)

    # The banana dataset ships with its own COLMAP data,
    # but we want to test our full pipeline, so just use the images.
    images_dir = os.path.join(extract_dir, "images")
    if not os.path.isdir(images_dir):
        # Some versions extract flat — look for image files
        for sub in Path(extract_dir).rglob("*.jpg"):
            images_dir = str(sub.parent)
            break
    print(f"  Test images at: {images_dir}")
    return images_dir


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    colmap_ok = detect_system()
    if not colmap_ok:
        print("\n[FATAL] COLMAP is not installed. Run:  bash setup.sh")
        sys.exit(1)

    frames_dir = args.frames
    output_dir = os.path.abspath(args.output)
    colmap_ws  = os.path.join(SCRIPT_DIR, "colmap_workspace")
    os.makedirs(output_dir, exist_ok=True)

    total_steps = 4 + (1 if args.commands else 0)
    timings = {}

    # ── Step 1: Prepare frames ───────────────────────────────────────────────
    _banner(1, total_steps, "Preparing frames")
    if os.path.isdir(os.path.join(colmap_ws, "images")) and not args.force:
        num_existing = len(os.listdir(os.path.join(colmap_ws, "images")))
        if num_existing > 10:
            print(f"  [SKIP] {num_existing} frames already in {colmap_ws}/images/ (use --force to redo)")
            step1_summary = {"selected": num_existing, "total": "?", "resolution": "?", "output_dir": os.path.join(colmap_ws, "images")}
        else:
            args.force = True  # too few frames, redo

    if args.force or not os.path.isdir(os.path.join(colmap_ws, "images")) or len(os.listdir(os.path.join(colmap_ws, "images"))) <= 10:
        with _timed("Step 1") as t1:
            step1_summary = prepare_frames(frames_dir, colmap_ws, target_max=args.max_frames)
        timings["step1"] = t1.elapsed
    else:
        timings["step1"] = 0.0

    # ── Step 2: COLMAP ───────────────────────────────────────────────────────
    _banner(2, total_steps, "COLMAP sparse reconstruction")
    sparse_dir = os.path.join(colmap_ws, "sparse", "0")
    if os.path.isfile(os.path.join(sparse_dir, "cameras.bin")) and not args.force:
        print(f"  [SKIP] Sparse model already at {sparse_dir} (use --force to redo)")
        step2_summary = {"success": True, "sparse_dir": sparse_dir}
        timings["step2"] = 0.0
    else:
        with _timed("Step 2") as t2:
            step2_summary = run_colmap(colmap_ws)
        timings["step2"] = t2.elapsed
        if not step2_summary.get("success"):
            print(f"\n[FATAL] COLMAP failed: {step2_summary.get('error')}")
            sys.exit(1)
        sparse_dir = step2_summary["sparse_dir"]

    # ── Step 3: Gaussian splat training ──────────────────────────────────────
    _banner(3, total_steps, "Gaussian splat training")
    ply_path = os.path.join(output_dir, "splat.ply")
    if os.path.isfile(ply_path) and not args.force:
        size_mb = os.path.getsize(ply_path) / (1024 * 1024)
        print(f"  [SKIP] splat.ply exists ({size_mb:.1f} MB) — use --force to redo")
        step3_summary = {"success": True, "ply_path": ply_path, "file_size_mb": round(size_mb, 1)}
        timings["step3"] = 0.0
    else:
        with _timed("Step 3") as t3:
            step3_summary = train_splat(sparse_dir, output_dir, args.iterations)
        timings["step3"] = t3.elapsed
        if not step3_summary.get("success"):
            print(f"\n[FATAL] Splat training failed: {step3_summary.get('error')}")
            sys.exit(1)
        ply_path = step3_summary["ply_path"]

    # ── Step 4: Generate viewer ──────────────────────────────────────────────
    furniture_glbs = []

    # Step 5 (optional): furniture from commands.json
    if args.commands:
        _banner(total_steps, total_steps, "Adding furniture from voice commands")
        with _timed("Step 5") as t5:
            furn_dir = os.path.join(output_dir, "furniture")
            furniture_glbs = process_add_commands(args.commands, furn_dir, args.tripo_key)
        timings["step5"] = t5.elapsed

    _banner(total_steps - (1 if not args.commands else 0) + (1 if args.commands else 0),
            total_steps, "Generating web viewer")
    with _timed("Step 4") as t4:
        viewer_path = generate_viewer(ply_path, output_dir, furniture_glbs or None)
    timings["step4"] = t4.elapsed

    # ── Summary ──────────────────────────────────────────────────────────────
    size_mb = os.path.getsize(ply_path) / (1024 * 1024)
    est_gauss = int(os.path.getsize(ply_path) / 248)
    frames_used = step1_summary.get("selected", "?")
    frames_total = step1_summary.get("total", "?")
    reg_images = step2_summary.get("registered_images", "?")

    print(f"""
{'═' * 60}
  Pipeline complete!
{'═' * 60}
  Frames used:     {frames_used} / {frames_total}
  COLMAP images:   {reg_images} registered
  Splat file:      {ply_path} ({size_mb:.1f} MB)
  Gaussians:       ~{est_gauss:,}
  Viewer:          {viewer_path}
  Furniture:       {len(furniture_glbs)} mesh(es)

  Timings:
    Step 1 (frames):  {timings.get('step1', 0):.1f}s
    Step 2 (COLMAP):  {timings.get('step2', 0):.1f}s
    Step 3 (splat):   {timings.get('step3', 0):.1f}s
    Step 4 (viewer):  {timings.get('step4', 0):.1f}s
    Step 5 (furnish): {timings.get('step5', 0):.1f}s

  View your splat:
    → Local:  http://localhost:8080/viewer.html
    → Upload: https://playcanvas.com/supersplat/editor
{'═' * 60}
""")

    if args.serve:
        serve(output_dir, args.port)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline: frames → COLMAP → splat → viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frames", help="Directory containing extracted video frames")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "output"), help="Output directory")
    parser.add_argument("--iterations", "-n", type=int, default=2000, help="Splat training iterations")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames for COLMAP (default 300)")
    parser.add_argument("--commands", default=None, help="Path to commands.json for furniture generation")
    parser.add_argument("--tripo-key", default=None, help="Tripo3D API key (or set TRIPO3D_API_KEY env)")
    parser.add_argument("--serve", action="store_true", help="Start local HTTP server when done")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default 8080)")
    parser.add_argument("--force", action="store_true", help="Force re-run all steps even if outputs exist")
    parser.add_argument("--test", action="store_true", help="Download banana test dataset and run pipeline")
    args = parser.parse_args()

    if args.test:
        test_dir = str(SCRIPT_DIR / "test_data")
        images_dir = download_test_dataset(test_dir)
        args.frames = images_dir
        args.output = str(SCRIPT_DIR / "test_output")
        print(f"  Test mode: frames={args.frames}  output={args.output}")

    if not args.frames:
        parser.error("--frames is required (or use --test)")

    if not os.path.isdir(args.frames):
        parser.error(f"Frames directory not found: {args.frames}")

    run(args)


if __name__ == "__main__":
    main()
