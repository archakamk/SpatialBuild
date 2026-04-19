#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end orchestrator for the 3D reconstruction pipeline.

  frames → COLMAP → gaussian splat → web viewer (+ optional furniture)

Default invocation (uses test frames, no args needed):
    python run_pipeline.py

Full invocation:
    python run_pipeline.py \\
        --frames /workspace/SpatialBuild/vision_pipeline/data/frames/ \\
        --output /workspace/SpatialBuild/reconstruction_pipeline/output/ \\
        --iterations 2000 \\
        --serve
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make sibling modules importable when run as a script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from step1_prepare_frames import prepare_frames
from step2_colmap import run_colmap
from step3_splat import train_splat, OPENSPLAT_BIN
from step4_viewer import generate_viewer, serve, serve_background
from step5_add_furniture import process_add_commands

# ── Hardcoded project paths ──────────────────────────────────────────────────
DEFAULT_FRAMES  = "/workspace/SpatialBuild/vision_pipeline/data/frames/"
DEFAULT_OUTPUT  = "/workspace/SpatialBuild/reconstruction_pipeline/output/"
DEFAULT_COLMAP_WS = "/workspace/SpatialBuild/reconstruction_pipeline/colmap_workspace/"
DEFAULT_COMMANDS = "/workspace/SpatialBuild/audio_pipeline/outputs/commands.json"


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
    """Print GPU / ROCm / tool info at startup.  Returns True if COLMAP found."""
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
            print("  GPU arch:    (rocminfo unavailable)")
    else:
        print("  ROCm:        not found")

    # COLMAP
    colmap_ok = shutil.which("colmap") is not None
    print(f"  COLMAP:      {shutil.which('colmap') or 'NOT FOUND'}")

    # OpenSplat
    osplat_ok = os.path.isfile(OPENSPLAT_BIN)
    print(f"  OpenSplat:   {OPENSPLAT_BIN}  ({'OK' if osplat_ok else 'NOT FOUND'})")

    # Python + torch
    print(f"  Python:      {sys.version.split()[0]}")
    try:
        import torch
        print(f"  PyTorch:     {torch.__version__}  (CUDA/HIP: {torch.cuda.is_available()})")
    except ImportError:
        print("  PyTorch:     not installed")

    # Disk space
    try:
        st = os.statvfs("/workspace")
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        print(f"  Disk free:   {free_gb:.1f} GB  (/workspace)")
    except Exception:
        pass

    print("═" * 60)
    return colmap_ok


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    colmap_ok = detect_system()
    if not colmap_ok:
        print("\n[FATAL] COLMAP is not installed.  Run:  bash setup.sh")
        sys.exit(1)

    frames_dir = args.frames
    output_dir = os.path.abspath(args.output)
    colmap_ws  = os.path.abspath(args.workspace)
    os.makedirs(output_dir, exist_ok=True)

    total_steps = 4 + (1 if args.commands else 0)
    timings = {}
    step1_summary: dict = {}

    # ── Step 1: Prepare frames ───────────────────────────────────────────────
    _banner(1, total_steps, "Preparing frames")
    images_dir = os.path.join(colmap_ws, "images")
    skip_step1 = False

    if os.path.isdir(images_dir) and not args.force:
        num_existing = len([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if num_existing > 10:
            print(f"  [SKIP] {num_existing} frames already in {images_dir} (use --force to redo)")
            step1_summary = {
                "selected": num_existing, "total": "?",
                "resolution": "?", "output_dir": images_dir,
            }
            skip_step1 = True
            timings["step1"] = 0.0

    if not skip_step1:
        with _timed("Step 1") as t1:
            step1_summary = prepare_frames(frames_dir, colmap_ws, target_max=args.max_frames)
        timings["step1"] = t1.elapsed

    # ── Step 2: COLMAP ───────────────────────────────────────────────────────
    _banner(2, total_steps, "COLMAP sparse reconstruction")
    sparse_dir = os.path.join(colmap_ws, "sparse", "0")
    step2_summary: dict = {}

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

    # ── Step 5 (optional): furniture from commands.json ──────────────────────
    furniture_glbs: list[str] = []
    if args.commands and os.path.isfile(args.commands):
        _banner(5, total_steps, "Adding furniture from voice commands")
        with _timed("Step 5") as t5:
            furn_dir = os.path.join(output_dir, "furniture")
            furniture_glbs = process_add_commands(args.commands, furn_dir, args.tripo_key)
        timings["step5"] = t5.elapsed

    # ── Step 4: Generate viewer ──────────────────────────────────────────────
    _banner(4, total_steps, "Generating web viewer")
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
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py                           # uses default test frames\n"
            "  python run_pipeline.py --serve                   # … and start viewer server\n"
            "  python run_pipeline.py --frames /my/frames/ -n 5000 --force\n"
        ),
    )
    parser.add_argument("--frames", default=DEFAULT_FRAMES,
                        help=f"Directory of extracted video frames (default: {DEFAULT_FRAMES})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--workspace", default=DEFAULT_COLMAP_WS,
                        help=f"COLMAP workspace directory (default: {DEFAULT_COLMAP_WS})")
    parser.add_argument("--iterations", "-n", type=int, default=2000,
                        help="Splat training iterations (default: 2000)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to subsample for COLMAP (default: 300)")
    parser.add_argument("--commands", default=None,
                        help=f"Path to commands.json for furniture (default: None, try {DEFAULT_COMMANDS})")
    parser.add_argument("--tripo-key", default=None,
                        help="Tripo3D API key (or set TRIPO3D_API_KEY env var)")
    parser.add_argument("--serve", action="store_true",
                        help="Start local HTTP server when done")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP server port (default: 8080)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps even if outputs exist")
    args = parser.parse_args()

    if not os.path.isdir(args.frames):
        print(f"[ERROR] Frames directory not found: {args.frames}")
        print(f"  Default path: {DEFAULT_FRAMES}")
        print(f"  Pass --frames /path/to/your/frames/  to override.")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
