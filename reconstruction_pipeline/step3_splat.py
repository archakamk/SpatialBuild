#!/usr/bin/env python3
"""
step3_splat.py — Train a gaussian splat from a COLMAP sparse model using OpenSplat.

Locates the OpenSplat binary (ROCm or CPU build), runs training, and validates
the output .ply file.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


SEARCH_PATHS = [
    # populated by setup.sh
    None,
    "/workspace/SpatialBuild/OpenSplat/build/opensplat",
    os.path.expanduser("~/OpenSplat/build/opensplat"),
]


def _read_config_bin() -> str | None:
    """Read OpenSplat binary path from setup.sh config."""
    config = os.path.join(os.path.dirname(__file__), ".pipeline_config")
    if not os.path.isfile(config):
        return None
    for line in open(config):
        if line.startswith("OPENSPLAT_BIN="):
            val = line.strip().split("=", 1)[1]
            return val if val and os.path.isfile(val) else None
    return None


def find_opensplat() -> str | None:
    """Search for the opensplat binary in known locations."""
    # Config file first
    from_config = _read_config_bin()
    if from_config:
        return from_config
    # Known paths
    for p in SEARCH_PATHS:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # $PATH
    which = shutil.which("opensplat")
    if which:
        return which
    return None


def detect_hip_env() -> dict[str, str]:
    """Return environment variables needed for AMD HIP/ROCm."""
    env = dict(os.environ)
    if os.path.isdir("/opt/rocm"):
        env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
        # Detect override version from rocminfo if possible
        try:
            info = subprocess.check_output(
                ["/opt/rocm/bin/rocminfo"], text=True, timeout=10
            )
            for line in info.splitlines():
                if "gfx" in line.lower():
                    gfx = line.strip().split()[-1]
                    # Map to major.minor.patch for HSA override
                    if gfx.startswith("gfx"):
                        digits = gfx[3:]
                        if len(digits) >= 3:
                            env["HSA_OVERRIDE_GFX_VERSION"] = f"{digits[0]}.{digits[1]}.{digits[2]}"
                    break
        except Exception:
            pass
    return env


def train_splat(
    colmap_dir: str,
    output_dir: str,
    iterations: int = 2000,
    opensplat_bin: str | None = None,
) -> dict:
    """
    Train gaussian splat.  Returns summary dict:
      { "success", "ply_path", "file_size_mb", "est_gaussians" }
    """
    if opensplat_bin is None:
        opensplat_bin = find_opensplat()

    if opensplat_bin is None:
        msg = (
            "OpenSplat binary not found.\n"
            "  Run:  bash setup.sh   to build it, or set --opensplat-bin manually.\n"
            "  For CPU-only (slow):  build OpenSplat without ROCm flags.\n"
        )
        print(f"[ERROR] {msg}")
        return {"success": False, "error": "OpenSplat not found"}

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "splat.ply")

    # The COLMAP dir must contain cameras.bin, images.bin, points3D.bin.
    # OpenSplat expects the *parent* of sparse/0 (the project root with images/).
    # Detect whether user passed sparse/0 directly or the workspace root.
    if os.path.isfile(os.path.join(colmap_dir, "cameras.bin")):
        # User passed sparse/0 — OpenSplat wants the workspace root (two levels up).
        project_root = os.path.dirname(os.path.dirname(colmap_dir))
    else:
        project_root = colmap_dir

    cmd = [
        opensplat_bin,
        project_root,
        "--output", ply_path,
        "-n", str(iterations),
        "--save-every", "0",
    ]

    env = detect_hip_env()
    print(f"    ▸ Training gaussian splat ({iterations} iters) …")
    print(f"      $ {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ✗ OpenSplat failed ({elapsed:.1f}s)")
        for line in result.stderr.strip().splitlines()[-20:]:
            print(f"      {line}")
        for line in result.stdout.strip().splitlines()[-20:]:
            print(f"      {line}")
        return {"success": False, "error": "OpenSplat training failed"}

    print(f"    ✓ Training done ({elapsed:.1f}s)")

    if not os.path.isfile(ply_path):
        # OpenSplat may have written to a default name in the project root
        alt = os.path.join(project_root, "splat.ply")
        if os.path.isfile(alt):
            shutil.move(alt, ply_path)
        else:
            print("    ✗ splat.ply not found after training")
            return {"success": False, "error": "No output PLY file"}

    size_mb = os.path.getsize(ply_path) / (1024 * 1024)
    est_gaussians = int(os.path.getsize(ply_path) / 248)
    print(f"  Output: {ply_path}")
    print(f"  Size:   {size_mb:.1f} MB  (~{est_gaussians:,} gaussians)")

    return {
        "success": True,
        "ply_path": ply_path,
        "file_size_mb": round(size_mb, 1),
        "est_gaussians": est_gaussians,
        "training_time_s": round(elapsed, 1),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train gaussian splat with OpenSplat")
    parser.add_argument("--colmap-dir", required=True, help="Path to COLMAP sparse/0/ or workspace root")
    parser.add_argument("--output", required=True, help="Output directory for splat.ply")
    parser.add_argument("--iterations", "-n", type=int, default=2000)
    parser.add_argument("--opensplat-bin", default=None, help="Path to opensplat binary")
    args = parser.parse_args()
    summary = train_splat(args.colmap_dir, args.output, args.iterations, args.opensplat_bin)
    print(f"\n  Result: {summary}")


if __name__ == "__main__":
    main()
