#!/usr/bin/env python3
"""
step3_splat.py — Train a gaussian splat from a COLMAP sparse model using OpenSplat.

The OpenSplat binary is pre-built at the hardcoded path below.
Sets HIP_VISIBLE_DEVICES for AMD Instinct MI300 (gfx942) with ROCm 7.0.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

OPENSPLAT_BIN = (
    "/workspace/SpatialBuild/reconstruction_pipeline/"
    "OpenSplat/build/OpenSplat/build/opensplat"
)


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
    binary = opensplat_bin or OPENSPLAT_BIN

    if not os.path.isfile(binary):
        print(f"[ERROR] OpenSplat binary not found at: {binary}")
        print("  Expected location: " + OPENSPLAT_BIN)
        print("  Run  bash setup.sh  or rebuild OpenSplat manually.")
        return {"success": False, "error": f"OpenSplat not found at {binary}"}

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "splat.ply")

    # OpenSplat expects the COLMAP workspace root (the dir containing images/
    # and sparse/).  If the caller passed sparse/0 directly, walk up two levels.
    if os.path.isfile(os.path.join(colmap_dir, "cameras.bin")):
        project_root = os.path.dirname(os.path.dirname(colmap_dir))
    else:
        project_root = colmap_dir

    cmd = [
        binary,
        project_root,
        "--output", ply_path,
        "-n", str(iterations),
        "--save-every", "5000",
    ]

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")

    print(f"    ▸ Training gaussian splat ({iterations} iters) …")
    print(f"      $ {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    ✗ OpenSplat failed ({elapsed:.1f}s)")
        for line in result.stderr.strip().splitlines()[-30:]:
            print(f"      {line}")
        for line in result.stdout.strip().splitlines()[-20:]:
            print(f"      {line}")
        return {"success": False, "error": "OpenSplat training failed"}

    print(f"    ✓ Training done ({elapsed:.1f}s)")

    # OpenSplat may write splat.ply into the project root instead of --output
    if not os.path.isfile(ply_path):
        alt = os.path.join(project_root, "splat.ply")
        if os.path.isfile(alt):
            shutil.move(alt, ply_path)
        else:
            print("    ✗ splat.ply not found after training")
            return {"success": False, "error": "No output PLY file"}

    size_bytes = os.path.getsize(ply_path)
    size_mb = size_bytes / (1024 * 1024)
    est_gaussians = int(size_bytes / 248)
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
    parser.add_argument("--colmap-dir", required=True,
                        help="COLMAP sparse/0/ directory or workspace root")
    parser.add_argument("--output", required=True, help="Output directory for splat.ply")
    parser.add_argument("--iterations", "-n", type=int, default=30000)
    parser.add_argument("--opensplat-bin", default=None, help="Override path to opensplat binary")
    args = parser.parse_args()
    summary = train_splat(args.colmap_dir, args.output, args.iterations, args.opensplat_bin)
    print(f"\n  Result: {summary}")


if __name__ == "__main__":
    main()