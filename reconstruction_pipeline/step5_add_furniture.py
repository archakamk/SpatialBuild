#!/usr/bin/env python3
"""
step5_add_furniture.py — Generate 3D furniture meshes from text descriptions
via the Tripo3D API and composite them into the viewer.

Falls back to placeholder .glb files if the API is unavailable.
"""

import argparse
import json
import os
import sys
import time

import requests


TRIPO_API_BASE = "https://api.tripo3d.ai/v2/openapi"
# Set your API key via env var or pass --api-key
TRIPO_API_KEY_ENV = "TRIPO3D_API_KEY"


def _get_api_key(override: str | None = None) -> str | None:
    if override:
        return override
    return os.environ.get(TRIPO_API_KEY_ENV)


def generate_mesh_tripo(prompt: str, api_key: str, output_path: str, timeout: int = 300) -> bool:
    """
    Call Tripo3D API: create a task from text, poll until done, download .glb.
    Returns True on success.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Step 1: Create generation task
    print(f"    ▸ Tripo3D: requesting mesh for "{prompt}" …")
    try:
        resp = requests.post(
            f"{TRIPO_API_BASE}/task",
            headers=headers,
            json={"type": "text_to_model", "prompt": prompt, "model_version": "default", "output_format": "glb"},
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json()["data"]["task_id"]
    except Exception as e:
        print(f"    ✗ Tripo3D task creation failed: {e}")
        return False

    # Step 2: Poll until done
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = requests.get(f"{TRIPO_API_BASE}/task/{task_id}", headers=headers, timeout=15)
            data = resp.json().get("data", {})
            status = data.get("status", "unknown")
            if status == "success":
                model_url = data.get("output", {}).get("model", "")
                if not model_url:
                    print("    ✗ Task succeeded but no model URL in response")
                    return False
                # Download
                print(f"    ▸ Downloading mesh …")
                dl = requests.get(model_url, timeout=60)
                with open(output_path, "wb") as f:
                    f.write(dl.content)
                print(f"    ✓ Mesh saved to {output_path} ({len(dl.content)/1024:.0f} KB)")
                return True
            elif status in ("failed", "cancelled"):
                print(f"    ✗ Tripo3D task {status}")
                return False
            else:
                print(f"      … status: {status} ({int(time.time()-t0)}s)")
        except Exception as e:
            print(f"      … poll error: {e}")
        time.sleep(10)

    print("    ✗ Tripo3D timed out")
    return False


def create_placeholder_glb(output_path: str, label: str = "furniture"):
    """
    Write a minimal placeholder .glb so the pipeline can proceed without the API.
    Uses trimesh to create a simple coloured box.
    """
    try:
        import trimesh
        box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        box.visual.face_colors = [100, 149, 237, 255]  # cornflower blue
        box.export(output_path, file_type="glb")
        print(f"    ⚠  Placeholder box written to {output_path} (API unavailable)")
        return True
    except ImportError:
        print("    ✗ trimesh not installed — cannot create placeholder")
        return False


def process_add_commands(
    commands_json: str,
    output_dir: str,
    api_key: str | None = None,
) -> list[str]:
    """
    Read a commands.json, find all "add" actions, generate .glb for each.
    Returns list of .glb file paths.
    """
    with open(commands_json, "r") as f:
        data = json.load(f)

    add_cmds = [c for c in data.get("commands", []) if c.get("action") == "add"]
    if not add_cmds:
        print("  No 'add' commands found in commands JSON.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    key = _get_api_key(api_key)
    glb_paths: list[str] = []

    for i, cmd in enumerate(add_cmds):
        target = cmd.get("target", "object")
        style = cmd.get("params", {}).get("style", "")
        prompt = f"{style} {target}".strip() if style else target
        out_file = os.path.join(output_dir, f"furniture_{i}_{target}.glb")

        success = False
        if key:
            success = generate_mesh_tripo(prompt, key, out_file)

        if not success:
            create_placeholder_glb(out_file, target)

        if os.path.isfile(out_file):
            glb_paths.append(out_file)

    return glb_paths


def add_single(
    description: str,
    output_dir: str,
    api_key: str | None = None,
) -> str | None:
    """Generate a single furniture mesh from a free-text description."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in description)[:40].strip().replace(" ", "_")
    out_file = os.path.join(output_dir, f"{safe_name or 'furniture'}.glb")

    key = _get_api_key(api_key)
    success = False
    if key:
        success = generate_mesh_tripo(description, key, out_file)
    if not success:
        create_placeholder_glb(out_file, description)

    return out_file if os.path.isfile(out_file) else None


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate 3D furniture from text (Tripo3D)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--command", help='Free text description, e.g. "blue velvet mid-century sofa"')
    group.add_argument("--commands-json", help="Path to commands.json with add actions")
    parser.add_argument("--output", default="./output/furniture", help="Output directory for .glb files")
    parser.add_argument("--api-key", default=None, help="Tripo3D API key (or set TRIPO3D_API_KEY)")
    args = parser.parse_args()

    if args.commands_json:
        paths = process_add_commands(args.commands_json, args.output, args.api_key)
        print(f"\n  Generated {len(paths)} mesh(es): {paths}")
    else:
        path = add_single(args.command, args.output, args.api_key)
        print(f"\n  Generated: {path}")


if __name__ == "__main__":
    main()
