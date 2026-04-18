#!/usr/bin/env python3
"""
step4_viewer.py — Generate a self-contained HTML viewer for gaussian splat PLY
files and optionally serve it over HTTP.

Reads viewer_template.html, injects the splat file reference and any furniture
mesh URLs, and writes a ready-to-open viewer.html.
"""

import argparse
import http.server
import os
import shutil
import socketserver
import sys
import threading
from pathlib import Path


TEMPLATE_NAME = "viewer_template.html"


def _find_template() -> str:
    """Locate the viewer template relative to this script."""
    here = Path(__file__).resolve().parent
    candidate = here / TEMPLATE_NAME
    if candidate.is_file():
        return str(candidate)
    print(f"[ERROR] Template not found at {candidate}")
    sys.exit(1)


def generate_viewer(
    splat_path: str,
    output_dir: str,
    furniture_glbs: list[str] | None = None,
    before_splat: str | None = None,
) -> str:
    """
    Build viewer.html from template.

    Args:
        splat_path:      Path to splat.ply
        output_dir:      Directory to write viewer.html + copy assets into
        furniture_glbs:  Optional list of .glb file paths to include
        before_splat:    Optional path to a "before" splat for A/B toggle

    Returns:
        Path to the generated viewer.html
    """
    os.makedirs(output_dir, exist_ok=True)
    template = _find_template()

    with open(template, "r") as f:
        html = f.read()

    # Copy the splat into the serve directory
    splat_dest = os.path.join(output_dir, "splat.ply")
    if os.path.abspath(splat_path) != os.path.abspath(splat_dest):
        shutil.copy2(splat_path, splat_dest)
    html = html.replace("/*__SPLAT_FILE__*/", "splat.ply")

    # Before/after toggle
    if before_splat and os.path.isfile(before_splat):
        before_dest = os.path.join(output_dir, "splat_before.ply")
        shutil.copy2(before_splat, before_dest)
        toggle_html = (
            '<button id="btnBefore" onclick="toggleSplat(\'before\')" class="active">Before</button>'
            '<button id="btnAfter" onclick="toggleSplat(\'after\')">After</button>'
        )
        html = html.replace("<!--__BEFORE_AFTER_TOGGLE__-->", toggle_html)

    # Furniture meshes
    furn_js_list = []
    if furniture_glbs:
        for i, glb in enumerate(furniture_glbs):
            fname = f"furniture_{i}.glb"
            dest = os.path.join(output_dir, fname)
            shutil.copy2(glb, dest)
            furn_js_list.append(f"'{fname}'")
    html = html.replace("/*__FURNITURE_LIST__*/", ", ".join(furn_js_list))

    viewer_path = os.path.join(output_dir, "viewer.html")
    with open(viewer_path, "w") as f:
        f.write(html)

    print(f"  Viewer written to {viewer_path}")
    return viewer_path


def serve(directory: str, port: int = 8080):
    """Start a local HTTP server so the viewer can load assets."""
    os.chdir(directory)

    handler = http.server.SimpleHTTPRequestHandler
    # Allow CORS for local file loading
    class CORSHandler(handler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

    with socketserver.TCPServer(("0.0.0.0", port), CORSHandler) as httpd:
        url = f"http://localhost:{port}/viewer.html"
        print(f"\n  Serving at {url}")
        print(f"  Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


def serve_background(directory: str, port: int = 8080) -> threading.Thread:
    """Start the HTTP server in a background thread (non-blocking)."""
    t = threading.Thread(target=serve, args=(directory, port), daemon=True)
    t.start()
    return t


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate and serve gaussian splat viewer")
    parser.add_argument("--splat", required=True, help="Path to splat.ply")
    parser.add_argument("--output", default=None, help="Output directory (default: ./output)")
    parser.add_argument("--meshes", nargs="*", help="Furniture .glb files to include")
    parser.add_argument("--before-splat", default=None, help="Before-edit splat for A/B toggle")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server after generation")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    out = args.output or os.path.join(os.path.dirname(__file__), "output")
    viewer = generate_viewer(args.splat, out, args.meshes, args.before_splat)

    if args.serve:
        serve(out, args.port)


if __name__ == "__main__":
    main()
