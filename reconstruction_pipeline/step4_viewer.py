#!/usr/bin/env python3
"""
step4_viewer.py — Generate a self-contained HTML viewer for gaussian splat PLY
files and optionally serve it over HTTP.

Reads viewer_template.html, injects the splat file reference and any furniture
mesh URLs, and writes a ready-to-open viewer.html.

For Jupyter environments: use display_in_notebook() to render inline, or
serve_background() + Jupyter's proxy URL to access from your local browser.
"""

import argparse
import base64
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


# ═══════════════════════════════════════════════════════════════════════════════
# Serving — works for both local and Jupyter/remote environments
# ═══════════════════════════════════════════════════════════════════════════════

def _get_jupyter_base_url() -> str | None:
    """Try to detect the Jupyter server base URL for proxy access."""
    # Check common env vars set by Jupyter Hub / cloud notebook platforms
    for var in ("JUPYTERHUB_SERVICE_PREFIX", "JUPYTER_BASE_URL", "NB_PREFIX"):
        val = os.environ.get(var)
        if val:
            return val.rstrip("/")
    return None


def serve(directory: str, port: int = 8080):
    """Start a local HTTP server.  Prints proxy URL hints for Jupyter."""
    os.chdir(directory)

    handler = http.server.SimpleHTTPRequestHandler

    class CORSHandler(handler):
        """Allow cross-origin requests so the viewer can load .ply files."""
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()
        def log_message(self, fmt, *a):
            if int(a[1]) >= 400:
                super().log_message(fmt, *a)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), CORSHandler) as httpd:
        local_url = f"http://localhost:{port}/viewer.html"

        print(f"\n  ┌───────────────────────────────────────────────────────┐")
        print(f"  │  Server running on port {port:<30d}│")
        print(f"  │                                                       │")
        print(f"  │  Local:   {local_url:<45s}│")

        # Jupyter proxy hint
        jupyter_base = _get_jupyter_base_url()
        if jupyter_base:
            proxy_url = f"{jupyter_base}/proxy/{port}/viewer.html"
            print(f"  │  Jupyter: {proxy_url:<45s}│")
        else:
            print(f"  │                                                       │")
            print(f"  │  Jupyter notebook? Access via one of:                 │")
            print(f"  │    <your-server-url>/proxy/{port}/viewer.html          │")
            print(f"  │    or install jupyter-server-proxy                    │")

        print(f"  │                                                       │")
        print(f"  │  Press Ctrl+C to stop                                 │")
        print(f"  └───────────────────────────────────────────────────────┘\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


def serve_background(directory: str, port: int = 8080) -> threading.Thread:
    """Start the HTTP server in a background daemon thread (non-blocking)."""
    t = threading.Thread(target=serve, args=(directory, port), daemon=True)
    t.start()
    return t


def display_in_notebook(viewer_html_path: str, width: int = 960, height: int = 640):
    """
    Display the viewer inline in a Jupyter notebook using an iframe.
    Starts a background server and points the iframe at it.
    """
    from IPython.display import HTML, display

    directory = os.path.dirname(os.path.abspath(viewer_html_path))
    port = 8080

    # Start server in background
    serve_background(directory, port)

    jupyter_base = _get_jupyter_base_url()
    if jupyter_base:
        src = f"{jupyter_base}/proxy/{port}/viewer.html"
    else:
        src = f"/proxy/{port}/viewer.html"

    iframe = f'<iframe src="{src}" width="{width}" height="{height}" style="border:1px solid #333; border-radius:8px;"></iframe>'
    display(HTML(iframe))
    print(f"  Server running on port {port}")
    print(f"  If the iframe is blank, open this URL in a new tab: {src}")


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
