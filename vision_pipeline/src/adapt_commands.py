"""Translate audio-pipeline commands into the vision-pipeline format.

The audio pipeline emits::

    {"timestamp": 15.2, "action": "recolor", "target": "wall",
     "params": {"color": "red"}, "raw_utterance": "…"}

The vision pipeline (edit_router) expects::

    {"t": 15.2, "frame_idx": 456, "action": "recolor", "target": "wall",
     "params": {"color": [255, 0, 0]}}

This module bridges the two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config

# ── Colour name → RGB ───────────────────────────────────────────────────────
COLOR_TABLE: dict[str, list[int]] = {
    "red":        [255, 0, 0],
    "green":      [0, 128, 0],
    "blue":       [0, 0, 255],
    "white":      [255, 255, 255],
    "black":      [0, 0, 0],
    "gray":       [128, 128, 128],
    "grey":       [128, 128, 128],
    "brown":      [139, 69, 19],
    "beige":      [245, 245, 220],
    "yellow":     [255, 255, 0],
    "orange":     [255, 165, 0],
    "purple":     [128, 0, 128],
    "pink":       [255, 192, 203],
    "navy":       [0, 0, 128],
    "teal":       [0, 128, 128],
    "maroon":     [128, 0, 0],
    "gold":       [255, 215, 0],
    "cream":      [255, 253, 208],
    "ivory":      [255, 255, 240],
    "steel blue": [70, 130, 180],
    "coral":      [255, 127, 80],
    "salmon":     [250, 128, 114],
    "olive":      [128, 128, 0],
    "cyan":       [0, 255, 255],
    "magenta":    [255, 0, 255],
    "lavender":   [230, 230, 250],
    "tan":        [210, 180, 140],
    "charcoal":   [54, 69, 79],
    "slate":      [112, 128, 144],
    "sage":       [188, 184, 138],
    "terracotta": [204, 78, 92],
    "sand":       [194, 178, 128],
    "mint":       [189, 252, 201],
    "peach":      [255, 218, 185],
    "burgundy":   [128, 0, 32],
    "sky blue":   [135, 206, 235],
    "forest green": [34, 139, 34],
    "dark gray":  [64, 64, 64],
    "light gray": [192, 192, 192],
}


def _resolve_color(name: str) -> list[int]:
    """Look up a colour name, case-insensitive.  Falls back to middle gray."""
    key = name.strip().lower()
    if key in COLOR_TABLE:
        return COLOR_TABLE[key]
    # Try without trailing/leading whitespace variants
    for k, v in COLOR_TABLE.items():
        if k.replace(" ", "") == key.replace(" ", ""):
            return v
    print(f"  [warn] Unknown colour '{name}', defaulting to gray")
    return [128, 128, 128]


def _convert_one(cmd: dict, fps: float) -> dict:
    """Convert a single audio-pipeline command dict to vision-pipeline format."""
    timestamp = cmd.get("timestamp", 0.0)
    action = cmd.get("action", "")
    target = cmd.get("target", "")
    params = dict(cmd.get("params", {}))

    out: dict = {
        "t": timestamp,
        "frame_idx": round(timestamp * fps),
        "action": action,
        "target": target,
        "params": {},
    }

    if action == "recolor":
        color_str = params.get("color", "gray")
        out["params"]["color"] = _resolve_color(color_str)

    elif action in ("replace", "retexture"):
        out["action"] = "retexture"
        material = (
            params.get("with")
            or params.get("replacement")
            or params.get("texture_file", "")
        )
        # Strip trailing .jpg/.png if the audio pipeline already included one
        material = material.strip()
        if material.endswith((".jpg", ".png")):
            out["params"]["texture_file"] = material
        else:
            # Normalise to a filename: "hardwood floor" → "hardwood_floor.jpg"
            out["params"]["texture_file"] = material.replace(" ", "_") + ".jpg"

    elif action == "remove":
        out["params"] = {}

    elif action == "add":
        item = (params.get("item") or "").lower()
        if any(kw in item for kw in ("painting", "picture", "image", "photo", "poster")):
            out["action"] = "place_image"
            out["params"]["image"] = params.get("image", "painting.jpg")
        else:
            out["params"] = params

    else:
        out["params"] = params

    return out


def adapt_commands(
    input_path: str | Path,
    output_path: str | Path,
    fps: float = 30.0,
) -> list[dict]:
    """Read audio-pipeline commands, convert, and write vision-pipeline format.

    Parameters
    ----------
    input_path : str | Path
        Path to the audio pipeline's ``.commands.json``.
    output_path : str | Path
        Where to write the converted commands JSON.
    fps : float
        Frame rate used to compute ``frame_idx`` from ``timestamp``.

    Returns
    -------
    list[dict]
        The converted command list (also written to *output_path*).
    """
    with open(input_path) as f:
        raw = json.load(f)

    # Handle both bare list and PipelineInput-style {"commands": [...]}
    if isinstance(raw, dict) and "commands" in raw:
        source = raw["commands"]
        fps = raw.get("video_fps", fps)
    elif isinstance(raw, list):
        source = raw
    else:
        raise ValueError(f"Unexpected JSON structure in {input_path}")

    converted = [_convert_one(cmd, fps) for cmd in source]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} commands  ({input_path} → {output_path})")
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio-pipeline commands to vision-pipeline format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to audio pipeline .commands.json",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(config.PROJECT_ROOT / "commands.json"),
        help="Output path (default: commands.json at project root)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=config.FRAME_RATE,
        help=f"Frame rate for timestamp → frame_idx (default: {config.FRAME_RATE})",
    )
    args = parser.parse_args()

    result = adapt_commands(args.input, args.output, args.fps)
    for cmd in result:
        print(
            f"  t={cmd['t']:>6.1f}  frame={cmd['frame_idx']:>5d}  "
            f"action={cmd['action']:<10s}  target={cmd['target']:<15s}  "
            f"params={cmd['params']}"
        )
