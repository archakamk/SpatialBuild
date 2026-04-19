"""Extract edit commands from a word-level transcript using the Gemini API.

Drop-in replacement for the Qwen LLM step in extract_audio.py.
Reads a ``.words.json`` file and writes a ``.commands.json`` file.

Requires:
    pip install google-genai
    export GEMINI_API_KEY="..."

Usage:
    python extract_commands_gemini.py \\
        --input  outputs/session.words.json \\
        --output outputs/session.commands.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """\
You convert a spoken walkthrough of a 3D room into a list of concrete scene-edit commands.

You will be given the transcript as a JSON array of words with timestamps (seconds):
[{"word": "...", "start": <float>, "end": <float>}, ...]

Your job:
- Identify every explicit edit the speaker wants performed on the scene.
- An "edit" is an instruction to change, add, or remove something in the room.
  Examples of edits: "paint this wall red", "replace the carpet with wood",
  "add a lamp in the corner", "remove the sofa".
- Preserve the speaker's exact spatial language for targets.
  Use "back wall" not "wall", "carpet near the door" not "carpet",
  "left window" not "window". Keep the spatial specificity the speaker provides.
- IGNORE everything that is not an edit command, including:
    * descriptions or opinions  ("this is a beautiful room", "looks great",
      "otherwise pretty good")
    * filler / stall words       ("so", "um", "okay", "you know")
    * greetings, narration, commentary about the walkthrough itself
- Deduplicate: if the same edit is said twice (e.g. the speaker repeats
  "paint this wall red"), emit it only once.

Output format (strict):
Return ONLY a JSON array. No prose, no markdown, no code fences.
Each element MUST be an object with exactly these keys:
  - "timestamp": number  -> the `end` time (seconds) of the LAST word of the
                            command, so we have the full intent.
  - "action":    string  -> one of: "recolor", "replace", "add", "remove".
  - "target":    string  -> the object being edited, as spoken (e.g. "back wall",
                            "carpet near the door", "sofa in the corner").
  - "params":    object  -> action-specific details. Examples:
        recolor -> {"color": "red"}
        replace -> {"with": "wood"} or {"with": "wood floor", "material": "wood"}
        add     -> {"item": "lamp", "location": "corner"}
        remove  -> {}

Example:
Input words (abbreviated):
[{"word":"paint","start":14.1,"end":14.3},
 {"word":"this","start":14.3,"end":14.5},
 {"word":"wall","start":14.5,"end":14.8},
 {"word":"red","start":14.9,"end":15.2}]
Expected output:
[{"timestamp": 15.2, "action": "recolor", "target": "wall", "params": {"color": "red"}}]

If there are no edit commands in the transcript, return [].
"""


def extract_commands_gemini(
    words: list[dict],
    commands_path: Path | str,
    raw_path: Path | str | None = None,
) -> list[dict]:
    """Send word-level transcript to Gemini and write extracted commands.

    Parameters
    ----------
    words : list[dict]
        Word-level transcript: ``[{"word": str, "start": float, "end": float}, ...]``
    commands_path : Path | str
        Where to write the ``.commands.json`` output.
    raw_path : Path | str | None
        Optional path to save Gemini's raw reply for debugging.

    Returns
    -------
    list[dict]
        The extracted command list.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.stderr.write(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Export it with:  export GEMINI_API_KEY='...'\n"
        )
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    user_prompt = (
        "Here is the word-level transcript with timestamps (seconds):\n"
        f"{json.dumps(words, ensure_ascii=False)}\n\n"
        "Extract the edit commands."
    )

    print(f"       Sending {len(words)} words to {GEMINI_MODEL}...")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    raw_text = response.text or "[]"

    if raw_path is not None:
        Path(raw_path).write_text(raw_text + "\n", encoding="utf-8")

    commands = json.loads(raw_text)
    if not isinstance(commands, list):
        commands = []

    commands_path = Path(commands_path)
    commands_path.write_text(
        json.dumps(commands, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"       Extracted {len(commands)} commands -> {commands_path.name}")
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract edit commands from a .words.json transcript using Gemini"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the .words.json file",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output .commands.json path (default: replace .words.json suffix)",
    )
    parser.add_argument(
        "--raw", default=None,
        help="Optional path to save Gemini's raw reply",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Input file not found: {input_path}\n")
        return 1

    words = json.loads(input_path.read_text(encoding="utf-8"))

    if args.output:
        output_path = Path(args.output)
    else:
        name = input_path.name.replace(".words.json", ".commands.json")
        output_path = input_path.parent / name

    raw_path = Path(args.raw) if args.raw else None

    commands = extract_commands_gemini(words, output_path, raw_path)

    print(f"\nDone. {len(commands)} commands written to {output_path}")
    for cmd in commands:
        print(
            f"  t={cmd.get('timestamp', 0):>6.1f}  "
            f"action={cmd.get('action', '?'):<10s}  "
            f"target={cmd.get('target', '?'):<20s}  "
            f"params={cmd.get('params', {})}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
