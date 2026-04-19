"""ElevenLabs voice confirmations for completed edit commands.

Generates short spoken confirmations after each edit and a final summary.

Requires:
    pip install elevenlabs
    export ELEVENLABS_API_KEY="..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config

CONFIRMATIONS_DIR = config.DATA_DIR / "output" / "confirmations"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George — calm male voice
MODEL_ID = "eleven_v3"
OUTPUT_FORMAT = "mp3_44100_128"


def _build_confirmation_text(command: dict) -> str:
    """Turn a command dict into a natural spoken confirmation."""
    action = command.get("action", "")
    target = command.get("target", "the surface")
    params = command.get("params", {})

    if action == "recolor":
        color = params.get("color", "a new color")
        if isinstance(color, list):
            color = "the new color"
        return f"Done. I've painted the {target} {color}."

    if action == "retexture":
        material = (
            params.get("texture_file", "")
            .replace(".jpg", "")
            .replace(".png", "")
            .replace("_", " ")
        ) or params.get("with", "a new material")
        return f"Done. I've replaced the {target} with {material}."

    if action == "remove":
        return f"Done. I've removed the {target} from the scene."

    if action == "add":
        item = params.get("item", "an item")
        location = params.get("location", "the room")
        return f"Done. I've added a {item} to {location}."

    return f"Done. I've applied the {action} edit to the {target}."


def _build_summary_text(commands: list[dict]) -> str:
    """Build a single spoken summary of all completed edits."""
    if not commands:
        return "No edits were applied."

    parts: list[str] = []
    for cmd in commands:
        action = cmd.get("action", "")
        target = cmd.get("target", "the surface")
        params = cmd.get("params", {})

        if action == "recolor":
            color = params.get("color", "a new color")
            if isinstance(color, list):
                color = "the new color"
            parts.append(f"painted the {target} {color}")
        elif action == "retexture":
            material = (
                params.get("texture_file", "")
                .replace(".jpg", "")
                .replace(".png", "")
                .replace("_", " ")
            ) or params.get("with", "a new material")
            parts.append(f"replaced the {target} with {material}")
        elif action == "remove":
            parts.append(f"removed the {target}")
        elif action == "add":
            item = params.get("item", "an item")
            parts.append(f"added a {item}")
        else:
            parts.append(f"edited the {target}")

    count = len(parts)
    if count == 1:
        edits_str = parts[0]
    elif count == 2:
        edits_str = f"{parts[0]} and {parts[1]}"
    else:
        edits_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    noun = "edit" if count == 1 else "edits"
    return (
        f"I've completed {count} {noun}: {edits_str}. "
        "Your remodeled space is ready to explore."
    )


class VoiceFeedback:
    """Generate spoken confirmations via ElevenLabs TTS."""

    def __init__(self, api_key: str | None = None):
        from elevenlabs.client import ElevenLabs

        key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            sys.stderr.write(
                "ELEVENLABS_API_KEY environment variable is not set.\n"
                "Export it with:  export ELEVENLABS_API_KEY='...'\n"
            )
            sys.exit(1)

        self.client = ElevenLabs(api_key=key)
        self.voice_id = VOICE_ID
        self.model_id = MODEL_ID

    def _synthesize(self, text: str, out_path: Path) -> str:
        """Call ElevenLabs TTS and write the result to *out_path*."""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        audio_iter = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format=OUTPUT_FORMAT,
        )

        with open(out_path, "wb") as f:
            for chunk in audio_iter:
                if chunk:
                    f.write(chunk)

        return str(out_path)

    def generate_confirmation(self, command: dict, idx: int = 0) -> str:
        """Generate a voice confirmation for a single command.

        Parameters
        ----------
        command : dict
            A command dict from commands.json.
        idx : int
            Command index (used for the output filename).

        Returns
        -------
        str
            Path to the saved .mp3 file.
        """
        text = _build_confirmation_text(command)
        out_path = CONFIRMATIONS_DIR / f"command_{idx}.mp3"
        print(f"  [voice] \"{text}\"")
        return self._synthesize(text, out_path)

    def generate_summary(self, commands: list[dict]) -> str:
        """Generate a single summary message for all commands.

        Returns
        -------
        str
            Path to the saved summary .mp3 file.
        """
        text = _build_summary_text(commands)
        out_path = CONFIRMATIONS_DIR / "summary.mp3"
        print(f"  [voice] \"{text}\"")
        return self._synthesize(text, out_path)


if __name__ == "__main__":
    test_command = {
        "t": 2.5,
        "frame_idx": 75,
        "action": "recolor",
        "target": "wall",
        "params": {"color": "steel blue"},
    }

    message = _build_confirmation_text(test_command)
    print(f"Message: {message}")

    vf = VoiceFeedback()
    out = vf._synthesize(message, CONFIRMATIONS_DIR / "test.mp3")
    print(f"Saved to: {out}")
