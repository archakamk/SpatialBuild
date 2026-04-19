"""Generate a structured documentation report of what the pipeline did.

Uses the Gemini API to produce both a JSON report and a Markdown document
suitable for display on the frontend.

Requires:
    pip install google-genai
    export GEMINI_API_KEY="..."

CLI:
    python -m src.generate_report
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import config

GEMINI_MODEL = "gemini-2.5-flash"
REPORT_JSON_PATH = config.DATA_DIR / "output" / "report.json"
REPORT_MD_PATH = config.DATA_DIR / "output" / "report.md"

REPORT_PROMPT = """\
You are documenting an AI-powered spatial editing pipeline. Given the \
following edit commands and pipeline statistics, generate a structured \
project report.

For each command, write:
- A human-readable description of what was done
- The technical steps involved (which models were used and how)
- A brief note on the result quality

Also write:
- An executive summary (2-3 sentences describing the overall project)
- A technical overview paragraph explaining the pipeline architecture
- A list of technologies used with one-line descriptions of each

Return JSON with this structure:
{
    "executive_summary": "string",
    "technical_overview": "string",
    "edits": [
        {
            "command": "string (what the user said)",
            "description": "string (what the pipeline did)",
            "technical_steps": ["step1", "step2", ...],
            "timestamp": float,
            "frame_idx": int,
            "action": "string"
        }
    ],
    "technologies": [
        {"name": "string", "description": "string", "role": "string"}
    ],
    "stats": {
        "total_frames": int,
        "frames_edited": int,
        "processing_time": "string (human readable)",
        "edits_applied": int
    }
}

Here are the edit commands that were executed:
{commands_json}

Here are the pipeline statistics:
{stats_json}
"""


class PipelineReporter:
    """Generate documentation reports via the Gemini API."""

    def __init__(self, api_key: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.stderr.write(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Export it with:  export GEMINI_API_KEY='...'\n"
            )
            sys.exit(1)

        self.client = genai.Client(api_key=key)

    def generate_report(
        self,
        commands: list[dict],
        pipeline_stats: dict,
    ) -> dict:
        """Send commands and stats to Gemini, return a structured report dict.

        The raw JSON is also saved to ``data/output/report.json``.
        """
        from google.genai import types

        prompt = REPORT_PROMPT.format(
            commands_json=json.dumps(commands, indent=2),
            stats_json=json.dumps(pipeline_stats, indent=2),
        )

        print("  [report] Generating structured report via Gemini …")

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        raw_text = response.text or "{}"
        report = json.loads(raw_text)

        REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  [report] Saved JSON → {REPORT_JSON_PATH}")

        return report

    def generate_markdown_report(
        self,
        commands: list[dict],
        pipeline_stats: dict,
    ) -> str:
        """Generate the JSON report then format it as Markdown.

        The Markdown is saved to ``data/output/report.md``.
        """
        report = self.generate_report(commands, pipeline_stats)
        md = _format_markdown(report)

        REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_MD_PATH.write_text(md, encoding="utf-8")
        print(f"  [report] Saved Markdown → {REPORT_MD_PATH}")

        return md


def _format_markdown(report: dict) -> str:
    """Convert the structured report dict into a clean Markdown document."""
    lines: list[str] = []

    lines.append("# SpatialBuild Pipeline Report\n")

    if summary := report.get("executive_summary"):
        lines.append("## Executive Summary\n")
        lines.append(f"{summary}\n")

    if overview := report.get("technical_overview"):
        lines.append("## Technical Overview\n")
        lines.append(f"{overview}\n")

    if edits := report.get("edits"):
        lines.append("## Edits Applied\n")
        for i, edit in enumerate(edits, 1):
            action = edit.get("action", "")
            desc = edit.get("description", "")
            cmd_text = edit.get("command", "")
            ts = edit.get("timestamp", 0)
            fidx = edit.get("frame_idx", 0)

            lines.append(f"### {i}. {action.title()} — {cmd_text}\n")
            lines.append(f"- **Timestamp:** {ts:.1f}s (frame {fidx})")
            lines.append(f"- **Description:** {desc}")

            if steps := edit.get("technical_steps"):
                lines.append("- **Technical steps:**")
                for step in steps:
                    lines.append(f"  1. {step}")

            lines.append("")

    if techs := report.get("technologies"):
        lines.append("## Technologies Used\n")
        lines.append("| Technology | Role | Description |")
        lines.append("|---|---|---|")
        for t in techs:
            lines.append(
                f"| {t.get('name', '')} "
                f"| {t.get('role', '')} "
                f"| {t.get('description', '')} |"
            )
        lines.append("")

    if stats := report.get("stats"):
        lines.append("## Pipeline Statistics\n")
        lines.append(f"- **Total frames:** {stats.get('total_frames', '—')}")
        lines.append(f"- **Frames edited:** {stats.get('frames_edited', '—')}")
        lines.append(f"- **Processing time:** {stats.get('processing_time', '—')}")
        lines.append(f"- **Edits applied:** {stats.get('edits_applied', '—')}")
        lines.append("")

    lines.append("---\n*Generated by SpatialBuild Pipeline Reporter*\n")

    return "\n".join(lines)


if __name__ == "__main__":
    example_commands = [
        {
            "t": 2.5,
            "frame_idx": 75,
            "action": "recolor",
            "target": "wall",
            "params": {"color": [70, 130, 180]},
        },
        {
            "t": 5.0,
            "frame_idx": 150,
            "action": "retexture",
            "target": "floor",
            "params": {"texture_file": "hardwood.jpg"},
        },
        {
            "t": 8.0,
            "frame_idx": 240,
            "action": "remove",
            "target": "bed",
            "params": {},
        },
    ]

    example_stats = {
        "total_frames": 625,
        "frames_edited": 624,
        "processing_time_seconds": 132.6,
        "models_used": ["Grounding DINO", "SAM 2.1", "LaMa", "ElevenLabs"],
        "input_video": "test_video.mp4",
        "output_video": "output.mp4",
    }

    reporter = PipelineReporter()
    md = reporter.generate_markdown_report(example_commands, example_stats)
    print("\n" + md)
