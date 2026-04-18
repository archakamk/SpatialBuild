"""Data contracts for the vision pipeline.

Shared interface between the voice pipeline (producer of edit commands)
and the vision pipeline (consumer that renders edited video).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EditCommand(BaseModel):
    """A single edit instruction emitted by the voice pipeline."""

    timestamp: float = Field(..., description="Seconds into the video when the command was issued.")
    action: Literal["recolor", "replace", "remove"]
    target: str = Field(..., description="What to edit, e.g. 'wall', 'floor', 'couch'.")
    params: dict = Field(
        default_factory=dict,
        description="Action-specific parameters, e.g. {'color': 'navy'} or {'replacement': 'leather couch'}.",
    )
    raw_utterance: str = Field(..., description="Original spoken text, kept for debugging.")


class PipelineInput(BaseModel):
    """Full input payload to the vision pipeline."""

    video_path: str = Field(..., description="Path to a normalized 720p mp4.")
    commands: list[EditCommand]
    video_fps: float
    video_duration: float


class PipelineOutput(BaseModel):
    """Artifacts produced by the vision pipeline."""

    output_video_path: str
    before_after_path: str
    masks_debug_dir: str


def load_input(path: str | Path) -> PipelineInput:
    """Load a PipelineInput from a JSON file on disk."""
    with open(path, "r") as f:
        data = json.load(f)
    return PipelineInput.model_validate(data)


def save_output(output: PipelineOutput, path: str | Path) -> None:
    """Serialize a PipelineOutput to a JSON file on disk."""
    with open(path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)


# Example input JSON (what load_input() expects to read):
#
# {
#   "video_path": "data/normalized/living_room_720p.mp4",
#   "video_fps": 30.0,
#   "video_duration": 12.5,
#   "commands": [
#     {
#       "timestamp": 1.2,
#       "action": "recolor",
#       "target": "wall",
#       "params": {"color": "navy"},
#       "raw_utterance": "make the wall navy blue"
#     },
#     {
#       "timestamp": 4.8,
#       "action": "replace",
#       "target": "couch",
#       "params": {"replacement": "leather couch"},
#       "raw_utterance": "replace the couch with a leather couch"
#     },
#     {
#       "timestamp": 7.3,
#       "action": "remove",
#       "target": "coffee table",
#       "params": {},
#       "raw_utterance": "remove the coffee table"
#     }
#   ]
# }
