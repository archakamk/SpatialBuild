"""Edit router — reads commands.json and dispatches to the correct editor.

Delegates each command to:
  • surface_edit.recolor      for action == "recolor"
  • surface_edit.retexture    for action == "retexture"
  • object_edit.remove_object for action == "remove"

Public API
----------
load_commands(path)
    Load and validate the commands JSON file.

    Parameters
    ----------
    path : str | Path
        Path to commands.json.

    Returns
    -------
    list[dict]
        Validated command dicts, each with keys:
        t, frame_idx, action, target, params.

route(command, image, mask)
    Apply a single edit command to an image using the appropriate editor.

    Parameters
    ----------
    command : dict
        A single command dict from load_commands().
    image : np.ndarray
        BGR image (H, W, 3).
    mask : np.ndarray
        Binary mask (H, W), uint8 0/255.

    Returns
    -------
    np.ndarray
        Edited BGR image (H, W, 3).
"""

from __future__ import annotations

import numpy as np


def load_commands(path) -> list[dict]:
    """Load and validate commands from a JSON file."""
    raise NotImplementedError


def route(command: dict, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dispatch *command* to the correct editing function."""
    raise NotImplementedError
