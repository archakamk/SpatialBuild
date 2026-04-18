"""Surface editing — recolor and retexture using masks.

Handles two actions from contracts.EditCommand:
  • recolor  — flat colour fill blended into the masked region
  • retexture — texture tiled/warped onto the masked region

Public API
----------
recolor(image, mask, color)
    Apply a solid colour to the masked region of an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3).
    mask : np.ndarray
        Binary mask (H, W), uint8 0/255.
    color : tuple[int, int, int]
        Target colour in RGB.

    Returns
    -------
    np.ndarray
        Edited BGR image (H, W, 3).

retexture(image, mask, texture_path)
    Tile / warp a texture image onto the masked region.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3).
    mask : np.ndarray
        Binary mask (H, W), uint8 0/255.
    texture_path : str | Path
        Path to the texture image file (resolved via config.TEXTURES_DIR).

    Returns
    -------
    np.ndarray
        Edited BGR image (H, W, 3).
"""

from __future__ import annotations

import numpy as np


def recolor(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Fill the masked region of *image* with a solid *color*."""
    raise NotImplementedError


def retexture(
    image: np.ndarray,
    mask: np.ndarray,
    texture_path,
) -> np.ndarray:
    """Warp a texture from *texture_path* onto the masked region."""
    raise NotImplementedError
