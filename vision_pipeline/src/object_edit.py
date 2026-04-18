"""Object removal via LaMa inpainting.

Uses the simple-lama-inpainting package (pip install simple-lama-inpainting).

Public API
----------
remove_object(image, mask)
    Inpaint the masked region to remove an object from the scene.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3).
    mask : np.ndarray
        Binary mask (H, W), uint8 0/255.  White (255) = region to remove.

    Returns
    -------
    np.ndarray
        Inpainted BGR image (H, W, 3) with the object removed.
"""

from __future__ import annotations

import numpy as np


def remove_object(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint the masked region of *image* using LaMa."""
    raise NotImplementedError
