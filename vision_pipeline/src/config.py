"""Pipeline configuration — paths, model configs, device settings.

Directory paths are derived relative to this file so they resolve correctly
on both the local dev machine and the remote AMD GPU instance.
"""

import os
from pathlib import Path

# ── Project layout (relative to this file) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
EDITED_FRAMES_DIR = DATA_DIR / "edited_frames"
MASKS_DIR = DATA_DIR / "masks"
TEXTURES_DIR = DATA_DIR / "textures"
TEST_OUTPUTS_DIR = DATA_DIR / "test_outputs"

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda"  # AMD GPU via ROCm; PyTorch maps to "cuda"

# ── SAM 2 ───────────────────────────────────────────────────────────────────
SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/workspace/SpatialBuild/vision_pipeline/sam2/checkpoints/sam2.1_hiera_large.pt",
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ── Grounding DINO (HuggingFace transformers) ──────────────────────────────
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"

# ── Video ───────────────────────────────────────────────────────────────────
FRAME_RATE = 30
