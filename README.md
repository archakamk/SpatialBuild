# SpatialBuild

**Voice-native spatial editing for the wearable era.**

Walk through any room wearing Ray-Ban Meta smart glasses, describe the changes you want out loud, and get back an edited 3D Gaussian splat of your remodeled space. No mouse. No menus. Just your voice and your gaze.

> Built at Realities Hack 2026

---

## Demo

🎥 [Watch the 2-minute demo](https://devpost.com/software/spatialbuild)
🌐 [Live frontend](https://spatialbuild.github.io)

### What it looks like

| Voice command | Before | After |
|---|---|---|
| *"Paint this wall green"* | ![before](assets/before_green.jpg) | ![after](assets/after_green.jpg) |
| *"Put a painting on this wall"* | ![before](assets/before_mona.jpg) | ![after](assets/after_mona.jpg) |
| *"Remove this desk"* | ![before](assets/before_desk.jpg) | ![after](assets/after_desk.jpg) |

---

## How It Works

```
Ray-Ban Meta → Audio Pipeline → Vision Pipeline → 3D Reconstruction → Interactive Viewer
```

### The Pipeline

**1. Capture** — Record a walkthrough of any room using Ray-Ban Meta smart glasses. Video and audio are captured simultaneously — the camera points where you look, so gaze direction is embedded in the footage.

**2. Audio Pipeline** — Whisper transcribes the audio into word-level timestamps. The Gemini API parses natural language commands ("paint this wall red", "remove that desk") into structured JSON with action, target, timestamp, and parameters.

**3. Vision Pipeline** — For each command, the system:
- Uses **Grounding DINO** to detect the target object ("wall", "desk") in the frame at the command's timestamp
- Applies **center-bias scoring** — since the glasses point where you look, the intended object is near frame center
- Uses **SAM 2** to segment the object and propagate the mask across all video frames
- Applies the edit: **recolor** (HSV-preserving color swap), **remove** (LaMa inpainting), **retexture** (tiled texture with luminance matching), or **place_image** (composite an image onto a surface)

**4. 3D Reconstruction** — COLMAP recovers camera poses via structure-from-motion. OpenSplat trains a Gaussian splat from the edited frames, producing a navigable 3D scene with all edits baked in.

**5. Voice Feedback** — ElevenLabs generates natural voice confirmations after each edit ("Done. I've painted the wall green."), closing the feedback loop through the glasses' speakers.

**6. Interactive Viewer** — A WebGL frontend lets you orbit the remodeled space, view the edit timeline with before/after comparisons, and read auto-generated documentation.

### The Key Insight: Gaze as Intent

When a user says *"this wall"* while wearing smart glasses, we know **which wall** they mean — because the camera points where they're looking. We exploit this by running object detection at the command's timestamp and selecting the detection closest to frame center.

This is something no phone app or desktop tool can do. It makes the interaction completely hands-free and spatial.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CAPTURE LAYER                           │
│  Ray-Ban Meta ──Bluetooth──▶ Phone ──USB/AirDrop──▶ Laptop  │
│  (video + audio recorded together)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ffmpeg split (720p + 16kHz wav)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│  AUDIO PIPELINE  │     │  VISION PIPELINE │
│                  │     │                  │
│  Whisper (STT)   │     │  Grounding DINO  │
│       │          │     │  (text → bbox)   │
│       ▼          │     │       │          │
│  Gemini API      │     │       ▼          │
│  (intent parse)  │     │  SAM 2           │
│       │          │     │  (segment +      │
│       ▼          │     │   propagate)     │
│  commands.json   │     │       │          │
│  [{t, action,    │     │       ▼          │
│    target,       │────▶│  Edit Engine     │
│    params}]      │     │  (recolor/remove │
│                  │     │   /retexture/    │
│  ElevenLabs      │     │   place_image)   │
│  (voice confirm) │     │       │          │
└──────────────────┘     │       ▼          │
                         │  edited_frames/  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   3D RECONSTRUCTION        │
                    │                            │
                    │   COLMAP (SfM → poses)     │
                    │          │                 │
                    │          ▼                 │
                    │   OpenSplat (Gaussian      │
                    │    splatting training)      │
                    │          │                 │
                    │          ▼                 │
                    │   scene.ply + viewer       │
                    └───────────────────────────┘
```

---

## Project Structure

```
SpatialBuild/
├── audio_pipeline/
│   ├── extract_audio.py              # MOV → MP4 → MP3 → transcript → commands
│   ├── extract_commands_gemini.py     # Gemini API command extraction
│   └── outputs/                      # Transcripts, word JSON, commands JSON
│
├── vision_pipeline/
│   ├── src/
│   │   ├── config.py                 # Paths, model configs, device settings
│   │   ├── grounding.py              # Grounding DINO (HuggingFace transformers)
│   │   ├── segmentation.py           # SAM 2 image + video mask propagation
│   │   ├── surface_edit.py           # Recolor, retexture, place_image
│   │   ├── object_edit.py            # LaMa inpainting for object removal
│   │   ├── edit_router.py            # Command dispatch + frame orchestration
│   │   ├── run_pipeline.py           # Main entry point
│   │   ├── adapt_commands.py         # Audio → vision command format bridge
│   │   ├── voice_feedback.py         # ElevenLabs voice confirmations
│   │   ├── generate_report.py        # Gemini-powered documentation
│   │   ├── contracts.py              # Pydantic models
│   │   └── video_io.py               # Frame extraction and video assembly
│   ├── data/
│   │   ├── frames/                   # Extracted video frames
│   │   ├── edited_frames/            # Pipeline output
│   │   ├── masks/                    # Intermediate segmentation masks
│   │   ├── textures/                 # Texture files for retexturing
│   │   └── output/                   # Final videos, reports, voice confirmations
│   └── _sam2_repo/                   # SAM 2 (installed via pip install -e)
│
├── reconstruction_pipeline/
│   ├── step1_prepare_frames.py       # Subsample and organize frames for COLMAP
│   ├── step2_colmap.py               # COLMAP sparse reconstruction
│   ├── step3_splat.py                # OpenSplat Gaussian splatting training
│   └── step4_viewer.py               # 3D viewer setup
│
└── frontend/
    ├── SpatialBuild.html             # Main demo page
    ├── before.mp4                    # Original capture video
    └── after.mp4                     # Edited video
```

---

## Quick Start

### Prerequisites

- AMD GPU with ROCm (tested on Instinct MI300X) or NVIDIA GPU with CUDA
- Python 3.10+
- ffmpeg
- COLMAP
- API keys: [Gemini](https://aistudio.google.com/app/apikey), [ElevenLabs](https://elevenlabs.io/app/settings/api-keys)

### Installation

```bash
git clone https://github.com/your-team/SpatialBuild.git
cd SpatialBuild

# Install Python dependencies
pip install torch torchvision torchaudio  # ROCm or CUDA version
pip install transformers simple-lama-inpainting opencv-python pillow
pip install google-genai elevenlabs
pip install imageio-ffmpeg

# Install SAM 2
cd vision_pipeline/_sam2_repo
SAM2_BUILD_CUDA=0 pip install -e .
cd ../..

# Download SAM 2 checkpoints
cd vision_pipeline/_sam2_repo/checkpoints
./download_ckpts.sh
cd ../../..
```

### Set API Keys

```bash
export GEMINI_API_KEY="your-gemini-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

### Run the Full Pipeline

```bash
# 1. Place your .MOV file in the project root
cp your_video.MOV SpatialBuild/

# 2. Audio pipeline — extract voice commands
cd audio_pipeline
cp ../your_video.MOV .
python extract_audio.py
cat outputs/*.commands.json

# 3. Extract video frames
cd ../vision_pipeline
ffmpeg -i ../your_video.MOV -vf "scale=720:-2" -q:v 2 data/frames/%06d.jpg

# 4. Convert commands to vision pipeline format
python -m src.adapt_commands \
    --input ../audio_pipeline/outputs/*.commands.json \
    --output commands.json --fps 30

# 5. Run vision pipeline
python -m src.run_pipeline \
    --frames data/frames/ \
    --commands commands.json \
    --output data/edited_frames/ \
    --voice

# 6. Assemble edited video
ffmpeg -framerate 30 -i data/edited_frames/%06d.jpg \
    -c:v libx264 -pix_fmt yuv420p -y data/output/output.mp4

# 7. 3D reconstruction (run by partner)
cd ../reconstruction_pipeline
python step1_prepare_frames.py --frames-dir ../vision_pipeline/data/edited_frames/
python step2_colmap.py --workspace colmap_workspace
python step3_splat.py --colmap-dir colmap_workspace --output splat_output --iterations 30000
```

---

## Supported Edit Commands

| Command type | Example voice input | What it does |
|---|---|---|
| **Recolor** | "Paint this wall green" | HSV-preserving color change with luminance matching |
| **Remove** | "Remove that desk" | SAM 2 mask + LaMa inpainting to erase objects |
| **Retexture** | "Replace the carpet with hardwood" | Tiled texture application with shadow preservation |
| **Place image** | "Put a painting on that wall" | Composite an image onto a surface with perspective |

---

## Technical Details

### Gaze-Based Center Bias Grounding

Standard object detection returns all instances of "wall" in a frame. We score detections by proximity to frame center:

```
combined_score = detection_confidence × 0.4 + center_proximity × 0.6
```

This heavily weights the object the user is looking at, making "this wall" unambiguous.

### Luminance-Preserving Recolor

Rather than a flat color fill, we blend the target color at 50% strength while preserving 70% of the original V channel (HSV). This keeps shadow gradients, highlights, and surface texture visible through the new color.

### Temporal Mask Propagation

SAM 2 segments the target on one anchor frame, then propagates the mask bidirectionally across the entire video. One detection → consistent edits across thousands of frames at 14.6 frames/sec on MI300X.

### Object Removal Pipeline

SAM 2 mask → 15px dilation (catches shadows and edge artifacts) → LaMa inpainting. The dilation step is critical for clean removal — without it, shadow remnants and edge halos remain visible.

---

## Built With

| Technology | Role |
|---|---|
| **Gemini API** | Voice command parsing + documentation generation |
| **ElevenLabs** | Natural voice confirmations |
| **SAM 2** | Video-aware segmentation + mask propagation |
| **Grounding DINO** | Open-vocabulary text-to-bounding-box detection |
| **LaMa** | Large mask inpainting for object removal |
| **Whisper** | Speech-to-text transcription |
| **COLMAP** | Structure-from-motion camera pose estimation |
| **OpenSplat** | Gaussian splatting 3D reconstruction |
| **Ray-Ban Meta** | Wearable capture device (video + audio) |
| **AMD MI300X** | GPU compute (192GB VRAM, ROCm) |
| **Three.js** | WebGL 3D viewer |
| **PyTorch** | ML framework (ROCm backend) |

---

## Performance

| Metric | Value |
|---|---|
| Frames processed | 5,491 |
| Mask propagation speed | 14.6 fps |
| Recolor (full video) | ~2 min |
| Object removal (full video) | ~16 min |
| COLMAP reconstruction | ~7 min |
| OpenSplat training (30K iter) | ~15 min |
| GPU | AMD Instinct MI300X (192GB) |
| GPU utilization during editing | 99% |
| VRAM usage | 75% peak |

---

## Market Opportunity

- **TAM**: $146B global interior design market (2025)
- **SAM**: $72B renovation & remodeling segment (11.8% CAGR)
- **SOM**: $8B visualization & rendering services
- **Pain point**: Professional 3D renderings cost $500–$2,500 per image and take 3–7 days
- **Our solution**: Voice-guided scan → edited 3D scene in minutes for ~$50

---

## Team

Built at StarkHacks 2026.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Meta AI](https://github.com/facebookresearch/sam2) for SAM 2
- [IDEA Research](https://github.com/IDEA-Research/GroundingDINO) for Grounding DINO
- [Google](https://ai.google.dev/) for Gemini API
- [ElevenLabs](https://elevenlabs.io/) for voice synthesis
- [OpenSplat](https://github.com/pierotofy/OpenSplat) for Gaussian splatting
- [AMD](https://www.amd.com/en/products/accelerators/instinct/mi300x.html) for MI300X compute
