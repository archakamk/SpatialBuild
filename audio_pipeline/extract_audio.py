"""MOV -> MP4 -> MP3 -> transcript (.txt + word-level .words.json) -> LLM edit commands (.commands.json).

Pipeline stages:
  1. Find the first .MOV in the script folder or its parent.
  2. Convert MOV -> MP4 (H.264 + AAC) with ffmpeg.
  3. Extract MP4 -> MP3 with ffmpeg.
  4. Transcribe MP3 with openai/whisper-tiny (Hugging Face):
       - <stem>.txt           : plain text transcript
       - <stem>.words.json    : word-level timestamps, e.g.
         [{"word": "paint", "start": 14.1, "end": 14.3}, ...]
  5. Use Qwen/Qwen2.5-3B-Instruct to convert the word-level transcript
     into edit commands:
       - <stem>.commands.json : list of {timestamp, action, target, params}
       - <stem>.llm_raw.txt   : raw LLM reply (for debugging)

All outputs are written to ./outputs/ next to this script.

Dependencies:
    pip install imageio-ffmpeg "transformers>=4.42" accelerate torch librosa soundfile

First run downloads:
    openai/whisper-tiny         (~150 MB)
    Qwen/Qwen2.5-3B-Instruct    (~6 GB)
Models are cached under ~/.cache/huggingface/hub.
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:
    sys.stderr.write(
        "imageio-ffmpeg is not installed. Install it with:\n"
        "    pip install imageio-ffmpeg\n"
    )
    sys.exit(1)


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _expose_ffmpeg_as_standard_name() -> None:
    """Ensure a binary named literally ``ffmpeg`` is on PATH.

    Hugging Face's ASR pipeline spawns ``subprocess.Popen(["ffmpeg", ...])``,
    but the imageio-ffmpeg binary has a versioned name. Symlink it under a
    per-user shim directory and prepend that directory to PATH.
    """
    from shutil import which

    if which("ffmpeg"):
        return

    shim_dir = Path.home() / ".cache" / "spatialbuild" / "ffmpeg-shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "ffmpeg"
    try:
        if shim.is_symlink() or shim.exists():
            shim.unlink()
        shim.symlink_to(FFMPEG)
    except OSError:
        pass
    os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")


_expose_ffmpeg_as_standard_name()
WHISPER_MODEL_ID = "openai/whisper-tiny"
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

LLM_INSTRUCTION = """You convert a spoken walkthrough of a 3D room into a list of concrete scene-edit commands.

You will be given the transcript as a JSON array of words with timestamps (seconds):
[{"word": "...", "start": <float>, "end": <float>}, ...]

Your job:
- Identify every explicit edit the speaker wants performed on the scene.
- An "edit" is an instruction to change, add, or remove something in the room.
  Examples of edits: "paint this wall red", "replace the carpet with wood",
  "add a lamp in the corner", "remove the sofa".
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
  - "target":    string  -> the object being edited, as spoken (e.g. "wall",
                            "carpet", "sofa in the corner").
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


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------
#helper

def find_mov_file(search_dirs: list[Path]) -> Path:
    x = []
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        candidates = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".mov"
        )
        if candidates:
            return candidates[0]
    tried = ", ".join(str(p) for p in search_dirs)
    raise FileNotFoundError(f"No .MOV file found in: {tried}")


def run_ffmpeg(args: list[str]) -> None:
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-stats", *args]
    subprocess.run(cmd, check=True)


def convert_to_mp4(src: Path, dst: Path) -> None:
    print(f"[1/4] Converting {src.name} -> {dst.name}")
    run_ffmpeg([
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ])


def extract_audio(mp4_path: Path, audio_path: Path) -> None:
    print(f"[2/4] Extracting audio {mp4_path.name} -> {audio_path.name}")
    run_ffmpeg([
        "-i", str(mp4_path),
        "-vn",
        "-c:a", "libmp3lame",
        "-q:a", "2",
        str(audio_path),
    ])


# ---------------------------------------------------------------------------
# Device / torch helpers
# ---------------------------------------------------------------------------

def pick_device_dtype():
    """Return (device, dtype) suited to the host. Imports torch lazily."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # fp32 on MPS is the safest option for these models.
        return "mps", torch.float32
    return "cpu", torch.float32


def free_torch_memory() -> None:
    """Free any cached GPU/MPS memory between large model loads."""
    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Whisper step
# ---------------------------------------------------------------------------

def transcribe_with_whisper(
    audio_path: Path,
    txt_path: Path,
    words_json_path: Path,
) -> list[dict]:
    print(f"[3/4] Transcribing {audio_path.name} -> {txt_path.name} / {words_json_path.name}")
    try:
        import torch  # noqa: F401  (used via pick_device_dtype)
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
    except ImportError as e:
        sys.stderr.write(
            f"Missing dependency: {e.name}\n"
            "Install with:\n"
            '    pip install "transformers>=4.42" accelerate torch librosa soundfile\n'
        )
        sys.exit(1)

    device, torch_dtype = pick_device_dtype()
    print(f"       Loading {WHISPER_MODEL_ID} on {device} ({torch_dtype})...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(
        str(audio_path),
        return_timestamps="word",
        generate_kwargs={"task": "transcribe"},
    )

    text = (result.get("text") or "").strip()
    txt_path.write_text(text + "\n", encoding="utf-8")

    words: list[dict] = []
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp")
        if not ts:
            continue
        start, end = ts
        words.append({
            "word": (chunk.get("text") or "").strip(),
            "start": start,
            "end": end,
        })
    words_json_path.write_text(
        json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Release whisper before loading the LLM.
    del pipe, model, processor
    free_torch_memory()

    return words


# ---------------------------------------------------------------------------
# LLM command-extraction step
# ---------------------------------------------------------------------------

def _parse_json_array(text: str):
    """Best-effort JSON array extraction from free-form LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[\s*(?:\{.*?\}\s*,?\s*)+\]", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("Could not parse JSON array from model output.")


def extract_commands_with_llm(
    words: list[dict],
    commands_path: Path,
    raw_path: Path,
) -> None:
    print(f"[4/4] Extracting edit commands with {LLM_MODEL_ID} -> {commands_path.name}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.stderr.write(
            f"Missing dependency: {e.name}\n"
            "Install with:\n"
            '    pip install "transformers>=4.42" accelerate torch\n'
        )
        sys.exit(1)

    device, torch_dtype = pick_device_dtype()
    print(f"       Loading {LLM_MODEL_ID} on {device} ({torch_dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    user_content = (
        f"{LLM_INSTRUCTION}\n\n"
        "Input: word-level transcript with timestamps (seconds):\n"
        f"{json.dumps(words, ensure_ascii=False)}"
    )
    messages = [{"role": "user", "content": user_content}]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    raw_path.write_text(reply + "\n", encoding="utf-8")

    try:
        commands = _parse_json_array(reply)
    except (ValueError, json.JSONDecodeError) as e:
        sys.stderr.write(
            f"Warning: failed to parse JSON from LLM output ({e}).\n"
            f"Raw reply saved to {raw_path}.\n"
        )
        commands = []

    commands_path.write_text(
        json.dumps(commands, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    del model, tokenizer
    free_torch_memory()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    mov_path = find_mov_file([script_dir, script_dir.parent])
    stem = mov_path.stem
    mp4_path = outputs_dir / f"{stem}.mp4"
    audio_path = outputs_dir / f"{stem}.mp3"
    txt_path = outputs_dir / f"{stem}.txt"
    words_json_path = outputs_dir / f"{stem}.words.json"
    commands_path = outputs_dir / f"{stem}.commands.json"
    llm_raw_path = outputs_dir / f"{stem}.llm_raw.txt"

    if mp4_path.exists():
        print(f"[1/4] Skipping (already exists): {mp4_path.name}")
    else:
        convert_to_mp4(mov_path, mp4_path)

    if audio_path.exists():
        print(f"[2/4] Skipping (already exists): {audio_path.name}")
    else:
        extract_audio(mp4_path, audio_path)

    if words_json_path.exists() and txt_path.exists():
        print(f"[3/4] Skipping (already exists): {words_json_path.name}")
        words = json.loads(words_json_path.read_text(encoding="utf-8"))
    else:
        words = transcribe_with_whisper(audio_path, txt_path, words_json_path)

    extract_commands_with_llm(words, commands_path, llm_raw_path)

    print("\nDone.")
    print(f"  MP4:        {mp4_path}")
    print(f"  Audio:      {audio_path}")
    print(f"  Transcript: {txt_path}")
    print(f"  Words JSON: {words_json_path}")
    print(f"  Commands:   {commands_path}")
    print(f"  LLM raw:    {llm_raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
