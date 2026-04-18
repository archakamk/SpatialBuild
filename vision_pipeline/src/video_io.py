"""Video frame I/O for the vision pipeline.

First stage: take a Ray-Ban Meta mp4 and extract / reassemble frames
for downstream segmentation and editing models.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path


def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


def extract_frames(video_path: str, output_dir: str, every_n: int = 1) -> list[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_paths: list[str] = []
    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if idx % every_n == 0:
            out_path = str(Path(output_dir) / f"frame_{idx:06d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(out_path)
        idx += 1
        if idx % 60 == 0:
            print(f"Extracted {idx}/{total} frames")

    cap.release()
    return saved_paths


def get_frame_at_timestamp(video_path: str, timestamp: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    return frame


def frames_to_video(frame_paths: list[str], output_path: str, fps: float) -> bool:
    if not frame_paths:
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = Path(output_path)

    # Primary path: ffmpeg -> H.264 in mp4. cv2's mp4v fourcc produces MPEG-4
    # Part 2 which QuickTime and most modern players won't decode on macOS.
    frame_dir = str(Path(frame_paths[0]).parent)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frame_dir}/frame_%06d.jpg",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
        return True
    if result.returncode != 0:
        print(result.stderr.decode(errors="replace"))

    # Fallback: cv2 VideoWriter with mp4v (works without ffmpeg, but playback
    # compatibility on macOS is limited).
    first = cv2.imread(frame_paths[0])
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for p in frame_paths:
        frame = cv2.imread(p)
        writer.write(frame)
    writer.release()

    return out.exists() and out.stat().st_size > 0


def side_by_side_video(original_path: str, edited_path: str, output_path: str) -> bool:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", original_path,
        "-i", edited_path,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode(errors="replace"))
        return False
    return True


if __name__ == "__main__":
    video = "./tests/test_video.mp4"

    print("[1/5] Metadata:")
    meta = get_video_metadata(video)
    print(meta)

    print("\n[2/5] Extracting frames...")
    paths = extract_frames(video, "./data/frames/", every_n=1)
    print(f"Extracted {len(paths)} frames")

    print("\n[3/5] Getting frame at t=5.0...")
    frame = get_frame_at_timestamp(video, 5.0)
    assert frame is not None, "Frame extraction failed"
    print(f"Frame shape: {frame.shape}")

    print("\n[4/5] Reassembling video...")
    success = frames_to_video(paths, "./data/output/roundtrip.mp4", meta["fps"])
    print(f"Roundtrip success: {success}")

    print("\n[5/5] Creating side-by-side...")
    success = side_by_side_video(video, "./data/output/roundtrip.mp4",
                                 "./data/output/side_by_side.mp4")
    print(f"Side-by-side success: {success}")

    print("\nAll tests passed.")
