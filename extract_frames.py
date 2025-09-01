#!/usr/bin/env python3
"""
extract_frames.py
-----------------
Download a YouTube video and save frames to an output folder.

Dependencies:
  - yt-dlp (downloads YouTube videos)
  - opencv-python (reads video & writes images)
  - ffmpeg (must be installed on your system for yt-dlp to merge streams)

Usage examples:
  python extract_frames.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o frames --every 0.5
  python extract_frames.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o frames --fps 2
  python extract_frames.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o frames --start 5 --end 20 --every 1
  python extract_frames.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o frames --every 1 --resize 1280x720 --image-format png

Notes:
  - --every and --fps are mutually exclusive; if both omitted, defaults to --every 1.0 (one frame per second).
  - Filenames include both a sequential index and a timestamp (seconds) for easy reference.
  - Respect the content's terms of service and copyright.
"""
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Third-party
import cv2  # type: ignore
import yt_dlp  # type: ignore


def sanitize_filename(name: str) -> str:
    """Sanitize a string so it can be used as part of a filename."""
    name = name.strip()
    # Replace invalid filename characters with underscores
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    # Collapse whitespace
    name = re.sub(r"\s+", "_", name)
    return name or "video"


def download_video(url: str, tmpdir: Path) -> Tuple[Path, str]:
    """
    Download the YouTube video with yt-dlp into tmpdir and return (video_path, title).
    Forces MP4 merge when possible (requires ffmpeg in PATH).
    """
    ydl_opts = {
        "format": "bv*+ba/b",         # best video + best audio, fallback to best
        "merge_output_format": "mp4", # try to produce mp4
        "outtmpl": str(tmpdir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")
        # Determine final path (post-processed ext might differ)
        base = Path(ydl.prepare_filename(info)).with_suffix("")  # drop ext
    # Find the actual downloaded file with common video extensions
    candidates = list(tmpdir.glob(base.name + ".*"))
    # Prefer mp4, then mkv, webm, anything else
    preferred_exts = ["mp4", "mkv", "webm", "mov"]
    video_path = None
    for ext in preferred_exts:
        try_path = tmpdir / f"{base.name}.{ext}"
        if try_path.exists():
            video_path = try_path
            break
    if video_path is None:
        # fallback: first candidate
        if not candidates:
            raise FileNotFoundError("Could not locate the downloaded video file.")
        video_path = candidates[0]
    return video_path, title


def parse_resize(resize: Optional[str]) -> Optional[Tuple[int, int]]:
    if not resize:
        return None
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", resize, flags=re.I)
    if not m:
        raise ValueError("Invalid --resize format. Use WIDTHxHEIGHT, e.g., 1280x720")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError("Resize dimensions must be positive integers.")
    return (w, h)


def save_frame(
    frame,
    dest_dir: Path,
    prefix: str,
    index: int,
    t_seconds: float,
    image_format: str,
    jpeg_quality: int,
) -> Path:
    """Save a single frame to disk and return the written path."""
    timestamp = f"{t_seconds:010.3f}s"
    ext = image_format.lower()
    filename = f"{prefix}_{index:06d}_{timestamp}.{ext}"
    out_path = dest_dir / filename

    params = []
    if ext in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif ext == "png":
        # 0-9 where 9 is maximum compression
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    ok = cv2.imwrite(str(out_path), frame, params)
    if not ok:
        raise IOError(f"Failed to write image: {out_path}")
    return out_path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_seconds: Optional[float],
    per_second: Optional[float],
    start: float,
    end: Optional[float],
    max_frames: Optional[int],
    image_format: str,
    jpeg_quality: int,
    resize_xy: Optional[Tuple[int, int]],
    prefix: str,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        # fallback if fps is not reported
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Determine sampling interval
    if per_second is not None and per_second > 0:
        sample_every_sec = 1.0 / per_second
    else:
        sample_every_sec = max(every_seconds or 1.0, 1e-9)

    # frame interval is best-effort for constant-fps streams
    frame_interval = max(1, int(round(fps * sample_every_sec)))
    start_frame = max(0, int(round(start * fps)))
    end_frame = None if end is None else max(start_frame, int(round(end * fps)))

    # Fast seek to start_frame if we know where to go
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # current position
    next_to_save = frame_idx  # first candidate frame index

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps if fps > 0 else 0.0

        # stop if we've reached the end range
        if end_frame is not None and frame_idx > end_frame:
            break

        if frame_idx >= next_to_save:
            # Resize if requested
            if resize_xy:
                frame = cv2.resize(frame, resize_xy, interpolation=cv2.INTER_AREA)
            save_frame(
                frame=frame,
                dest_dir=output_dir,
                prefix=prefix,
                index=saved,
                t_seconds=t,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            saved += 1
            next_to_save += frame_interval
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video and extract frames to an output folder."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o",
        "--output",
        default="output_frames",
        help="Output directory (default: output_frames)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--every",
        type=float,
        default=1.0,
        help="Save a frame every N seconds (default: 1.0)",
    )
    group.add_argument(
        "--fps",
        type=float,
        help="Save N frames per second (mutually exclusive with --every)",
    )

    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in seconds (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (default: until video ends)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to save",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png", "jpeg"],
        default="jpg",
        help="Image format for frames (default: jpg)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize to WIDTHxHEIGHT, e.g. 1280x720 (optional)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for saved frames (default: based on video title)",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep the downloaded video file (saved next to output folder)",
    )

    args = parser.parse_args()

    try:
        resize_xy = parse_resize(args.resize)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        print("[info] Downloading video... (this may take a moment)")
        try:
            video_path, title = download_video(args.url, tmpdir)
        except Exception as e:
            print(f"[error] Failed to download video: {e}", file=sys.stderr)
            return 1

        prefix = sanitize_filename(args.prefix if args.prefix else title)
        print(f"[info] Video title: {title}")
        print(f"[info] Local file: {video_path}")
        print(f"[info] Saving frames to: {output_dir}")

        try:
            saved = extract_frames(
                video_path=video_path,
                output_dir=output_dir,
                every_seconds=args.every if args.fps is None else None,
                per_second=args.fps,
                start=max(0.0, float(args.start or 0.0)),
                end=(None if args.end is None else max(0.0, float(args.end))),
                max_frames=args.max_frames,
                image_format=args.image_format.lower(),
                jpeg_quality=max(1, min(100, int(args.jpeg_quality))),
                resize_xy=resize_xy,
                prefix=prefix,
            )
        except Exception as e:
            print(f"[error] Frame extraction failed: {e}", file=sys.stderr)
            return 1

        print(f"[done] Saved {saved} frames.")

        if args.keep_video:
            # Move the downloaded video next to the output_dir for user reference
            final_video_path = output_dir.parent / f"{prefix}{video_path.suffix}"
            try:
                video_path.replace(final_video_path)
                print(f"[info] Kept video at: {final_video_path}")
            except Exception:
                # If replace fails across devices, fallback to copy
                import shutil
                shutil.copy2(video_path, final_video_path)
                print(f"[info] Kept video at: {final_video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
