"""
sliding_window.py
-----------------
Cuts a football match video into overlapping 60-second clips.
Slide every 30 seconds (50% overlap) so no action gets split across clips.

Usage:
    python sliding_window.py                        # uses 720p by default
    python sliding_window.py --video 224p.mp4       # use smaller video for testing
    python sliding_window.py --duration 60          # clip length in seconds (default 60)
    python sliding_window.py --step 30              # slide step in seconds (default 30)
    python sliding_window.py --test                 # quick test: only first 3 clips

Each clip is saved temporarily to data/temp/clip_current.mp4
After action_recognizer.py processes it, the clip is deleted.
This folder never holds more than 1 clip at a time.
"""

import os
import subprocess
import time
import argparse
import json
from pathlib import Path
from datetime import datetime


# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent  # current-football-project/
DATA_DIR    = BASE_DIR / "data"
MATCH_DIR   = DATA_DIR / "2019-10-01 - Blackburn Rovers - Nottingham Forest"
TEMP_DIR    = DATA_DIR / "temp"
LOG_DIR     = BASE_DIR / "logs"

DEFAULT_VIDEO   = MATCH_DIR / "720p.mp4"
TEST_VIDEO      = MATCH_DIR / "224p.mp4"


# ── helpers ────────────────────────────────────────────────────────────────────

def get_video_duration(video_path: Path) -> float:
    """Use ffprobe to get the total duration of the video in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info   = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_clip(video_path: Path, start_sec: float, duration: int, out_path: Path) -> bool:
    """
    Cut a single clip from video_path starting at start_sec for `duration` seconds.
    Uses ffmpeg stream copy (no re-encoding) — very fast, no GPU needed.
    Returns True if clip was created successfully.
    """
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),          # seek to start
        "-i", str(video_path),          # input file
        "-t", str(duration),            # duration of clip
        "-c", "copy",                   # stream copy = no re-encode, very fast
        "-avoid_negative_ts", "1",
        "-y",                           # overwrite if exists
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def seconds_to_gametime(seconds: float) -> str:
    """
    Convert raw video seconds to match game time string.
    e.g. 554.3 → "1st 09:14"
    Assumes first half = 0-2700s (45 min), second half = 2700s+
    """
    if seconds < 2700:
        half    = "1st"
        minutes = int(seconds // 60)
        secs    = int(seconds % 60)
    else:
        half    = "2nd"
        adj     = seconds - 2700
        minutes = int(adj // 60)
        secs    = int(adj % 60)
    return f"{half} {minutes:02d}:{secs:02d}"


def delete_clip(clip_path: Path):
    """Delete the temp clip after it has been processed."""
    if clip_path.exists():
        clip_path.unlink()


# ── main window generator ──────────────────────────────────────────────────────

def run_sliding_window(
    video_path: Path,
    clip_duration: int = 60,
    step: int = 30,
    test_mode: bool = False,
    on_clip_ready=None         # callback: called with (clip_path, start_sec, end_sec)
):
    """
    Main loop. Slides a window across the video and yields clips one by one.

    Args:
        video_path    : path to the .mp4 file
        clip_duration : length of each clip in seconds (default 60)
        step          : how many seconds to advance each time (default 30)
        test_mode     : if True, only process first 3 clips then stop
        on_clip_ready : callback function called after each clip is extracted
                        signature: on_clip_ready(clip_path, start_sec, end_sec, gametime)
    """

    # setup folders
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # log file for this run
    log_path = LOG_DIR / f"sliding_window_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\n{'='*60}")
    print(f"  Soccer EKG — Sliding Window")
    print(f"{'='*60}")
    print(f"  Video     : {video_path.name}")
    print(f"  Clip size : {clip_duration}s  |  Step : {step}s  |  Overlap : {clip_duration - step}s")
    print(f"  Temp dir  : {TEMP_DIR}")
    if test_mode:
        print(f"  TEST MODE : only first 3 clips")
    print(f"{'='*60}\n")

    # get total duration
    total_duration = get_video_duration(video_path)
    total_clips    = int((total_duration - clip_duration) / step) + 1
    print(f"  Match duration : {total_duration/60:.1f} min  →  ~{total_clips} clips total\n")

    clip_count = 0
    start_sec  = 0.0

    while start_sec + clip_duration <= total_duration:

        end_sec   = start_sec + clip_duration
        gametime  = seconds_to_gametime(start_sec)
        clip_path = TEMP_DIR / "clip_current.mp4"

        print(f"[Clip {clip_count+1:03d}]  {gametime}  ({start_sec:.0f}s → {end_sec:.0f}s)")

        # extract clip
        t0      = time.time()
        success = extract_clip(video_path, start_sec, clip_duration, clip_path)
        elapsed = time.time() - t0

        if not success:
            print(f"  ⚠ ffmpeg failed for clip at {start_sec}s — skipping")
            start_sec += step
            continue

        print(f"  ✓ Clip ready  ({elapsed:.2f}s to extract)  →  {clip_path.name}")

        # log entry
        with open(log_path, "a") as f:
            f.write(f"{clip_count+1},{start_sec},{end_sec},{gametime},{clip_path}\n")

        # call action recognizer (or any callback)
        if on_clip_ready:
            on_clip_ready(clip_path, start_sec, end_sec, gametime)

        # delete clip after processing
        delete_clip(clip_path)
        print(f"  ✓ Clip deleted\n")

        clip_count += 1
        start_sec  += step

        # test mode: stop after 3 clips
        if test_mode and clip_count >= 3:
            print("  [TEST MODE] Stopping after 3 clips.")
            break

    print(f"\n{'='*60}")
    print(f"  Done. Processed {clip_count} clips.")
    print(f"  Log saved to: {log_path}")
    print(f"{'='*60}\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Soccer EKG — Sliding Window")
    parser.add_argument("--video",    type=str, default=None,  help="Path to video file")
    parser.add_argument("--duration", type=int, default=60,    help="Clip duration in seconds (default 60)")
    parser.add_argument("--step",     type=int, default=30,    help="Slide step in seconds (default 30)")
    parser.add_argument("--test",     action="store_true",     help="Test mode: only 3 clips")
    args = parser.parse_args()

    # pick video
    if args.video:
        video_path = Path(args.video)
    elif args.test:
        video_path = TEST_VIDEO      # use 224p for quick testing
        print("  [TEST MODE] Using 224p.mp4 for faster testing")
    else:
        video_path = DEFAULT_VIDEO   # use 720p for real run

    if not video_path.exists():
        print(f"  ERROR: Video not found at {video_path}")
        exit(1)

    # example callback — just prints, action_recognizer will replace this
    def dummy_callback(clip_path, start_sec, end_sec, gametime):
        print(f"  → [callback] clip ready at gametime {gametime} — action_recognizer would run here")

    run_sliding_window(
        video_path    = video_path,
        clip_duration = args.duration,
        step          = args.step,
        test_mode     = args.test,
        on_clip_ready = dummy_callback
    )