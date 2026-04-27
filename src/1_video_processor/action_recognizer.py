"""
action_recognizer.py
--------------------
Detects soccer actions from a 60-second video clip using VideoMAE.

Model: MCG-NJU/videomae-small-finetuned-kinetics (Kinetics-400)
       Loaded once at startup, reused for every clip.

Input:  path to a .mp4 clip
Output: list of detected actions with confidence scores
        [
            {"action": "Shot", "confidence": 0.87, "time_in_clip": 14.2},
            {"action": "Foul", "confidence": 0.73, "time_in_clip": 42.8},
        ]

How it works:
    1. Sample 16 frames evenly from the clip
    2. Run VideoMAE on those 16 frames
    3. Get top-K predictions
    4. Map Kinetics labels → SoccerNet action types
    5. Filter by confidence threshold
    6. Return matched actions with estimated time_in_clip

Quick test:
    python action_recognizer.py --clip path/to/clip.mp4
    python action_recognizer.py --test   (uses a short synthetic clip)
"""

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


# ═══════════════════════════════════════════════════════════════════════════
# KINETICS → SOCCER ACTION MAPPING
# ═══════════════════════════════════════════════════════════════════════════
# Maps Kinetics-400 labels to SoccerNet-style action types.
# One Kinetics label can map to one soccer action.
# Labels NOT in this map are ignored (not soccer relevant).

KINETICS_TO_SOCCER = {
    # direct soccer labels only — no more javelin/hammer/discus
    "shooting goal (soccer)"              : "Goal",
    "kicking soccer ball"                 : "Shot",
    "juggling soccer ball"                : "Shot",
    "kicking field goal"                  : "Shot",

    # celebration = goal just happened
    "celebrating"                         : "Goal",
    "pumping fist"                        : "Goal",

    # physical contact → foul
    "high kick"                           : "Foul",
    "headbutting"                         : "Foul",
    "drop kicking"                        : "Foul",
    "side kick"                           : "Foul",

    # jumping → corner
    "high jump"                           : "Corner",
    "long jump"                           : "Corner",
    "triple jump"                         : "Corner",
    "jumping into pool"                   : "Corner",

    # passing motion
    "catching or throwing baseball"       : "Pass",
    "catching or throwing frisbee"        : "Pass",
    "throwing ball"                       : "Pass",
    "passing American football (in game)" : "Pass",
}

CONFIDENCE_THRESHOLD = 0.15

# How many frames to sample from the clip
NUM_FRAMES = 16   # VideoMAE was trained on 16 frames

# Model name
MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER (singleton — load once, reuse)
# ═══════════════════════════════════════════════════════════════════════════

_model     = None
_processor = None
_device    = None


def load_model():
    """Load VideoMAE model and processor. Called once at startup."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    print(f"  [model] loading {MODEL_NAME}...")
    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    _model     = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
    _model     = _model.to(_device)
    _model.eval()
    print(f"  [model] loaded on {_device} ✓")

    return _model, _processor, _device


# ═══════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_frames(clip_path: str, num_frames: int = NUM_FRAMES) -> Optional[List]:
    """
    Sample `num_frames` frames evenly from a video clip.
    Returns list of RGB numpy arrays, or None if video can't be opened.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [frames] ERROR: cannot open {clip_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames < num_frames:
        print(f"  [frames] WARNING: only {total_frames} frames, need {num_frames}")
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if len(frames) < num_frames:
        # pad by repeating last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])

    return frames, duration_sec


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(frames: List, model, processor, device: str) -> List[Dict]:
    """
    Run VideoMAE on a list of frames.
    Returns top predictions with Kinetics labels and confidence scores.
    """
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits

    probs   = torch.softmax(logits, dim=-1)[0]
    top_k   = torch.topk(probs, k=10)

    results = []
    for score, idx in zip(top_k.values, top_k.indices):
        label = model.config.id2label[idx.item()]
        results.append({
            "kinetics_label" : label,
            "confidence"     : round(score.item(), 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAP KINETICS → SOCCER
# ═══════════════════════════════════════════════════════════════════════════

def map_to_soccer_actions(
    predictions: List[Dict],
    clip_duration_sec: float,
    clip_start_sec: float = 0.0,       # ← add this
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> List[Dict]:
    soccer_actions = []
    seen_actions   = set()

    for pred in predictions:
        kinetics_label = pred["kinetics_label"]
        confidence     = pred["confidence"]

        if confidence < confidence_threshold:
            continue

        soccer_action = KINETICS_TO_SOCCER.get(kinetics_label)
        if soccer_action is None:
            continue

        if soccer_action in seen_actions:
            continue
        seen_actions.add(soccer_action)

        # use middle of clip as time estimate
        # absolute video time = clip start + half clip duration
        time_in_clip    = round(clip_duration_sec / 2.0, 1)
        abs_video_time  = clip_start_sec + time_in_clip   # ← absolute time

        soccer_actions.append({
            "action"         : soccer_action,
            "confidence"     : confidence,
            "time_in_clip"   : time_in_clip,
            "video_time"     : abs_video_time,    # ← NEW: absolute seconds
            "kinetics_label" : kinetics_label,
        })

    return soccer_actions

# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — called by pipeline
# ═══════════════════════════════════════════════════════════════════════════

def detect_actions(clip_path: str, clip_start_sec: float = 0.0) -> List[Dict]:
    model, processor, device = load_model()
    result = extract_frames(clip_path, NUM_FRAMES)
    if result is None:
        return []
    frames, duration_sec = result
    predictions  = run_inference(frames, model, processor, device)
    soccer_actions = map_to_soccer_actions(predictions, duration_sec, clip_start_sec)
    return soccer_actions

# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, default=None, help="Path to clip to test")
    parser.add_argument("--test", action="store_true",    help="Run on first 60s of match video")
    args = parser.parse_args()

    # find clip to test
    if args.clip:
        clip_path = args.clip
    elif args.test:
        # extract a quick test clip from the match video
        import subprocess, os
        BASE     = Path(__file__).parent.parent.parent
        video    = str(BASE / "data/2019-10-01 - Blackburn Rovers - Nottingham Forest/224p.mp4")
        clip_path = str(BASE / "data/temp/test_clip.mp4")
        os.makedirs(str(BASE / "data/temp"), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-ss", "540", "-i", video,
            "-t", "60", "-c", "copy", "-y", clip_path
        ], capture_output=True)
        print(f"  [test] extracted clip from 9:00 → 10:00 of match video")
    else:
        print("Usage: python action_recognizer.py --clip path/to/clip.mp4")
        print("       python action_recognizer.py --test")
        exit(0)

    print(f"\n  Clip: {clip_path}")
    print(f"  Running detection...\n")

    actions = detect_actions(clip_path)

    if actions:
        print(f"  Detected {len(actions)} soccer action(s):")
        for a in actions:
            print(f"    {a['action']:<12} conf={a['confidence']:.3f}  "
                  f"t={a['time_in_clip']}s  (from: '{a['kinetics_label']}')")
    else:
        print("  No soccer actions detected above threshold.")
        print("  (This is normal for a quiet clip — try a clip around a goal or foul)")

    print("\n  Done!")