"""
action_recognizer.py
--------------------
Detects soccer actions from a 60-second video clip using Qwen2-VL.

Model: Qwen/Qwen2-VL-7B-Instruct
       Vision-Language Model — reads jersey numbers, describes actions,
       identifies team colors from video frames.

Input:  path to a .mp4 clip + clip_start_sec
Output: list of detected events
        [
            {
                "action"      : "Shot",
                "jersey"      : "7",
                "team"        : "Blackburn Rovers",
                "team_color"  : "blue/white",
                "description" : "Player #7 takes a right-footed shot...",
                "video_time"  : 554.2,
                "time_in_clip": 30.0,
                "confidence"  : 0.85,
            }
        ]

Why Qwen2-VL over VideoMAE:
    - VideoMAE: generic Kinetics-400, cannot read jerseys, no text output
    - Qwen2-VL: strong OCR (reads jersey numbers), generates natural language
                descriptions, identifies team by kit color

Setup (run once on GPU node):
    pip install transformers>=4.37 qwen-vl-utils

Quick test:
    python action_recognizer.py --test
    python action_recognizer.py --clip path/to/clip.mp4
"""

import cv2
import re
import json
import torch
import subprocess
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# ── model config ───────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2-VL-7B-Instruct"
NUM_FRAMES  = 8      # sample 8 frames per clip (spread evenly)
             # Qwen2-VL works well with 4-16 frames
             # 8 is a good balance of speed vs accuracy

# soccer action keywords to look for in VLM response
ACTION_KEYWORDS = {
    "Goal"        : ["goal", "scores", "scored", "into the net", "back of the net"],
    "Shot"        : ["shot", "shoots", "attempt", "saved", "blocked", "wide", "header"],
    "Foul"        : ["foul", "tackle", "brings down", "trips", "pushes", "holds"],
    "Corner"      : ["corner", "corner kick"],
    "Free_Kick"   : ["free kick", "free-kick"],
    "Offside"     : ["offside"],
    "Substitution": ["substitut", "comes on", "replaces", "comes off"],
}

# team color hints (kit colors for this specific match)
TEAM_COLOR_MAP = {
    "blue"        : "Blackburn Rovers",
    "blue/white"  : "Blackburn Rovers",
    "dark blue"   : "Blackburn Rovers",
    "red"         : "Nottingham Forest",
    "red/white"   : "Nottingham Forest",
    "white"       : None,   # ambiguous
}


# ── model singleton ────────────────────────────────────────────────────────
_model     = None
_processor = None
_device    = None


def load_model():
    """Load Qwen2-VL model and processor. Called once at startup."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    print(f"  [model] loading {MODEL_NAME}...")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model     = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.float16 if _device == "cuda" else torch.float32,
        device_map  = "auto",
        trust_remote_code = True,
    )
    _model.eval()
    print(f"  [model] loaded on {_device} ✓")

    return _model, _processor, _device


# ═══════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_frames(clip_path: str, num_frames: int = NUM_FRAMES):
    """Sample num_frames evenly from a video clip. Returns (frames, duration_sec)."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [frames] ERROR: cannot open {clip_path}")
        return None, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_sec = total_frames / fps

    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # pad if needed
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])

    return frames, duration_sec


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a soccer match analyst. 
Analyze video frames from a soccer match and identify key events.
Always respond in valid JSON format only — no other text."""

ACTION_PROMPT = """Analyze these video frames from a soccer match clip.

Identify if any key soccer action is happening:
- Shot (attempt on goal, header, blocked shot)
- Goal (ball crosses the line, celebration)  
- Foul (tackle, trip, push, handball)
- Corner (corner kick being taken)
- Free_Kick (free kick being taken)
- Substitution (player being replaced)

For the main player involved:
- Read their jersey number from the shirt if visible
- Note the kit color (e.g. "blue/white", "red")

Respond ONLY with this JSON (no other text):
{
  "action": "Shot" or "Goal" or "Foul" or "Corner" or "Free_Kick" or "Substitution" or "None",
  "jersey": "7" or null,
  "team_color": "blue/white" or "red" or null,
  "description": "one sentence describing what is happening",
  "confidence": 0.0 to 1.0
}

If no clear soccer action is visible, return:
{"action": "None", "jersey": null, "team_color": null, "description": "No clear action", "confidence": 0.0}"""


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(frames: List, model, processor, device: str) -> Optional[Dict]:
    """
    Run Qwen2-VL on a list of frames.
    Returns parsed JSON dict or None if parsing fails.
    """
    from qwen_vl_utils import process_vision_info
    from PIL import Image as PILImage

    # convert numpy frames to PIL images
    pil_images = [PILImage.fromarray(f) for f in frames]

    # build message with images + prompt
    content = []
    for img in pil_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": ACTION_PROMPT})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]

    # prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text   = [text],
        images = image_inputs,
        videos = video_inputs,
        padding = True,
        return_tensors = "pt",
    ).to(device)

    # generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = 200,
            do_sample      = False,
            temperature    = None,
            top_p          = None,
        )

    # decode
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response  = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # parse JSON
    try:
        # strip any markdown code fences if present
        response = re.sub(r"```json|```", "", response).strip()
        return json.loads(response)
    except json.JSONDecodeError:
        # try to extract JSON from response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        print(f"  [vlm] JSON parse failed: {response[:100]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# RESOLVE TEAM FROM COLOR
# ═══════════════════════════════════════════════════════════════════════════

def resolve_team(team_color: Optional[str]) -> Optional[str]:
    """Map kit color to team name using TEAM_COLOR_MAP."""
    if not team_color:
        return None
    color_lower = team_color.lower().strip()
    for key, team in TEAM_COLOR_MAP.items():
        if key in color_lower:
            return team
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_actions(clip_path: str, clip_start_sec: float = 0.0) -> List[Dict]:
    """
    Main entry point for the pipeline.
    Takes a clip path, returns list of detected soccer events.

    Args:
        clip_path      : path to .mp4 clip file
        clip_start_sec : absolute start time of clip in the full match video

    Returns:
        [
            {
                "action"      : "Shot",
                "jersey"      : "7",
                "team"        : "Blackburn Rovers",
                "team_color"  : "blue/white",
                "description" : "Player #7 takes a right-footed shot...",
                "video_time"  : 554.2,
                "time_in_clip": 30.0,
                "confidence"  : 0.85,
            },
            ...
        ]
        Empty list if no soccer action detected.
    """
    model, processor, device = load_model()

    # extract frames
    frames, duration_sec = extract_frames(clip_path, NUM_FRAMES)
    if not frames:
        return []

    # run VLM
    result = run_inference(frames, model, processor, device)
    if not result:
        return []

    # filter out "None" actions
    action = result.get("action", "None")
    if action in ("None", "none", "", None):
        return []

    # resolve team from color
    team_color = result.get("team_color")
    team       = resolve_team(team_color)

    # absolute video time = clip start + midpoint of clip
    time_in_clip = round(duration_sec / 2.0, 1)
    video_time   = clip_start_sec + time_in_clip

    return [{
        "action"      : action,
        "jersey"      : result.get("jersey"),
        "team"        : team,
        "team_color"  : team_color,
        "description" : result.get("description", ""),
        "video_time"  : video_time,
        "time_in_clip": time_in_clip,
        "confidence"  : float(result.get("confidence", 0.5)),
    }]


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Extract test clip from match video at 9:00")
    args = parser.parse_args()

    if args.clip:
        clip_path = args.clip

    elif args.test:
        import os
        BASE      = Path(__file__).parent.parent.parent
        video     = str(BASE / "data/2019-10-01 - Blackburn Rovers - Nottingham Forest/720p.mp4")
        clip_path = str(BASE / "data/temp/test_clip_vlm.mp4")
        os.makedirs(str(BASE / "data/temp"), exist_ok=True)

        print("  [test] extracting clip at 9:00 → 10:00 (Armstrong's shot)")
        subprocess.run([
            "ffmpeg", "-ss", "540", "-i", video,
            "-t", "60", "-c", "copy", "-y", clip_path
        ], capture_output=True)
        clip_start = 540.0

    else:
        print("Usage:")
        print("  python action_recognizer.py --test")
        print("  python action_recognizer.py --clip path/to/clip.mp4")
        exit(0)

    print(f"\n  Clip   : {clip_path}")
    print(f"  Running Qwen2-VL detection...\n")

    actions = detect_actions(clip_path, clip_start_sec=clip_start if args.test else 0.0)

    if actions:
        print(f"  Detected {len(actions)} action(s):\n")
        for a in actions:
            print(f"  Action      : {a['action']}")
            print(f"  Jersey      : #{a['jersey']}" if a['jersey'] else "  Jersey      : not detected")
            print(f"  Team        : {a['team'] or a['team_color'] or 'unknown'}")
            print(f"  Description : {a['description']}")
            print(f"  Video time  : {a['video_time']:.1f}s")
            print(f"  Confidence  : {a['confidence']:.2f}")
    else:
        print("  No soccer action detected in this clip.")

    print("\n  Done!")