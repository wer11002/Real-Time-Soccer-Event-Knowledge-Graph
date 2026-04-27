"""
buffer.py
---------
A rolling buffer that holds video-detected actions until the ESPN scrape
arrives. When ESPN data comes in, the buffer is flushed for matching.

Why we need it:
    Video clips overlap by 50% (60-sec clips, step 30-sec)
    → same action may be detected in two consecutive clips
    → dedup: keep only one if within ±5 sec

Why we need a buffer at all:
    Video processes every 30 seconds
    ESPN scrapes every ~5 minutes
    → buffer holds the video events until ESPN arrives
    → flush + match + clear

Quick test:
    python buffer.py
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VideoEvent:
    """One action detected in the video by the model."""
    video_time  : float    # absolute seconds from match start (e.g. 554.2)
    action      : str      # "Shot", "Foul", "Corner"...
    confidence  : float    # 0.0 to 1.0
    gametime    : str      # e.g. "1st 09:14"
    clip_start  : float    # start_sec of the clip this came from
    detected_at : str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY HELPER — converts action_recognizer output → VideoEvent
# ═══════════════════════════════════════════════════════════════════════════

def make_video_event(detected: dict, clip_start_sec: float, gametime: str) -> VideoEvent:
    """
    Convert one action_recognizer output dict into a VideoEvent for the buffer.

    Args:
        detected      : dict from detect_actions() with keys:
                        action, confidence, time_in_clip, video_time, kinetics_label
        clip_start_sec: absolute start time of the clip in seconds
        gametime      : human-readable time string e.g. "1st 09:14"

    Returns:
        VideoEvent ready to be added to the buffer

    Example:
        detected = {"action": "Shot", "confidence": 0.37,
                    "time_in_clip": 30.0, "video_time": 554.2,
                    "kinetics_label": "javelin throw"}

        event = make_video_event(detected, clip_start_sec=540.0, gametime="1st 09:00")
        buffer.add(event)
    """
    return VideoEvent(
        video_time  = detected["video_time"],
        action      = detected["action"],
        confidence  = detected["confidence"],
        gametime    = gametime,
        clip_start  = clip_start_sec,
    )


# ═══════════════════════════════════════════════════════════════════════════
# BUFFER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class EventBuffer:
    """
    Holds video events until ESPN data arrives.
    Deduplicates overlapping detections automatically on add.
    """

    def __init__(self, dedup_window_sec: float = 5.0):
        """
        Args:
            dedup_window_sec : if same action is seen within this many seconds
                               of an existing entry, skip it (keep higher confidence)
        """
        self._events      : List[VideoEvent] = []
        self.dedup_window : float = dedup_window_sec

    # ── add events ─────────────────────────────────────────────────────────

    def add(self, event: VideoEvent) -> bool:
        """
        Add a new event. Returns True if added, False if it was a duplicate.
        If duplicate found with higher confidence → replaces the old one.
        """
        duplicate = self._find_duplicate(event)

        if duplicate is None:
            self._events.append(event)
            return True

        # duplicate found — keep the higher-confidence version
        if event.confidence > duplicate.confidence:
            self._events.remove(duplicate)
            self._events.append(event)
            return True

        return False

    def add_many(self, events: List[VideoEvent]) -> int:
        """Add multiple events. Returns how many were actually added (after dedup)."""
        added = 0
        for e in events:
            if self.add(e):
                added += 1
        return added

    def add_from_detections(
        self,
        detections  : List[dict],
        clip_start_sec : float,
        gametime    : str,
    ) -> int:
        """
        Convenience method — converts action_recognizer output directly
        into VideoEvents and adds them to the buffer.

        Args:
            detections     : list of dicts from detect_actions()
            clip_start_sec : absolute start time of the clip in seconds
            gametime       : human-readable time string e.g. "1st 09:00"

        Returns:
            number of events added after dedup

        Example (how main.py uses this):
            detections = detect_actions(clip_path, clip_start_sec)
            added = buffer.add_from_detections(detections, clip_start_sec, gametime)
        """
        events = [make_video_event(d, clip_start_sec, gametime) for d in detections]
        return self.add_many(events)

    # ── dedup helper ───────────────────────────────────────────────────────

    def _find_duplicate(self, new: VideoEvent) -> Optional[VideoEvent]:
        """Return an existing event that matches (same action, within window)."""
        for e in self._events:
            same_action = e.action == new.action
            close_time  = abs(e.video_time - new.video_time) <= self.dedup_window
            if same_action and close_time:
                return e
        return None

    # ── access & flush ─────────────────────────────────────────────────────

    def get_all(self) -> List[VideoEvent]:
        """Return all buffered events sorted by video_time."""
        return sorted(self._events, key=lambda e: e.video_time)

    def size(self) -> int:
        return len(self._events)

    def flush(self) -> List[VideoEvent]:
        """Return all events and clear the buffer."""
        events       = self.get_all()
        self._events = []
        return events

    def clear(self):
        self._events = []

    # ── debug ──────────────────────────────────────────────────────────────

    def __repr__(self):
        return f"<EventBuffer size={len(self._events)}>"

    def summary(self) -> str:
        """Human-readable summary of buffer contents."""
        if not self._events:
            return "buffer is empty"
        lines = [f"buffer has {len(self._events)} events:"]
        for e in self.get_all():
            lines.append(
                f"  {e.gametime:<12} {e.action:<10} "
                f"conf={e.confidence:.2f}  t={e.video_time:.1f}s"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── EventBuffer self-test ───\n")

    buf = EventBuffer(dedup_window_sec=5.0)

    # ── test 1: basic add ──────────────────────────────────────────────────
    print("→ clip 1 (0-60s):  Shot at 14s")
    buf.add(VideoEvent(
        video_time=14.2, action="Shot", confidence=0.85,
        gametime="1st 00:14", clip_start=0.0
    ))

    # ── test 2: duplicate (lower confidence) → should be skipped ──────────
    print("→ clip 2 (30-90s): Shot at 14s  (DUPLICATE — should be skipped)")
    buf.add(VideoEvent(
        video_time=14.5, action="Shot", confidence=0.82,
        gametime="1st 00:14", clip_start=30.0
    ))

    # ── test 3: new action ─────────────────────────────────────────────────
    print("→ clip 2 (30-90s): Foul at 42s")
    buf.add(VideoEvent(
        video_time=42.8, action="Foul", confidence=0.73,
        gametime="1st 00:42", clip_start=30.0
    ))

    # ── test 4: same type far apart → NOT a duplicate ──────────────────────
    print("→ clip 3 (60-120s): Shot at 75s  (different time → new event)")
    buf.add(VideoEvent(
        video_time=75.1, action="Shot", confidence=0.78,
        gametime="1st 01:15", clip_start=60.0
    ))

    # ── test 5: duplicate with HIGHER confidence → should REPLACE ──────────
    print("→ clip 4: Foul at 43s with higher confidence (0.91 vs 0.73) — should REPLACE")
    buf.add(VideoEvent(
        video_time=43.0, action="Foul", confidence=0.91,
        gametime="1st 00:43", clip_start=30.0
    ))

    print("\n" + buf.summary())

    # ── test 6: add_from_detections (simulates main.py usage) ─────────────
    print("\n── Testing add_from_detections (simulates action_recognizer output) ──")
    fake_detections = [
        {"action": "Goal", "confidence": 0.92, "time_in_clip": 30.0,
         "video_time": 3780.0, "kinetics_label": "shooting goal (soccer)"},
    ]
    added = buf.add_from_detections(fake_detections, clip_start_sec=3750.0, gametime="1st 63:00")
    print(f"  added {added} event(s) from detections")

    # ── flush ──────────────────────────────────────────────────────────────
    print("\n→ ESPN arrives, flushing buffer...")
    flushed = buf.flush()
    print(f"  flushed {len(flushed)} events")
    print(f"  buffer is now empty: {buf.size() == 0}")

    print("\n✓ all good!")