"""
align.py
--------
Matches video-detected events (from buffer.py) to ESPN scraped events.

Matching rules:
    1. Time must be within ±2 minutes
    2. Action types must be compatible (fuzzy mapping — see ACTION_MAP)
    3. Pick the candidate with the smallest time difference

Output per buffered event:
    - matched   : True/False
    - player    : name (if matched)
    - team      : team (if matched)
    - espn_time : ESPN-reported time (if matched)
    - all original fields preserved

Quick test:
    python align.py
"""

import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
# FUZZY ACTION MAPPING
# ═══════════════════════════════════════════════════════════════════════════
# Maps video model action labels → compatible ESPN action labels
# espn_scraper.py normalizes ESPN actions to: Shot, Goal, Foul, Corner,
# Offside, Free_Kick, Substitution

ACTION_MAP: Dict[str, List[str]] = {

    # shots and goals
    "Shot"        : ["Shot", "Goal"],
    "Goal"        : ["Goal", "Shot"],       # a goal is also a shot
    "Penalty"     : ["Goal", "Shot"],

    # fouls and cards
    "Foul"        : ["Foul", "Free_Kick"],  # foul often leads to free kick
    "Free_Kick"   : ["Free_Kick", "Foul"],

    # set pieces
    "Corner"      : ["Corner"],
    "Offside"     : ["Offside"],

    # other
    "Substitution": ["Substitution"],
    "Pass"        : ["Shot", "Foul"],       # pass motion sometimes looks like shot/foul
}


def action_matches(video_action: str, espn_action: str) -> bool:
    """
    Check if a video-detected action is compatible with an ESPN action.
    Uses ACTION_MAP, falls back to direct string comparison.
    """
    va = video_action.strip()
    ea = espn_action.strip()

    compatible = ACTION_MAP.get(va, [va])
    return ea in compatible


# ═══════════════════════════════════════════════════════════════════════════
# MATCHED EVENT — output of alignment
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MatchedEvent:
    """Result of aligning one video event against the ESPN data."""

    # original video-side fields
    video_time : float
    action     : str
    confidence : float
    gametime   : str

    # alignment result
    matched    : bool
    player     : Optional[str]  = None
    team       : Optional[str]  = None
    espn_time  : Optional[float]= None   # ESPN-reported minute (e.g. 9.0)
    espn_text  : Optional[str]  = None   # raw commentary text
    time_diff  : Optional[float]= None   # |video_min - espn_min| in minutes

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# CORE MATCHING LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def match_event(
    video_event        : dict,
    espn_events        : List[dict],
    time_tolerance_min : float = 2.0,
) -> MatchedEvent:
    """
    Match one video event against the list of ESPN events.

    Args:
        video_event        : dict with video_time, action, confidence, gametime
        espn_events        : list of dicts from espn_scraper.py
        time_tolerance_min : max time difference in minutes (default ±2 min)

    Returns:
        MatchedEvent with matched=True if a match was found
    """
    video_minute = video_event["video_time"] / 60.0
    video_action = video_event["action"]

    # find all compatible ESPN events within time window
    candidates = []
    for e in espn_events:
        time_diff = abs(e["time"] - video_minute)
        if time_diff <= time_tolerance_min and action_matches(video_action, e["action"]):
            candidates.append((time_diff, e))

    # no candidate → unmatched (becomes gray node in KG)
    if not candidates:
        return MatchedEvent(
            video_time = video_event["video_time"],
            action     = video_action,
            confidence = video_event["confidence"],
            gametime   = video_event["gametime"],
            matched    = False,
        )

    # pick candidate with smallest time difference
    candidates.sort(key=lambda x: x[0])
    best_diff, best = candidates[0]

    return MatchedEvent(
        video_time = video_event["video_time"],
        action     = video_action,
        confidence = video_event["confidence"],
        gametime   = video_event["gametime"],
        matched    = True,
        player     = best.get("player"),
        team       = best.get("team"),
        espn_time  = best.get("time"),
        espn_text  = best.get("full_text"),
        time_diff  = round(best_diff, 2),
    )


def align_buffer(
    buffer_events      : List,
    espn_events        : List[dict],
    time_tolerance_min : float = 2.0,
) -> List[MatchedEvent]:
    """
    Align all buffered video events against ESPN data.

    Args:
        buffer_events : list of VideoEvent objects OR plain dicts (both work)
        espn_events   : list of dicts from espn_scraper.py
    """
    results = []
    for v in buffer_events:
        # handle both VideoEvent dataclass objects and plain dicts
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            v_dict = {
                "video_time" : v.video_time,
                "action"     : v.action,
                "confidence" : v.confidence,
                "gametime"   : v.gametime,
            }
        else:
            v_dict = v

        matched = match_event(v_dict, espn_events, time_tolerance_min)
        results.append(matched)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY HELPER
# ═══════════════════════════════════════════════════════════════════════════

def summarize(matched_events: List[MatchedEvent]) -> str:
    """Human-readable summary of alignment results."""
    if not matched_events:
        return "no events aligned"

    matched_count = sum(1 for e in matched_events if e.matched)
    lines = [f"\n── Alignment results: {matched_count}/{len(matched_events)} matched ──"]

    for e in matched_events:
        if e.matched:
            lines.append(
                f"  ✓ {e.gametime:<12} {e.action:<10} → {e.player or '—':<20} "
                f"({e.team})  Δt={e.time_diff}min"
            )
        else:
            lines.append(
                f"  ✗ {e.gametime:<12} {e.action:<10} → UNKNOWN (no ESPN match)"
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# quick self-test — uses REAL ESPN data via espn_scraper
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "2_web_scraper"))
    from espn_scraper import ESPNScraper

    print("─── align.py self-test (real ESPN data) ───\n")

    # load real ESPN data
    scraper = ESPNScraper()
    scraper.find_and_load("2019-10-01", "Blackburn Rovers", "Nottingham Forest")
    espn_events = scraper.get_all_events()
    print(f"ESPN: {len(espn_events)} events loaded\n")

    # simulate video buffer events (what action_recognizer would produce)
    video_events = [
        {"video_time":  554.2, "action": "Shot",   "confidence": 0.37, "gametime": "1st 09:14"},
        {"video_time":  720.0, "action": "Foul",   "confidence": 0.73, "gametime": "1st 12:00"},
        {"video_time": 1080.5, "action": "Goal",   "confidence": 0.09, "gametime": "1st 18:00"},
        {"video_time": 1400.0, "action": "Corner", "confidence": 0.04, "gametime": "1st 23:20"},
        {"video_time": 1800.0, "action": "Shot",   "confidence": 0.65, "gametime": "1st 30:00"},
    ]

    print(f"Video buffer: {len(video_events)} events\n")

    # run alignment
    results = align_buffer(video_events, espn_events, time_tolerance_min=2.0)
    print(summarize(results))

    # fuzzy match check
    print("\n── Fuzzy-match check ──")
    print(f"  'Goal' ↔ 'Shot'   : {action_matches('Goal',   'Shot')}   (should be True)")
    print(f"  'Shot' ↔ 'Goal'   : {action_matches('Shot',   'Goal')}   (should be True)")
    print(f"  'Corner' ↔ 'Foul' : {action_matches('Corner', 'Foul')}  (should be False)")
    print(f"  'Foul' ↔ 'Free_Kick': {action_matches('Foul', 'Free_Kick')}  (should be True)")

    print("\n✓ all good!")