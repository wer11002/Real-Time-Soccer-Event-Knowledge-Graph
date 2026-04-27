"""
espn_scraper.py
---------------
Fetches live match events from ESPN's hidden API.

The API returns structured play-by-play data with:
- Exact match time (seconds + display "9'")
- Player names (from participants array)
- Team name
- Action type (from play.type)
- Full commentary text

Usage in pipeline:
    scraper = ESPNScraper()
    scraper.find_and_load("2019-10-01", "Blackburn Rovers", "Nottingham Forest")
    
    # simulate ESPN tick — get events up to current match minute
    events = scraper.get_events_up_to(minute=10.0)
    
    # get new events since last scrape
    events = scraper.get_events_in_window(from_min=5.0, to_min=10.0)

Fallback:
    If API returns empty or fails → loads from CSV automatically.

Quick test:
    python espn_scraper.py
"""

import os
import re
import csv
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from thefuzz import fuzz
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# ── configuration & paths ──────────────────────────────────────────────────────
API_BASE = os.getenv("ESPN_API_BASE_URL", "http://site.api.espn.com/apis/site/v2/sports/soccer")
LEAGUES_STR = os.getenv("ESPN_LEAGUES", "eng.2,eng.1,esp.1,ita.1,ger.1,fra.1,uefa.champions")
LEAGUES = LEAGUES_STR.split(",")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_csv_path = os.getenv("FOOTBALL_CSV_PATH", "data/blackburn_forest_2019-10-01.csv")
CSV_PATH = BASE_DIR / env_csv_path

# ESPN play type → our action labels
PLAY_TYPE_MAP = {
    "shot-blocked"   : "Shot",
    "shot-wide"      : "Shot",
    "shot-saved"     : "Shot",
    "shot-on-target" : "Shot",
    "shot-off-target": "Shot",
    "shot"           : "Shot",
    "goal"           : "Goal",
    "penalty-scored" : "Goal",
    "penalty-missed" : "Shot",
    "foul"           : "Foul",
    "yellow-card"    : "Foul",
    "red-card"       : "Foul",
    "corner"         : "Corner",
    "offside"        : "Offside",
    "free-kick"      : "Free_Kick",
    "substitution"   : "Substitution",
    "kickoff"        : "Other",
    "end-period"     : "Other",
    "start-period"   : "Other",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def parse_time_min(time_str: str) -> float:
    """'45+2'' → 47.0,  '9'' → 9.0"""
    t = re.sub(r"'", "", str(time_str).strip())
    if "+" in t:
        base, extra = t.split("+", 1)
        try:
            return float(base.strip()) + float(extra.strip())
        except ValueError:
            return 0.0
    try:
        return float(t.strip())
    except ValueError:
        return 0.0


def normalize_action(play_type: str) -> str:
    return PLAY_TYPE_MAP.get(play_type.lower().strip(), "Other")


def parse_commentary_event(item: dict) -> Optional[Dict]:
    """Parse one ESPN commentary item into our standard event dict."""
    play = item.get("play", {})
    if not play:
        return None

    clock        = play.get("clock", {})
    time_value   = clock.get("value", 0.0)
    time_display = clock.get("displayValue", "")
    period       = play.get("period", {}).get("number", 1)

    # convert clock seconds → absolute match minutes
    if period == 1:
        time_min = time_value / 60.0
    else:
        time_min = 45.0 + (time_value / 60.0)

    if time_min <= 0 and not time_display:
        return None

    # action type
    play_type = play.get("type", {})
    type_str  = play_type.get("type", play_type.get("text", "other"))
    action    = normalize_action(type_str)

    if action == "Other":
        return None

    # player (first participant = main actor)
    participants = play.get("participants", [])
    player = None
    if participants:
        athlete = participants[0].get("athlete", {})
        player  = athlete.get("displayName")

    # team
    team = play.get("team", {}).get("displayName")

    # card flags
    yellow = "yellow" in type_str.lower()
    red    = "red"    in type_str.lower()

    full_text = item.get("text", play.get("text", ""))

    return {
        "time"      : round(time_min, 2),
        "time_raw"  : time_display or f"{int(time_min)}'",
        "player"    : player,
        "team"      : team,
        "action"    : action,
        "yellow"    : yellow,
        "red"       : red,
        "full_text" : full_text,
        "period"    : period,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ESPN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════

class ESPNScraper:
    """
    Fetches live match events from ESPN API.
    Falls back to CSV if API returns no data.
    """

    def __init__(self, csv_fallback: Path = CSV_PATH):
        self.csv_fallback = csv_fallback
        self._events      : List[Dict] = []
        self._loaded      : bool = False
        self._source      : str  = "none"

    def _find_game(self, date_str: str, team1: str, team2: str) -> Tuple[Optional[str], Optional[str]]:
        """Search ESPN API for a match. Returns (game_id, league) or (None, None)."""
        for league in LEAGUES:
            url = f"{API_BASE}/{league}/scoreboard?dates={date_str}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    continue
                for event in r.json().get("events", []):
                    name = event.get("name", "").lower()
                    s1   = fuzz.token_set_ratio(team1.lower(), name)
                    s2   = fuzz.token_set_ratio(team2.lower(), name)
                    if s1 > 80 and s2 > 80:
                        return event.get("id"), league
            except Exception:
                continue
        return None, None

    def _load_from_api(self, game_id: str, league: str) -> int:
        """Load events from ESPN API for a given game_id."""
        url = f"{API_BASE}/{league}/summary?event={game_id}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return 0
            data       = r.json()
            commentary = data.get("commentary", []) or data.get("keyEvents", [])
            events     = [parse_commentary_event(i) for i in commentary]
            events     = [e for e in events if e is not None]
            if events:
                self._events = sorted(events, key=lambda e: e["time"])
                self._source = "api"
                return len(self._events)
        except Exception as e:
            print(f"  [espn] API error: {e}")
        return 0

    def _load_from_csv(self) -> int:
        """Fallback: load events from pre-scraped CSV."""
        if not self.csv_fallback.exists():
            return 0
        events = []
        with open(self.csv_fallback, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                time_min = parse_time_min(row.get("Time", "0"))
                player   = row.get("Player", "").strip()
                team     = row.get("Team",   "").strip()
                action   = row.get("Action_Type", "").strip()
                if not action or action == "None":
                    continue
                events.append({
                    "time"      : time_min,
                    "time_raw"  : row.get("Time", "").strip(),
                    "player"    : player if player not in ("None", "") else None,
                    "team"      : team   if team   not in ("None", "") else None,
                    "action"    : action,
                    "yellow"    : row.get("Yellow_Card", "0") == "1",
                    "red"       : row.get("Red_Card",    "0") == "1",
                    "full_text" : row.get("Full_Text", "").strip(),
                    "period"    : 1 if time_min <= 45 else 2,
                })
        self._events = sorted(events, key=lambda e: e["time"])
        self._source = "csv"
        return len(self._events)

    def find_and_load(self, date: str, team1: str, team2: str) -> int:
        """
        Find the match on ESPN and load all events.
        Falls back to CSV if API fails.
        Returns number of events loaded.
        """
        date_str = date.replace("-", "")
        print(f"  [espn] searching for {team1} vs {team2} on {date}...")

        game_id, league = self._find_game(date_str, team1, team2)

        if game_id:
            print(f"  [espn] found game_id={game_id} in {league}")
            n = self._load_from_api(game_id, league)
            if n > 0:
                print(f"  [espn] loaded {n} events from API ✓")
                self._loaded = True
                return n
            print(f"  [espn] API empty — falling back to CSV")
        else:
            print(f"  [espn] match not found — falling back to CSV")

        n = self._load_from_csv()
        if n > 0:
            print(f"  [espn] loaded {n} events from CSV ✓")
            self._loaded = True
        return n

    # ── query methods ──────────────────────────────────────────────────────

    def get_all_events(self) -> List[Dict]:
        self._ensure_loaded()
        return self._events

    def get_events_up_to(self, minute: float) -> List[Dict]:
        """All events from 0 to minute — simulates ESPN scrape tick."""
        self._ensure_loaded()
        return [e for e in self._events if e["time"] <= minute]

    def get_events_in_window(self, from_min: float, to_min: float) -> List[Dict]:
        """Events in [from_min, to_min] window."""
        self._ensure_loaded()
        return [e for e in self._events if from_min <= e["time"] <= to_min]

    def get_events_near(self, minute: float, tolerance: float = 2.0) -> List[Dict]:
        """Events within ±tolerance of minute."""
        self._ensure_loaded()
        return [e for e in self._events if abs(e["time"] - minute) <= tolerance]

    def total_events(self) -> int:
        return len(self._events)

    def source(self) -> str:
        return self._source

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call find_and_load() first.")

    def summary(self) -> str:
        lines = [f"ESPNScraper [{self._source}] — {len(self._events)} events"]
        for e in self._events[:5]:
            player = e["player"] or "—"
            lines.append(f"  [{e['time_raw']:>6}]  {e['action']:<12}  {player}")
        if len(self._events) > 5:
            lines.append(f"  ... and {len(self._events)-5} more")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def load_espn_events(date: str, team1: str, team2: str) -> List[Dict]:
    """One-liner to load all ESPN events for a match."""
    scraper = ESPNScraper()
    scraper.find_and_load(date, team1, team2)
    return scraper.get_all_events()


# ══════════════════════════════════════════════════════════════════════════
# quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── ESPNScraper self-test ───\n")

    scraper = ESPNScraper()
    n = scraper.find_and_load(
        date  = "2019-10-01",
        team1 = "Blackburn Rovers",
        team2 = "Nottingham Forest",
    )

    print(f"\nLoaded {n} events from: {scraper.source()}\n")
    print(scraper.summary())

    print("\n── Events in first 10 minutes ──")
    for e in scraper.get_events_up_to(10):
        player = e["player"] or "—"
        print(f"  [{e['time_raw']:>6}]  {e['action']:<12}  {player:<22}  {e['team']}")

    print("\n── Events near minute 9 (±2 min) ──")
    for e in scraper.get_events_near(9.0, tolerance=2.0):
        player = e["player"] or "—"
        print(f"  [{e['time_raw']:>6}]  {e['action']:<12}  {player:<22}  {e['team']}")

    print("\n── Yellow cards ──")
    for e in scraper.get_all_events():
        if e["yellow"]:
            print(f"  [{e['time_raw']:>6}]  {e['player']}  🟨")

    print("\n✓ done!")