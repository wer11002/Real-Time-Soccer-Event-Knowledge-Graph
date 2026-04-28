"""
roster_lookup.py
----------------
Maps jersey numbers to player names using ESPN roster data.

Loaded once at pipeline startup.
Used by action_recognizer.py after Qwen2-VL detects a jersey number.

Usage:
    lookup = RosterLookup()
    lookup.load_from_espn("2019-10-01", "Blackburn Rovers", "Nottingham Forest")

    # direct lookup by jersey + team name
    player = lookup.find(jersey="7", team="Blackburn Rovers")
    # → "Adam Armstrong"

    # lookup by jersey + team color (when VLM says "blue/white kit")
    player = lookup.find_by_color(jersey="7", color="blue")
    # → "Adam Armstrong"  (if Blackburn plays in blue)

    # fuzzy team name (VLM might say "Blackburn" not full name)
    player = lookup.find(jersey="7", team="Blackburn")
    # → "Adam Armstrong"

Quick test:
    python roster_lookup.py
"""

import sys
import re
import requests
from pathlib import Path
from typing import Optional, Dict, List
from thefuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "2_web_scraper"))

# leagues to search
LEAGUES = ["eng.2", "eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions"]


# ═══════════════════════════════════════════════════════════════════════════
# ROSTER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

class RosterLookup:
    """
    Loads match rosters from ESPN API.
    Maps (jersey_number, team_name) → player_name.
    """

    def __init__(self):
        # structure: {team_name: {jersey_number: player_name}}
        self._rosters   : Dict[str, Dict[str, str]] = {}
        self._team_colors: Dict[str, str] = {}   # team_name → kit color hint
        self._loaded    : bool = False
        self._game_id   : Optional[str] = None

    # ── load from ESPN ─────────────────────────────────────────────────────

    def _find_game(self, date_str: str, team1: str, team2: str):
        for league in LEAGUES:
            url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
                   f"/{league}/scoreboard?dates={date_str}")
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

    def load_from_espn(self, date: str, team1: str, team2: str) -> int:
        """
        Load rosters for a match from ESPN API.
        Returns total number of players loaded.
        """
        date_str = date.replace("-", "")
        print(f"  [roster] loading rosters for {team1} vs {team2}...")

        game_id, league = self._find_game(date_str, team1, team2)
        if not game_id:
            print(f"  [roster] match not found in API")
            return 0

        self._game_id = game_id
        url = (f"http://site.api.espn.com/apis/site/v2/sports/soccer"
               f"/{league}/summary?event={game_id}")

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return 0

            data    = r.json()
            rosters = data.get("rosters", [])
            total   = 0

            for team_data in rosters:
                team_name = team_data.get("team", {}).get("displayName", "")
                players   = team_data.get("roster", [])

                self._rosters[team_name] = {}

                for p in players:
                    jersey  = str(p.get("jersey", "")).strip()
                    athlete = p.get("athlete", {})
                    name    = athlete.get("displayName", "").strip()
                    if jersey and name:
                        self._rosters[team_name][jersey] = name
                        total += 1

                print(f"  [roster] {team_name}: {len(players)} players loaded")

            self._loaded = True
            return total

        except Exception as e:
            print(f"  [roster] error: {e}")
            return 0

    def load_manual(self, rosters: Dict[str, Dict[str, str]]):
        """
        Load rosters manually (for testing or offline use).
        rosters format: {"Team Name": {"jersey": "Player Name"}}
        """
        self._rosters = rosters
        self._loaded  = True

    # ── lookup methods ─────────────────────────────────────────────────────

    def find(self, jersey: str, team: str) -> Optional[str]:
        """
        Find player by jersey number and team name.
        Uses fuzzy matching on team name (handles partial names from VLM).

        Args:
            jersey : jersey number as string e.g. "7" or "#7"
            team   : team name (can be partial e.g. "Blackburn")

        Returns:
            Player name or None if not found
        """
        jersey = str(jersey).strip().lstrip("#")

        # find best matching team
        best_team = self._find_team(team)
        if not best_team:
            return None

        return self._rosters[best_team].get(jersey)

    def find_by_color(self, jersey: str, color: str) -> Optional[str]:
        """
        Find player by jersey number and kit color.
        Useful when VLM identifies team by color ("blue/white kit")
        rather than team name.

        Currently returns candidates from all teams — caller picks best.
        """
        jersey = str(jersey).strip().lstrip("#")
        results = []

        for team_name, players in self._rosters.items():
            if jersey in players:
                results.append({
                    "player" : players[jersey],
                    "team"   : team_name,
                })

        return results  # caller resolves ambiguity with ESPN event data

    def get_all_players(self, team: str = None) -> List[Dict]:
        """Return all players, optionally filtered by team."""
        results = []
        for team_name, players in self._rosters.items():
            if team and not self._team_matches(team, team_name):
                continue
            for jersey, name in players.items():
                results.append({
                    "jersey" : jersey,
                    "name"   : name,
                    "team"   : team_name,
                })
        return sorted(results, key=lambda p: int(p["jersey"]) if p["jersey"].isdigit() else 99)

    def get_teams(self) -> List[str]:
        """Return list of loaded team names."""
        return list(self._rosters.keys())

    # ── helpers ────────────────────────────────────────────────────────────

    def _find_team(self, team_query: str) -> Optional[str]:
        """Fuzzy match team query against loaded team names."""
        best_score = 0
        best_team  = None
        for team_name in self._rosters:
            score = fuzz.token_set_ratio(team_query.lower(), team_name.lower())
            if score > best_score:
                best_score = score
                best_team  = team_name
        return best_team if best_score > 60 else None

    def _team_matches(self, query: str, team_name: str) -> bool:
        return fuzz.token_set_ratio(query.lower(), team_name.lower()) > 60

    def __repr__(self):
        teams = ", ".join(self._rosters.keys())
        total = sum(len(p) for p in self._rosters.values())
        return f"<RosterLookup teams=[{teams}] players={total}>"


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── RosterLookup self-test ───\n")

    lookup = RosterLookup()
    n = lookup.load_from_espn(
        date  = "2019-10-01",
        team1 = "Blackburn Rovers",
        team2 = "Nottingham Forest",
    )
    print(f"\nLoaded {n} players total")
    print(repr(lookup))

    print("\n── Full rosters ──")
    for team in lookup.get_teams():
        print(f"\n{team}:")
        for p in lookup.get_all_players(team):
            print(f"  #{p['jersey']:<3}  {p['name']}")

    print("\n── Direct jersey lookups ──")
    tests = [
        ("7",  "Blackburn Rovers",  "Adam Armstrong"),
        ("23", "Nottingham Forest", "Joe Lolley"),
        ("19", "Blackburn Rovers",  "Stewart Downing"),
        ("21", "Nottingham Forest", "Samba Sow"),
        ("26", "Blackburn",         "Darragh Lenihan"),  # fuzzy team name
        ("99", "Blackburn Rovers",  None),               # unknown jersey
    ]

    all_passed = True
    for jersey, team, expected in tests:
        result = lookup.find(jersey=jersey, team=team)
        ok     = result == expected
        status = "✓" if ok else "✗"
        if not ok:
            all_passed = False
        print(f"  {status} #{jersey:<3} {team:<22} → {result}  (expected: {expected})")

    print("\n── Color-based lookup (jersey #7, any team) ──")
    candidates = lookup.find_by_color(jersey="7", color="blue")
    for c in candidates:
        print(f"  #{7}  {c['player']:<22} ({c['team']})")

    print(f"\n{'✓ all tests passed!' if all_passed else '✗ some tests failed'}")