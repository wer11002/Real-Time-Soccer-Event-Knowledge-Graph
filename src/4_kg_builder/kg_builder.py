"""
kg_builder.py — Soccer EKG builder (RDF/OWL version)
─────────────────────────────────────────────────────

Two entry points:

1. CSV MODE (original, for testing with pre-scraped data)
       python kg_builder.py
       python kg_builder.py --fast

2. REAL-TIME MODE (for the pipeline)
       from kg_builder import ingest_matched_event
       ingest_matched_event(matched_event, match_name, ekg)

Output: data/kg_output/ekg.ttl (Turtle RDF)

What changed from dict-based version:
  - Underlying storage is now rdflib Graph (RDF triples)
  - T-Box loaded at startup (ontology)
  - A-Box populated incrementally (instance triples)
  - Save format is .ttl instead of .csv
  - Queries are SPARQL
"""

import csv
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

from rdflib import Literal, RDF, RDFS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from ekg_schema import EKG_Graph, EKG, INST


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent.parent.parent
DATA_DIR  = BASE_DIR / "data"
CSV_PATH  = DATA_DIR / "blackburn_forest_2019-10-01.csv"
OUT_DIR   = DATA_DIR / "kg_output"
TTL_PATH  = OUT_DIR / "ekg.ttl"


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_id(text: str) -> str:
    """'Joe Lolley' -> 'joe_lolley'"""
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")


def parse_time_min(time_str: str) -> float:
    """'45+2'' -> 47.0"""
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


def extract_assist(full_text: str) -> Optional[str]:
    m = re.search(r"Assisted by ([A-Z][a-zA-Z\s\-'\.]+?)(?:\.|with\b)", full_text or "")
    return m.group(1).strip() if m else None


# ═══════════════════════════════════════════════════════════════════════════
# A-BOX INSTANCE CREATORS (populate graph with instance triples)
# ═══════════════════════════════════════════════════════════════════════════

def get_or_create_match(match_name: str, ekg: EKG_Graph) -> str:
    """Create a Match instance if not already present. Returns match_id."""
    mid = normalize_id(match_name)
    if mid in ekg._seen_matches:
        return mid

    parts = match_name.split(" - ", 1)
    date  = parts[0].strip()
    match_uri = ekg.match_uri(mid)

    ekg.g.add((match_uri, RDF.type, EKG.Match))
    ekg.g.add((match_uri, RDFS.label, Literal(match_name)))
    ekg.g.add((match_uri, EKG.hasDate, Literal(date)))

    ekg._seen_matches.add(mid)
    return mid


def get_or_create_team(team_name: str, ekg: EKG_Graph) -> str:
    """Create a Team instance if not already present."""
    tid = normalize_id(team_name)
    if tid in ekg._seen_teams:
        return tid

    team_uri = ekg.team_uri(tid)
    ekg.g.add((team_uri, RDF.type, EKG.Team))
    ekg.g.add((team_uri, RDFS.label, Literal(team_name)))

    ekg._seen_teams.add(tid)
    return tid


def get_or_create_player(player_name: str, team_id: Optional[str], ekg: EKG_Graph) -> Tuple[str, bool]:
    """Create a Player instance if not already present. Returns (pid, is_new)."""
    pid = normalize_id(player_name)
    is_new = pid not in ekg._seen_players

    if is_new:
        player_uri = ekg.player_uri(pid)
        ekg.g.add((player_uri, RDF.type,   EKG.Player))
        ekg.g.add((player_uri, RDFS.label, Literal(player_name)))

        if team_id:
            team_uri = ekg.team_uri(team_id)
            ekg.g.add((player_uri, EKG.PLAYS_FOR, team_uri))

        ekg._seen_players.add(pid)

    return pid, is_new


def add_participates_in(player_id: str, match_id: str, ekg: EKG_Graph):
    """Ensure player --PARTICIPATED_IN--> match edge exists (only once)."""
    player_uri = ekg.player_uri(player_id)
    match_uri  = ekg.match_uri(match_id)
    # check if already exists
    if (player_uri, EKG.PARTICIPATED_IN, match_uri) not in ekg.g:
        ekg.g.add((player_uri, EKG.PARTICIPATED_IN, match_uri))


# ═══════════════════════════════════════════════════════════════════════════
# CORE: CREATE ONE EVENT NODE + ITS TRIPLES
# ═══════════════════════════════════════════════════════════════════════════

def _create_event_node(
    ekg         : EKG_Graph,
    match_id    : str,
    time_raw    : str,
    event_type  : str,
    player_id   : Optional[str],
    team_id     : Optional[str],
    full_text   : str = "",
    confidence  : Optional[float] = None,
    matched     : bool = True,
    last_event  : dict = None,
) -> Tuple[str, list]:
    """
    Create ONE Event instance + wire its edges.
    Returns (event_id, list_of_edge_descriptions_for_display).
    """
    ekg._event_count += 1
    event_id  = f"{ekg._event_count:04d}"
    event_uri = ekg.event_uri(event_id)

    # choose class: ActionEvent or CardEvent
    if event_type in ("YellowCard", "RedCard"):
        event_class = EKG.CardEvent
    else:
        event_class = EKG.ActionEvent

    # ── event instance + datatype properties ─────────────────────────────
    ekg.g.add((event_uri, RDF.type, event_class))
    ekg.g.add((event_uri, EKG.hasEventType, Literal(event_type)))
    ekg.g.add((event_uri, EKG.hasTime,      Literal(time_raw)))
    ekg.g.add((event_uri, EKG.hasTimeMin,   Literal(parse_time_min(time_raw))))
    ekg.g.add((event_uri, EKG.isMatched,    Literal(matched)))
    if full_text:
        ekg.g.add((event_uri, EKG.hasFullText, Literal(full_text)))
    if confidence is not None:
        ekg.g.add((event_uri, EKG.hasConfidence, Literal(float(confidence))))

    new_edges = []

    # event -[IN_MATCH]-> match
    match_uri = ekg.match_uri(match_id)
    ekg.g.add((event_uri, EKG.IN_MATCH, match_uri))
    new_edges.append(f"event_{event_id} --[IN_MATCH]--> {match_id}")

    # event -[PRECEDED_BY]-> previous event
    if last_event is not None and match_id in last_event:
        prev_id  = last_event[match_id]
        prev_uri = ekg.event_uri(prev_id)
        ekg.g.add((event_uri, EKG.PRECEDED_BY, prev_uri))
        new_edges.append(f"event_{event_id} --[PRECEDED_BY]--> event_{prev_id}")
    if last_event is not None:
        last_event[match_id] = event_id

    # team -[INVOLVED_IN]-> event
    if team_id:
        team_uri = ekg.team_uri(team_id)
        ekg.g.add((team_uri, EKG.INVOLVED_IN, event_uri))
        new_edges.append(f"{team_id} --[INVOLVED_IN]--> event_{event_id}")

    # player -[PERFORMED]-> event
    if player_id:
        player_uri = ekg.player_uri(player_id)
        ekg.g.add((player_uri, EKG.PERFORMED, event_uri))
        new_edges.append(f"{player_id} --[PERFORMED]--> event_{event_id}")

    return event_id, new_edges


# ═══════════════════════════════════════════════════════════════════════════
# INGEST — CSV MODE (for testing)
# ═══════════════════════════════════════════════════════════════════════════

def ingest_event(row: dict, ekg: EKG_Graph, last_event: dict) -> dict:
    """Process one CSV row. Card events become separate nodes (TRIGGERED edge)."""
    match_name  = row["Match"]
    team_name   = row["Team"]   if row["Team"]   != "None" else None
    time_raw    = row["Time"]
    player_name = row["Player"] if row["Player"] != "None" else None
    action_type = row["Action_Type"]
    full_text   = row["Full_Text"]
    yellow      = row["Yellow_Card"] == "1"
    red         = row["Red_Card"]    == "1"

    new_edges     = []
    is_new_player = False

    # nodes
    match_id = get_or_create_match(match_name, ekg)
    team_id  = get_or_create_team(team_name, ekg) if team_name else None

    player_id = None
    if player_name:
        player_id, is_new_player = get_or_create_player(player_name, team_id, ekg)
        if is_new_player and team_id:
            new_edges.append(f"{player_name} --[PLAYS_FOR]--> {team_name}")
        add_participates_in(player_id, match_id, ekg)

    # main action event
    main_event_id, edges = _create_event_node(
        ekg, match_id, time_raw,
        event_type = action_type,
        player_id  = player_id,
        team_id    = team_id,
        full_text  = full_text,
        matched    = True,
        last_event = last_event,
    )
    new_edges.extend(edges)

    # assist
    assist_name = extract_assist(full_text)
    if assist_name and assist_name != player_name:
        assist_pid, assist_new = get_or_create_player(assist_name, team_id, ekg)
        main_event_uri = ekg.event_uri(main_event_id)
        assist_uri     = ekg.player_uri(assist_pid)
        ekg.g.add((main_event_uri, EKG.ASSISTED_BY, assist_uri))
        new_edges.append(f"event_{main_event_id} --[ASSISTED_BY]--> {assist_name}")
        if assist_new:
            is_new_player = True

    # card event (first-class node)
    card_type = None
    if red:    card_type = "RedCard"
    elif yellow: card_type = "YellowCard"

    if card_type:
        card_event_id, card_edges = _create_event_node(
            ekg, match_id, time_raw,
            event_type = card_type,
            player_id  = player_id,
            team_id    = team_id,
            full_text  = f"{card_type} following {action_type}",
            matched    = True,
            last_event = last_event,
        )
        new_edges.extend(card_edges)

        # foul event -[TRIGGERED]-> card event
        main_event_uri = ekg.event_uri(main_event_id)
        card_event_uri = ekg.event_uri(card_event_id)
        ekg.g.add((main_event_uri, EKG.TRIGGERED, card_event_uri))
        new_edges.append(f"event_{main_event_id} --[TRIGGERED]--> event_{card_event_id} ({card_type})")

    return {
        "time"          : time_raw,
        "player"        : player_name,
        "action_type"   : action_type,
        "team"          : team_name,
        "is_new_player" : is_new_player,
        "yellow"        : yellow,
        "red"           : red,
        "card"          : card_type,
        "new_edges"     : new_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# INGEST — REAL-TIME MODE (for the pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def ingest_matched_event(matched, match_name: str, ekg: EKG_Graph,
                         last_event: dict) -> dict:
    """Ingest one MatchedEvent object from align.py into the RDF graph."""
    action_type = matched.action
    time_raw    = matched.gametime or f"{matched.video_time/60:.1f}'"
    full_text   = matched.espn_text or ""
    confidence  = matched.confidence

    new_edges     = []
    is_new_player = False

    match_id = get_or_create_match(match_name, ekg)

    if matched.matched:
        team_id = get_or_create_team(matched.team, ekg) if matched.team else None
        player_id, is_new_player = get_or_create_player(matched.player, team_id, ekg)
        if is_new_player and team_id:
            new_edges.append(f"{matched.player} --[PLAYS_FOR]--> {matched.team}")
        add_participates_in(player_id, match_id, ekg)
    else:
        player_id = None
        team_id   = None

    event_id, edges = _create_event_node(
        ekg, match_id, time_raw,
        event_type = action_type,
        player_id  = player_id,
        team_id    = team_id,
        full_text  = full_text,
        confidence = confidence,
        matched    = matched.matched,
        last_event = last_event,
    )
    new_edges.extend(edges)

    # parse card from ESPN text if matched
    card_type = None
    if matched.matched and full_text:
        if re.search(r"\bred card\b", full_text, re.IGNORECASE):
            card_type = "RedCard"
        elif re.search(r"\byellow card\b|booked", full_text, re.IGNORECASE):
            card_type = "YellowCard"

    if card_type:
        card_event_id, card_edges = _create_event_node(
            ekg, match_id, time_raw,
            event_type = card_type,
            player_id  = player_id,
            team_id    = team_id,
            full_text  = f"{card_type} following {action_type}",
            matched    = True,
            last_event = last_event,
        )
        new_edges.extend(card_edges)
        main_uri = ekg.event_uri(event_id)
        card_uri = ekg.event_uri(card_event_id)
        ekg.g.add((main_uri, EKG.TRIGGERED, card_uri))
        new_edges.append(f"event_{event_id} --[TRIGGERED]--> event_{card_event_id} ({card_type})")

    return {
        "time"          : time_raw,
        "player"        : matched.player if matched.matched else "UNKNOWN",
        "action_type"   : action_type,
        "team"          : matched.team,
        "is_new_player" : is_new_player,
        "matched"       : matched.matched,
        "card"          : card_type,
        "new_edges"     : new_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

ACTION_ICONS = {
    "Goal"        : "GOAL",
    "Shot"        : "shot",
    "Foul"        : "foul",
    "Corner"      : "corner",
    "Offside"     : "offside",
    "FreeKick"    : "free kick",
    "Substitution": "sub",
    "YellowCard"  : "🟨",
    "RedCard"     : "🟥",
    "Penalty"     : "penalty",
}


def print_event(res: dict, ekg: EKG_Graph):
    icon   = ACTION_ICONS.get(res["action_type"], res["action_type"])
    player = res.get("player") or "(no player)"
    team   = res.get("team")   or "—"
    tag    = " << NEW PLAYER >>" if res.get("is_new_player") else ""
    card   = f" [{res.get('card')}]" if res.get("card") else ""

    print(f"\n[{res['time']:>6}]  {icon:<12}  {player}{card}{tag}")
    print(f"          Team: {team}")
    for edge in res["new_edges"]:
        print(f"          + {edge}")
    print(f"          KG -> {ekg.stats()}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV MODE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def stream_simulation(csv_path: Path, scale: float = 1.0, fast: bool = False):
    with open(csv_path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f),
                      key=lambda r: parse_time_min(r.get("Time", "0")))

    ekg        = EKG_Graph()
    last_event = {}
    prev_min   = None

    match_name = rows[0]["Match"] if rows else "Unknown"
    mode_label = "FAST (no delay)" if fast else f"scale={scale}s/match-min"

    print(f"\n{'='*62}")
    print(f"  MATCH : {match_name}")
    print(f"  EVENTS: {len(rows)}  |  MODE: {mode_label}")
    print(f"  T-Box : {len(ekg.g)} triples at start")
    print(f"{'='*62}")

    for i, row in enumerate(rows):
        curr_min = parse_time_min(row.get("Time", "0"))

        if not fast and prev_min is not None:
            gap  = curr_min - prev_min
            wait = max(0.1, min(5.0, gap * scale))
            time.sleep(wait)

        prev_min = curr_min
        result   = ingest_event(row, ekg, last_event)
        print_event(result, ekg)

        if (i + 1) % 20 == 0:
            ekg.save(TTL_PATH)

    ekg.save(TTL_PATH)

    print(f"\n{'='*62}")
    print(f"  MATCH ENDED")
    print(f"  Final EKG : {ekg.stats()}")
    yellow = ekg.events_by_type("YellowCard")
    red    = ekg.events_by_type("RedCard")
    print(f"  Yellow cards : {len(yellow)}")
    print(f"  Red cards    : {len(red)}")
    print(f"  Saved to     : {TTL_PATH}")
    print(f"{'='*62}\n")

    return ekg


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer EKG builder (RDF)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fast",  action="store_true")
    args = parser.parse_args()

    stream_simulation(CSV_PATH, scale=args.scale, fast=args.fast)