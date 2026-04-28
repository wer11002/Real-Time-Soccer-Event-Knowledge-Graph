"""
ekg_schema.py — RDF/OWL Schema for Soccer Event Knowledge Graph
────────────────────────────────────────────────────────────────

Defines the ontology (T-Box) and provides the EKG container (A-Box holder).

T-Box: Classes and properties of the soccer domain
   Classes    : Player, Team, Match, Event, ActionEvent, CardEvent
   Properties : PERFORMED, PLAYS_FOR, PARTICIPATED_IN, IN_MATCH,
                PRECEDED_BY, TRIGGERED, ASSISTED_BY, INVOLVED_IN

A-Box: Instance data populated incrementally during the pipeline
   (added by kg_builder.py as events stream in)

TKG layer: validFrom / validUntil on PLAYS_FOR edges
VLM layer: hasDescription / hasJersey on Event nodes

Serialization: Turtle (.ttl) by default.

Quick test:
    python ekg_schema.py
"""

from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD


# ═══════════════════════════════════════════════════════════════════════════
# NAMESPACES
# ═══════════════════════════════════════════════════════════════════════════

EKG  = Namespace("http://soccerekg.org/ontology#")
INST = Namespace("http://soccerekg.org/data#")


# ═══════════════════════════════════════════════════════════════════════════
# CLASSES (T-Box)
# ═══════════════════════════════════════════════════════════════════════════

CLASSES = {
    "Player"      : EKG.Player,
    "Team"        : EKG.Team,
    "Match"       : EKG.Match,
    "Event"       : EKG.Event,
    "ActionEvent" : EKG.ActionEvent,
    "CardEvent"   : EKG.CardEvent,
}


# ═══════════════════════════════════════════════════════════════════════════
# OBJECT PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

OBJECT_PROPERTIES = {
    # Ren-ev: entity → event
    "PERFORMED"       : (EKG.PERFORMED,       EKG.Player,      EKG.Event),
    "INVOLVED_IN"     : (EKG.INVOLVED_IN,     EKG.Team,        EKG.Event),
    "ASSISTED_BY"     : (EKG.ASSISTED_BY,     EKG.Event,       EKG.Player),
    # Ren-en: entity → entity
    "PLAYS_FOR"       : (EKG.PLAYS_FOR,       EKG.Player,      EKG.Team),
    "PARTICIPATED_IN" : (EKG.PARTICIPATED_IN, EKG.Player,      EKG.Match),
    "IN_MATCH"        : (EKG.IN_MATCH,        EKG.Event,       EKG.Match),
    # Rev-ev: event → event
    "PRECEDED_BY"     : (EKG.PRECEDED_BY,     EKG.Event,       EKG.Event),
    "TRIGGERED"       : (EKG.TRIGGERED,       EKG.ActionEvent, EKG.CardEvent),
}


# ═══════════════════════════════════════════════════════════════════════════
# DATATYPE PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

DATATYPE_PROPERTIES = {
    # ── existing ──────────────────────────────────────────────────────────
    "hasTime"        : XSD.string,    # "9'" or "45+2'"
    "hasTimeMin"     : XSD.float,     # 9.0
    "hasEventType"   : XSD.string,    # "Shot", "YellowCard"
    "hasConfidence"  : XSD.float,     # 0.87
    "isMatched"      : XSD.boolean,   # True/False (gray node if False)
    "hasFullText"    : XSD.string,    # commentary text from ESPN
    "hasDate"        : XSD.string,    # "2019-10-01"

    # ── TKG layer (temporal validity on PLAYS_FOR edges) ──────────────────
    "validFrom"      : XSD.date,      # when player joined team
    "validUntil"     : XSD.date,      # when player left team

    # ── VLM layer (from Qwen2-VL output) ─────────────────────────────────
    "hasDescription" : XSD.string,    # "Player #7 takes a right-footed shot..."
    "hasJersey"      : XSD.string,    # "7" — jersey number read by VLM
}


# ═══════════════════════════════════════════════════════════════════════════
# T-BOX BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_tbox(g: Graph):
    """Populate the graph with T-Box definitions. Called once at startup."""

    g.bind("ekg",  EKG)
    g.bind("data", INST)
    g.bind("owl",  OWL)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # classes
    for name, uri in CLASSES.items():
        g.add((uri, RDF.type,   OWL.Class))
        g.add((uri, RDFS.label, Literal(name)))

    g.add((EKG.ActionEvent, RDFS.subClassOf, EKG.Event))
    g.add((EKG.CardEvent,   RDFS.subClassOf, EKG.Event))

    # object properties
    for name, (uri, domain, range_) in OBJECT_PROPERTIES.items():
        g.add((uri, RDF.type,    OWL.ObjectProperty))
        g.add((uri, RDFS.label,  Literal(name)))
        g.add((uri, RDFS.domain, domain))
        g.add((uri, RDFS.range,  range_))

    # datatype properties
    for name, range_ in DATATYPE_PROPERTIES.items():
        uri = EKG[name]
        g.add((uri, RDF.type,   OWL.DatatypeProperty))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, RDFS.range, range_))

    return g


# ═══════════════════════════════════════════════════════════════════════════
# EKG CONTAINER
# ═══════════════════════════════════════════════════════════════════════════

class EKG_Graph:
    """Wraps an rdflib Graph with T-Box pre-loaded. Grows A-Box in real-time."""

    def __init__(self):
        self.g = Graph()
        build_tbox(self.g)
        self._seen_players : set = set()
        self._seen_teams   : set = set()
        self._seen_matches : set = set()
        self._event_count  : int = 0

    # ── URI helpers ────────────────────────────────────────────────────────

    @staticmethod
    def player_uri(player_id: str) -> URIRef:
        return INST[f"player_{player_id}"]

    @staticmethod
    def team_uri(team_id: str) -> URIRef:
        return INST[f"team_{team_id}"]

    @staticmethod
    def match_uri(match_id: str) -> URIRef:
        return INST[f"match_{match_id}"]

    @staticmethod
    def event_uri(event_id: str) -> URIRef:
        return INST[f"event_{event_id}"]

    @staticmethod
    def plays_for_uri(player_id: str, team_id: str, date: str) -> URIRef:
        """URI for a time-bounded PLAYS_FOR edge (TKG reification)."""
        return INST[f"plays_for_{player_id}_{team_id}_{date}"]

    # ── stats ──────────────────────────────────────────────────────────────

    def stats(self) -> str:
        return (
            f"{len(self._seen_players)} players | "
            f"{self._event_count} events | "
            f"{len(self._seen_teams)} teams | "
            f"{len(self.g)} triples"
        )

    def triple_count(self) -> int:
        return len(self.g)

    # ── SPARQL query helpers ───────────────────────────────────────────────

    def events_by_type(self, event_type: str) -> list:
        q = """
        SELECT ?e WHERE {
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?etype)
        }
        """
        return [row[0] for row in self.g.query(
            q, initBindings={"etype": Literal(event_type)})]

    def count_cards(self, player_id: str, color: str = "YellowCard") -> int:
        q = """
        SELECT (COUNT(?e) AS ?c) WHERE {
            ?p ekg:PERFORMED ?e .
            ?e a ekg:CardEvent .
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?color)
        }
        """
        result = self.g.query(q, initBindings={
            "p"     : self.player_uri(player_id),
            "color" : Literal(color),
        })
        for row in result:
            return int(row[0])
        return 0

    def events_for_player(self, player_id: str) -> list:
        q = "SELECT ?e WHERE { ?p ekg:PERFORMED ?e . }"
        return [row[0] for row in self.g.query(
            q, initBindings={"p": self.player_uri(player_id)})]

    def player_team_at(self, player_id: str, date: str) -> list:
        """
        TKG query: which team was a player on at a given date?
        Uses validFrom / validUntil on PLAYS_FOR edges.
        """
        q = """
        SELECT ?team WHERE {
            ?edge ekg:subject  ?p .
            ?edge ekg:object   ?team .
            ?edge ekg:validFrom  ?from .
            OPTIONAL { ?edge ekg:validUntil ?until }
            FILTER (?from <= ?date)
            FILTER (!BOUND(?until) || ?until >= ?date)
        }
        """
        return [row[0] for row in self.g.query(q, initBindings={
            "p"    : self.player_uri(player_id),
            "date" : Literal(date, datatype=XSD.date),
        })]

    # ── save ───────────────────────────────────────────────────────────────

    def save(self, out_path: Path, format: str = "turtle"):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.g.serialize(destination=str(out_path), format=format)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("─── ekg_schema.py self-test ───\n")

    ekg = EKG_Graph()
    print(f"T-Box loaded: {len(ekg.g)} triples\n")

    print("── T-Box classes ──")
    for name, uri in CLASSES.items():
        print(f"  {name:<14} {uri}")

    print("\n── T-Box object properties ──")
    for name, (uri, domain, range_) in OBJECT_PROPERTIES.items():
        print(f"  {name:<18} {domain.split('#')[-1]} → {range_.split('#')[-1]}")

    print("\n── T-Box datatype properties ──")
    for name, range_ in DATATYPE_PROPERTIES.items():
        tag = ""
        if name in ("validFrom", "validUntil"):
            tag = "  ← TKG"
        elif name in ("hasDescription", "hasJersey"):
            tag = "  ← VLM"
        print(f"  {name:<16} → {range_.split('#')[-1]}{tag}")

    print("\n── Test A-Box (with TKG + VLM triples) ──")
    from rdflib import RDF, RDFS

    lolley  = ekg.player_uri("joe_lolley")
    team    = ekg.team_uri("nottingham_forest")
    event   = ekg.event_uri("0001")
    edge    = ekg.plays_for_uri("joe_lolley", "nottingham_forest", "2019-10-01")

    # player
    ekg.g.add((lolley, RDF.type,   EKG.Player))
    ekg.g.add((lolley, RDFS.label, Literal("Joe Lolley")))

    # team
    ekg.g.add((team, RDF.type,   EKG.Team))
    ekg.g.add((team, RDFS.label, Literal("Nottingham Forest")))

    # TKG edge: PLAYS_FOR with validFrom
    ekg.g.add((edge, RDF.type,        EKG.PLAYS_FOR))
    ekg.g.add((edge, EKG.subject,     lolley))
    ekg.g.add((edge, EKG.object,      team))
    ekg.g.add((edge, EKG.validFrom,   Literal("2017-07-01", datatype=XSD.date)))
    ekg.g.add((edge, EKG.validUntil,  Literal("2021-06-30", datatype=XSD.date)))

    # event with VLM description + jersey
    ekg.g.add((event, RDF.type,           EKG.ActionEvent))
    ekg.g.add((event, EKG.hasEventType,   Literal("Shot")))
    ekg.g.add((event, EKG.hasTime,        Literal("1'")))
    ekg.g.add((event, EKG.hasJersey,      Literal("23")))
    ekg.g.add((event, EKG.hasDescription, Literal(
        "Player #23 in red kit takes a left-footed shot from the centre of the box")))

    ekg.g.add((lolley, EKG.PERFORMED, event))

    print(f"  {ekg.stats()}")

    out = Path("data/kg_output/test_ekg.ttl")
    ekg.save(out)
    print(f"  Saved to: {out}")

    print("\n── Sample Turtle output ──")
    with open(out) as f:
        for i, line in enumerate(f):
            if i > 35: print("  ..."); break
            print(f"  {line.rstrip()}")

    print("\n✓ all good!")