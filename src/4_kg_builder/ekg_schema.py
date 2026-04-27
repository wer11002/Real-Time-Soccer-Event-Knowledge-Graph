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

Serialization: Turtle (.ttl) by default.

Quick test:
    python ekg_schema.py
"""

from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD


# ═══════════════════════════════════════════════════════════════════════════
# NAMESPACES
# ═══════════════════════════════════════════════════════════════════════════

# Our project namespace
EKG = Namespace("http://soccerekg.org/ontology#")

# Instance namespace (for actual players, events, matches etc.)
INST = Namespace("http://soccerekg.org/data#")


# ═══════════════════════════════════════════════════════════════════════════
# CLASSES (T-Box)
# ═══════════════════════════════════════════════════════════════════════════

CLASSES = {
    "Player"      : EKG.Player,
    "Team"        : EKG.Team,
    "Match"       : EKG.Match,
    "Event"       : EKG.Event,
    "ActionEvent" : EKG.ActionEvent,   # Shot, Goal, Foul, Corner...
    "CardEvent"   : EKG.CardEvent,     # YellowCard, RedCard
}


# ═══════════════════════════════════════════════════════════════════════════
# OBJECT PROPERTIES (edges between instances)
# ═══════════════════════════════════════════════════════════════════════════

OBJECT_PROPERTIES = {
    # ── Ren-ev: entity → event ───────────────────────────────────────────
    "PERFORMED"       : (EKG.PERFORMED,       EKG.Player, EKG.Event),
    "INVOLVED_IN"     : (EKG.INVOLVED_IN,     EKG.Team,   EKG.Event),
    "ASSISTED_BY"     : (EKG.ASSISTED_BY,     EKG.Event,  EKG.Player),

    # ── Ren-en: entity → entity ──────────────────────────────────────────
    "PLAYS_FOR"       : (EKG.PLAYS_FOR,       EKG.Player, EKG.Team),
    "PARTICIPATED_IN" : (EKG.PARTICIPATED_IN, EKG.Player, EKG.Match),
    "IN_MATCH"        : (EKG.IN_MATCH,        EKG.Event,  EKG.Match),

    # ── Rev-ev: event → event ────────────────────────────────────────────
    "PRECEDED_BY"     : (EKG.PRECEDED_BY,     EKG.Event,       EKG.Event),
    "TRIGGERED"       : (EKG.TRIGGERED,       EKG.ActionEvent, EKG.CardEvent),
}


# ═══════════════════════════════════════════════════════════════════════════
# DATATYPE PROPERTIES (attributes on instances)
# ═══════════════════════════════════════════════════════════════════════════

DATATYPE_PROPERTIES = {
    "hasTime"       : XSD.string,   # "9'" or "45+2'"
    "hasTimeMin"    : XSD.float,    # 9.0
    "hasEventType"  : XSD.string,   # "Shot", "YellowCard"
    "hasConfidence" : XSD.float,    # 0.87
    "isMatched"     : XSD.boolean,  # True/False
    "hasFullText"   : XSD.string,   # commentary text
    "hasDate"       : XSD.string,   # "2019-10-01"
}


# ═══════════════════════════════════════════════════════════════════════════
# T-BOX BUILDER — called once at pipeline startup
# ═══════════════════════════════════════════════════════════════════════════

def build_tbox(g: Graph):
    """
    Populate the graph with the T-Box (ontology definitions).
    This is called once at the start — never changes after that.
    """

    # bind prefixes for pretty Turtle output
    g.bind("ekg",  EKG)
    g.bind("data", INST)
    g.bind("owl",  OWL)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # ── declare classes ──────────────────────────────────────────────────
    for name, uri in CLASSES.items():
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.label, Literal(name)))

    # ActionEvent and CardEvent are subclasses of Event
    g.add((EKG.ActionEvent, RDFS.subClassOf, EKG.Event))
    g.add((EKG.CardEvent,   RDFS.subClassOf, EKG.Event))

    # ── declare object properties with domain/range ──────────────────────
    for name, (uri, domain, range_) in OBJECT_PROPERTIES.items():
        g.add((uri, RDF.type,    OWL.ObjectProperty))
        g.add((uri, RDFS.label,  Literal(name)))
        g.add((uri, RDFS.domain, domain))
        g.add((uri, RDFS.range,  range_))

    # ── declare datatype properties ──────────────────────────────────────
    for name, range_ in DATATYPE_PROPERTIES.items():
        uri = EKG[name]
        g.add((uri, RDF.type,   OWL.DatatypeProperty))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, RDFS.range, range_))

    return g


# ═══════════════════════════════════════════════════════════════════════════
# EKG CONTAINER — holds the live graph + convenient query helpers
# ═══════════════════════════════════════════════════════════════════════════

class EKG_Graph:
    """
    Wraps an rdflib Graph with the T-Box pre-loaded.
    Provides helper methods for the pipeline + SPARQL queries.
    """

    def __init__(self):
        self.g = Graph()
        build_tbox(self.g)

        # track seen instances to avoid redundant work
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

    # ── stats ──────────────────────────────────────────────────────────────

    def stats(self) -> str:
        """Short one-line summary."""
        n_triples = len(self.g)
        n_players = len(self._seen_players)
        n_teams   = len(self._seen_teams)
        n_events  = self._event_count
        return (
            f"{n_players} players | "
            f"{n_events} events | "
            f"{n_teams} teams | "
            f"{n_triples} triples"
        )

    def triple_count(self) -> int:
        return len(self.g)

    # ── query helpers (equivalent of the old ones) ────────────────────────

    def events_by_type(self, event_type: str) -> list:
        """Return all event URIs with the given hasEventType."""
        q = """
        SELECT ?e WHERE {
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?etype)
        }
        """
        return [row[0] for row in self.g.query(q, initBindings={"etype": Literal(event_type)})]

    def count_cards(self, player_id: str, color: str = "YellowCard") -> int:
        """Count how many yellow or red cards a player got (SPARQL query)."""
        q = """
        SELECT (COUNT(?e) AS ?c) WHERE {
            ?p ekg:PERFORMED ?e .
            ?e a ekg:CardEvent .
            ?e ekg:hasEventType ?t .
            FILTER (STR(?t) = ?color)
        }
        """
        player_uri = self.player_uri(player_id)
        result = self.g.query(q, initBindings={
            "p"     : player_uri,
            "color" : Literal(color),
        })
        for row in result:
            return int(row[0])
        return 0

    def events_for_player(self, player_id: str) -> list:
        """Return all event URIs performed by a player."""
        q = "SELECT ?e WHERE { ?p ekg:PERFORMED ?e . }"
        player_uri = self.player_uri(player_id)
        return [row[0] for row in self.g.query(q, initBindings={"p": player_uri})]

    # ── save to disk ───────────────────────────────────────────────────────

    def save(self, out_path: Path, format: str = "turtle"):
        """Save the graph to disk. Default Turtle (.ttl)."""
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

    # show the T-Box
    print("── T-Box classes declared ──")
    for cls_name, uri in CLASSES.items():
        print(f"  {cls_name:<14} {uri}")

    print("\n── T-Box object properties declared ──")
    for name, (uri, domain, range_) in OBJECT_PROPERTIES.items():
        d_label = domain.split("#")[-1]
        r_label = range_.split("#")[-1]
        print(f"  {name:<18} {d_label} → {r_label}")

    print("\n── T-Box datatype properties declared ──")
    for name, range_ in DATATYPE_PROPERTIES.items():
        r_label = range_.split("#")[-1]
        print(f"  {name:<15} → {r_label}")

    # try adding a few manual triples to see how A-Box works
    print("\n── Adding test A-Box triples ──")
    lolley = ekg.player_uri("joe_lolley")
    team   = ekg.team_uri("nottingham_forest")
    event  = ekg.event_uri("0001")

    ekg.g.add((lolley, RDF.type, EKG.Player))
    ekg.g.add((lolley, RDFS.label, Literal("Joe Lolley")))
    ekg.g.add((team,   RDF.type, EKG.Team))
    ekg.g.add((team,   RDFS.label, Literal("Nottingham Forest")))
    ekg.g.add((event,  RDF.type, EKG.ActionEvent))
    ekg.g.add((event,  EKG.hasEventType, Literal("Shot")))
    ekg.g.add((event,  EKG.hasTime,      Literal("1'")))

    ekg.g.add((lolley, EKG.PLAYS_FOR,  team))
    ekg.g.add((lolley, EKG.PERFORMED,  event))

    print(f"  {ekg.stats()}")
    print(f"  Total triples: {ekg.triple_count()}")

    # save to disk
    out_path = Path("data/kg_output/test_ekg.ttl")
    ekg.save(out_path)
    print(f"\n  Saved Turtle to: {out_path}")

    # show the first few lines of the output
    print("\n── First lines of Turtle output ──")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i > 30:
                print("  ...")
                break
            print(f"  {line.rstrip()}")

    print("\n✓ all good!")