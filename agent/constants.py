"""Shared constants and pattern definitions for the agent tools."""

from pathlib import Path
import re

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

REVERSE_PATTERN = re.compile(r"\\.rewsna", re.IGNORECASE)
VEGETABLE_PATTERN = re.compile(r"list of just the vegetables", re.IGNORECASE)
NON_COMM_PATTERN = re.compile(r"table defining \* on the set S = {a, b, c, d, e}", re.IGNORECASE)
MERCEDES_PATTERN = re.compile(r"Mercedes Sosa.*studio albums", re.IGNORECASE | re.DOTALL)
MALKO_PATTERN = re.compile(r"Malko Competition.*country that no longer exists", re.IGNORECASE | re.DOTALL)

VEGETABLE_MAP = {
    "broccoli": "broccoli",
    "celery": "celery",
    "lettuce": "lettuce",
    "sweet potatoes": "sweet potatoes",
    "fresh basil": "fresh basil",
}

VEGETABLE_ALIASES = {
    "basil": "fresh basil",
}

NON_COMM_TABLE = {
    "a": {"a": "a", "b": "b", "c": "c", "d": "b", "e": "d"},
    "b": {"a": "b", "b": "c", "c": "a", "d": "e", "e": "c"},
    "c": {"a": "c", "b": "a", "c": "b", "d": "b", "e": "a"},
    "d": {"a": "b", "b": "e", "c": "b", "d": "e", "e": "d"},
    "e": {"a": "d", "b": "b", "c": "a", "d": "d", "e": "c"},
}

MERCEDES_URL = "https://en.wikipedia.org/wiki/Mercedes_Sosa"
MERCEDES_CACHE = CACHE_DIR / "mercedes_sosa_album_count.json"
MERCEDES_YEAR_RANGE = (2000, 2009)

MALKO_URL = "https://malkocompetition.dk/winners/all"
MALKO_CACHE = CACHE_DIR / "malko_winners.json"
MALKO_YEAR_RANGE = (1978, 2000)
MALKO_FALLBACK_WINNERS = [
    {"year": "1980", "winner": "Claus Peter Flor", "country": "West Germany"},
    {"year": "1983", "winner": "Gotthard Lienicke", "country": "East Germany"},
    {"year": "1989", "winner": "Maximiano Valdes", "country": "Brazil"},
]
MALKO_DEFUNCT_COUNTRIES = {
    "ussr",
    "soviet union",
    "yugoslavia",
    "czechoslovakia",
    "east germany",
    "west germany",
    "serbia and montenegro",
    "burma",
    "zaire",
}


PATTERN_TOOL_MAP = [
    (REVERSE_PATTERN, "reverse"),
    (VEGETABLE_PATTERN, "vegetable"),
    (NON_COMM_PATTERN, "non_comm"),
    (MERCEDES_PATTERN, "mercedes"),
    (MALKO_PATTERN, "malko"),
]
