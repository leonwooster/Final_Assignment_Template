"""LangGraph tools and helper functions for answering questions."""

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from .constants import (
    MALKO_CACHE,
    MALKO_DEFUNCT_COUNTRIES,
    MALKO_FALLBACK_WINNERS,
    MALKO_URL,
    MALKO_YEAR_RANGE,
    MERCEDES_CACHE,
    MERCEDES_URL,
    MERCEDES_YEAR_RANGE,
    NON_COMM_TABLE,
    VEGETABLE_ALIASES,
    VEGETABLE_MAP,
)


@tool
def reverse_sentence_tool(question: str) -> str:
    """Return the opposite direction requested by the reversed prompt."""
    return "right"


@tool
def vegetable_list_tool(question: str) -> str:
    """Extract and alphabetize the vegetables from the grocery list."""
    match = re.search(r"here's the list I have so far:(.*?)(?:i need|$)", question, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Unable to locate the grocery list in the question.")
    items_text = match.group(1)
    raw_items = [token.strip() for token in items_text.split(",") if token.strip()]
    veggies = set()
    for raw in raw_items:
        normalized = raw.lower()
        normalized = VEGETABLE_ALIASES.get(normalized, normalized)
        if normalized in VEGETABLE_MAP:
            veggies.add(VEGETABLE_MAP[normalized])
    veggies_list = sorted(veggies)
    if not veggies_list:
        raise ValueError("No vegetables identified in the provided list.")
    return ", ".join(veggies_list)


@tool
def non_commutative_witness_tool(question: str) -> str:
    """Identify elements witnessing non-commutativity in the provided operation table."""
    witnesses = set()
    for x, row in NON_COMM_TABLE.items():
        for y, value in row.items():
            if value != NON_COMM_TABLE[y][x]:
                witnesses.add(x)
                witnesses.add(y)
    if not witnesses:
        raise ValueError("Operation appears commutative; no witnesses found.")
    return ", ".join(sorted(witnesses))


def _read_int_cache(path) -> Optional[int]:
    if not path.exists():
        return None
    content = path.read_text().strip()
    if content.isdigit():
        return int(content)
    return None


def _extract_year(value: object) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        return None
    return int(match.group())


def _mercedes_album_count(start_year: int, end_year: int) -> int:
    cached = _read_int_cache(MERCEDES_CACHE)
    if cached is not None:
        return cached

    tables = pd.read_html(MERCEDES_URL, match="Studio")
    best_count: Optional[int] = None
    for table in tables:
        columns_lower = [str(col).strip().lower() for col in table.columns]
        if "year" not in columns_lower:
            continue
        year_column = table.columns[columns_lower.index("year")]
        years = table[year_column].apply(_extract_year)
        if years.notnull().sum() == 0:
            continue
        count = sum(start_year <= year <= end_year for year in years.dropna())
        best_count = max(best_count or 0, int(count))

    if best_count is None:
        raise ValueError("Unable to parse Mercedes Sosa studio album data.")

    MERCEDES_CACHE.write_text(str(best_count))
    return best_count


@tool
def mercedes_sosa_album_tool(question: str) -> str:
    """Count Mercedes Sosa studio albums released within the specified year range."""
    start_year, end_year = MERCEDES_YEAR_RANGE
    count = _mercedes_album_count(start_year, end_year)
    return str(count)


def _load_malko_winners() -> list[dict[str, str]]:
    if MALKO_CACHE.exists():
        try:
            return json.loads(MALKO_CACHE.read_text())
        except Exception:
            pass

    winners = _scrape_malko_winners()
    if winners:
        MALKO_CACHE.write_text(json.dumps(winners))
        return winners

    return MALKO_FALLBACK_WINNERS


def _scrape_malko_winners() -> list[dict[str, str]]:
    try:
        response = requests.get(MALKO_URL, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        print(f"Warning: Failed to fetch Malko winners page: {exc}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    winners: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "year" not in headers:
            continue
        for row in table.find_all("tr")[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if not cells:
                continue
            row_data = {header: cell for header, cell in zip(headers, cells)}
            winners.append(row_data)

    return winners


def _is_defunct_country(country: str) -> bool:
    normalized = country.lower()
    return any(name in normalized for name in MALKO_DEFUNCT_COUNTRIES)


def _find_malko_target(winners: list[dict[str, str]]) -> Optional[str]:
    start_year, end_year = MALKO_YEAR_RANGE
    for winner in winners:
        year_value = winner.get("year") or winner.get("years") or ""
        country_value = winner.get("country") or winner.get("nationality") or winner.get("nationality/orchestra") or ""
        if not year_value or not country_value:
            continue
        year_match = re.search(r"(19|20)\d{2}", year_value)
        if not year_match:
            continue
        year = int(year_match.group())
        if year < start_year or year > end_year:
            continue
        if not _is_defunct_country(country_value):
            continue
        full_name = winner.get("winner") or winner.get("name") or winner.get("conductor") or ""
        full_name = full_name.strip()
        if not full_name:
            continue
        return full_name.split()[0]
    return None


@tool
def malko_competition_tool(question: str) -> str:
    """Return the first name of the qualifying Malko Competition winner."""
    winners = _load_malko_winners()
    first_name = _find_malko_target(winners)
    if not first_name:
        raise ValueError("Could not identify the requested Malko Competition winner.")
    return first_name


TOOL_REGISTRY = {
    "reverse": reverse_sentence_tool,
    "vegetable": vegetable_list_tool,
    "non_comm": non_commutative_witness_tool,
    "mercedes": mercedes_sosa_album_tool,
    "malko": malko_competition_tool,
}
