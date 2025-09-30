"""Shared constants for the agent."""

from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
