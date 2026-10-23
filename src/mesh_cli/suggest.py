"""'Did you mean' hints for mistyped commands, service names, config fields and file names."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


def closest(word: str, options: Iterable[str]) -> str | None:
    matches = difflib.get_close_matches(word, list(options), n=1, cutoff=0.6)
    return matches[0] if matches else None


def did_you_mean(word: str, options: Iterable[str]) -> str:
    """` Did you mean 'x'?` (with a leading space) or an empty string when nothing is close."""
    match = closest(word, options)
    return f" Did you mean '{match}'?" if match else ""
