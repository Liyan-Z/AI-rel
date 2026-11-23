"""Extract the chosen option from a model reply. Failures are kept (as None) and reported, never guessed."""
from __future__ import annotations

import re

ARABIC_LETTERS = {"أ": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D", "هـ": "E", "ه": "E"}
_PAT = re.compile(r"(?:answer|الإجابة|الاجابة)?\s*(?:is|هي|:)?\s*\(?\b([A-Ea-e])\b\)?", re.I)


def parse_choice(text: str, labels: list[str]) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    first = t[0]
    if first in ARABIC_LETTERS and (len(t) == 1 or not t[1].isalpha()):
        c = ARABIC_LETTERS[first]
        return c if c in labels else None
    m = re.match(r"^\(?([A-Ea-e])\)?(?:[\s\.\):,]|$)", t)
    if m and m.group(1).upper() in labels:
        rest = t[m.end(1):]
        if re.match(r"^\)?\s*(?:or|/|,|and)\s*\(?[A-Ea-e]\b", rest, re.I):
            return None  # hedged ("A or B"): count as a parse failure, don't guess
        return m.group(1).upper()
    found = {x.upper() for x in re.findall(r"\b([A-E])\b", t[:80])}
    found &= set(labels)
    return found.pop() if len(found) == 1 else None
