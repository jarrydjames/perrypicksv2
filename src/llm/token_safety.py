from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class TokenBudgetResult:
    tokens: int
    truncated: bool
    content: str


def count_tokens(text: str, *, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a given text.

    Uses tiktoken when available; falls back to a conservative heuristic.

    NOTE: model choice affects encoding.
    """

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Heuristic fallback: ~4 chars per token in English-ish text.
        return max(1, len(text) // 4)


def truncate_to_token_budget(
    text: str,
    *,
    max_tokens: int,
    model: str = "gpt-4o-mini",
    suffix: str = "\n\n[TRUNCATED]",
) -> TokenBudgetResult:
    """Truncate a string to fit within max_tokens.

    This is the simplest (and safest) pre-flight guard to avoid API 400 errors.
    If you want smarter trimming, do summarization in a *separate* call.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    t = count_tokens(text, model=model)
    if t <= max_tokens:
        return TokenBudgetResult(tokens=t, truncated=False, content=text)

    # Binary search truncation by character length.
    lo, hi = 0, len(text)
    best = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid] + suffix
        tc = count_tokens(candidate, model=model)
        if tc <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    return TokenBudgetResult(tokens=count_tokens(best, model=model), truncated=True, content=best)
