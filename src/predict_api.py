from __future__ import annotations

from typing import Any, Dict


def predict_game(game_input: str, use_binned_intervals: bool = True) -> Dict[str, Any]:
    """
    Single public entrypoint used by app.py.

    Production runtime: uses compact sklearn models shipped in-repo.

    Model use-cases:
    - margin/spread/ML: ridge twohead (calibration-first)
    - game total: gbt twohead (small + stable)
    - team totals: derived from total+margin

    Returns the rich dict (status, bands80, normal, labels, text, etc).
    """
    from src.predict_from_gameid_v3_runtime import predict_from_game_id

    # `use_binned_intervals` kept for backwards compatibility; runtime predictor
    # already bakes in model-specific sigmas.
    _ = use_binned_intervals
    return predict_from_game_id(game_input)
