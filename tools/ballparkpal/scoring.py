"""Ballpark Pal validation and overlay scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverlayRule:
    field: str
    neutral: float
    scale: float
    weight: float
    favor_when_above: bool = True


BALLPARKPAL_MODEL_BLEND_WEIGHT: float = 0.10
BALLPARKPAL_BALLPARK_BLEND_WEIGHT: float = 0.90

BALLPARKPAL_OVERLAY_RULES: tuple[OverlayRule, ...] = (
    OverlayRule(
        field="ballparkpal_home_run_probability",
        neutral=0.10,
        scale=1.0,
        weight=100.0,
        favor_when_above=True,
    ),
)

BALLPARKPAL_HOME_RUN_NEUTRAL_SCORE: float = 10.0


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _bounded_centered_delta(value: float, *, neutral: float, scale: float, favor_when_above: bool) -> float:
    if scale <= 0:
        return 0.0
    raw = (value - neutral) / scale
    if not favor_when_above:
        raw = -raw
    return max(-1.0, min(1.0, raw))


def compute_ballparkpal_overlay(row: Mapping[str, Any], *, model_score: float | None = None) -> dict[str, Any]:
    rule = BALLPARKPAL_OVERLAY_RULES[0]
    factor_details: dict[str, Any] = {rule.field: None}
    home_run_probability = _to_float(row.get(rule.field))
    model_score_value = _to_float(model_score if model_score is not None else row.get("predicted_hr_score"))

    if home_run_probability is None:
        signed_score = 0.0
        display_score = 50.0
        adjusted_score = model_score_value if model_score_value is not None else 50.0
    else:
        display_score = max(0.0, min(100.0, home_run_probability * 100.0))
        signed_score = display_score - BALLPARKPAL_HOME_RUN_NEUTRAL_SCORE
        factor_details[rule.field] = {
            "value": home_run_probability,
            "neutral": rule.neutral,
            "scale": rule.scale,
            "weight": rule.weight,
            "score": display_score,
            "delta": signed_score,
        }
        if model_score_value is None:
            adjusted_score = display_score
        else:
            adjusted_score = (
                (BALLPARKPAL_MODEL_BLEND_WEIGHT * model_score_value)
                + (BALLPARKPAL_BALLPARK_BLEND_WEIGHT * display_score)
            )

    direction = "neutral"
    if signed_score > 0:
        direction = "favorable"
    elif signed_score < 0:
        direction = "unfavorable"

    return {
        "ballparkpal_overlay_signed_score": round(signed_score, 3),
        "ballparkpal_overlay_display_score": round(max(0.0, min(100.0, display_score)), 1),
        "ballparkpal_overlay_adjusted_score": round(adjusted_score, 1) if adjusted_score is not None else None,
        "ballparkpal_overlay_direction": direction,
        "ballparkpal_overlay_model_score": round(model_score_value, 1) if model_score_value is not None else None,
        "ballparkpal_overlay_blend_weights": {
            "model": BALLPARKPAL_MODEL_BLEND_WEIGHT,
            "ballpark": BALLPARKPAL_BALLPARK_BLEND_WEIGHT,
        },
        "ballparkpal_overlay_factor_details": factor_details,
    }


def apply_ballparkpal_overlay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    annotated = frame.copy()
    overlay_rows = annotated.apply(lambda row: pd.Series(compute_ballparkpal_overlay(row.to_dict())), axis=1)
    for column in overlay_rows.columns:
        annotated[column] = overlay_rows[column]
    return annotated
