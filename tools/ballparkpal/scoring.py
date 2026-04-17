"""Fixed-weight Ballpark Pal validation and overlay scoring."""

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


BALLPARKPAL_OVERLAY_RULES: tuple[OverlayRule, ...] = (
    OverlayRule(field="ballparkpal_home_run_probability", neutral=0.10, scale=0.05, weight=12.0, favor_when_above=True),
    OverlayRule(field="ballparkpal_hit_probability", neutral=0.70, scale=0.10, weight=6.0, favor_when_above=True),
    OverlayRule(field="ballparkpal_runs_allowed", neutral=4.5, scale=1.0, weight=8.0, favor_when_above=True),
    OverlayRule(field="ballparkpal_home_runs_allowed", neutral=0.80, scale=0.35, weight=4.0, favor_when_above=True),
)


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
    signed_score = 0.0
    factor_details: dict[str, Any] = {}
    for rule in BALLPARKPAL_OVERLAY_RULES:
        value = _to_float(row.get(rule.field))
        if value is None:
            factor_details[rule.field] = None
            continue
        centered = _bounded_centered_delta(
            value,
            neutral=rule.neutral,
            scale=rule.scale,
            favor_when_above=rule.favor_when_above,
        )
        contribution = centered * rule.weight
        signed_score += contribution
        factor_details[rule.field] = {
            "value": value,
            "neutral": rule.neutral,
            "scale": rule.scale,
            "weight": rule.weight,
            "centered": centered,
            "contribution": contribution,
        }

    max_abs = sum(rule.weight for rule in BALLPARKPAL_OVERLAY_RULES)
    display_score = 50.0 if max_abs <= 0 else ((signed_score + max_abs) / (2.0 * max_abs)) * 100.0
    model_score_value = _to_float(model_score if model_score is not None else row.get("predicted_hr_score"))
    if model_score_value is None:
        adjusted_score = None
    else:
        adjusted_score = max(0.0, min(100.0, model_score_value + signed_score))

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
