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


BALLPARKPAL_MODEL_BLEND_WEIGHT: float = 0.90
BALLPARKPAL_BALLPARK_BLEND_WEIGHT: float = 0.10

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
BALLPARKPAL_NORMALIZED_SCORE_NEUTRAL: float = 50.0


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
        raw_score = None
        display_score = None
        adjusted_score = model_score_value
    else:
        raw_score = max(0.0, min(100.0, home_run_probability * 100.0))
        display_score = raw_score
        signed_score = raw_score - BALLPARKPAL_HOME_RUN_NEUTRAL_SCORE
        factor_details[rule.field] = {
            "value": home_run_probability,
            "neutral": rule.neutral,
            "scale": rule.scale,
            "weight": rule.weight,
            "raw_score": raw_score,
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
        "ballparkpal_overlay_raw_score": round(raw_score, 1) if raw_score is not None else None,
        "ballparkpal_overlay_display_score": round(display_score, 1) if display_score is not None else None,
        "ballparkpal_overlay_adjusted_score": round(adjusted_score, 1) if adjusted_score is not None else None,
        "ballparkpal_overlay_direction": direction,
        "ballparkpal_overlay_model_score": round(model_score_value, 1) if model_score_value is not None else None,
        "ballparkpal_overlay_blend_weights": {
            "model": BALLPARKPAL_MODEL_BLEND_WEIGHT,
            "ballpark": BALLPARKPAL_BALLPARK_BLEND_WEIGHT,
        },
        "ballparkpal_overlay_factor_details": factor_details,
    }


def normalize_ballparkpal_overlay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "ballparkpal_overlay_raw_score" not in frame.columns:
        return frame

    normalized = frame.copy()
    raw_scores = pd.to_numeric(normalized["ballparkpal_overlay_raw_score"], errors="coerce")
    valid_scores = raw_scores.dropna()
    if valid_scores.empty:
        normalized["ballparkpal_overlay_display_score"] = np.nan
        normalized["ballparkpal_overlay_direction"] = "neutral"
        normalized["ballparkpal_overlay_adjusted_score"] = pd.to_numeric(
            normalized.get("predicted_hr_score"), errors="coerce"
        )
        return normalized

    min_score = float(valid_scores.min())
    max_score = float(valid_scores.max())
    if np.isclose(min_score, max_score):
        normalized_scores = pd.Series(BALLPARKPAL_NORMALIZED_SCORE_NEUTRAL, index=normalized.index, dtype="float64")
        normalized_scores = normalized_scores.where(raw_scores.notna(), np.nan)
    else:
        normalized_scores = ((raw_scores - min_score) / (max_score - min_score)) * 100.0

    normalized["ballparkpal_overlay_display_score"] = normalized_scores.clip(lower=0.0, upper=100.0)
    normalized["ballparkpal_overlay_direction"] = normalized["ballparkpal_overlay_display_score"].apply(
        lambda value: "neutral"
        if pd.isna(value)
        else (
            "neutral"
            if np.isclose(float(value), BALLPARKPAL_NORMALIZED_SCORE_NEUTRAL)
            else ("favorable" if float(value) > BALLPARKPAL_NORMALIZED_SCORE_NEUTRAL else "unfavorable")
        )
    )
    model_scores = pd.to_numeric(normalized.get("predicted_hr_score"), errors="coerce")
    normalized["ballparkpal_overlay_adjusted_score"] = (
        (BALLPARKPAL_MODEL_BLEND_WEIGHT * model_scores)
        + (BALLPARKPAL_BALLPARK_BLEND_WEIGHT * normalized["ballparkpal_overlay_display_score"])
    )
    normalized["ballparkpal_overlay_adjusted_score"] = normalized["ballparkpal_overlay_adjusted_score"].where(
        normalized["ballparkpal_overlay_display_score"].notna(),
        model_scores,
    )
    return normalized


def apply_ballparkpal_overlay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    annotated = frame.copy()
    overlay_rows = annotated.apply(lambda row: pd.Series(compute_ballparkpal_overlay(row.to_dict())), axis=1)
    for column in overlay_rows.columns:
        annotated[column] = overlay_rows[column]
    return normalize_ballparkpal_overlay_frame(annotated)
