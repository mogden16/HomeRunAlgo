"""Normalize and join Ballpark Pal exports onto batter-game model rows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPORT_TYPES = ("batters", "pitchers", "teams", "games")

PICK_BATTER_FEATURE_COLUMNS = [
    "bp_batter_home_run_probability",
    "bp_batter_hit_probability",
]

PICK_PITCHER_FEATURE_COLUMNS = [
    "bp_pitcher_runs_allowed",
    "bp_pitcher_home_runs_allowed",
]

BALLPARKPAL_PICK_OVERLAY_RULES = {
    "bp_batter_home_run_probability": {
        "neutral": 0.10,
        "scale": 0.05,
        "weight": 12.0,
        "label": "batter_hr_probability",
    },
    "bp_batter_hit_probability": {
        "neutral": 0.70,
        "scale": 0.10,
        "weight": 6.0,
        "label": "batter_hit_probability",
    },
    "bp_pitcher_runs_allowed": {
        "neutral": 4.5,
        "scale": 1.0,
        "weight": 8.0,
        "label": "pitcher_runs_allowed",
    },
    "bp_pitcher_home_runs_allowed": {
        "neutral": 0.80,
        "scale": 0.35,
        "weight": 4.0,
        "label": "pitcher_home_runs_allowed",
    },
}

BALLPARKPAL_PICK_OVERLAY_MAX_ABS_SCORE = sum(rule["weight"] for rule in BALLPARKPAL_PICK_OVERLAY_RULES.values())


def _snake_case(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def _standardize_team_code(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def _standardize_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _load_export(path: Path, export_type: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [_snake_case(col) for col in df.columns]
    df["export_type"] = export_type
    df["source_file"] = path.name
    df["archive_date"] = path.parent.name
    return df


def _prefix_nonkey_columns(df: pd.DataFrame, prefix: str, key_columns: set[str]) -> pd.DataFrame:
    rename_map = {
        column: f"{prefix}_{column}"
        for column in df.columns
        if column not in key_columns and not str(column).startswith(f"{prefix}_") and not str(column).startswith("bp_")
    }
    return df.rename(columns=rename_map)


def _normalize_batters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = df["game_date"].map(_standardize_date)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["team"] = df["team"].map(_standardize_team_code)
    df["opponent"] = df["opponent"].map(_standardize_team_code)
    df["is_home"] = df["side"].astype(str).str.upper().eq("H").astype("Int64")

    rename_map = {
        "full_name": "bp_batter_full_name",
        "last_name": "bp_batter_last_name",
        "batter_stand": "bp_batter_stand",
        "batting_position": "bp_batter_batting_position",
        "plate_appearances": "bp_batter_plate_appearances",
        "at_bats": "bp_batter_at_bats",
        "hits": "bp_batter_hits",
        "bases": "bp_batter_bases",
        "strikeouts": "bp_batter_strikeouts",
        "walks": "bp_batter_walks",
        "singles": "bp_batter_singles",
        "doubles": "bp_batter_doubles",
        "triples": "bp_batter_triples",
        "home_runs": "bp_batter_home_runs",
        "rb_is": "bp_batter_rbis",
        "runs": "bp_batter_runs",
        "stolen_base_attempts": "bp_batter_stolen_base_attempts",
        "stolen_base_successes": "bp_batter_stolen_base_successes",
        "points_dk": "bp_batter_points_dk",
        "points_fd": "bp_batter_points_fd",
        "home_run_probability": "bp_batter_home_run_probability",
        "hit_probability": "bp_batter_hit_probability",
        "stolen_base_probability": "bp_batter_stolen_base_probability",
    }
    return _prefix_nonkey_columns(df.rename(columns=rename_map), "bp_batter", {"game_date", "game_pk", "player_id", "team", "opponent", "is_home", "archive_date", "source_file", "export_type"})


def _normalize_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = df["game_date"].map(_standardize_date)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["team"] = df["team"].map(_standardize_team_code)
    df["opponent"] = df["opponent"].map(_standardize_team_code)
    df["is_home"] = df["side"].astype(str).str.upper().eq("H").astype("Int64")

    rename_map = {
        "full_name": "bp_pitcher_full_name",
        "last_name": "bp_pitcher_last_name",
        "pitcher_hand": "bp_pitcher_hand",
        "batters_faced": "bp_pitcher_batters_faced",
        "innings": "bp_pitcher_innings",
        "win_pct": "bp_pitcher_win_pct",
        "loss_pct": "bp_pitcher_loss_pct",
        "nd_pct": "bp_pitcher_nd_pct",
        "quality_start": "bp_pitcher_quality_start",
        "points_dk": "bp_pitcher_points_dk",
        "points_fd": "bp_pitcher_points_fd",
        "runs_allowed": "bp_pitcher_runs_allowed",
        "hits_allowed": "bp_pitcher_hits_allowed",
        "strikeouts": "bp_pitcher_strikeouts",
        "walks": "bp_pitcher_walks",
        "home_runs_allowed": "bp_pitcher_home_runs_allowed",
        "stolen_bases_allowed": "bp_pitcher_stolen_bases_allowed",
    }
    return _prefix_nonkey_columns(df.rename(columns=rename_map), "bp_pitcher", {"game_date", "game_pk", "player_id", "team", "opponent", "is_home", "archive_date", "source_file", "export_type"})


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = df["game_date"].map(_standardize_date)
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["team"] = df["team"].map(_standardize_team_code)
    df["opponent"] = df["opponent"].map(_standardize_team_code)
    df["is_home"] = df["side"].astype(str).str.upper().eq("H").astype("Int64")

    rename_map = {
        "runs": "bp_team_runs",
        "win_percent": "bp_team_win_pct",
        "win_margin2": "bp_team_win_margin_2",
        "win_margin3": "bp_team_win_margin_3",
        "loss_margin2": "bp_team_loss_margin_2",
        "loss_margin3": "bp_team_loss_margin_3",
        "home_runs": "bp_team_home_runs",
        "triples": "bp_team_triples",
        "doubles": "bp_team_doubles",
        "singles": "bp_team_singles",
        "walks": "bp_team_walks",
        "strikeouts": "bp_team_strikeouts",
    }
    return _prefix_nonkey_columns(df.rename(columns=rename_map), "bp_team", {"game_date", "game_pk", "team", "opponent", "is_home", "archive_date", "source_file", "export_type"})


def _normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = df["game_date"].map(_standardize_date)
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["away_team"] = df["away_team"].map(_standardize_team_code)
    df["home_team"] = df["home_team"].map(_standardize_team_code)

    rename_map = {
        "runs_away": "bp_game_runs_away",
        "runs_home": "bp_game_runs_home",
        "away_win_pct": "bp_game_away_win_pct",
        "home_win_pct": "bp_game_home_win_pct",
        "away_win_margin3": "bp_game_away_win_margin_3",
        "away_win_margin2": "bp_game_away_win_margin_2",
        "away_win_margin1": "bp_game_away_win_margin_1",
        "home_win_margin3": "bp_game_home_win_margin_3",
        "home_win_margin2": "bp_game_home_win_margin_2",
        "home_win_margin1": "bp_game_home_win_margin_1",
        "runs_first_inning_pct": "bp_game_runs_first_inning_pct",
        "runs_first5_away": "bp_game_runs_first5_away",
        "runs_first5_home": "bp_game_runs_first5_home",
        "away_win_first5": "bp_game_away_win_first5",
        "home_win_first5": "bp_game_home_win_first5",
    }
    df = df.rename(columns=rename_map)
    df["bp_game_total_runs"] = pd.to_numeric(df["bp_game_runs_away"], errors="coerce") + pd.to_numeric(df["bp_game_runs_home"], errors="coerce")
    df["bp_game_run_diff"] = pd.to_numeric(df["bp_game_runs_home"], errors="coerce") - pd.to_numeric(df["bp_game_runs_away"], errors="coerce")
    df["bp_game_win_pct_gap"] = pd.to_numeric(df["bp_game_home_win_pct"], errors="coerce") - pd.to_numeric(df["bp_game_away_win_pct"], errors="coerce")
    df["bp_game_first5_total_runs"] = pd.to_numeric(df["bp_game_runs_first5_away"], errors="coerce") + pd.to_numeric(df["bp_game_runs_first5_home"], errors="coerce")
    df["bp_game_first5_run_diff"] = pd.to_numeric(df["bp_game_runs_first5_home"], errors="coerce") - pd.to_numeric(df["bp_game_runs_first5_away"], errors="coerce")
    return _prefix_nonkey_columns(df, "bp_game", {"game_date", "game_pk", "away_team", "home_team", "archive_date", "source_file", "export_type"})


def load_ballparkpal_archive(root_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and concatenate all archived Ballpark Pal exports under ``root_dir``."""

    frames: dict[str, list[pd.DataFrame]] = {export_type: [] for export_type in EXPORT_TYPES}
    for date_dir in sorted([path for path in root_dir.iterdir() if path.is_dir() and len(path.name) == 10]):
        for export_type in EXPORT_TYPES:
            matches = sorted(date_dir.glob(f"*_{export_type}.xlsx"))
            if not matches:
                continue
            # Use the latest timestamped file for a given day.
            frame = _load_export(matches[-1], export_type)
            frames[export_type].append(frame)

    return {
        export_type: pd.concat(items, ignore_index=True) if items else pd.DataFrame()
        for export_type, items in frames.items()
    }


def normalize_ballparkpal_exports(root_dir: Path) -> dict[str, pd.DataFrame]:
    """Return normalized export frames keyed by export type."""

    loaded = load_ballparkpal_archive(root_dir)
    return {
        "batters": _normalize_batters(loaded["batters"]) if not loaded["batters"].empty else pd.DataFrame(),
        "pitchers": _normalize_pitchers(loaded["pitchers"]) if not loaded["pitchers"].empty else pd.DataFrame(),
        "teams": _normalize_teams(loaded["teams"]) if not loaded["teams"].empty else pd.DataFrame(),
        "games": _normalize_games(loaded["games"]) if not loaded["games"].empty else pd.DataFrame(),
    }


def load_ballparkpal_snapshot(snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and normalize the four exports from one date folder."""

    frames: dict[str, pd.DataFrame] = {}
    for export_type in EXPORT_TYPES:
        matches = sorted(snapshot_dir.glob(f"*_{export_type}.xlsx"))
        if not matches:
            frames[export_type] = pd.DataFrame()
            continue
        loaded = _load_export(matches[-1], export_type)
        if export_type == "batters":
            frames[export_type] = _normalize_batters(loaded)
        elif export_type == "pitchers":
            frames[export_type] = _normalize_pitchers(loaded)
        elif export_type == "teams":
            frames[export_type] = _normalize_teams(loaded)
        else:
            frames[export_type] = _normalize_games(loaded)
    return frames


BALLPARKPAL_BATTER_FEATURE_COLUMNS = [
    "bp_batter_home_run_probability",
    "bp_batter_hit_probability",
    "bp_batter_points_dk",
    "bp_batter_points_fd",
    "bp_batter_plate_appearances",
    "bp_batter_at_bats",
    "bp_batter_hits",
    "bp_batter_bases",
    "bp_batter_strikeouts",
    "bp_batter_walks",
    "bp_batter_runs",
    "bp_batter_rbis",
    "bp_batter_stolen_base_attempts",
]

BALLPARKPAL_PITCHER_FEATURE_COLUMNS = [
    "bp_pitcher_points_dk",
    "bp_pitcher_points_fd",
    "bp_pitcher_batters_faced",
    "bp_pitcher_innings",
    "bp_pitcher_runs_allowed",
    "bp_pitcher_hits_allowed",
    "bp_pitcher_strikeouts",
    "bp_pitcher_walks",
    "bp_pitcher_home_runs_allowed",
]

BALLPARKPAL_TEAM_FEATURE_COLUMNS = [
    "bp_team_runs",
    "bp_team_win_pct",
    "bp_team_home_runs",
    "bp_team_triples",
    "bp_team_doubles",
    "bp_team_singles",
    "bp_team_walks",
    "bp_team_strikeouts",
]

BALLPARKPAL_GAME_FEATURE_COLUMNS = [
    "bp_game_runs_away",
    "bp_game_runs_home",
    "bp_game_total_runs",
    "bp_game_run_diff",
    "bp_game_away_win_pct",
    "bp_game_home_win_pct",
    "bp_game_win_pct_gap",
    "bp_game_runs_first_inning_pct",
    "bp_game_runs_first5_away",
    "bp_game_runs_first5_home",
    "bp_game_first5_total_runs",
    "bp_game_first5_run_diff",
]

BALLPARKPAL_FEATURE_COLUMNS = [
    *BALLPARKPAL_BATTER_FEATURE_COLUMNS,
    *BALLPARKPAL_PITCHER_FEATURE_COLUMNS,
    *BALLPARKPAL_TEAM_FEATURE_COLUMNS,
    *BALLPARKPAL_GAME_FEATURE_COLUMNS,
]


PICK_ENRICHMENT_FEATURE_COLUMNS = [
    *PICK_BATTER_FEATURE_COLUMNS,
    *PICK_PITCHER_FEATURE_COLUMNS,
]


@dataclass
class BallparkPalJoinCoverage:
    rows_in: int
    rows_out: int
    batter_coverage: float
    pitcher_coverage: float
    team_coverage: float
    game_coverage: float
    missing_exports: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BallparkPalPickCoverage:
    rows_in: int
    rows_out: int
    batter_coverage: float
    pitcher_coverage: float
    any_feature_coverage: float
    missing_exports: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BallparkPalPickOverlaySummary:
    rows_in: int
    rows_out: int
    support_rate: float
    against_rate: float
    neutral_rate: float
    mean_signed_score: float
    mean_display_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_pick_frame(picks_df: pd.DataFrame) -> pd.DataFrame:
    df = picks_df.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "game_pk" in df.columns:
        df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    if "batter_id" in df.columns:
        df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce").astype("Int64")
    if "pitcher_id" in df.columns:
        df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64")
    if "team" in df.columns:
        df["team"] = df["team"].map(_standardize_team_code)
    if "opponent_team" in df.columns:
        df["opponent"] = df["opponent_team"].map(_standardize_team_code)
    elif "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(_standardize_team_code)
    if "is_home" in df.columns:
        df["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").astype("Int64")
    return df


def _score_overlay_component(series: pd.Series, *, neutral: float, scale: float, weight: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    centered = ((numeric - neutral) / scale).clip(-1.0, 1.0)
    signed = centered * weight
    display = ((centered + 1.0) / 2.0 * 100.0).clip(0.0, 100.0)
    return signed.fillna(0.0), display.fillna(50.0), centered.fillna(0.0)


def _attach_pick_overlay(enriched: pd.DataFrame) -> tuple[pd.DataFrame, BallparkPalPickOverlaySummary]:
    working = enriched.copy()
    alignment_columns: list[str] = []
    signed_components: list[pd.Series] = []

    for source_column, rule in BALLPARKPAL_PICK_OVERLAY_RULES.items():
        signed_component, display_component, centered_component = _score_overlay_component(
            working[source_column],
            neutral=rule["neutral"],
            scale=rule["scale"],
            weight=rule["weight"],
        )
        base_name = f"bp_overlay_{rule['label']}"
        working[f"{base_name}_signed"] = signed_component
        working[f"{base_name}_display"] = display_component
        working[f"{base_name}_centered"] = centered_component
        working[f"{base_name}_alignment"] = np.select(
            [signed_component > 0, signed_component < 0],
            ["in_favor", "not_in_favor"],
            default="neutral",
        )
        alignment_columns.append(f"{base_name}_alignment")
        signed_components.append(signed_component)

    working["ballparkpal_overlay_signed_score"] = sum(signed_components)
    working["ballparkpal_overlay_display_score"] = (
        (working["ballparkpal_overlay_signed_score"] + BALLPARKPAL_PICK_OVERLAY_MAX_ABS_SCORE)
        / (2 * BALLPARKPAL_PICK_OVERLAY_MAX_ABS_SCORE)
        * 100.0
    ).clip(0.0, 100.0)

    if "predicted_hr_score" in working.columns:
        model_score = pd.to_numeric(working["predicted_hr_score"], errors="coerce")
    elif "predicted_hr_probability" in working.columns:
        model_score = pd.to_numeric(working["predicted_hr_probability"], errors="coerce") * 100.0
    else:
        model_score = pd.Series(0.0, index=working.index)
    working["ballparkpal_overlay_adjusted_score"] = (model_score.fillna(0.0) + working["ballparkpal_overlay_signed_score"]).clip(0.0, 100.0)
    if "game_date" in working.columns:
        working["ballparkpal_overlay_adjusted_rank"] = (
            working.groupby("game_date")["ballparkpal_overlay_adjusted_score"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
    else:
        working["ballparkpal_overlay_adjusted_rank"] = (
            working["ballparkpal_overlay_adjusted_score"].rank(method="first", ascending=False).astype("Int64")
        )
    working["ballparkpal_overlay_alignment_count"] = working[alignment_columns].eq("in_favor").sum(axis=1)
    working["ballparkpal_overlay_grade"] = pd.cut(
        working["ballparkpal_overlay_signed_score"],
        bins=[-np.inf, -5.0, 5.0, np.inf],
        labels=["against", "mixed", "supportive"],
        include_lowest=True,
    ).astype("string")

    summary = BallparkPalPickOverlaySummary(
        rows_in=len(enriched),
        rows_out=len(working),
        support_rate=float((working["ballparkpal_overlay_signed_score"] > 0).mean()) if len(working) else 0.0,
        against_rate=float((working["ballparkpal_overlay_signed_score"] < 0).mean()) if len(working) else 0.0,
        neutral_rate=float((working["ballparkpal_overlay_signed_score"] == 0).mean()) if len(working) else 0.0,
        mean_signed_score=float(working["ballparkpal_overlay_signed_score"].mean()) if len(working) else 0.0,
        mean_display_score=float(working["ballparkpal_overlay_display_score"].mean()) if len(working) else 0.0,
    )
    return working, summary


def enrich_picks_with_ballparkpal(
    picks_df: pd.DataFrame,
    snapshot_dir: Path,
) -> tuple[pd.DataFrame, BallparkPalPickCoverage, BallparkPalPickOverlaySummary]:
    """Attach the requested Ballpark Pal fields to a picks dataframe."""

    exports = load_ballparkpal_snapshot(snapshot_dir)
    picks = _normalize_pick_frame(picks_df)
    enriched = picks.copy()

    batter_source = exports["batters"].copy()
    if not batter_source.empty:
        batter_source = batter_source[[
            "game_date",
            "game_pk",
            "player_id",
            "team",
            "opponent",
            *PICK_BATTER_FEATURE_COLUMNS,
        ]].drop_duplicates(["game_date", "game_pk", "player_id", "team", "opponent"])
        enriched = enriched.merge(
            batter_source,
            left_on=["game_date", "game_pk", "batter_id", "team", "opponent"],
            right_on=["game_date", "game_pk", "player_id", "team", "opponent"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["player_id"], errors="ignore")
    else:
        for column in PICK_BATTER_FEATURE_COLUMNS:
            enriched[column] = pd.NA

    pitcher_source = exports["pitchers"].copy()
    if not pitcher_source.empty:
        pitcher_source = pitcher_source[[
            "game_date",
            "game_pk",
            "player_id",
            "team",
            "opponent",
            *PICK_PITCHER_FEATURE_COLUMNS,
        ]].drop_duplicates(["game_date", "game_pk", "player_id", "team", "opponent"])
        enriched = enriched.merge(
            pitcher_source,
            left_on=["game_date", "game_pk", "pitcher_id", "opponent", "team"],
            right_on=["game_date", "game_pk", "player_id", "team", "opponent"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_bp_pitcher"),
        ).drop(columns=["player_id"], errors="ignore")
    else:
        for column in PICK_PITCHER_FEATURE_COLUMNS:
            enriched[column] = pd.NA

    missing_exports = [name for name, frame in exports.items() if frame.empty]
    batter_coverage = float(enriched["bp_batter_home_run_probability"].notna().mean()) if len(enriched) else 0.0
    pitcher_coverage = float(enriched["bp_pitcher_runs_allowed"].notna().mean()) if len(enriched) else 0.0
    any_feature_coverage = float(enriched[PICK_ENRICHMENT_FEATURE_COLUMNS].notna().any(axis=1).mean()) if len(enriched) else 0.0
    coverage = BallparkPalPickCoverage(
        rows_in=len(picks_df),
        rows_out=len(enriched),
        batter_coverage=batter_coverage,
        pitcher_coverage=pitcher_coverage,
        any_feature_coverage=any_feature_coverage,
        missing_exports=missing_exports,
    )
    enriched, overlay_summary = _attach_pick_overlay(enriched)
    return enriched, coverage, overlay_summary


def augment_model_dataset_with_ballparkpal(
    base_df: pd.DataFrame,
    archive_root: Path,
) -> tuple[pd.DataFrame, BallparkPalJoinCoverage]:
    """Join Ballpark Pal features onto a batter-game dataset."""

    exports = normalize_ballparkpal_exports(archive_root)
    augmented = base_df.copy()
    augmented["game_date"] = pd.to_datetime(augmented["game_date"]).dt.strftime("%Y-%m-%d")
    augmented["game_pk"] = pd.to_numeric(augmented["game_pk"], errors="coerce").astype("Int64")
    augmented["player_id"] = pd.to_numeric(augmented["player_id"], errors="coerce").astype("Int64")
    augmented["team"] = augmented["team"].map(_standardize_team_code)
    augmented["opponent"] = augmented["opponent"].map(_standardize_team_code)
    if "is_home" in augmented.columns:
        augmented["is_home"] = pd.to_numeric(augmented["is_home"], errors="coerce").astype("Int64")
    else:
        augmented["is_home"] = np.where(augmented["team"].notna() & augmented["opponent"].notna(), 0, np.nan)
    augmented["bp_join_side"] = np.where(augmented["is_home"].astype("Int64") == 1, "H", "A")

    batter = exports["batters"].copy()
    pitcher = exports["pitchers"].copy()
    teams = exports["teams"].copy()
    games = exports["games"].copy()

    joined = augmented.merge(
        batter[[
            "game_date",
            "game_pk",
            "player_id",
            "team",
            "opponent",
            "is_home",
            *BALLPARKPAL_BATTER_FEATURE_COLUMNS,
        ]].drop_duplicates(["game_date", "game_pk", "player_id", "team", "opponent", "is_home"]),
        on=["game_date", "game_pk", "player_id", "team", "opponent", "is_home"],
        how="left",
        validate="many_to_one",
    )

    pitcher_join = pitcher[[
        "game_date",
        "game_pk",
        "player_id",
        "team",
        "opponent",
        "is_home",
        *BALLPARKPAL_PITCHER_FEATURE_COLUMNS,
    ]].drop_duplicates(["game_date", "game_pk", "player_id", "team", "opponent", "is_home"])

    joined = joined.merge(
        pitcher_join,
        left_on=["game_date", "game_pk", "opp_pitcher_id", "team", "opponent", "is_home"],
        right_on=["game_date", "game_pk", "player_id", "team", "opponent", "is_home"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_bp_pitcher"),
    ).drop(columns=["player_id_bp_pitcher"], errors="ignore")

    team_join = teams[[
        "game_date",
        "game_pk",
        "team",
        "opponent",
        "is_home",
        *BALLPARKPAL_TEAM_FEATURE_COLUMNS,
    ]].drop_duplicates(["game_date", "game_pk", "team", "opponent", "is_home"])
    joined = joined.merge(
        team_join,
        on=["game_date", "game_pk", "team", "opponent", "is_home"],
        how="left",
        validate="many_to_one",
    )

    game_join = games[[
        "game_date",
        "game_pk",
        *BALLPARKPAL_GAME_FEATURE_COLUMNS,
    ]].drop_duplicates(["game_date", "game_pk"])
    joined = joined.merge(
        game_join,
        on=["game_date", "game_pk"],
        how="left",
        validate="many_to_one",
    )

    missing_exports = [name for name, frame in exports.items() if frame.empty]
    coverage = BallparkPalJoinCoverage(
        rows_in=len(base_df),
        rows_out=len(joined),
        batter_coverage=float(joined["bp_batter_home_run_probability"].notna().mean()) if len(joined) else 0.0,
        pitcher_coverage=float(joined["bp_pitcher_points_dk"].notna().mean()) if len(joined) else 0.0,
        team_coverage=float(joined["bp_team_runs"].notna().mean()) if len(joined) else 0.0,
        game_coverage=float(joined["bp_game_total_runs"].notna().mean()) if len(joined) else 0.0,
        missing_exports=missing_exports,
    )
    return joined, coverage
