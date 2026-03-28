"""Build Cloudflare Pages dashboard artifacts from published public picks."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_START_DATE = "2026-03-25"
DEFAULT_CURRENT_PICKS_PATH = Path("data/live/current_picks.json")
DEFAULT_HISTORY_PATH = Path("data/live/pick_history.json")
DEFAULT_OUTPUT_DIR = Path("cloudflare-app/data")
DEFAULT_MODEL_FAMILY = "2024-25 trained"
DEFAULT_DATA_NOTE = "Public dashboard tracking begins on Opening Night, March 25, 2026. Trained on 2024 and 2025 season data."
DEFAULT_REFRESH_SCHEDULE = {
    "timezone": "ET",
    "runs": [
        {
            "time_et": "2:00 AM ET",
            "type": "daily",
            "label": "Daily refresh",
            "description": "Refreshes data, retrains, settles yesterday, and rolls the slate forward.",
        },
        {
            "time_et": "11:00 AM ET",
            "type": "publish",
            "label": "Publish",
            "description": "Updates the public picks for the current slate.",
        },
        {
            "time_et": "1:00 PM ET",
            "type": "publish",
            "label": "Publish",
            "description": "Updates the public picks for the current slate.",
        },
        {
            "time_et": "3:00 PM ET",
            "type": "publish",
            "label": "Publish",
            "description": "Updates the public picks for the current slate.",
        },
        {
            "time_et": "6:00 PM ET",
            "type": "publish",
            "label": "Publish",
            "description": "Updates the public picks for the current slate.",
        },
    ],
}

DISPLAY_COLUMNS = [
    "pick_id",
    "game_pk",
    "game_date",
    "rank",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_name",
    "confidence_tier",
    "predicted_hr_probability",
    "predicted_hr_score",
    "actual_hit_hr",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "result_label",
]
CURRENT_PICK_COLUMNS = [
    "pick_id",
    "published_at",
    "game_pk",
    "game_date",
    "rank",
    "batter_id",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_id",
    "pitcher_name",
    "confidence_tier",
    "predicted_hr_probability",
    "predicted_hr_score",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "result",
]
HISTORY_COLUMNS = [
    "pick_id",
    "published_at",
    "game_pk",
    "game_date",
    "rank",
    "batter_id",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_id",
    "pitcher_name",
    "confidence_tier",
    "predicted_hr_probability",
    "predicted_hr_score",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "result_label",
    "actual_hit_hr",
]


def score_sort_value(row: dict[str, Any]) -> float:
    score = row.get("predicted_hr_score")
    if score is None:
        return float("-inf")
    return float(score)


def current_pick_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (-score_sort_value(row), int(row.get("rank") or 999), str(row.get("batter_name") or ""))


def history_sort_key(row: dict[str, Any]) -> tuple[str, float, int, str]:
    return (
        str(row.get("game_date") or ""),
        -score_sort_value(row),
        int(row.get("rank") or 999),
        str(row.get("batter_name") or ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-picks-path", default=str(DEFAULT_CURRENT_PICKS_PATH), help="Path to the latest published picks JSON.")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH), help="Path to the public pick ledger JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where dashboard JSON will be written.")
    parser.add_argument("--tracking-start-date", default=DEFAULT_TRACKING_START_DATE, help="Only picks on or after this date are published.")
    parser.add_argument("--latest-count", type=int, default=12, help="Number of latest picks shown on the landing page.")
    parser.add_argument("--history-per-date", type=int, default=10, help="Number of historical picks preserved per date in the dashboard.")
    parser.add_argument("--min-player-picks", type=int, default=2, help="Minimum settled picks required for a player leaderboard row.")
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return [row for row in data if isinstance(row, dict)]


def serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_result(value: Any) -> tuple[str, int | None]:
    token = str(value or "").strip().lower()
    if token in {"1", "hr", "hit", "home_run", "home run", "success"}:
        return "HR", 1
    if token in {"0", "miss", "no hr", "no_hr", "failed", "failure"}:
        return "No HR", 0
    return "Pending", None


def build_pick_id(row: dict[str, Any]) -> str:
    game_date = str(row.get("game_date") or row.get("pick_date") or "").strip()
    game_pk = str(row.get("game_pk") or "").strip()
    batter_id = str(row.get("batter_id") or "").strip()
    batter_name = str(row.get("batter_name") or "").strip().lower().replace(" ", "-")
    pitcher_id = str(row.get("pitcher_id") or "").strip()
    pitcher_name = str(row.get("pitcher_name") or "").strip().lower().replace(" ", "-")
    identity = batter_id or batter_name
    matchup = pitcher_id or pitcher_name
    if game_pk:
        return str(row.get("pick_id") or f"{game_date}:{game_pk}:{identity}:{matchup}")
    return str(row.get("pick_id") or f"{game_date}:{identity}:{matchup}")


def normalize_pick(row: dict[str, Any], tracking_start_date: str) -> dict[str, Any] | None:
    game_date = str(row.get("game_date") or row.get("pick_date") or "").strip()
    if not game_date or game_date < tracking_start_date:
        return None

    probability = parse_float(row.get("predicted_hr_probability"))
    score = parse_float(row.get("predicted_hr_score"))
    if score is None and probability is not None:
        score = probability * 100.0

    result_label, actual_hit_hr = normalize_result(row.get("result") or row.get("result_label"))
    normalized = {
        "pick_id": build_pick_id(row),
        "published_at": str(row.get("published_at") or datetime.now(timezone.utc).isoformat()),
        "game_pk": parse_int(row.get("game_pk")),
        "game_date": game_date,
        "rank": parse_int(row.get("rank")) or 999,
        "batter_id": parse_int(row.get("batter_id")),
        "batter_name": str(row.get("batter_name") or "Unknown hitter"),
        "team": str(row.get("team") or ""),
        "opponent_team": str(row.get("opponent_team") or row.get("opponent") or ""),
        "pitcher_id": parse_int(row.get("pitcher_id")),
        "pitcher_name": str(row.get("pitcher_name") or ""),
        "confidence_tier": str(row.get("confidence_tier") or "watch").lower(),
        "predicted_hr_probability": probability,
        "predicted_hr_score": score,
        "top_reason_1": str(row.get("top_reason_1") or ""),
        "top_reason_2": str(row.get("top_reason_2") or ""),
        "top_reason_3": str(row.get("top_reason_3") or ""),
        "result_label": result_label,
        "actual_hit_hr": actual_hit_hr,
    }
    return normalized


def clean_current_pick_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in sorted(rows, key=current_pick_sort_key):
        cleaned.append({column: serialize_value(row.get(column)) for column in CURRENT_PICK_COLUMNS[:-1]})
        cleaned[-1]["result"] = str(row.get("result_label") or "Pending")
    return cleaned


def clean_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {column: serialize_value(row.get(column)) for column in HISTORY_COLUMNS}
        for row in sorted(rows, key=history_sort_key)
    ]


def upsert_history(existing_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_dates = {str(row["game_date"]) for row in current_rows}
    retained_existing = [
        row
        for row in existing_rows
        if not (str(row["game_date"]) in current_dates and row.get("result_label") == "Pending")
    ]
    by_id = {str(row["pick_id"]): dict(row) for row in retained_existing}
    for row in current_rows:
        key = str(row["pick_id"])
        previous = by_id.get(key)
        if previous:
            if previous.get("result_label") in {"HR", "No HR"} and row.get("result_label") == "Pending":
                row["result_label"] = previous["result_label"]
                row["actual_hit_hr"] = previous.get("actual_hit_hr")
            by_id[key].update(row)
        else:
            by_id[key] = dict(row)
    return sorted(by_id.values(), key=history_sort_key)


def top_k_by_date(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=history_sort_key):
        grouped.setdefault(row["game_date"], []).append(row)
    trimmed: list[dict[str, Any]] = []
    for game_date in sorted(grouped):
        trimmed.extend(grouped[game_date][:k])
    return trimmed


def summarize_top_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    subset = top_k_by_date(rows, k)
    settled = [row for row in subset if row["actual_hit_hr"] is not None]
    homers = sum(int(row["actual_hit_hr"]) for row in settled)
    hit_rate = (homers / len(settled)) if settled else None
    avg_score = (
        sum(float(row["predicted_hr_score"]) for row in subset if row["predicted_hr_score"] is not None) / len(subset)
        if subset
        else None
    )
    return {
        "top_k": k,
        "dates": len({row["game_date"] for row in subset}),
        "picks": len(subset),
        "homers": homers,
        "hit_rate": serialize_value(hit_rate),
        "avg_score": serialize_value(avg_score),
    }


def summarize_confidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"elite": 0, "strong": 1, "watch": 2, "longshot": 3}
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        tier = str(row["confidence_tier"] or "watch")
        bucket = buckets.setdefault(tier, {"confidence_tier": tier, "picks": 0, "homers": 0, "prob_total": 0.0, "prob_count": 0})
        bucket["picks"] += 1
        if row["actual_hit_hr"] is not None:
            bucket["homers"] += int(row["actual_hit_hr"])
        if row["predicted_hr_probability"] is not None:
            bucket["prob_total"] += float(row["predicted_hr_probability"])
            bucket["prob_count"] += 1
    rows_out: list[dict[str, Any]] = []
    for tier, bucket in buckets.items():
        picks = int(bucket["picks"])
        rows_out.append(
            {
                "confidence_tier": tier,
                "picks": picks,
                "homers": int(bucket["homers"]),
                "hit_rate": serialize_value((bucket["homers"] / picks) if picks else None),
                "avg_probability": serialize_value((bucket["prob_total"] / bucket["prob_count"]) if bucket["prob_count"] else None),
            }
        )
    return sorted(rows_out, key=lambda row: (order.get(row["confidence_tier"], 99), row["confidence_tier"]))


def build_player_leaderboard(rows: list[dict[str, Any]], min_player_picks: int) -> list[dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["batter_name"], row["team"])
        bucket = stats.setdefault(
            key,
            {"batter_name": row["batter_name"], "team": row["team"], "picks": 0, "homers": 0, "score_total": 0.0, "score_count": 0},
        )
        bucket["picks"] += 1
        bucket["homers"] += int(row["actual_hit_hr"] or 0)
        if row["predicted_hr_score"] is not None:
            bucket["score_total"] += float(row["predicted_hr_score"])
            bucket["score_count"] += 1
    leaderboard: list[dict[str, Any]] = []
    for bucket in stats.values():
        if bucket["picks"] < min_player_picks:
            continue
        leaderboard.append(
            {
                "batter_name": bucket["batter_name"],
                "team": bucket["team"],
                "picks": bucket["picks"],
                "homers": bucket["homers"],
                "hit_rate": serialize_value(bucket["homers"] / bucket["picks"]),
                "avg_score": serialize_value((bucket["score_total"] / bucket["score_count"]) if bucket["score_count"] else None),
            }
        )
    return sorted(leaderboard, key=lambda row: (-row["homers"], -(row["hit_rate"] or 0), -row["picks"]))[:20]


def to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: serialize_value(row.get(column)) for column in DISPLAY_COLUMNS} for row in rows]


def build_refresh_schedule() -> dict[str, Any]:
    return {
        "timezone": str(DEFAULT_REFRESH_SCHEDULE["timezone"]),
        "runs": [dict(run) for run in DEFAULT_REFRESH_SCHEDULE["runs"]],
    }


def build_dashboard_artifacts(
    *,
    current_picks_path: Path = DEFAULT_CURRENT_PICKS_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tracking_start_date: str = DEFAULT_TRACKING_START_DATE,
    latest_count: int = 12,
    history_per_date: int = 10,
    min_player_picks: int = 2,
    persist_history: bool = True,
    latest_available_date_override: str | None = None,
) -> Path:
    current_picks_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_input = load_json_array(current_picks_path)
    history_input = load_json_array(history_path)

    current_rows = [row for row in (normalize_pick(item, tracking_start_date) for item in current_input) if row is not None]
    history_rows = [row for row in (normalize_pick(item, tracking_start_date) for item in history_input) if row is not None]
    current_rows = sorted(current_rows, key=current_pick_sort_key)
    merged_history = upsert_history(history_rows, current_rows)

    current_picks_path.write_text(json.dumps(clean_current_pick_rows(current_rows), indent=2), encoding="utf-8")
    if persist_history:
        history_path.write_text(json.dumps(clean_history_rows(merged_history), indent=2), encoding="utf-8")

    latest_game_date = latest_available_date_override or max((row["game_date"] for row in current_rows), default=tracking_start_date)
    latest_picks = current_rows[:latest_count]
    dashboard_history = sorted(
        merged_history,
        key=lambda row: (-int(str(row["game_date"]).replace("-", "")), -score_sort_value(row), int(row["rank"]), str(row["batter_name"])),
    )
    settled_rows = [row for row in merged_history if row["actual_hit_hr"] is not None]
    recent_successes = [
        row
        for row in sorted(settled_rows, key=lambda item: (item["game_date"], score_sort_value(item), -item["rank"]), reverse=True)
        if row["actual_hit_hr"] == 1
    ][:25]

    settled_count = len(settled_rows)
    homers = sum(int(row["actual_hit_hr"] or 0) for row in settled_rows)
    hit_rate = (homers / settled_count) if settled_count else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracking_start_date": tracking_start_date,
        "model_family": DEFAULT_MODEL_FAMILY,
        "latest_available_date": latest_game_date,
        "data_note": DEFAULT_DATA_NOTE,
        "refresh_schedule": build_refresh_schedule(),
        "overview": {
            "latest_slate_size": len(latest_picks),
            "tracked_dates": len({row["game_date"] for row in merged_history}),
            "tracked_picks": len(merged_history),
            "settled_picks": settled_count,
            "tracked_homers": homers,
            "tracked_hit_rate": serialize_value(hit_rate),
            "open_picks": len([row for row in merged_history if row["actual_hit_hr"] is None]),
        },
        "top_k_summary": [summarize_top_k(settled_rows, k) for k in (1, 3, 5, history_per_date)],
        "confidence_summary": summarize_confidence(settled_rows),
        "latest_picks": to_records(latest_picks),
        "history": to_records(dashboard_history),
        "player_leaderboard": build_player_leaderboard(settled_rows, min_player_picks),
        "recent_successes": to_records(recent_successes),
    }

    output_path = output_dir / "dashboard.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_dashboard_artifacts(
        current_picks_path=Path(args.current_picks_path),
        history_path=Path(args.history_path),
        output_dir=Path(args.output_dir),
        tracking_start_date=args.tracking_start_date,
        latest_count=args.latest_count,
        history_per_date=args.history_per_date,
        min_player_picks=args.min_player_picks,
    )
    print(f"Wrote dashboard artifact to {output_path}")


if __name__ == "__main__":
    main()
