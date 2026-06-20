"""Verify forward-only public/live artifacts and repo-side rollout assumptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_dashboard_artifacts import (
    CURRENT_PICK_COLUMNS as PUBLIC_CURRENT_PICK_COLUMNS,
    DEFAULT_TRACKING_START_DATE,
    DISPLAY_COLUMNS as PUBLIC_DISPLAY_COLUMNS,
    HISTORY_COLUMNS as PUBLIC_HISTORY_COLUMNS,
    current_pick_sort_key as public_current_pick_sort_key,
)

TRACKING_START_DATE = DEFAULT_TRACKING_START_DATE
CURRENT_PICK_COLUMNS = set(PUBLIC_CURRENT_PICK_COLUMNS)
HISTORY_COLUMNS = set(PUBLIC_HISTORY_COLUMNS)
DISPLAY_COLUMNS = list(PUBLIC_DISPLAY_COLUMNS)
LEGACY_OPTIONAL_PREDICTION_COLUMNS = {"positive_call_threshold", "positive_hr_call"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-picks", default="data/live/current_picks.json")
    parser.add_argument("--pick-history", default="data/live/pick_history.json")
    parser.add_argument("--dashboard", default="cloudflare-app/data/dashboard.json")
    parser.add_argument("--refresh-script", default="scripts/refresh_dashboard.ps1")
    parser.add_argument("--live-pipeline", default="scripts/live_pipeline.py")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score_sort_value(row: dict[str, Any]) -> float:
    score = row.get("predicted_hr_score")
    return float(score) if score is not None else float("-inf")


def current_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return public_current_pick_sort_key(row)


def history_sort_key(row: dict[str, Any]) -> tuple[str, float, int, str]:
    return (
        str(row.get("game_date") or ""),
        int(row.get("original_rank") or row.get("rank") or 999),
        -score_sort_value({"predicted_hr_score": row.get("original_score", row.get("predicted_hr_score"))}),
        str(row.get("batter_name") or ""),
    )


def dashboard_history_sort_key(row: dict[str, Any]) -> tuple[str, float, int, str]:
    return (
        -int(str(row.get("game_date") or "0").replace("-", "") or 0),
        int(row.get("original_rank") or row.get("rank") or 999),
        -score_sort_value({"predicted_hr_score": row.get("original_score", row.get("predicted_hr_score"))}),
        str(row.get("batter_name") or ""),
    )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_migrating_columns(actual: set[str], expected: set[str], path: Path, label: str) -> None:
    extra = actual - expected
    missing = expected - actual
    assert_true(not extra, f"{path} contains unexpected {label} fields: {sorted(extra)}")
    assert_true(
        missing <= LEGACY_OPTIONAL_PREDICTION_COLUMNS,
        f"{path} is missing required {label} fields: {sorted(missing)}",
    )


def verify_current_picks(path: Path) -> None:
    rows = load_json(path)
    assert_true(isinstance(rows, list), f"{path} must contain a JSON array.")
    assert_true(rows == sorted(rows, key=current_sort_key), f"{path} is not sorted by stable board order.")
    for row in rows:
        verify_migrating_columns(set(row), CURRENT_PICK_COLUMNS, path, "current-pick")
        assert_true(str(row.get("game_date") or "") >= TRACKING_START_DATE, f"{path} contains pre-tracking row {row.get('pick_id')}.")


def verify_pick_history(path: Path) -> None:
    rows = load_json(path)
    assert_true(isinstance(rows, list), f"{path} must contain a JSON array.")
    assert_true(rows == sorted(rows, key=history_sort_key), f"{path} is not deterministically ordered.")
    seen_pick_ids: set[str] = set()
    for row in rows:
        verify_migrating_columns(set(row), HISTORY_COLUMNS, path, "history")
        assert_true(str(row.get("game_date") or "") >= TRACKING_START_DATE, f"{path} contains pre-tracking row {row.get('pick_id')}.")
        pick_id = str(row.get("pick_id") or "")
        assert_true(pick_id not in seen_pick_ids, f"{path} contains duplicate pick_id {pick_id}.")
        seen_pick_ids.add(pick_id)


def verify_dashboard(path: Path) -> None:
    payload = load_json(path)
    assert_true(isinstance(payload, dict), f"{path} must contain a JSON object.")
    assert_true(str(payload.get("tracking_start_date")) == TRACKING_START_DATE, f"{path} tracking_start_date drifted from {TRACKING_START_DATE}.")
    latest_picks = payload.get("latest_picks", [])
    history = payload.get("history", [])
    assert_true(isinstance(latest_picks, list), f"{path} latest_picks must be a list.")
    assert_true(isinstance(history, list), f"{path} history must be a list.")
    assert_true(latest_picks == sorted(latest_picks, key=current_sort_key), f"{path} latest_picks are not sorted by stable board order.")
    assert_true(
        history == sorted(
            history,
            key=dashboard_history_sort_key,
        ),
        f"{path} history is not sorted by date desc and stable board order.",
    )
    for row in latest_picks + history:
        actual_columns = list(row)
        expected_columns = [column for column in DISPLAY_COLUMNS if column in row]
        verify_migrating_columns(set(actual_columns), set(DISPLAY_COLUMNS), path, "dashboard row")
        assert_true(
            actual_columns == expected_columns,
            f"{path} dashboard field order drifted for pick {row.get('pick_id')}.",
        )
        assert_true(str(row.get("game_date") or "") >= TRACKING_START_DATE, f"{path} contains pre-tracking dashboard row {row.get('pick_id')}.")


def verify_refresh_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        '"data/live/current_picks.json"',
        '"data/live/pick_history.json"',
        '"cloudflare-app/data/dashboard.json"',
        '"settle"',
        '"prepare"',
        "scripts\\run_refresh_mode.py",
        "git add -- $trackedFiles",
        "git push",
    ]
    for snippet in required_snippets:
        assert_true(snippet in text, f"{path} is missing required rollout snippet: {snippet}")
    assert_true("git status --porcelain -- $trackedFiles" in text, f"{path} still scans a broader git status scope than the public artifact contract.")


def verify_live_pipeline(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert_true("LIVE_PRODUCTION_FEATURE_COLUMNS" in text, f"{path} no longer references LIVE_PRODUCTION_FEATURE_COLUMNS.")
    assert_true("from train_model import (" in text and "LIVE_PRODUCTION_FEATURE_COLUMNS," in text, f"{path} import block drifted from the live production feature contract.")
    assert_true("from train_model import (\n    FEATURE_COLUMNS," not in text, f"{path} still imports offline FEATURE_COLUMNS into the live publish flow.")


def print_operator_checklist() -> None:
    print("Operator checklist:")
    print("- Cloudflare Pages build command is blank.")
    print("- Cloudflare Pages output directory is cloudflare-app.")
    print("- Cloudflare Pages production branch is master.")
    print("- Prepare should run once after 6:00 AM ET using scripts\\refresh_dashboard.ps1 -Mode prepare.")
    print("- Scheduled mixed refreshes should update statuses and major alerts without reshuffling the morning board.")
    print("- Settle should rerun every 5 minutes after the last scheduled first pitch until the slate is final using scripts\\refresh_dashboard.ps1 -Mode settle.")
    print("- This machine can git push to origin/master non-interactively.")


def verify_public_live_artifacts(
    *,
    current_picks: Path = Path("data/live/current_picks.json"),
    pick_history: Path = Path("data/live/pick_history.json"),
    dashboard: Path = Path("cloudflare-app/data/dashboard.json"),
    refresh_script: Path = Path("scripts/refresh_dashboard.ps1"),
    live_pipeline: Path = Path("scripts/live_pipeline.py"),
) -> None:
    verify_current_picks(current_picks)
    verify_pick_history(pick_history)
    verify_dashboard(dashboard)
    verify_refresh_script(refresh_script)
    verify_live_pipeline(live_pipeline)


def main() -> None:
    args = parse_args()
    verify_public_live_artifacts(
        current_picks=Path(args.current_picks),
        pick_history=Path(args.pick_history),
        dashboard=Path(args.dashboard),
        refresh_script=Path(args.refresh_script),
        live_pipeline=Path(args.live_pipeline),
    )
    print("Repo-side public/live artifact verification passed.")
    print_operator_checklist()


if __name__ == "__main__":
    main()
