"""Inspect archived Ballpark Pal exports for completeness and schema health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.ballparkpal.validation import validate_export_file

EXPORT_KEYS = ("batters", "pitchers", "teams", "games")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="data/ballparkpal/raw", help="Archive root directory.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any issues are detected.")
    return parser.parse_args()


def _latest_file_for_key(date_dir: Path, export_key: str) -> Path | None:
    files = sorted(date_dir.glob(f"*_{export_key}.xlsx"))
    return files[-1] if files else None


def inspect_date(date_dir: Path) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    for export_key in EXPORT_KEYS:
        file_path = _latest_file_for_key(date_dir, export_key)
        if file_path is None:
            exports[export_key] = {"status": "missing", "path": None, "validation": None}
            continue
        validation = validate_export_file(file_path, export_key).to_dict()
        exports[export_key] = {
            "status": "valid" if validation["is_valid"] else "invalid",
            "path": str(file_path),
            "columns": validation["detected_columns"][:25],
            "validation": validation,
        }
    complete_set = all(exports[key]["status"] in {"valid", "invalid"} for key in EXPORT_KEYS)
    all_valid = all(exports[key]["status"] == "valid" for key in EXPORT_KEYS)
    return {
        "date": date_dir.name,
        "complete_set": complete_set,
        "all_valid": all_valid,
        "exports": exports,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    date_dirs = sorted([path for path in root_dir.iterdir() if path.is_dir() and len(path.name) == 10])
    results = [inspect_date(path) for path in date_dirs]

    total = len(results)
    complete = sum(1 for row in results if row["complete_set"])
    valid = sum(1 for row in results if row["all_valid"])
    print(f"Scanned dates: {total}")
    print(f"Complete 4-file sets: {complete}")
    print(f"All 4 files valid: {valid}")
    print("")
    for row in results:
        status = "OK" if row["all_valid"] else ("PARTIAL" if row["complete_set"] else "MISSING")
        print(f"{row['date']}: {status}")
        for export_key in EXPORT_KEYS:
            export = row["exports"][export_key]
            print(f"  - {export_key}: {export['status']}")
            if export["validation"] and not export["validation"]["is_valid"]:
                print(f"    error: {export['validation']['error']}")

    report_path = root_dir / "inspection_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote inspection report: {report_path}")

    if args.strict and any(not row["all_valid"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

