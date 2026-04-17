from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from tools.ballparkpal.manifest import build_manifest
from tools.ballparkpal.overlay import build_validation_overlay
from tools.ballparkpal.validator import validate_workbook_file


class BallparkPalValidationTests(unittest.TestCase):
    def _write_workbook(self, path: Path, *, headers: list[str], rows: list[list[object]], sheet_name: str = "Batters") -> Path:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = sheet_name
        worksheet.append(headers)
        for row in rows:
            worksheet.append(row)
        workbook.save(path)
        return path

    def test_valid_xlsx_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "batters.xlsx"
            self._write_workbook(
                path,
                headers=["GameDate", "Batter", "HomeRunProbability", "HitProbability"],
                rows=[["2026-04-17", "Alpha", 0.21, 0.54]],
            )

            finding = validate_workbook_file(path, requested_date="2026-04-17", export_name="batters")

            self.assertTrue(finding.valid)
            self.assertEqual(finding.workbook_date, "2026-04-17")
            self.assertEqual(finding.row_count, 1)

    def test_html_masquerading_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "batters.xlsx"
            path.write_text("<html><body>login required</body></html>", encoding="utf-8")

            finding = validate_workbook_file(path, requested_date="2026-04-17", export_name="batters")

            self.assertFalse(finding.valid)
            self.assertTrue(any("HTML" in error or "zip archive" in error for error in finding.errors))

    def test_workbook_date_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "games.xlsx"
            self._write_workbook(
                path,
                headers=["GameDate", "HomeTeam", "AwayTeam"],
                rows=[["2026-04-16", "NYY", "BOS"]],
                sheet_name="Games",
            )

            finding = validate_workbook_file(path, requested_date="2026-04-17", export_name="games")

            self.assertFalse(finding.valid)
            self.assertTrue(any("does not match requested date" in error or "mismatched dates" in error for error in finding.errors))

    def test_overlay_output_builds_only_requested_date_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            batters_path = base / "batters.xlsx"
            pitchers_path = base / "pitchers.xlsx"
            teams_path = base / "teams.xlsx"
            games_path = base / "games.xlsx"
            self._write_workbook(
                batters_path,
                headers=["GameDate", "Batter", "HomeRunProbability", "HitProbability"],
                rows=[
                    ["2026-04-17", "Alpha", 0.21, 0.54],
                    ["2026-04-16", "Old Alpha", 0.11, 0.40],
                ],
                sheet_name="Batters",
            )
            self._write_workbook(
                pitchers_path,
                headers=["GameDate", "Pitcher", "RunsAllowed", "HomeRunsAllowed"],
                rows=[["2026-04-17", "Pitcher A", 2.1, 0.3]],
                sheet_name="Pitchers",
            )
            self._write_workbook(
                teams_path,
                headers=["GameDate", "Team", "RunsAllowed"],
                rows=[["2026-04-17", "NYY", 3.0]],
                sheet_name="Teams",
            )
            self._write_workbook(
                games_path,
                headers=["GameDate", "HomeTeam", "AwayTeam"],
                rows=[["2026-04-17", "NYY", "BOS"]],
                sheet_name="Games",
            )

            payload = build_validation_overlay(
                {
                    "batters": batters_path,
                    "pitchers": pitchers_path,
                    "teams": teams_path,
                    "games": games_path,
                },
                "2026-04-17",
            )

            self.assertEqual(payload["requested_date"], "2026-04-17")
            self.assertEqual(len(payload["batters"]), 1)
            self.assertEqual(payload["batters"][0]["name"], "Alpha")
            self.assertEqual(len(payload["pitchers"]), 1)

    def test_manifest_contains_validation_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "batters.xlsx"
            self._write_workbook(
                path,
                headers=["GameDate", "Batter", "HomeRunProbability", "HitProbability"],
                rows=[["2026-04-17", "Alpha", 0.21, 0.54]],
            )
            finding = validate_workbook_file(path, requested_date="2026-04-17", export_name="batters")
            manifest = build_manifest(requested_date="2026-04-17", validations=[finding])
            self.assertTrue(manifest.overall_valid)
            self.assertEqual(manifest.downloads[0]["validation_result"], "valid")

