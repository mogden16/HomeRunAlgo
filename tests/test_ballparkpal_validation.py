from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.ballparkpal.validation import validate_export_file


def test_validate_export_file_accepts_valid_xlsx(tmp_path: Path) -> None:
    output_path = tmp_path / "batters.xlsx"
    frame = pd.DataFrame(
        {
            "Player Name": ["A", "B"],
            "Team": ["NYY", "LAD"],
            "Opponent": ["BOS", "SFG"],
            "HR Probability": [0.12, 0.09],
        }
    )
    frame.to_excel(output_path, index=False)

    result = validate_export_file(output_path, "batters")
    assert result.is_valid
    assert result.workbook_open_ok
    assert result.expected_columns_matched >= 2


def test_validate_export_file_rejects_html_masked_as_xlsx(tmp_path: Path) -> None:
    output_path = tmp_path / "teams.xlsx"
    output_path.write_text("<html><body>login</body></html>", encoding="utf-8")

    result = validate_export_file(output_path, "teams")
    assert not result.is_valid
    assert result.error is not None
    assert "signature" in result.error.lower() or "workbook" in result.error.lower()

