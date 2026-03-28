from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import check_cloudflare_dashboard_freshness


class CloudflareFreshnessTests(unittest.TestCase):
    def test_main_returns_zero_when_remote_matches_local_generated_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "dashboard.json"
            local_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-03-28T06:03:47.753303+00:00",
                        "latest_available_date": "2026-03-28",
                        "overview": {"latest_slate_size": 12},
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with patch(
                "scripts.check_cloudflare_dashboard_freshness.fetch_json",
                return_value=(
                    {
                        "generated_at": "2026-03-28T06:03:47.753303+00:00",
                        "latest_available_date": "2026-03-28",
                        "overview": {"latest_slate_size": 12},
                    },
                    {"cf-cache-status": "MISS", "age": "0"},
                ),
            ):
                with patch(
                    "sys.argv",
                    [
                        "check_cloudflare_dashboard_freshness.py",
                        "--dashboard-url",
                        "https://example.com/data/dashboard.json",
                        "--local-dashboard",
                        str(local_path),
                    ],
                ):
                    with redirect_stdout(stdout):
                        result = check_cloudflare_dashboard_freshness.main()

            self.assertEqual(result, 0)
            self.assertIn("Cloudflare dashboard is fresh", stdout.getvalue())

    def test_main_returns_one_when_remote_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "dashboard.json"
            local_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-03-28T06:03:47.753303+00:00",
                        "latest_available_date": "2026-03-28",
                        "overview": {"latest_slate_size": 12},
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch(
                "scripts.check_cloudflare_dashboard_freshness.fetch_json",
                return_value=(
                    {
                        "generated_at": "2026-03-28T05:50:00+00:00",
                        "latest_available_date": "2026-03-27",
                        "overview": {"latest_slate_size": 10},
                    },
                    {"cf-cache-status": "HIT", "age": "611"},
                ),
            ):
                with patch(
                    "sys.argv",
                    [
                        "check_cloudflare_dashboard_freshness.py",
                        "--dashboard-url",
                        "https://example.com/data/dashboard.json",
                        "--local-dashboard",
                        str(local_path),
                    ],
                ):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        result = check_cloudflare_dashboard_freshness.main()

            self.assertEqual(result, 1)
            self.assertIn("Cloudflare dashboard is stale", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
