from pathlib import Path
import unittest


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "manual-live-refresh.yml"


class WorkflowScheduleTests(unittest.TestCase):
    def test_live_refresh_keeps_high_frequency_schedule_and_morning_safety_schedule(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn('- cron: "*/5 * * * *"', workflow)
        self.assertIn('- cron: "17 10,11 * * *"', workflow)


if __name__ == "__main__":
    unittest.main()
