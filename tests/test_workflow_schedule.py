from pathlib import Path
import unittest


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "manual-live-refresh.yml"
RECOVERY_WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "morning-live-recovery.yml"


class WorkflowScheduleTests(unittest.TestCase):
    def test_live_refresh_keeps_high_frequency_schedule_and_morning_safety_schedule(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn('- cron: "*/5 * * * *"', workflow)
        self.assertIn('- cron: "17 10,11 * * *"', workflow)

    def test_morning_recovery_is_triggered_after_scheduled_refreshes(self) -> None:
        workflow = RECOVERY_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("workflow_run:", workflow)
        self.assertIn("- Live Refresh", workflow)
        self.assertIn("- completed", workflow)
        self.assertIn("github.event.workflow_run.event == 'schedule'", workflow)
        self.assertIn("manual-live-refresh.yml", workflow)
        self.assertIn("inputs: { mode: 'auto' }", workflow)


if __name__ == "__main__":
    unittest.main()
