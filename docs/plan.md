# Elite-Pick Precision Plan

## Goal

Improve held-out elite-pick precision without introducing leakage.

## Success Criteria

- Increase elite-pick precision by at least 8% versus the current held-out baseline.
- Maintain or improve PR-AUC.
- Do not reduce elite-pick sample size by more than 25% unless precision improves by at least 15%.
- Keep all scripts runnable.
- Document all changes and experiment outcomes.

## Milestones

1. Establish the current baseline.
   - Run the existing backtest path on the current engineered dataset.
   - Record model family, feature profile, threshold, PR-AUC, and elite-pick precision on the held-out test window.
   - Record elite-pick sample size and false-positive counts.

2. Diagnose elite false positives.
   - Review holdout ranked outputs for elite picks that missed.
   - Check whether misses cluster around weak absolute probabilities, sparse handedness splits, or high-missingness features.
   - Reconfirm that training, thresholding, calibration, and live-style tier assignment remain time-safe.

3. Test precision-focused improvements one at a time.
   - Thresholding logic: evaluate absolute probability gates and date-slate-aware elite selection rules.
   - Calibration: test whether calibrated probabilities improve elite precision without hurting PR-AUC.
   - Feature filtering: remove or downweight unstable high-missingness or low-sample features if they drive false positives.
   - Leakage checks: confirm no same-day or post-outcome information enters training or elite selection.
   - Class balance handling: test class weighting only if it helps precision without degrading ranking quality.

4. Validate and keep only winning changes.
   - Run full held-out validation after each experiment.
   - Append each result to `docs/progress.md`.
   - Retain only changes that improve elite-pick precision while satisfying the constraints.

5. Finish with final verification.
   - Run the relevant test suite and the final backtest path successfully.
   - Summarize final metrics, key code changes, and remaining risks.

6. Prepare the production rollout.
   - Fix live-pipeline and dashboard blockers that would prevent a safe publish.
   - Update the dashboard semantics so `elite` is clearly presented as a narrower subset while keeping the tier contract stable.
   - Verify the `master`-driven prepare/publish/dashboard path before release.
