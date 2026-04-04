# Elite-Pick Precision Progress

## Baseline

- Date: 2026-04-03
- Command path: `train_model.run_backtest(data/mlb_player_game_real.csv, model_name="logistic", feature_profile="live_shrunk", compare_against="live_plus", calibration="sigmoid", selection_metric="pr_auc")`
- Selected candidate: `logistic::live_shrunk`
- Missingness threshold: `0.35`
- Holdout window: `2024-08-01` through `2024-09-30` (`16,118` rows across `61` dates)
- Holdout PR-AUC: `0.1599`
- Elite policy: top `1%` of each date slate
- Elite sample size: `194`
- Elite HR count: `46`
- Elite false positives: `148`
- Elite precision: `0.2371`

Key baseline observations:

- Elite labeling was percentile-only, so the model promoted several daily top-ranked sluggers to elite even when their absolute score was not especially strong.
- False positives were concentrated among repeat superstar bats, which is consistent with a ranking rule that is too permissive at the very top of each slate.
- The historical engineered dataset still does not contain the intended `*_shrunk` columns, so the selected `live_shrunk` profile is only using the non-missing overlap of that profile.
- Logistic search already considered `class_weight="balanced"` and rejected it; the selected baseline remained unweighted.
- Calibration already stayed inside the existing time-aware training-only path, so no leakage issue was found there.

## Experiments

### Experiment 1: Conservative elite cap from training OOF

- Change: learn elite policy from training OOF scores only, but require at least `75%` of baseline elite sample size.
- Selected policy under that rule: `elite_top_k=3`, no probability floor.
- Holdout PR-AUC: `0.1599`
- Holdout elite sample size: `169`
- Holdout elite precision: `0.2485`
- Precision lift vs baseline: `+4.8%`
- Decision: rejected. Precision improved, but not enough to reach the `+8%` target.

### Experiment 2: Elite cap aligned to repo success rule

- Change: keep the same training-OOF elite-policy search, but allow sample reductions larger than `25%` when training elite precision improves by at least `15%`, matching the mission rule.
- Selected policy: `elite_top_k=1`, no probability floor.
- Holdout PR-AUC: `0.1599`
- Holdout elite sample size: `61`
- Holdout elite HR count: `18`
- Holdout elite false positives: `43`
- Holdout elite precision: `0.2951`
- Precision lift vs baseline: `+24.5%`
- Sample-size change vs baseline: `-68.6%`
- Decision: kept. The sample reduction is larger than `25%`, but the precision gain is larger than the required `15%`, so it satisfies the stated constraint.

What was investigated but not retained:

- Calibration: kept the existing time-aware search. No separate calibration change was needed once elite selection was tightened.
- Feature filtering: the biggest issue was not a single noisy feature but the percentile-only elite labeling. The missing historical `*_shrunk` features remain a known limitation rather than a retained change.
- Leakage checks: the new elite rule is learned only from training OOF predictions and then applied to the holdout window by date, so the change does not introduce holdout leakage.
- Class balance handling: the existing logistic search already evaluated balanced weighting and the selected model still preferred `class_weight=None`.

## Final

- Final retained change: elite picks now use a training-OOF-tuned confidence policy instead of raw top-1%-by-date percentile labeling.
- Final policy: `elite_top_k=1`, `elite_probability_floor=None`, `elite_percentile_floor=0.99`
- Final holdout PR-AUC: `0.1599` (`unchanged`)
- Final elite precision: `0.2951` (baseline `0.2371`, `+24.5%`)
- Final elite sample size: `61` (baseline `194`, `-68.6%`)
- Final elite false positives: `43` (baseline `148`)

Verification:

- `python -m unittest tests.test_model_search` passed.
- `python -m unittest tests.test_live_pipeline` passed after fixing the rollout-critical fast-refit, settlement, and dashboard builder regressions.

Production rollout implementation:

- `fast_refit` now reuses persisted live metadata, including `confidence_policy`, instead of silently drifting back to default feature-profile assumptions.
- When live metadata is missing, the training path bootstraps by running the full search flow once and then persists the approved configuration for later fast refits.
- Dashboard artifact generation now always upserts current rows into history before trimming the active board, which restores the settled-history behavior expected by publish failures and local dashboard rebuilds.
- The public dashboard keeps the same tier names and default filters (`elite + strong`), but the copy and legend now describe `elite` as the narrowest high-conviction subset instead of the whole public slate.
- `dashboard.json` now carries additive `confidence_policy` and `tier_guide` fields so the frontend can explain the current elite policy without changing the existing row contract.

Residual risks:

- Historical `live_shrunk` remains only partially realized until the missing `*_shrunk` features are actually engineered into the offline dataset.
