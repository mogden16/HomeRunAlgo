# Ballpark Pal Feature Experiment

This experiment checks whether Ballpark Pal's pregame projections add signal beyond the current HomeRunAlgo feature set.

## What it does
- Loads a season-level batter-game dataset.
- Joins Ballpark Pal `batters`, `pitchers`, `teams`, and `games` exports onto each row.
- Compares the current model against several Ballpark Pal-augmented variants.
- Writes an augmented dataset, a comparison report, and a coefficient ranking report.

## Why this is separate from production
- The archived Ballpark Pal data may not be a guaranteed point-in-time snapshot.
- This should be used to measure incremental lift first, not to replace the current model blindly.

## Required inputs
- A matching season batter-game dataset, for example a generated 2025 dataset.
- The Ballpark Pal export archive under `data/ballparkpal/raw`.

## How to generate the matching season dataset
Example for 2025:
```powershell
python generate_data.py --start-date 2025-03-18 --end-date 2025-09-28 --output data/live/model_training_dataset_2025.csv
```

## How to run the experiment
```powershell
python tools/ballparkpal/compare_model_features.py --source-data-path data/live/model_training_dataset_2025.csv --archive-root data/ballparkpal/raw --output-dir data/ballparkpal/analysis
```

## Outputs
- `ballparkpal_augmented_dataset.csv`
- `ballparkpal_comparison_report.json`
- `ballparkpal_feature_coefficients.csv`

## What to look for
- Whether `baseline_plus_bp_all` improves PR-AUC / ROC-AUC over `baseline`
- Whether the smaller `baseline_plus_bp_core` set gives most of the lift
- Which Ballpark Pal features have the strongest coefficients in the augmented logistic model

## Practical interpretation
- If the full Ballpark Pal set wins but the core set is nearly as good, prefer the smaller set first.
- If only the batter probabilities help, keep the batter projections and skip the broader team/game fields.
- If the lift is tiny or unstable, keep Ballpark Pal as an archive reference, not a live model input.
- If the comparison script reports zero join coverage, do not trust the archive for backtesting yet; it is likely a non-point-in-time snapshot rather than a historical daily archive.
