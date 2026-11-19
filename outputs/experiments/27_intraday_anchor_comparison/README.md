# Daily anchors for the 15:30 ridge update

This experiment compares the original 58-feature daily LightGBM, the tuned
61-feature daily LightGBM, and the fixed 50/50 log-space blend of tuned LightGBM
and the Notebook 16e CNN–Inception–LSTM hybrid.

## Design

- Each daily anchor forecasts with information available at the previous close.
  The two newer daily anchors are refitted from the repository recipes. The
  2022–2025 folds reuse the published historical validation selections.
  The 2021 stacking forecasts select settings on 2020 and refit through 2020.
- The regular-session evaluation uses the original 15:30 checkpoint feature
  query, the same eligible date/PERMNO keys and labels for every anchor, and exact
  daily fallback on missing checkpoints.
- Each refitted annual ridge predicts the difference between log1p auction
  shares and log1p of its own daily anchor. Every refit uses the original 73
  source features with the corresponding daily forecast as one input. The
  rolling cutoff, training row caps, penalty grid, and historical clipping are
  shared. The original saved ridge is also reconstructed and verified against
  its published annual metrics.
- The refitted ridges use the same ordered, seeded historical samples. The
  original saved ridge used its own saved fit. The matched refit of the original
  anchor gives a comparison that holds the ridge samples fixed as well.
- Report log RMSE, stock-level WAPE, and signed share bias. Development metrics
  average 2022–2024 years equally. Paired intervals use 5,000 circular
  moving-block draws of 20 trading dates within years. These are retrospective
  comparisons, including the previously inspected 2025 period.

## Run from the project root

Use the model dependencies in `requirements-models.txt`. The hybrid stage uses
Apple MPS when available.

```sh
.venv/bin/python notebooks/compare_intraday_anchors.py tree
.venv/bin/python notebooks/compare_intraday_anchors.py hybrid
.venv/bin/python notebooks/compare_intraday_anchors.py compare
```

The standalone runner uses the standard Notebook 05 panel, or the copy in the
midterm cache if the standard path is absent. Set `AUCTION_DAILY_PANEL` to use a
different local copy. The `compare` stage can score available complete anchors
while the other daily refits are in progress. It verifies coverage, key and
label agreement, the original annual metrics, and missing-checkpoint fallback.
`annual_metrics.csv`, `period_metrics.csv`, and `paired_intervals.csv` summarize
the latest run. Ridge sample fingerprints, validation trials, fitted parameters,
and matched prediction Parquets are saved alongside those tables.

The daily refits use local historical samples and fitted models. This run uses
LightGBM 4.6.0, while Notebook 16a recorded 4.7.0. The hybrid uses PyTorch 2.15.0
on Apple MPS. `daily_refit_audit.csv` records differences from the published
daily scores on their full universe. The matched intraday results use the
actual refitted forecasts and the common regular-session universe.

## Tuned LightGBM result

| Daily anchor and update | 2022–24 daily log RMSE | 2022–24 15:30 log RMSE | 2022–24 15:30 WAPE | 2025 15:30 log RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original LightGBM and saved ridge | 0.46949 | 0.45995 | 34.59% | 0.40916 |
| Original LightGBM and matched ridge refit | 0.46949 | 0.45912 | 34.53% | 0.40887 |
| Tuned LightGBM and matched ridge refit | 0.46407 | 0.45471 | 33.98% | 0.40418 |

The tuned pairing beats the original matched refit by 0.00441 log RMSE in
development (95% paired date-block interval of −0.00646 to −0.00247). Its own
15:30 update reduces development log RMSE by 0.00937 from the tuned daily
forecast. The tuned pairing's development share bias is −12.24%, versus
−11.80% for the original saved pairing. Both updates are evaluated on the
same 1,260,168 regular-session stock-days across 2022–2025.

## Complete three-anchor comparison

| Pairing | 2022–24 log RMSE | 2022–24 WAPE | 2022–24 bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original daily LightGBM + saved ridge | 0.45995 | 34.59% | −11.80% | 0.40916 | 31.78% |
| Tuned daily LightGBM + refitted ridge | 0.45471 | 33.98% | −12.24% | 0.40418 | 31.46% |
| 50/50 daily neural blend + refitted ridge | 0.45396 | 34.03% | −12.25% | 0.40156 | 31.01% |

The blend's development log-RMSE advantage over the tuned pairing is 0.00074,
with a 95% paired date-block interval of −0.00344 to +0.00300. Its WAPE is
0.050 percentage points higher. The 2025 log-RMSE advantage is 0.00263, with
interval −0.00773 to +0.00206. The blend's 2025 share bias is −7.99%, compared
with −5.84% for the tuned pairing.

The tuned LightGBM plus ridge is the preferred main pairing given the small,
uncertain development gain and added complexity of the neural blend. The blend
remains a challenger. This recommendation uses development evidence and reports
2025 as previously inspected confirmation evidence.

## Verification and interpretation

All three anchors were scored on 1,260,168 common regular-session stock-days.
The original daily and intraday scores reconcile with the saved publication.
Chronology, key and label matching, checkpoint coverage, forecast validity, and
exact missing-checkpoint fallback passed the runner's checks. A synthetic
known-residual calculation also verified the seven-penalty selection and fallback.
Saved ridge parameter reconstruction was checked against the written predictions.

Daily models were rebuilt because their original fitted artifacts and row-level
forecasts were absent from this checkout. These fresh daily refits differ slightly
from the published daily scores, as recorded in daily_refit_audit.csv. Original
daily parameters, feature definitions and historical validation choices were reused.
The ridge comparison holds its source-feature specification and sampling fixed.
The three new ratio features influence the ridge through the daily anchor forecast.

The date-block intervals condition on fitted forecasts. They omit daily and
intraday refitting uncertainty and accumulated model-selection uncertainty.
Quarterly ridge refitting was outside this experiment's annual fitting design.
