# Audit of unused forward period availability

**No ready unused forward period was found in the audited local sources.** Missing post-2025 coverage in required components: auction_labels, crsp_daily, checkpoint_1530, market_daily, sector_daily, calendar.

```text
           family  files          first_date           last_date  forward_rows
   auction_labels     11 2015-01-02 00:00:00 2025-12-31 00:00:00           0.0
       crsp_daily     15 2011-01-03 00:00:00 2025-12-31 00:00:00           0.0
  checkpoint_1530     11 2015-01-02 00:00:00 2025-12-31 00:00:00           0.0
     market_daily      1 2011-01-03 00:00:00 2025-12-31 00:00:00           0.0
     sector_daily      1 2011-01-03 00:00:00 2025-12-31 00:00:00           0.0
         calendar      1 2011-01-03 00:00:00 2025-12-31 00:00:00           0.0
identifier_bridge     15 2011-01-03 00:00:00 2025-12-31 00:00:00           0.0
 industry_history      1                 NaN                 NaN           NaN
  security_master      1 2011-01-03 00:00:00 2025-12-31 00:00:00           0.0
```

The source configuration ends 2025-12-31. The FOMC event filename extends into 2026, but this is not auction/model coverage. Only schemas, date ranges, counts and filenames were read during this audit; no forward auction volume values, predictions or errors were inspected. The historical evaluation preceding this audit used only 2022–2025 saved forecasts.

## Required before a new score

Obtain complete compatible forward auction records, CRSP inputs, 15:30 aggregates, market/sector context and calendar/identifier support. Verify date/key coverage, early-close handling, extraction cutoffs and label-validity flags without exploratory outcome analysis. Record provenance and prior-use attestation. Freeze a non-overlapping evaluation window and minimum duration, forecast delivery rules, annual training and historical tuning procedure, and economically justified acceptance criteria before scoring. Use only earlier out-of-time forecasts to fit every correction. Log forecasts before outcomes when feasible.

A 2026 model would require the locked daily and intraday training procedures to be refitted using earlier data; a saved 2025 coefficient file must not silently be treated as that annual refit. Model coefficients and preprocessing, eligible universe and cutoff assumptions must be fingerprinted. Do not claim WAPE noninferiority without an independently justified margin. Score once at the fixed end, without optional stopping or retrospective subgroup changes.

No new forward period was scored. Existing 2025 data cannot be reused as a fresh holdout. Data stored outside this local workspace were not audited.
