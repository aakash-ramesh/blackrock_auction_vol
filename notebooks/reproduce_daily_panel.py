"""Rebuild Notebook 05 in isolation, then audit against saved research artifacts.

Run from the project root with .venv/bin/python notebooks/reproduce_daily_panel.py.
The historical notebook and outputs/eda artifacts are never overwritten.
"""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reproduction/01_daily_panel"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(OUT / ".ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(OUT / ".jupyter_runtime"))

import duckdb
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    started = datetime.now(timezone.utc).isoformat()
    source = ROOT / "notebooks/05_feature_engineering_enriched_daily_panel.ipynb"
    paths = sorted((ROOT / "blackrock_data/stock_daily/auction_daily").glob("year=*/*.parquet"))
    paths += sorted((ROOT / "blackrock_data/stock_daily/crsp_daily").glob("year=*/*.parquet"))
    paths += [ROOT / "blackrock_data" / name for name in [
        "market/market_daily.parquet", "sector/sector_daily.parquet",
        "events/event_calendar.parquet", "reference/industry_history.parquet",
        "quality/auction_source_anomalies.parquet",
    ]]
    assert len(paths) == 31, f"Expected 11 auction + 15 CRSP + 5 context files, got {len(paths)}"
    inventory = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                  "sha256": sha256(p)} for p in paths]
    pd.DataFrame(inventory).to_csv(OUT / "source_inventory.csv", index=False)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["duckdb", "numpy", "pandas", "matplotlib", "seaborn", "nbclient"]}}
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2))
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (OUT / "requirements.lock.txt").write_text(frozen)

    nb = nbformat.read(source, as_version=4)
    old = 'OUT = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering"'
    new = 'OUT = PROJECT_ROOT / "outputs" / "reproduction" / "01_daily_panel"'
    replacements = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            replacements += cell.source.count(old)
            cell.source = cell.source.replace(old, new)
            cell.outputs = []
            cell.execution_count = None
    assert replacements == 1, "Unexpected Notebook 05 setup; review before running."
    nb.metadata["reproduction"] = {"source_sha256": sha256(source),
                                  "change": "Output directory only; feature logic unchanged."}
    executed = OUT / "05_feature_engineering_rebuilt.ipynb"
    if "--audit-only" not in sys.argv:
        nbformat.write(nb, executed)
        print("Executing isolated Notebook 05...", flush=True)
        client = NotebookClient(nb, timeout=None, kernel_name="auction-local",
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, executed)
    print("Panel rebuilt; checking source reconciliation and historical features...", flush=True)

    con = duckdb.connect()
    con.execute("SET threads=4")
    panel = OUT / "enriched_daily_panel.parquet"
    con.execute(f"CREATE VIEW panel AS SELECT * FROM read_parquet('{panel}')")
    checks = []

    def check(name, observed, expected):
        passed = bool(observed == expected)
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": passed})

    overview = con.sql("""SELECT count(*) AS rows, count(DISTINCT permno) securities,
        count(DISTINCT date) trading_days, count(*) FILTER (WHERE eligible_target) eligible,
        count(*)-count(DISTINCT(date,permno)) duplicate_keys,
        count(*) FILTER (WHERE eligible_target AND (auction_volume IS NULL OR NOT observed_target
                          OR exclude_source_anomaly)) invalid_eligibility,
        count(*) FILTER (WHERE eligible_target AND date=DATE '2019-08-12') anomaly_eligible,
        count(*) FILTER (WHERE auction_volume IS NOT NULL AND auction_volume<=0) nonpositive_labels
        FROM panel""").df().iloc[0]
    for key, expected in {"rows": 3594404, "securities": 2149, "trading_days": 2766,
                          "eligible": 3499401, "duplicate_keys": 0, "invalid_eligibility": 0,
                          "anomaly_eligible": 0, "nonpositive_labels": 0}.items():
        check(key, int(overview[key]), expected)

    annual = con.sql("""SELECT year, count(*) stock_days,
        count(*) FILTER (WHERE eligible_target) eligible_targets FROM panel GROUP BY year ORDER BY year""").df()
    # The source coverage table's observed counts include the excluded anomaly labels.
    saved_eligible = [337566,325638,318040,313807,308240,303168,321564,328634,318679,316212,307853]
    check("annual_eligible_counts", annual.eligible_targets.tolist(), saved_eligible)
    annual.to_csv(OUT / "annual_reconciliation.csv", index=False)

    catalog = pd.read_csv(OUT / "approved_enriched_daily_features.csv")
    old_catalog = pd.read_csv(ROOT / "outputs/eda/05_feature_engineering/approved_enriched_daily_features.csv")
    check("feature_count", len(catalog), 99)
    check("feature_catalog_matches", catalog.equals(old_catalog), True)
    current = pd.read_csv(OUT / "enriched_feature_missingness.csv")
    old_missing = pd.read_csv(ROOT / "outputs/eda/05_feature_engineering/enriched_feature_missingness.csv")
    comparison = current.merge(old_missing, on="feature", suffixes=("_rebuilt", "_saved"), validate="one_to_one")
    comparison["missing_difference"] = comparison.missing_rebuilt - comparison.missing_saved
    comparison.to_csv(OUT / "missingness_reconciliation.csv", index=False)
    check("features_with_changed_null_counts", int(comparison.missing_difference.ne(0).sum()), 0)
    check("missingness_features_compared", len(comparison), 99)

    crsp = str(ROOT / "blackrock_data/stock_daily/crsp_daily/year=*/*.parquet")
    auction = str(ROOT / "blackrock_data/stock_daily/auction_daily/year=*/*.parquet")
    con.execute(f"CREATE VIEW source_crsp AS SELECT * FROM read_parquet('{crsp}', union_by_name=true)")
    con.execute(f"CREATE VIEW source_auction AS SELECT * FROM read_parquet('{auction}', union_by_name=true)")
    for table in ["source_crsp", "source_auction"]:
        duplicates = con.sql(f"SELECT count(*)-count(DISTINCT(date,permno)) FROM {table}").fetchone()[0]
        check(f"{table}_duplicate_keys", int(duplicates), 0)
    source_reconciliation = con.sql("""SELECT
        count(*) FILTER (WHERE p.auction_volume IS DISTINCT FROM a.auction_volume) target_mismatches,
        count(*) FILTER (WHERE p.observed_target IS DISTINCT FROM coalesce(a.target_found,false)) flag_mismatches
        FROM panel p LEFT JOIN source_auction a USING(date,permno)""").fetchone()
    check("target_values_match_source", int(source_reconciliation[0]), 0)
    check("target_flags_match_source", int(source_reconciliation[1]), 0)
    con.execute("CREATE TEMP TABLE sampled_stocks AS SELECT DISTINCT permno FROM panel ORDER BY hash(permno) LIMIT 16")
    # Separate pandas implementation of lag/window logic on raw rows, including pre-2015 history.
    raw = con.sql("""SELECT c.date,c.permno,c.daily_volume,a.auction_volume
        FROM source_crsp c LEFT JOIN source_auction a USING(date,permno)
        WHERE c.permno IN (SELECT permno FROM sampled_stocks) ORDER BY c.permno,c.date""").df()
    parts = []
    for _, group in raw.groupby("permno", sort=False):
        group = group.copy()
        prior = group.auction_volume.shift(1)
        group["expected_median20"] = prior.rolling(20, min_periods=1).median()
        group["expected_lag1_auction"] = prior
        group["expected_lag1_volume"] = group.daily_volume.shift(1)
        group["expected_history"] = prior.notna().cumsum()
        parts.append(group)
    expected = pd.concat(parts)
    actual = con.sql("""SELECT date,permno,median20_auction_volume,lag1_auction_volume,
        lag1_daily_volume,prior_auction_observations FROM panel
        WHERE permno IN (SELECT permno FROM sampled_stocks)""").df()
    aligned = actual.merge(expected, on=["date", "permno"], validate="one_to_one")
    timing_rows = []
    for feature, reference in [("median20_auction_volume", "expected_median20"),
                               ("lag1_auction_volume", "expected_lag1_auction"),
                               ("lag1_daily_volume", "expected_lag1_volume"),
                               ("prior_auction_observations", "expected_history")]:
        matches = np.isclose(aligned[feature].to_numpy(dtype=float), aligned[reference].to_numpy(dtype=float),
                             equal_nan=True, rtol=1e-12, atol=1e-9)
        mismatches = int((~matches).sum())
        check(f"independent_{feature}_mismatches", mismatches, 0)
        timing_rows.append({"feature": feature, "stocks": 16, "rows": len(aligned), "mismatches": mismatches})
    pd.DataFrame(timing_rows).to_csv(OUT / "independent_history_checks.csv", index=False)

    forbidden = {"auction_volume", "log1p_auction_volume", "auction_price", "auction_share_daily_volume",
                 "current_daily_volume", "current_return", "current_close"}
    check("forbidden_features_in_catalog", sorted(forbidden & set(catalog.variable)), [])
    numeric = con.sql("DESCRIBE panel").df()
    numeric_types = ("DOUBLE", "FLOAT", "DECIMAL", "BIGINT", "INTEGER", "SMALLINT", "TINYINT", "HUGEINT")
    features = [r.column_name for r in numeric.itertuples() if r.column_name in set(catalog.variable)
                and r.column_type.startswith(numeric_types)]
    finite_counts = con.sql("SELECT " + ",".join(
        f'count(*) FILTER (WHERE eligible_target AND "{f}" IS NOT NULL AND NOT isfinite("{f}")) AS "{f}"'
        for f in features) + " FROM panel").df()
    check("nonfinite_numeric_feature_values", int(finite_counts.iloc[0].sum()), 0)
    con.close()
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv", index=False)
    passed = all(row["passed"] for row in checks)
    manifest = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "source_notebook_sha256": sha256(source), "source_files": len(paths),
                "panel_bytes": panel.stat().st_size, "panel_sha256": sha256(panel),
                "all_checks_passed": passed, "checks": checks}
    (OUT / "validation_summary.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"all_checks_passed": passed, "checks": len(checks), "panel": str(panel)}, indent=2), flush=True)
    if not passed:
        raise AssertionError("Panel validation failed; inspect validation_checks.csv.")


if __name__ == "__main__":
    main()
