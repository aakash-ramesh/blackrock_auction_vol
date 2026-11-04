"""Fixed-model residual audit and prior-year out-of-time calibration.

Run with the local .venv and the same DYLD_LIBRARY_PATH used in steps 03/17.
Only a prerequisite 2021 historical model is fitted. The 2022-2025 base models
and their saved predictions remain unchanged.
"""
from pathlib import Path
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/experiments/18_error_diagnosis_calibration"
PREVIOUS = ROOT / "outputs/reproduction/03_compact_nonlinear"
PANEL = ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from notebooks.modeling_utils import fit_preprocessor, apply_preprocessor, forecast_metrics
from notebooks.compare_target_formulations import sha, parameters, offset, SEED

METHODS = ["uncalibrated", "global_share_ratio", "global_log_shift", "global_smearing", "shrunk_decile_ratio"]
FEATURES = json.loads((ROOT / "outputs/eda/09_global_finalists/frozen_global_benchmark.json").read_text())["features"]
TEMPLATE = json.loads((PREVIOUS / "models/preprocessor_2022.json").read_text())
TYPES = [list(TEMPLATE[k]) for k in ["continuous", "boolean", "categorical"]]
ALL_FEATURES = [f for group in TYPES for f in group]
CONTEXT = ["date", "permno", "ticker", "auction_volume", "log1p_auction_volume", "median20_auction_volume",
           "lag1_auction_volume", "lag1_daily_volume", "abnormal_volume_lag1", "lag1_return",
           "lag1_dividend_flag", "lag1_distribution_flag", "lag1_price_adjustment_flag",
           "size_percentile", "liquidity_percentile", "prior_auction_observations",
           "monthly_options_expiry", "quarterly_options_expiry", "is_last_trading_day_month", "is_early_close"]


def progress(message):
    text = datetime.now(timezone.utc).isoformat() + " " + message
    print(text, flush=True)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(text + "\n")


def save_frame(con, frame, path):
    con.register("write_frame", frame)
    con.execute(f"COPY write_frame TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("write_frame")


def setup():
    for folder in [OUT, OUT / "historical_2021", OUT / "predictions", OUT / "parameters"]:
        folder.mkdir(parents=True, exist_ok=True)
    verified = json.loads((PREVIOUS / "reproduction_summary.json").read_text())
    assert verified["all_checks_passed"]
    fingerprints = []
    for r in verified["predictions"]:
        assert sha(ROOT / r["path"]) == r["sha256"]
        fingerprints.append(r)
    for year in range(2022,2026):
        for path in [PREVIOUS / f"models/preprocessor_{year}.json", PREVIOUS / f"models/reconstruction_{year}.json",
                     PREVIOUS / f"models/lgbm_residual_58_l2_{year}.txt", PREVIOUS / f"models/feature_columns_lgbm_residual_58_l2_{year}.json"]:
            fingerprints.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path)})
    panel_record = pd.read_csv(ROOT / "outputs/reproduction/02_benchmark_reconciliation/input_fingerprints.csv")
    panel_hash = panel_record.loc[panel_record.path.str.endswith("enriched_daily_panel.parquet"),"sha256"].iloc[0]
    assert sha(PANEL) == panel_hash
    fingerprints.append({"path":str(PANEL.relative_to(ROOT)),"sha256":panel_hash})
    pd.DataFrame(fingerprints).to_csv(OUT / "input_fingerprints.csv",index=False)
    plan = {"created_utc":datetime.now(timezone.utc).isoformat(), "script_sha256":sha(Path(__file__)),
        "base_model":"Fixed 58-feature squared-error LightGBM with median-relative log target; use verified 2022-2025 forecasts unchanged",
        "development_years":[2022,2023,2024],"confirmation_year":2025,"confirmation_previously_inspected":True,
        "calibration_history":"For evaluation year Y, fit correction only on full-year out-of-time forecasts for Y-1",
        "2021_prerequisite":"Tune on 2020 using 2015-2019; refit on 2016-2020; predict full 2021. Original grid/seeds/sample sizes/processing unchanged.",
        "methods":{
            "uncalibrated":"P",
            "global_share_ratio":"P * sum(A)/sum(P)",
            "global_log_shift":"max(0, expm1(log1p(P) + mean(log1p(A)-log1p(P))))",
            "global_smearing":"max(0, (1+P)*mean((1+A)/(1+P)) - 1)",
            "shrunk_decile_ratio":"Prior-year forecast deciles; ratio sum(A)/sum(P) in each; blend with global ratio using n/(n+10000); multiply P"},
        "decile_shrinkage_pseudo_rows":10000,"calibration_factor_bounds":"none; no parameter tuning or test-year fitting",
        "forecast_bounds":"Original bounds retained in base model; no additional re-clipping after correction",
        "adoption_gate":"Mean development absolute annual aggregate bias improves; mean annual WAPE does not worsen; mean log RMSE deterioration <=0.002; no annual WAPE deterioration >0.002; paired daily-WAPE development CI upper <0",
        "paired_uncertainty":"2000 whole-date bootstrap draws; same-date dependence preserved, serial dependence and multiplicity not corrected",
        "diagnostics":["forecast-size calibration","size/history/events","ex-post actual-volume tail","largest stock/date errors","error concentration","clipping contribution"],
        "excluded_changes":["new target","new predictors","2022-2025 base model refitting","clipping-policy adjustment","anomaly-history adjustment"]}
    (OUT / "experiment_plan.json").write_text(json.dumps(plan,indent=2))
    (OUT / "environment.json").write_text(json.dumps({n:importlib.metadata.version(n) for n in ["numpy","pandas","duckdb","lightgbm"]},indent=2))
    return plan


def load(con, start, end, rows=None, seed=SEED):
    columns = list(dict.fromkeys(CONTEXT + ALL_FEATURES))
    names = ",".join('"'+f+'"' for f in columns)
    sample = f"USING SAMPLE {rows} ROWS (reservoir, {seed})" if rows else ""
    return con.sql(f"SELECT * FROM (SELECT {names} FROM read_parquet('{PANEL}') WHERE eligible_target AND date>=DATE '{start}' AND date<DATE '{end}') {sample}").df()


def history_2021(con):
    destination = OUT / "historical_2021/predictions_2021.parquet"
    if destination.exists():
        manifest_path = OUT / "historical_2021/manifest.json"
        assert manifest_path.exists(), "Historical output is incomplete"
        manifest = json.loads(manifest_path.read_text())
        assert sha(destination) == manifest["prediction_sha256"]
        progress("Reusing checked historical 2021 prerequisite")
        return destination
    progress("Building 2021 historical calibration forecasts: tune strictly on 2020")
    year = 2021
    train = load(con,"2015-01-01","2020-01-01",250000,SEED+year)
    validation = load(con,"2020-01-01","2021-01-01",120000,SEED+100+year)
    save_frame(con,train,OUT / "historical_2021/tune_sample.parquet")
    save_frame(con,validation,OUT / "historical_2021/validation_sample.parquet")
    xt, prep = fit_preprocessor(train,*TYPES)
    xv = apply_preprocessor(validation,prep,*TYPES)
    mask = np.array([f in set(FEATURES) for f in prep["output_sources"]])
    xt,xv = xt[:,mask],xv[:,mask]
    fallback = float(np.median(train.auction_volume))
    target = train.log1p_auction_volume.to_numpy()-offset(train,"median20_log_residual",fallback)
    val_base = offset(validation,"median20_log_residual",fallback)
    val_target = validation.log1p_auction_volume.to_numpy()-val_base
    clip = np.quantile(train.log1p_auction_volume,[0.001,0.999])
    choices = []
    for leaves in [15,31,63]:
        progress(f"2021 prerequisite: tuning leaves={leaves}")
        model = lgb.LGBMRegressor(**parameters(year,leaves,2000,True))
        model.fit(xt,target,eval_set=[(xv,val_target)],eval_metric="rmse",callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
        p = np.clip(val_base+model.predict(xv),*clip)
        choices.append({"leaves":leaves,"iterations":int(model.best_iteration_),"validation_rmse":float(np.sqrt(np.mean((validation.log1p_auction_volume.to_numpy()-p)**2)))})
    best = min(choices,key=lambda r:r["validation_rmse"])
    pd.DataFrame(choices).to_csv(OUT / "historical_2021/hyperparameter_validation.csv",index=False)
    del train,validation,xt,xv
    import gc
    gc.collect()
    refit = load(con,"2016-01-01","2021-01-01",350000,SEED+200+year)
    test = load(con,"2021-01-01","2022-01-01")
    save_frame(con,refit,OUT / "historical_2021/refit_sample.parquet")
    xr,prep = fit_preprocessor(refit,*TYPES)
    xe = apply_preprocessor(test,prep,*TYPES)
    mask = np.array([f in set(FEATURES) for f in prep["output_sources"]])
    fallback = float(np.median(refit.auction_volume))
    target = refit.log1p_auction_volume.to_numpy()-offset(refit,"median20_log_residual",fallback)
    clip = np.quantile(refit.log1p_auction_volume,[0.001,0.999])
    progress(f"2021 prerequisite: final fit leaves={best['leaves']}, trees={best['iterations']}")
    model = lgb.LGBMRegressor(**parameters(year,best["leaves"],max(25,best["iterations"]),False))
    model.fit(xr[:,mask],target)
    raw_log = offset(test,"median20_log_residual",fallback)+model.predict(xe[:,mask])
    frame = test[CONTEXT].copy()
    frame["uncalibrated"] = np.maximum(np.expm1(np.clip(raw_log,*clip)),0)
    save_frame(con,frame,destination)
    model.booster_.save_model(str(OUT / "historical_2021/model.txt"))
    (OUT / "historical_2021/preprocessor.json").write_text(json.dumps(prep,indent=2))
    (OUT / "historical_2021/manifest.json").write_text(json.dumps({"selected":best,"fallback":fallback,"clip_log":clip.tolist(),
        "latest_fitting_date":"2020-12-31","evaluation_year":2021,"rows":len(frame),"prediction_sha256":sha(destination)},indent=2))
    progress(f"2021 prerequisite complete: {len(frame):,} strictly forward forecasts")
    return destination


def fit_corrections(frame):
    actual = frame.auction_volume.to_numpy(dtype=float)
    pred = frame.uncalibrated.to_numpy(dtype=float)
    log_residual = np.log1p(actual)-np.log1p(pred)
    global_ratio = float(actual.sum()/pred.sum())
    edges = np.unique(np.quantile(pred,np.arange(0.1,1,0.1))).tolist()
    bins = np.searchsorted(edges,pred,side="right")
    factors,bin_records = [],[]
    for bucket in range(len(edges)+1):
        selected = bins==bucket
        n = int(selected.sum())
        raw = float(actual[selected].sum()/pred[selected].sum()) if n else global_ratio
        weight = n/(n+10000)
        pooled = weight*raw+(1-weight)*global_ratio
        factors.append(pooled)
        bin_records.append({"bin":bucket,"rows":n,"raw_ratio":raw,"pooled_ratio":pooled})
    return {"global_share_ratio":global_ratio,"global_log_shift":float(log_residual.mean()),
            "global_smearing":float(np.exp(log_residual).mean()),"decile_edges":edges,
            "decile_factors":factors,"decile_details":bin_records,
            "fit_start":str(frame.date.min()),"fit_end":str(frame.date.max()),"fit_rows":len(frame)}


def apply_corrections(pred, params):
    pred = np.asarray(pred,dtype=float)
    bins = np.searchsorted(params["decile_edges"],pred,side="right")
    return {"uncalibrated":pred,
            "global_share_ratio":pred*params["global_share_ratio"],
            "global_log_shift":np.maximum(np.expm1(np.log1p(pred)+params["global_log_shift"]),0),
            "global_smearing":np.maximum((1+pred)*params["global_smearing"]-1,0),
            "shrunk_decile_ratio":pred*np.array(params["decile_factors"])[bins]}


def run():
    setup()
    con = duckdb.connect()
    con.execute("SET threads=4")
    historical = history_2021(con)
    metrics, clipping, checks = [],[],[]
    def check(name,passed):
        checks.append({"check":name,"passed":bool(passed)})
    for year in range(2022,2026):
        progress(f"Fold {year}: diagnose fixed model and fit corrections using {year-1} only")
        if year==2022:
            previous = con.sql(f"SELECT * FROM read_parquet('{historical}')").df()
        else:
            previous = con.sql(f"SELECT date,permno,auction_volume,lgbm_residual_58_l2 AS uncalibrated FROM read_parquet('{PREVIOUS / 'predictions' / f'nonlinear_challenger_predictions_{year-1}.parquet'}')").df()
        params = fit_corrections(previous)
        check(f"{year}_calibration_is_prior_year",pd.Timestamp(params["fit_end"])<pd.Timestamp(f"{year}-01-01") and previous.date.dt.year.eq(year-1).all())
        (OUT / f"parameters/calibration_{year}.json").write_text(json.dumps(params,indent=2))
        source = load(con,f"{year}-01-01",f"{year+1}-01-01")
        frame = source[CONTEXT].copy()
        prior_path = PREVIOUS / f"predictions/nonlinear_challenger_predictions_{year}.parquet"
        prior = con.sql(f"SELECT date,permno,auction_volume AS saved_actual,lgbm_residual_58_l2 AS uncalibrated FROM read_parquet('{prior_path}')").df()
        original_keys = frame[["date","permno"]].copy()
        frame = frame.merge(prior,on=["date","permno"],validate="one_to_one",sort=False)
        check(f"{year}_row_order_and_labels",frame[["date","permno"]].equals(original_keys) and np.array_equal(frame.auction_volume,frame.saved_actual))
        frame = frame.drop(columns="saved_actual")
        prep = json.loads((PREVIOUS / f"models/preprocessor_{year}.json").read_text())
        reconstruction = json.loads((PREVIOUS / f"models/reconstruction_{year}.json").read_text())
        names = json.loads((PREVIOUS / f"models/feature_columns_lgbm_residual_58_l2_{year}.json").read_text())["output_names"]
        matrix = apply_preprocessor(source,prep,*[list(prep[k]) for k in ["continuous","boolean","categorical"]])
        indices = [prep["output_names"].index(name) for name in names]
        model = lgb.Booster(model_file=str(PREVIOUS / f"models/lgbm_residual_58_l2_{year}.txt"))
        raw_log = offset(source,"median20_log_residual",reconstruction["fallback"])+model.predict(matrix[:,indices])
        bounded = np.maximum(np.expm1(np.clip(raw_log,*reconstruction["clip_log"])),0)
        check(f"{year}_saved_base_model_reconstructs_all_rows",np.allclose(bounded,frame.uncalibrated,rtol=1e-10,atol=1e-8))
        frame["upper_clipped"] = raw_log>reconstruction["clip_log"][1]
        frame["lower_clipped"] = raw_log<reconstruction["clip_log"][0]
        raw_pred = np.expm1(raw_log)
        check(f"{year}_unclipped_diagnostic_finite",np.isfinite(raw_pred).all())
        clipping.append({"test_year":year,"rows":len(frame),"upper_clipped":int(frame.upper_clipped.sum()),"lower_clipped":int(frame.lower_clipped.sum()),
            "shares_removed_by_upper_bound":float(np.maximum(raw_pred-bounded,0).sum()),
            "net_shares_removed_by_bounds":float((raw_pred-bounded).sum()),
            "observed_base_shortfall":float(frame.auction_volume.sum()-frame.uncalibrated.sum())})
        for method,pred in apply_corrections(frame.uncalibrated,params).items():
            frame[method] = pred
            check(f"{year}_{method}_finite_nonnegative",np.isfinite(pred).all() and (pred>=0).all())
            metrics.append({"test_year":year,"method":method,**forecast_metrics(frame.auction_volume,pred)})
        save_frame(con,frame,OUT / f"predictions/calibrated_predictions_{year}.parquet")
        pd.DataFrame(metrics).to_csv(OUT / "annual_metrics.csv",index=False)
        pd.DataFrame(clipping).to_csv(OUT / "clipping_diagnostics.csv",index=False)
        pd.DataFrame(checks).to_csv(OUT / "fitting_checks.csv",index=False)
        progress(f"Fold {year}: saved {len(frame):,} complete forecasts for five methods")
        del source,matrix,frame,previous,prior
        import gc
        gc.collect()
    con.close()
    summarize()


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat
    con=duckdb.connect()
    con.execute("SET threads=4")
    checks=pd.read_csv(OUT / "fitting_checks.csv").to_dict("records")
    def check(name,passed):
        checks.append({"check":name,"passed":bool(passed)})
    annual=pd.read_csv(OUT / "annual_metrics.csv")
    files=[]
    for year in range(2022,2026):
        path=OUT / f"predictions/calibrated_predictions_{year}.parquet"
        frame=con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        params=json.loads((OUT / f"parameters/calibration_{year}.json").read_text())
        check(f"{year}_written_unique_keys",not frame.duplicated(["date","permno"]).any())
        reference=PREVIOUS / f"predictions/nonlinear_challenger_predictions_{year}.parquet"
        mismatches=con.sql(f'''SELECT count(*) FROM read_parquet('{path}') n FULL JOIN read_parquet('{reference}') r USING(date,permno)
            WHERE n.permno IS NULL OR r.permno IS NULL OR n.auction_volume IS DISTINCT FROM r.auction_volume
            OR n.uncalibrated IS DISTINCT FROM r.lgbm_residual_58_l2''').fetchone()[0]
        check(f"{year}_written_base_and_targets_unchanged",mismatches==0)
        for method,pred in apply_corrections(frame.uncalibrated,params).items():
            check(f"{year}_{method}_correction_replays",np.allclose(pred,frame[method],rtol=1e-12,atol=1e-9))
            fresh=forecast_metrics(frame.auction_volume,frame[method])
            old=annual[(annual.test_year==year)&annual.method.eq(method)].iloc[0]
            check(f"{year}_{method}_metrics_recompute",all(np.isclose(value,old[k],rtol=1e-10,atol=1e-9) for k,value in fresh.items()))
        files.append({"path":str(path.relative_to(ROOT)),"rows":len(frame),"sha256":sha(path)})
    history=OUT / "historical_2021/predictions_2021.parquet"
    hmanifest=json.loads((OUT / "historical_2021/manifest.json").read_text())
    check("historical_2021_model_fits_only_through_2020",pd.Timestamp(hmanifest["latest_fitting_date"])<pd.Timestamp("2021-01-01"))
    hbad=con.sql(f'''SELECT count(*) FROM read_parquet('{history}') h FULL JOIN
        (SELECT date,permno,auction_volume FROM read_parquet('{PANEL}') WHERE eligible_target AND year=2021) p USING(date,permno)
        WHERE h.permno IS NULL OR p.permno IS NULL OR h.auction_volume IS DISTINCT FROM p.auction_volume''').fetchone()[0]
    check("historical_2021_complete_correct_evaluation_keys",hbad==0)
    hdup=con.sql(f"SELECT count(*)-count(DISTINCT(date,permno)) FROM read_parquet('{history}')").fetchone()[0]
    check("historical_2021_unique_keys",hdup==0)
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv",index=False)
    assert all(r["passed"] for r in checks),"Forecast validation failed"
    progress("Written forecast verification passed; building error diagnostics and paired comparisons")

    glob=str(OUT / "predictions/calibrated_predictions_*.parquet")
    con.execute(f'''CREATE TEMP TABLE development AS SELECT *,
        ntile(10) OVER(PARTITION BY year(date) ORDER BY uncalibrated,date,permno) forecast_decile,
        percent_rank() OVER(PARTITION BY year(date) ORDER BY auction_volume) actual_percentile,
        abs(uncalibrated-auction_volume) base_abs_share_error,
        abs(ln(1+uncalibrated)-log1p_auction_volume) base_abs_log_error
        FROM read_parquet('{glob}') WHERE year(date) IN(2022,2023,2024)''')
    dimensions={
        "size":"CASE WHEN size_percentile<0.33 THEN 'small' WHEN size_percentile<0.67 THEN 'mid' ELSE 'large' END",
        "history":"CASE WHEN prior_auction_observations<5 THEN 'cold_0_4' WHEN prior_auction_observations<20 THEN 'developing_5_19' ELSE 'seasoned' END",
        "event":"CASE WHEN is_early_close THEN 'early_close' WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",
        "forecast_decile":"CAST(forecast_decile AS VARCHAR)",
        "actual_tail":"CASE WHEN actual_percentile>=0.99 THEN 'top_1pct' ELSE 'lower_99pct' END"}
    groups=[]
    for method in METHODS:
        for dimension,expression in dimensions.items():
            groups.append(con.sql(f'''SELECT '{method}' AS "method",'{dimension}' dimension,{expression} segment,count(*) observations,
                avg(auction_volume) mean_actual,avg("{method}") mean_forecast,
                sqrt(avg(pow(ln(1+"{method}")-log1p_auction_volume,2))) rmse_log,
                sum(abs("{method}"-auction_volume))/sum(auction_volume) wape,
                sum("{method}")/sum(auction_volume)-1 aggregate_bias,
                avg(log1p_auction_volume-ln(1+"{method}")) mean_log_residual
                FROM development GROUP BY 1,2,3''').df())
    groups=pd.concat(groups,ignore_index=True)
    groups.to_csv(OUT / "segment_diagnostics.csv",index=False)
    dates=con.sql('''SELECT date,count(*) stocks,sum(auction_volume) actual_shares,sum(uncalibrated) forecast_shares,
        sum(base_abs_share_error) absolute_share_error,sum(uncalibrated)/sum(auction_volume)-1 aggregate_bias,
        avg(base_abs_log_error) log_mae,max(CAST(monthly_options_expiry AS INT)) monthly_expiry,
        max(CAST(quarterly_options_expiry AS INT)) quarterly_expiry,max(CAST(is_last_trading_day_month AS INT)) month_end,
        max(CAST(is_early_close AS INT)) early_close FROM development GROUP BY date ORDER BY absolute_share_error DESC''').df()
    dates.to_csv(OUT / "date_error_summary.csv",index=False)
    stocks=con.sql('''SELECT permno,any_value(ticker) ticker,count(*) observations,
        sum(auction_volume) actual_shares,sum(uncalibrated) forecast_shares,sum(base_abs_share_error) absolute_share_error,
        sum(uncalibrated)/sum(auction_volume)-1 aggregate_bias,avg(log1p_auction_volume-ln(1+uncalibrated)) mean_log_residual,
        sqrt(avg(pow(ln(1+uncalibrated)-log1p_auction_volume,2))) rmse_log
        FROM development GROUP BY permno ORDER BY absolute_share_error DESC''').df()
    stocks.to_csv(OUT / "stock_error_summary.csv",index=False)
    export_columns="date,permno,ticker,auction_volume,uncalibrated,base_abs_share_error,base_abs_log_error,prior_auction_observations,size_percentile,quarterly_options_expiry,monthly_options_expiry,is_last_trading_day_month,is_early_close,lag1_return,abnormal_volume_lag1,lag1_dividend_flag,lag1_distribution_flag,lag1_price_adjustment_flag,upper_clipped,lower_clipped"
    for label,order in [("share","base_abs_share_error"),("log","base_abs_log_error")]:
        con.sql(f"SELECT {export_columns} FROM development ORDER BY {order} DESC,date,permno LIMIT 200").df().to_csv(OUT / f"largest_{label}_errors.csv",index=False)
    concentration=con.sql('''WITH ranked AS(SELECT *,row_number()OVER(ORDER BY base_abs_share_error DESC) error_rank,count(*)OVER() total FROM development)
        SELECT count(*) observations,sum(base_abs_share_error) total_absolute_share_error,
        sum(CASE WHEN error_rank<=ceil(total*.01) THEN base_abs_share_error ELSE 0 END)/sum(base_abs_share_error) top_1pct_error_rows_share_of_error,
        sum(CASE WHEN actual_percentile>=.99 THEN base_abs_share_error ELSE 0 END)/sum(base_abs_share_error) largest_actual_1pct_share_of_error,
        sum(CASE WHEN prior_auction_observations<20 THEN base_abs_share_error ELSE 0 END)/sum(base_abs_share_error) short_history_share_of_error
        FROM ranked''').df()
    concentration.to_csv(OUT / "error_concentration.csv",index=False)
    bootstrap=[]
    for number,method in enumerate(METHODS):
        if method=="uncalibrated":
            continue
        daily=con.sql(f'''SELECT date,year(date) test_year,
            (sum(abs("{method}"-auction_volume))-sum(abs(uncalibrated-auction_volume)))/sum(auction_volume) wape_delta,
            avg(abs(ln(1+"{method}")-log1p_auction_volume)-abs(ln(1+uncalibrated)-log1p_auction_volume)) log_mae_delta
            FROM read_parquet('{glob}') GROUP BY date,year(date) ORDER BY date''').df()
        for label,years in [("development",[2022,2023,2024]),("confirmation",[2025])]:
            for metric in ["wape_delta","log_mae_delta"]:
                values=daily.loc[daily.test_year.isin(years),metric].to_numpy()
                rng=np.random.default_rng(SEED+number)
                draws=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
                low,high=np.quantile(draws,[.025,.975])
                bootstrap.append({"method":method,"period":label,"metric":metric,"dates":len(values),"mean_difference":values.mean(),"ci_2_5":low,"ci_97_5":high})
    bootstrap=pd.DataFrame(bootstrap)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv",index=False)
    dev=annual[annual.test_year.isin([2022,2023,2024])].copy()
    dev["abs_bias"]=dev.aggregate_bias.abs()
    summary=dev.groupby("method",as_index=False).agg(development_rmse_log=("rmse_log","mean"),development_wape=("wape","mean"),
        development_bias=("aggregate_bias","mean"),development_abs_bias=("abs_bias","mean"))
    confirm=annual[annual.test_year.eq(2025)][["method","rmse_log","wape","aggregate_bias"]].rename(columns={"rmse_log":"confirmation_rmse_log","wape":"confirmation_wape","aggregate_bias":"confirmation_bias"})
    summary=summary.merge(confirm,on="method",validate="one_to_one")
    base=summary.set_index("method").loc["uncalibrated"]
    baseline_annual=annual[annual.method.eq("uncalibrated")].set_index("test_year")
    summary["passes_adoption_gate"]=False
    for index,row in summary.iterrows():
        if row.method=="uncalibrated":
            continue
        ci=bootstrap[(bootstrap.method==row.method)&bootstrap.period.eq("development")&bootstrap.metric.eq("wape_delta")].iloc[0]
        candidate_annual=annual[annual.method.eq(row.method)].set_index("test_year")
        worst_wape=(candidate_annual.loc[[2022,2023,2024],"wape"]-baseline_annual.loc[[2022,2023,2024],"wape"]).max()
        summary.loc[index,"log_rmse_change"]=row.development_rmse_log-base.development_rmse_log
        summary.loc[index,"wape_change"]=row.development_wape-base.development_wape
        summary.loc[index,"paired_daily_wape_ci_upper"]=ci.ci_97_5
        summary.loc[index,"passes_adoption_gate"]=bool(row.development_abs_bias<base.development_abs_bias and row.development_wape<=base.development_wape and row.development_rmse_log-base.development_rmse_log<=.002 and worst_wape<=.002 and ci.ci_97_5<0)
    summary.to_csv(OUT / "calibration_comparison.csv",index=False)
    params_rows=[]
    for year in range(2022,2026):
        p=json.loads((OUT / f"parameters/calibration_{year}.json").read_text())
        params_rows.append({"test_year":year,"fit_year":year-1,"fit_rows":p["fit_rows"],"global_share_ratio":p["global_share_ratio"],"global_log_shift":p["global_log_shift"],"global_smearing":p["global_smearing"]})
    pd.DataFrame(params_rows).to_csv(OUT / "calibration_parameter_summary.csv",index=False)
    clipping=pd.read_csv(OUT / "clipping_diagnostics.csv")
    clipdev=clipping[clipping.test_year.le(2024)]
    clip_fraction=float(clipdev.shares_removed_by_upper_bound.sum()/clipdev.observed_base_shortfall.sum())
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    deciles=groups[groups.dimension.eq("forecast_decile")].copy()
    deciles["number"]=deciles.segment.astype(int)
    for method in ["uncalibrated","global_log_shift","global_share_ratio","shrunk_decile_ratio"]:
        d=deciles[deciles.method.eq(method)].sort_values("number")
        axes[0].plot(d.number,1+d.aggregate_bias,marker="o",label=method)
    axes[0].axhline(1,color="black",linestyle="--")
    axes[0].set(xlabel="Base forecast decile",ylabel="Forecast / actual shares",title="Development mean-volume calibration")
    axes[0].legend(fontsize=7)
    ordered=summary.set_index("method").loc[METHODS]
    axes[1].barh(METHODS,ordered.development_rmse_log,color="#52769d")
    axes[1].set(title="Development log RMSE",xlabel="Lower is better")
    axes[2].barh(METHODS,100*ordered.development_wape,color="#c58a35")
    axes[2].set(title="Development WAPE (%)",xlabel="Lower is better")
    fig.tight_layout()
    fig.savefig(OUT / "calibration_comparison.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    passing=summary.loc[summary.passes_adoption_gate,"method"].tolist()
    manifest={"completed_utc":datetime.now(timezone.utc).isoformat(),"all_checks_passed":True,"checks":len(checks),"prediction_files":files,
        "total_evaluation_rows":sum(f["rows"] for f in files),"passing_corrections":passing,"upper_clipping_share_of_net_volume_shortfall":clip_fraction,
        "calibration_2021_prediction_sha256":sha(history),"plan_sha256":sha(OUT / "experiment_plan.json")}
    (OUT / "experiment_summary.json").write_text(json.dumps(manifest,indent=2))
    lines=["| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE | Gate |",
           "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in summary.set_index("method").loc[METHODS].reset_index().itertuples():
        lines.append(f"| {r.method} | {r.development_rmse_log:.5f} | {r.development_wape:.2%} | {r.development_bias:.2%} | {r.confirmation_rmse_log:.5f} | {r.confirmation_wape:.2%} | {'Pass' if r.passes_adoption_gate else 'Reference' if r.method=='uncalibrated' else 'Fail'} |")
    conclusion="No correction passes the prespecified accuracy/calibration gate; retain the uncalibrated forecast as the main reference." if not passing else "Corrections passing the prespecified research gate: " + ", ".join(passing) + "."
    c=concentration.iloc[0]
    report=f'''# Error diagnosis and historical calibration

{conclusion}

All {len(checks)} checks passed for {manifest['total_evaluation_rows']:,} evaluation stock-days. The verified 2022-2025 base forecasts and models were preserved. A prerequisite 2021 model was tuned on 2020, fitted through 2020, and used only to obtain earlier out-of-time calibration evidence.

{chr(10).join(lines)}

Development figures average annual 2022-2024 metrics, giving years equal weight. The previously inspected 2025 confirmation period is reported separately and does not determine the adoption gate.

## Residual findings

- The largest 1% of absolute-share-error rows contribute {c.top_1pct_error_rows_share_of_error:.2%} of total development absolute share error. These are ex-post error groups, not usable predictors.
- The top 1% of actual auction volumes contribute {c.largest_actual_1pct_share_of_error:.2%} of absolute share error.
- Fewer-than-20-history rows contribute {c.short_history_share_of_error:.2%} of absolute share error. Large relative cold-start errors should not be mistaken for the main source of universe-wide share error.
- The upper forecast bound removes shares equal to {clip_fraction:.2%} of the observed net development volume shortfall. This is a mechanical bound-effect diagnostic, not a claim about causal economic error drivers or a tested change in clipping policy.
- `segment_diagnostics.csv` separates stock size, history, forecast-size deciles, ex-post actual tails and disjoint calendar groups. Event precedence is early close, quarterly expiry, monthly expiry, month-end, regular.
- `largest_share_errors.csv` and `largest_log_errors.csv` contain different failure rankings with known calendar and lagged-market context. Those annotations do not establish why an error occurred.
- `stock_error_summary.csv` and `date_error_summary.csv` distinguish persistent stock-level bias from common-date misses. The stock table retains sparse stocks with their counts; no stock-specific correction was fitted.

## Calibration design

For evaluation year Y, all parameters are fitted on the full Y-1 out-of-time forecast sample. No correction uses the labels from the year it evaluates. Each historical forecast's base model was fitted before that forecast year. The common model specification was selected retrospectively in earlier research, so this chronology is not a claim of untouched specification validation.

Four fixed corrections were tested: global share-total scaling; a shift by mean historical log residual; mean exponential-residual smearing; and ten historical forecast-size bins with ratio factors shrunk toward the global factor using n/(n+10000). Historical forecast decile thresholds are retained when scoring later dates. No calibration parameters, bounds, shrinkage strengths, or extra candidates were tuned on evaluation results.

The underlying base-model clipping remains unchanged. Corrections apply to its saved forecasts without a second clipping step; otherwise the intended correction would be partly reversed. The unclipped model is reconstructed only to diagnose the mechanical effect of existing bounds, not introduced as an alternative forecast.

The gate requires lower mean absolute annual aggregate bias, no mean annual WAPE worsening, no more than 0.002 deterioration in mean log RMSE, no annual WAPE deterioration above 0.002, and an upper paired development daily-WAPE interval below zero. Bias alone cannot establish a better forecast. The date-level bootstrap preserves cross-sectional dependence but not serial dependence, and does not adjust for multiple comparisons. Daily bootstrap weighting differs from the equal-year headline aggregation.

## Interpretation

Near-zero average log residual and negative raw-share volume bias can coexist: predicting a typical log volume and predicting arithmetic expected shares are different objectives in a skewed distribution. Correcting total-volume bias may increase absolute or log errors. This experiment measures that tradeoff; it does not assign the shortfall entirely to retransformation, structural growth, or clipping.

## Outputs

- `annual_metrics.csv`, `calibration_comparison.csv`, `paired_bootstrap.csv`: annual results, adoption criteria and uncertainty.
- `parameters/`, `calibration_parameter_summary.csv`: prior-year fitting windows and exact correction factors.
- `predictions/`: full actuals, fixed base forecasts, four corrections, and diagnostic context.
- `historical_2021/`: prerequisite out-of-time model, samples, predictions and provenance.
- `segment_diagnostics.csv`, `error_concentration.csv`, `clipping_diagnostics.csv`: error structure.
- `largest_share_errors.csv`, `largest_log_errors.csv`, `stock_error_summary.csv`, `date_error_summary.csv`: detailed audit tables.
- `validation_checks.csv`, `input_fingerprints.csv`, `experiment_plan.json`, `experiment_summary.json`: chronology, exact-input and forecast checks.

Run `.venv/bin/python notebooks/diagnose_and_calibrate.py` with the step-03 OpenMP environment. Append `--summarize` to rebuild diagnostics from existing forecasts without model fitting.
'''
    (OUT / "README.md").write_text(report)
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# 18 — Error diagnosis and calibration\n\nThe experiment is executed locally through `diagnose_and_calibrate.py`. These cells inspect its saved results. See the frozen plan for chronology and adoption criteria."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'blackrock_data').exists())\nout = root / 'outputs/experiments/18_error_diagnosis_calibration'"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'calibration_comparison.csv'))\ndisplay(Image(filename=str(out / 'calibration_comparison.png')))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'error_concentration.csv'))\ndisplay(pd.read_csv(out / 'clipping_diagnostics.csv'))\ndisplay(pd.read_csv(out / 'segment_diagnostics.csv'))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'calibration_parameter_summary.csv'))\ndisplay(pd.read_csv(out / 'paired_bootstrap.csv'))\nchecks = pd.read_csv(out / 'validation_checks.csv')\nassert checks.passed.all()\ndisplay(checks)"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'largest_share_errors.csv').head(20))\ndisplay(pd.read_csv(out / 'date_error_summary.csv').head(20))")])
    notebook.metadata.kernelspec={"display_name":"Auction local","name":"auction-local","language":"python"}
    nbformat.write(notebook,ROOT / "notebooks/18_error_diagnosis_calibration.ipynb")
    con.close()
    progress("Error diagnosis, calibration comparisons and all checks complete")
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        run()
