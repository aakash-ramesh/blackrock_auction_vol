"""First 15:30 update: fixed daily anchor, honest historical stacking, same-row tests.

Run with .venv Python and the OpenMP environment used in preceding experiments.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT / "outputs/experiments/19_intraday_1530"
DAILY=ROOT / "outputs/reproduction/03_compact_nonlinear"
PANEL=ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
HISTORICAL=ROOT / "outputs/experiments/18_error_diagnosis_calibration/historical_2021/predictions_2021.parquet"
os.environ.setdefault("MPLCONFIGDIR",str(OUT / ".matplotlib"))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from notebooks.modeling_utils import fit_preprocessor,apply_preprocessor,fit_ridge,predict_ridge,forecast_metrics
from notebooks.compare_target_formulations import sha,parameters,SEED

DAILY_FEATURES=json.loads((ROOT / "outputs/eda/09_global_finalists/frozen_global_benchmark.json").read_text())["features"]
INTRADAY_FEATURES=["cp_cum_volume","cp_cum_dollar_volume","cp_cum_trade_count","cp_recent_30m_volume",
                   "cp_vwap","cp_recent_30m_volume_share","cp_volume_to_adv20","cp_volume_surprise20",
                   "cp_trade_count_surprise20","cp_vwap_return_prior_close"]
PRIOR_CHECKPOINT_FEATURES=["checkpoint_avg_volume20","checkpoint_avg_trades20","ratio20_auction_to_checkpoint","lag1_price"]
CONTROL_FEATURES=DAILY_FEATURES+["log_daily_forecast"]+PRIOR_CHECKPOINT_FEATURES
MODEL_FEATURES=CONTROL_FEATURES+INTRADAY_FEATURES
METHODS=["fixed_daily","checkpoint_ratio_rule","ridge_update","lgbm_daily_control","lgbm_intraday_update"]
CONTEXT=["date","permno","ticker","year","auction_volume","log1p_auction_volume","daily_forecast","log_daily_forecast",
         "cp_available","ratio20_auction_to_checkpoint","size_percentile","liquidity_percentile","prior_auction_observations",
         "quarterly_options_expiry","monthly_options_expiry","is_last_trading_day_month"]


def progress(message):
    line=datetime.now(timezone.utc).isoformat()+" "+message
    print(line,flush=True)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(line+"\n")


def save_frame(con,frame,path):
    con.register("output_frame",frame)
    con.execute(f"COPY output_frame TO '{path}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.unregister("output_frame")


def prepare():
    for folder in [OUT,OUT / "samples",OUT / "models",OUT / "predictions"]:
        folder.mkdir(parents=True,exist_ok=True)
    old=json.loads((DAILY / "reproduction_summary.json").read_text())
    calibration=json.loads((ROOT / "outputs/experiments/18_error_diagnosis_calibration/experiment_summary.json").read_text())
    assert old["all_checks_passed"] and calibration["all_checks_passed"]
    inputs=[]
    for item in old["predictions"]:
        assert sha(ROOT / item["path"])==item["sha256"]
        inputs.append(item)
    assert sha(HISTORICAL)==calibration["calibration_2021_prediction_sha256"]
    inputs.append({"path":str(HISTORICAL.relative_to(ROOT)),"sha256":sha(HISTORICAL)})
    inventory=pd.read_csv(ROOT / "outputs/reproduction/02_benchmark_reconciliation/input_fingerprints.csv")
    expected=inventory.loc[inventory.path.str.endswith("enriched_daily_panel.parquet"),"sha256"].iloc[0]
    assert sha(PANEL)==expected
    inputs.append({"path":str(PANEL.relative_to(ROOT)),"sha256":expected})
    for path in sorted((ROOT / "blackrock_data/stock_intraday/trade_checkpoints").glob("year=*/*.parquet")):
        inputs.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size})
    pd.DataFrame(inputs).to_csv(OUT / "input_fingerprints.csv",index=False)
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(Path(__file__)),
          "checkpoint":"15:30 ET","business_target":"NYSE official closing-auction shares",
          "development_years":[2022,2023,2024],"confirmation_year":2025,"confirmation_previously_inspected":True,
          "primary_universe":"All eligible regular-session stock-days; early closes excluded; missing checkpoint falls back to fixed daily forecast",
          "historical_stacking":"Use only saved annually out-of-time 2021-2025 daily LightGBM forecasts; no in-sample daily predictions",
          "update_history":"Expanding from 2021; 2022 tuning uses Jan-Sep 2021 and validates Oct-Dec 2021; later folds validate preceding full year and train on earlier years",
          "refit_history":"All available out-of-time update history from 2021 through preceding year",
          "training_universe":"Available valid 15:30 checkpoints; identical sampled rows across all learned candidates and daily-only control",
          "methods":METHODS,"daily_features":DAILY_FEATURES,"additional_prior_close_control_features":PRIOR_CHECKPOINT_FEATURES,"intraday_features":INTRADAY_FEATURES,
          "rule":"Current cumulative volume times prior-only 20-row median auction/checkpoint ratio; missing historical ratio uses final-fitting historical median; missing current checkpoint uses daily forecast",
          "learned_target":"log1p(actual auction) minus log1p(fixed daily forecast)",
          "grids":{"ridge_strength":[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],"lgbm_leaves":[15,31,63],"max_trees":2000},
          "sample_sizes":{"tune":250000,"validation":120000,"refit":350000},
          "bounds":"Rule and learned/control forecasts share update-training-only 0.1/99.9 percentile log-volume bounds; missing-checkpoint fallbacks retain daily forecasts exactly",
          "success_gate":"Intraday LightGBM improves mean development log RMSE >=0.002 vs both fixed daily and daily-only learned control, no annual log RMSE worsening >0.002 vs either, no mean WAPE worsening vs either, paired development date-bootstrap log-MAE upper CI <0 vs both",
          "limits":["2021 onward update history differs from five-year daily-model fitting history","first-fold quarterly validation differs from later annual validation",
                    "prior-only checkpoint features rely on upstream 15:30 extraction; tick timestamps are not in these aggregates",
                    "whole-date bootstrap does not account for cross-date serial dependence or multiple comparisons"],
          "excluded_changes":["daily model retraining","daily target or calibration changes","other checkpoints","new external datasets"]}
    (OUT / "experiment_plan.json").write_text(json.dumps(plan,indent=2))
    (OUT / "environment.json").write_text(json.dumps({n:importlib.metadata.version(n) for n in ["duckdb","numpy","pandas","lightgbm"]},indent=2))
    catalog=[{"feature":f,"information_time":"prior close / known calendar"} for f in DAILY_FEATURES]
    catalog.append({"feature":"log_daily_forecast","information_time":"saved out-of-time daily prediction using prior-close information"})
    catalog.extend({"feature":f,"information_time":"prior close; historical checkpoint normalization / prior price, available to both learned LightGBM candidates"} for f in PRIOR_CHECKPOINT_FEATURES)
    catalog.extend({"feature":f,"information_time":"observed by 15:30; normalization uses prior-only historical windows"} for f in INTRADAY_FEATURES)
    pd.DataFrame(catalog).to_csv(OUT / "feature_timing_catalog.csv",index=False)


def build_panel(con):
    progress("Auditing checkpoint keys and building prior-only 15:30 history")
    raw=str(ROOT / "blackrock_data/stock_intraday/trade_checkpoints/year=*/trade_checkpoints_*.parquet")
    con.execute(f"CREATE TEMP VIEW raw_cp AS SELECT * FROM read_parquet('{raw}',union_by_name=true) WHERE checkpoint='15:30'")
    assert con.sql("SELECT count(*)-count(DISTINCT(date,permno)) FROM raw_cp").fetchone()[0]==0
    forecasts=[f"SELECT date,permno,uncalibrated AS daily_forecast FROM read_parquet('{HISTORICAL}')"]
    for year in range(2022,2026):
        forecasts.append(f"SELECT date,permno,lgbm_residual_58_l2 AS daily_forecast FROM read_parquet('{DAILY / 'predictions' / f'nonlinear_challenger_predictions_{year}.parquet'}')")
    con.execute("CREATE TEMP VIEW daily_predictions AS "+" UNION ALL ".join(forecasts))
    assert con.sql("SELECT count(*)-count(DISTINCT(date,permno)) FROM daily_predictions").fetchone()[0]==0
    fields=[("cum_volume","cp_cum_volume"),("cum_dollar_volume","cp_cum_dollar_volume"),("cum_trade_count","cp_cum_trade_count"),
            ("recent_30m_volume","cp_recent_30m_volume"),("vwap","cp_vwap"),("recent_30m_volume_share","cp_recent_30m_volume_share")]
    valid="coalesce(c.checkpoint_data_found,false) AND NOT coalesce(c.checkpoint_after_market_close,true) AND NOT coalesce(p.is_early_close,true) AND c.cum_volume IS NOT NULL AND c.cum_volume>0"
    cp_select=",".join(f"CASE WHEN {valid} THEN c.{source} END AS {destination}" for source,destination in fields)
    query=f'''WITH joined AS(
        SELECT p.*, {valid} AS cp_available,{cp_select}
        FROM read_parquet('{PANEL}') p LEFT JOIN raw_cp c USING(date,permno)
    ), historical AS(
        SELECT *,avg(cp_cum_volume) OVER w20 AS checkpoint_avg_volume20,
        avg(cp_cum_trade_count) OVER w20 AS checkpoint_avg_trades20,
        median(CASE WHEN eligible_target AND cp_available THEN auction_volume*1.0/cp_cum_volume END) OVER w20 AS ratio20_auction_to_checkpoint
        FROM joined WINDOW w20 AS(PARTITION BY permno ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    )
    SELECT h.*,d.daily_forecast,ln(1+d.daily_forecast) log_daily_forecast,
        cp_cum_volume/nullif(avg20_daily_volume,0) cp_volume_to_adv20,
        cp_cum_volume/nullif(checkpoint_avg_volume20,0) cp_volume_surprise20,
        cp_cum_trade_count/nullif(checkpoint_avg_trades20,0) cp_trade_count_surprise20,
        cp_vwap/nullif(lag1_price,0)-1 cp_vwap_return_prior_close
    FROM historical h JOIN daily_predictions d USING(date,permno)
    WHERE h.eligible_target AND h.year BETWEEN 2021 AND 2025'''
    destination=OUT / "intraday_panel.parquet"
    con.execute(f"COPY ({query}) TO '{destination}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.execute(f"CREATE TEMP VIEW panel AS SELECT * FROM read_parquet('{destination}')")
    coverage=con.sql('''SELECT year,count(*) eligible_targets,count(*) FILTER(WHERE is_early_close) early_close_rows,
        count(*) FILTER(WHERE NOT is_early_close) regular_session_rows,
        count(*) FILTER(WHERE NOT is_early_close AND cp_available) valid_checkpoint_rows,
        count(*) FILTER(WHERE NOT is_early_close AND NOT cp_available) daily_fallback_rows,
        count(*) FILTER(WHERE daily_forecast IS NULL OR NOT isfinite(daily_forecast)) invalid_daily_forecasts
        FROM panel GROUP BY year ORDER BY year''').df()
    coverage.to_csv(OUT / "coverage_by_year.csv",index=False)
    assert coverage.invalid_daily_forecasts.eq(0).all()
    assert con.sql("SELECT count(*) FILTER(WHERE is_early_close AND cp_available) FROM panel").fetchone()[0]==0
    progress("Checkpoint panel ready; early closes excluded from evaluation and missing checkpoints retain daily fallback")
    return destination


def run():
    prepare()
    con=duckdb.connect()
    con.execute("SET threads=4")
    build_panel(con)
    template=json.loads((DAILY / "models/preprocessor_2022.json").read_text())
    booleans=[f for f in MODEL_FEATURES if f in template["boolean"]]
    categorical=[f for f in MODEL_FEATURES if f in template["categorical"]]
    continuous=[f for f in MODEL_FEATURES if f not in booleans+categorical]
    def load(start,end,sample=None,seed=SEED,available=True):
        sampling=f"USING SAMPLE {sample} ROWS (reservoir,{seed})" if sample else ""
        availability="AND cp_available" if available else ""
        return con.sql(f"SELECT * FROM (SELECT * FROM panel WHERE NOT is_early_close AND date>=DATE '{start}' AND date<DATE '{end}' {availability}) {sampling}").df()
    trial_rows,metric_rows=[],[]
    for year in range(2022,2026):
        progress(f"Fold {year}: loading honest out-of-time update training rows")
        validation_start="2021-10-01" if year==2022 else f"{year-1}-01-01"
        train=load("2021-01-01",validation_start,250000,SEED+year)
        validation=load(validation_start,f"{year}-01-01",120000,SEED+100+year)
        refit=load("2021-01-01",f"{year}-01-01",350000,SEED+200+year)
        test=load(f"{year}-01-01",f"{year+1}-01-01",available=False)
        assert train.date.max()<validation.date.min() and validation.date.max()<test.date.min() and refit.date.max()<test.date.min()
        for label,frame in [("tune",train),("validation",validation),("refit",refit)]:
            save_frame(con,frame,OUT / f"samples/{year}_{label}.parquet")
        xt,prep=fit_preprocessor(train,continuous,booleans,categorical)
        xv=apply_preprocessor(validation,prep,continuous,booleans,categorical)
        z=train.log1p_auction_volume.to_numpy()-train.log_daily_forecast.to_numpy()
        zv=validation.log1p_auction_volume.to_numpy()-validation.log_daily_forecast.to_numpy()
        clip_tune=np.quantile(train.log1p_auction_volume,[.001,.999])
        chosen={}
        for candidate in ["ridge_update","lgbm_daily_control","lgbm_intraday_update"]:
            allowed=CONTROL_FEATURES if candidate=="lgbm_daily_control" else MODEL_FEATURES
            mask=np.array([f in set(allowed) for f in prep["output_sources"]])
            choices=[]
            for parameter in [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1] if candidate=="ridge_update" else [15,31,63]:
                if candidate=="ridge_update":
                    coefficient=fit_ridge(xt[:,mask],z,parameter)
                    component=predict_ridge(xv[:,mask],coefficient)
                    iterations=0
                else:
                    progress(f"Fold {year}: tuning {candidate}, leaves={parameter}")
                    model=lgb.LGBMRegressor(**parameters(year,parameter,2000,True))
                    model.fit(xt[:,mask],z,eval_set=[(xv[:,mask],zv)],eval_metric="rmse",callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
                    component=model.predict(xv[:,mask]);iterations=int(model.best_iteration_)
                p=np.clip(validation.log_daily_forecast.to_numpy()+component,*clip_tune)
                row={"test_year":year,"candidate":candidate,"parameter":parameter,"iterations":iterations,
                     "validation_rmse_log":float(np.sqrt(np.mean((validation.log1p_auction_volume.to_numpy()-p)**2)))}
                choices.append(row);trial_rows.append(row)
            chosen[candidate]=min(choices,key=lambda r:r["validation_rmse_log"])
            progress(f"Fold {year}: selected {candidate}: {chosen[candidate]}")
        del train,validation,xt,xv
        import gc
        gc.collect()
        xr,prep=fit_preprocessor(refit,continuous,booleans,categorical)
        xe=apply_preprocessor(test,prep,continuous,booleans,categorical)
        z=refit.log1p_auction_volume.to_numpy()-refit.log_daily_forecast.to_numpy()
        clip=np.quantile(refit.log1p_auction_volume,[.001,.999])
        fallback_ratio=float(np.median(refit.auction_volume.to_numpy(dtype=float)/refit.cp_cum_volume.to_numpy(dtype=float)))
        result=test[CONTEXT].copy()
        result["fixed_daily"]=result.daily_forecast
        available=test.cp_available.to_numpy(dtype=bool)
        rule_ratio=test.ratio20_auction_to_checkpoint.astype(float).fillna(fallback_ratio).to_numpy()
        rule=test.cp_cum_volume.to_numpy(dtype=float)*rule_ratio
        rule=np.maximum(np.expm1(np.clip(np.log1p(rule),*clip)),0)
        result["checkpoint_ratio_rule"]=np.where(available,rule,result.daily_forecast)
        (OUT / f"models/preprocessor_{year}.json").write_text(json.dumps(prep,indent=2))
        for candidate in ["ridge_update","lgbm_daily_control","lgbm_intraday_update"]:
            progress(f"Fold {year}: final fit {candidate}")
            allowed=CONTROL_FEATURES if candidate=="lgbm_daily_control" else MODEL_FEATURES
            mask=np.array([f in set(allowed) for f in prep["output_sources"]])
            choice=chosen[candidate]
            if candidate=="ridge_update":
                coef=fit_ridge(xr[:,mask],z,choice["parameter"])
                np.save(OUT / f"models/{candidate}_{year}.npy",coef)
                component=predict_ridge(xe[:,mask],coef)
            else:
                model=lgb.LGBMRegressor(**parameters(year,int(choice["parameter"]),max(25,choice["iterations"]),False))
                model.fit(xr[:,mask],z)
                model.booster_.save_model(str(OUT / f"models/{candidate}_{year}.txt"))
                component=model.predict(xe[:,mask])
            p=np.maximum(np.expm1(np.clip(test.log_daily_forecast.to_numpy()+component,*clip)),0)
            result[candidate]=np.where(available,p,result.daily_forecast)
            (OUT / f"models/features_{candidate}_{year}.json").write_text(json.dumps(np.array(prep["output_names"])[mask].tolist()))
        (OUT / f"models/reconstruction_{year}.json").write_text(json.dumps({"clip_log":clip.tolist(),"fallback_ratio":fallback_ratio,"selected":chosen,
            "refit_start":str(refit.date.min()),"refit_end":str(refit.date.max()),"first_test_date":str(test.date.min())},indent=2))
        for candidate in METHODS:
            for sample_name,mask in [("all_regular_sessions",np.ones(len(result),dtype=bool)),("checkpoint_available",available)]:
                metric_rows.append({"test_year":year,"candidate":candidate,"sample":sample_name,
                    **forecast_metrics(result.auction_volume.to_numpy()[mask],result[candidate].to_numpy()[mask])})
        save_frame(con,result,OUT / f"predictions/intraday_predictions_{year}.parquet")
        pd.DataFrame(metric_rows).to_csv(OUT / "annual_metrics.csv",index=False)
        pd.DataFrame(trial_rows).to_csv(OUT / "hyperparameter_validation.csv",index=False)
        progress(f"Fold {year}: saved {len(result):,} regular-session rows with common daily fallback")
        del refit,test,xr,xe,result
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
    annual=pd.read_csv(OUT / "annual_metrics.csv")
    checks,files=[],[]
    def check(name,passed):
        checks.append({"check":name,"passed":bool(passed)})
    check("forbidden_current_daily_outcomes_not_inputs",not set(MODEL_FEATURES)&{"auction_volume","log1p_auction_volume","current_daily_volume","current_close","current_return","auction_price","auction_share_daily_volume"})
    for year in range(2022,2026):
        path=OUT / f"predictions/intraday_predictions_{year}.parquet"
        frame=con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        check(f"{year}_unique_keys",not frame.duplicated(["date","permno"]).any())
        check(f"{year}_proper_year",frame.date.dt.year.eq(year).all())
        mismatch=con.sql(f'''SELECT count(*) FROM read_parquet('{path}') n FULL JOIN
            (SELECT date,permno,auction_volume FROM read_parquet('{PANEL}') WHERE eligible_target AND year={year} AND NOT is_early_close) p USING(date,permno)
            WHERE n.permno IS NULL OR p.permno IS NULL OR n.auction_volume IS DISTINCT FROM p.auction_volume''').fetchone()[0]
        check(f"{year}_all_regular_session_keys_and_labels",mismatch==0)
        old=DAILY / f"predictions/nonlinear_challenger_predictions_{year}.parquet"
        mismatch=con.sql(f'''SELECT count(*) FROM read_parquet('{path}') n LEFT JOIN read_parquet('{old}') o USING(date,permno)
            WHERE n.fixed_daily IS DISTINCT FROM o.lgbm_residual_58_l2''').fetchone()[0]
        check(f"{year}_daily_reference_unchanged",mismatch==0)
        reconstruction=json.loads((OUT / f"models/reconstruction_{year}.json").read_text())
        check(f"{year}_refit_strictly_historical",pd.Timestamp(reconstruction["refit_end"])<pd.Timestamp(f"{year}-01-01"))
        available=frame.cp_available.to_numpy(dtype=bool)
        for candidate in METHODS:
            check(f"{year}_{candidate}_finite_nonnegative",np.isfinite(frame[candidate]).all() and frame[candidate].ge(0).all())
            if candidate!="fixed_daily":
                check(f"{year}_{candidate}_missing_checkpoint_exact_fallback",np.array_equal(frame.loc[~available,candidate],frame.loc[~available,"fixed_daily"]))
            for label,mask in [("all_regular_sessions",np.ones(len(frame),dtype=bool)),("checkpoint_available",available)]:
                fresh=forecast_metrics(frame.auction_volume.to_numpy()[mask],frame[candidate].to_numpy()[mask])
                recorded=annual[(annual.test_year==year)&annual.candidate.eq(candidate)&annual["sample"].eq(label)].iloc[0]
                check(f"{year}_{candidate}_{label}_written_metrics",all(np.isclose(value,recorded[key],atol=1e-9,rtol=1e-10) for key,value in fresh.items()))
        panel=OUT / "intraday_panel.parquet"
        sample=con.sql(f"SELECT * FROM read_parquet('{panel}') WHERE year={year} AND NOT is_early_close ORDER BY hash(date,permno) LIMIT 200").df()
        prep=json.loads((OUT / f"models/preprocessor_{year}.json").read_text())
        x=apply_preprocessor(sample,prep,*[list(prep[k]) for k in ["continuous","boolean","categorical"]])
        for candidate in ["ridge_update","lgbm_daily_control","lgbm_intraday_update"]:
            names=json.loads((OUT / f"models/features_{candidate}_{year}.json").read_text())
            indices=[prep["output_names"].index(name) for name in names]
            allowed=CONTROL_FEATURES if candidate=="lgbm_daily_control" else MODEL_FEATURES
            check(f"{year}_{candidate}_feature_allowlist",all(prep["output_sources"][i] in allowed for i in indices))
            if candidate=="ridge_update":
                component=predict_ridge(x[:,indices],np.load(OUT / f"models/{candidate}_{year}.npy"))
            else:
                component=lgb.Booster(model_file=str(OUT / f"models/{candidate}_{year}.txt")).predict(x[:,indices])
            pred=np.maximum(np.expm1(np.clip(sample.log_daily_forecast.to_numpy()+component,*reconstruction["clip_log"])),0)
            pred=np.where(sample.cp_available,pred,sample.daily_forecast)
            aligned=sample[["date","permno"]].merge(frame[["date","permno",candidate]],on=["date","permno"],validate="one_to_one")
            check(f"{year}_{candidate}_serialized_model_reconstruction",np.allclose(pred,aligned[candidate],rtol=1e-10,atol=1e-8))
        files.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"rows":len(frame)})
    # Independently recompute historical checkpoint windows from raw aggregates.
    progress("Forecasts verified; independently checking checkpoint history and paired gains")
    cp=str(ROOT / "blackrock_data/stock_intraday/trade_checkpoints/year=*/trade_checkpoints_*.parquet")
    sample_ids=con.sql(f"SELECT DISTINCT permno FROM read_parquet('{OUT / 'intraday_panel.parquet'}') ORDER BY hash(permno) LIMIT 16").df().permno.tolist()
    ids=",".join(str(int(p)) for p in sample_ids)
    raw=con.sql(f'''SELECT p.date,p.permno,p.auction_volume,p.eligible_target,p.is_early_close,c.cum_volume,c.cum_trade_count,
        c.checkpoint_data_found,c.checkpoint_after_market_close FROM read_parquet('{PANEL}') p LEFT JOIN
        (SELECT * FROM read_parquet('{cp}',union_by_name=true) WHERE checkpoint='15:30') c USING(date,permno)
        WHERE p.permno IN({ids}) ORDER BY p.permno,p.date''').df()
    valid=raw.checkpoint_data_found.fillna(False)&~raw.checkpoint_after_market_close.fillna(True)&~raw.is_early_close.fillna(True)&raw.cum_volume.gt(0).fillna(False)
    raw["valid_volume"]=raw.cum_volume.astype(float).where(valid)
    raw["valid_trades"]=raw.cum_trade_count.astype(float).where(valid)
    raw["auction_cp_ratio"]=(raw.auction_volume.astype(float)/raw.valid_volume).where(raw.eligible_target)
    parts=[]
    for _,part in raw.groupby("permno",sort=False):
        part=part.copy()
        part["expected_avg_volume"]=part.valid_volume.shift(1).rolling(20,min_periods=1).mean()
        part["expected_avg_trades"]=part.valid_trades.shift(1).rolling(20,min_periods=1).mean()
        part["expected_ratio"]=part.auction_cp_ratio.shift(1).rolling(20,min_periods=1).median()
        parts.append(part)
    expected=pd.concat(parts)
    actual=con.sql(f"SELECT date,permno,checkpoint_avg_volume20,checkpoint_avg_trades20,ratio20_auction_to_checkpoint FROM read_parquet('{OUT / 'intraday_panel.parquet'}') WHERE permno IN({ids})").df()
    aligned=actual.merge(expected,on=["date","permno"],validate="one_to_one")
    for feature,reference in [("checkpoint_avg_volume20","expected_avg_volume"),("checkpoint_avg_trades20","expected_avg_trades"),("ratio20_auction_to_checkpoint","expected_ratio")]:
        check(f"independent_prior_window_{feature}",np.allclose(aligned[feature].to_numpy(dtype=float),aligned[reference].to_numpy(dtype=float),equal_nan=True,rtol=1e-12,atol=1e-8))
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv",index=False)
    assert all(r["passed"] for r in checks),"Intraday verification failed; inspect validation_checks.csv"

    glob=str(OUT / "predictions/intraday_predictions_*.parquet")
    bootstrap=[]
    for number,candidate in enumerate(METHODS):
        if candidate=="fixed_daily":
            continue
        for reference in ["fixed_daily","lgbm_daily_control"]:
            if candidate==reference:
                continue
            daily=con.sql(f'''SELECT date,year(date) test_year,
                avg(abs(ln(1+"{candidate}")-log1p_auction_volume)-abs(ln(1+"{reference}")-log1p_auction_volume)) log_mae_delta,
                (sum(abs("{candidate}"-auction_volume))-sum(abs("{reference}"-auction_volume)))/sum(auction_volume) daily_wape_delta
                FROM read_parquet('{glob}') GROUP BY date,year(date) ORDER BY date''').df()
            for period,years in [("development",[2022,2023,2024]),("confirmation",[2025])]:
                for metric in ["log_mae_delta","daily_wape_delta"]:
                    values=daily.loc[daily.test_year.isin(years),metric].to_numpy()
                    rng=np.random.default_rng(SEED+number)
                    draws=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
                    low,high=np.quantile(draws,[.025,.975])
                    bootstrap.append({"candidate":candidate,"reference":reference,"period":period,"metric":metric,"dates":len(values),"mean_difference":values.mean(),"ci_2_5":low,"ci_97_5":high})
    bootstrap=pd.DataFrame(bootstrap)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv",index=False)
    main=annual[annual["sample"].eq("all_regular_sessions")]
    dev=main[main.test_year.isin([2022,2023,2024])].groupby("candidate",as_index=False).agg(
        development_rmse_log=("rmse_log","mean"),development_wape=("wape","mean"),development_bias=("aggregate_bias","mean"))
    confirmation=main[main.test_year.eq(2025)][["candidate","rmse_log","wape","aggregate_bias"]].rename(columns={"rmse_log":"confirmation_rmse_log","wape":"confirmation_wape","aggregate_bias":"confirmation_bias"})
    decision=dev.merge(confirmation,on="candidate",validate="one_to_one")
    target=decision.set_index("candidate").loc["lgbm_intraday_update"]
    gate_rows=[]
    for reference in ["fixed_daily","lgbm_daily_control"]:
        base=decision.set_index("candidate").loc[reference]
        pivot=main.pivot(index="candidate",columns="test_year",values="rmse_log")
        worst=float((pivot.loc["lgbm_intraday_update",[2022,2023,2024]]-pivot.loc[reference,[2022,2023,2024]]).max())
        ci=bootstrap[(bootstrap.candidate=="lgbm_intraday_update")&(bootstrap.reference==reference)&bootstrap.period.eq("development")&bootstrap.metric.eq("log_mae_delta")].iloc[0]
        change=float(target.development_rmse_log-base.development_rmse_log)
        gate_rows.append({"reference":reference,"rmse_change":change,"worst_annual_rmse_change":worst,"wape_change":float(target.development_wape-base.development_wape),
            "paired_log_mae_ci_upper":ci.ci_97_5,"passes":bool(change<=-.002 and worst<=.002 and target.development_wape<=base.development_wape and ci.ci_97_5<0)})
    pd.DataFrame(gate_rows).to_csv(OUT / "intraday_information_gate.csv",index=False)
    decision.to_csv(OUT / "intraday_comparison.csv",index=False)
    available=annual[annual["sample"].eq("checkpoint_available")&annual.test_year.isin([2022,2023,2024])].groupby("candidate",as_index=False)[["rmse_log","wape","aggregate_bias"]].mean()
    available.to_csv(OUT / "available_checkpoint_development_metrics.csv",index=False)
    segment_queries=[]
    for candidate in METHODS:
        for dimension,expression in {
            "size":"CASE WHEN size_percentile<.33 THEN 'small' WHEN size_percentile<.67 THEN 'mid' ELSE 'large' END",
            "history":"CASE WHEN prior_auction_observations<5 THEN 'cold_0_4' WHEN prior_auction_observations<20 THEN 'developing_5_19' ELSE 'seasoned' END",
            "actual_tail":"CASE WHEN actual_percentile>=.99 THEN 'top_1pct' ELSE 'lower_99pct' END",
            "event":"CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",
            "availability":"CASE WHEN cp_available THEN 'available' ELSE 'daily_fallback' END"}.items():
            segment_queries.append(f'''SELECT '{candidate}' candidate,'{dimension}' dimension,{expression} segment,count(*) observations,
                sqrt(avg(pow(ln(1+"{candidate}")-log1p_auction_volume,2))) rmse_log,
                sum(abs("{candidate}"-auction_volume))/sum(auction_volume) wape,
                sum("{candidate}")/sum(auction_volume)-1 aggregate_bias FROM ranked GROUP BY 1,2,3''')
    segment=con.sql(f'''WITH ranked AS(SELECT *,percent_rank()OVER(PARTITION BY year(date) ORDER BY auction_volume) actual_percentile
        FROM read_parquet('{glob}') WHERE year(date) IN(2022,2023,2024)) {" UNION ALL ".join(segment_queries)}''').df()
    segment.to_csv(OUT / "segment_diagnostics.csv",index=False)
    coverage=pd.read_csv(OUT / "coverage_by_year.csv")
    rows=coverage[coverage.year.between(2022,2025)]
    coverage_rate=float(rows.valid_checkpoint_rows.sum()/rows.regular_session_rows.sum())
    manifest={"completed_utc":datetime.now(timezone.utc).isoformat(),"all_checks_passed":True,"checks":len(checks),"prediction_files":files,
        "regular_session_rows":sum(f["rows"] for f in files),"checkpoint_coverage":coverage_rate,"intraday_information_gate_passed":all(r["passes"] for r in gate_rows),
        "independent_history_audit_rows":len(aligned),"independent_history_audit_stocks":len(sample_ids),"plan_sha256":sha(OUT / "experiment_plan.json")}
    (OUT / "experiment_summary.json").write_text(json.dumps(manifest,indent=2))
    samples=[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted((OUT / "samples").glob("*.parquet"))]
    pd.DataFrame(samples).to_csv(OUT / "training_sample_fingerprints.csv",index=False)
    selected_rows=[]
    for year in range(2022,2026):
        chosen=json.loads((OUT / f"models/reconstruction_{year}.json").read_text())["selected"]
        for candidate,settings in chosen.items():
            row={**settings}
            if candidate!="ridge_update":
                row["requested_final_trees"]=max(25,settings["iterations"])
                row["saved_final_trees"]=lgb.Booster(model_file=str(OUT / f"models/{candidate}_{year}.txt")).num_trees()
            selected_rows.append(row)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_hyperparameters.csv",index=False)
    pretty={"fixed_daily":"Fixed daily forecast","checkpoint_ratio_rule":"Checkpoint ratio rule","ridge_update":"Ridge intraday update",
            "lgbm_daily_control":"Daily-only LightGBM control","lgbm_intraday_update":"15:30 LightGBM update"}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    d=decision.set_index("candidate").loc[METHODS]
    for ax,metric,title,mult in [(axes[0],"development_rmse_log","Development log RMSE",1),(axes[1],"development_wape","Development WAPE (%)",100)]:
        ax.barh([pretty[c] for c in METHODS],mult*d[metric],color=["#7c8791","#c58b3b","#7395b6","#97adc1","#287b70"])
        ax.set_title(title+"; lower is better")
    fig.suptitle("15:30 update on identical regular-session stock-days, 2022-2024")
    fig.tight_layout()
    fig.savefig(OUT / "intraday_comparison.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    lines=["| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in d.reset_index().itertuples():
        lines.append(f"| {pretty[r.candidate]} | {r.development_rmse_log:.5f} | {r.development_wape:.2%} | {r.development_bias:.2%} | {r.confirmation_rmse_log:.5f} | {r.confirmation_wape:.2%} |")
    conclusion="The 15:30 LightGBM update passes the prespecified incremental-information gate against both references." if manifest["intraday_information_gate_passed"] else "The 15:30 LightGBM update does not pass every prespecified incremental-information criterion; retain the references and inspect the comparison."
    report=f'''# First 15:30 intraday forecast update

{conclusion}

All {len(checks)} verification checks passed. Evaluation includes {manifest['regular_session_rows']:,} eligible regular-session stock-days over 2022-2025. Valid checkpoints cover {coverage_rate:.2%}; missing checkpoints use the unchanged daily forecast for every method.

{chr(10).join(lines)}

Development metrics average annual 2022-2024 values with equal year weight. The previously inspected 2025 year is confirmation evidence. These daily-reference metrics differ from earlier whole-year tables because early-close sessions are excluded for all candidates here.

## Forecast and information set

The outcome remains official closing-auction shares. The 15:30 update sees current cumulative trading volume, dollar volume, trade count, VWAP, recent 30-minute activity, and prior-only normalizations. It does not see final daily volume, closing returns, auction outcomes, order imbalance, or later checkpoints. The extraction time boundary relies on the upstream checkpoint files; these aggregates do not contain underlying tick timestamps for an independent cutoff re-audit.

The daily-only learned control receives the same original 58 daily features, saved daily forecast, prior price and historical checkpoint summaries used for normalization. It receives no current-session checkpoint inputs. Its sample rows, preprocessing, search budget, target and forecast bounds match the intraday LightGBM model. Beating this control is the evidence for additional current-session information within this experiment, rather than attributing every gain over the fixed daily forecast to intraday information.

The simple rule multiplies today's cumulative checkpoint volume by the prior-only 20-stock-row median auction/checkpoint ratio. Missing ratio history uses the fitting sample's median ratio. Ridge and LightGBM learn a log1p correction around the fixed daily forecast. Original daily predictions and model files are preserved.

## Historical training and validation

Update training uses only genuinely out-of-time daily forecasts from 2021 onward. It never uses fitted in-sample daily predictions. For the 2022 fold, tuning fits Jan-Sep 2021 and validates Oct-Dec 2021; the final model fits available 2021 rows. Later folds tune on the previous full year, train on earlier years from 2021, and refit through the previous year. All model, imputation and clipping parameters are fitted before the evaluation year.

This expanding 2021-onward update history differs from the daily model's five-year history. The initial quarterly validation is also distinct from later annual validation. These are disclosed limitations of this first bounded intraday study. The base model specification and 2025 results had already been inspected, so none of this is claimed as untouched production validation.

The preceding experiments' minimum-25-tree final-refit convention is retained. The daily-only control selected five trees for 2023 and four for 2025, and was refitted with 25 in those folds. This can affect the learned-control comparison; it does not alter the fixed daily reference. Both learned LightGBM candidates follow the same rule. Exact selected and saved tree counts appear in `selected_hyperparameters.csv`. No post-result tree-count sensitivity or additional tuning was introduced.

Early-close sessions are excluded because 15:30 occurs after their auction. Missing or invalid regular-session checkpoints retain the daily forecast exactly. No model claims a usable 15:30 forecast on early-close sessions. The available-checkpoint-only summary is separate from the primary all-regular-session comparison.

All learned/control candidates and the simple rule share bounds derived from update-fitting log targets. The fixed daily reference retains its saved bounds. The daily-only control helps isolate the effect of adding checkpoint inputs from differences in the extra modeling layer and these common update rules.

## Verification and uncertainty

Written predictions were checked for exact eligible keys and labels, unchanged daily reference, no early-close inclusion, finite nonnegative forecasts, exact missing-checkpoint fallback, metric recalculation, and reconstruction from saved models. Independent pandas calculations matched three prior-only checkpoint history features on {len(aligned):,} rows from {len(sample_ids)} stocks.

The information gate requires at least 0.002 mean development log-RMSE improvement versus both the fixed daily forecast and the daily-only learned control, no annual deterioration beyond 0.002, no mean WAPE worsening, and paired log-MAE upper confidence bounds below zero against both. Whole-date bootstrap intervals preserve same-date dependence but not cross-date serial dependence and are not adjusted for multiple comparisons. The study measures forecast accuracy, not execution-cost savings.

## Outputs

- `annual_metrics.csv`, `intraday_comparison.csv`: common-row annual and development/confirmation results.
- `coverage_by_year.csv`, `available_checkpoint_development_metrics.csv`: coverage, exclusions and observed-checkpoint-only results.
- `intraday_information_gate.csv`, `paired_bootstrap.csv`: comparisons against both references.
- `predictions/`, `models/`, `samples/`: complete forecasts, fitted models, historical samples and preprocessing.
- `intraday_panel.parquet`, `feature_timing_catalog.csv`: engineered data and input availability contract.
- `segment_diagnostics.csv`: size, history, event, tail and availability diagnostics.
- `validation_checks.csv`, `input_fingerprints.csv`, `training_sample_fingerprints.csv`, `experiment_plan.json`, `experiment_summary.json`: checks and reproducibility records.

Run `.venv/bin/python notebooks/build_intraday_1530_update.py` with the preceding OpenMP runtime environment. Append `--summarize` to verify and rebuild reports from saved forecasts without model fitting.
'''
    (OUT / "README.md").write_text(report)
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# 19 — First 15:30 intraday update\n\nThe experiment was executed locally through `build_intraday_1530_update.py`. These cells inspect saved results. The daily-only control isolates the contribution of current-session information."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'blackrock_data').exists())\nout = root / 'outputs/experiments/19_intraday_1530'"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'intraday_comparison.csv'))\ndisplay(Image(filename=str(out / 'intraday_comparison.png')))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'coverage_by_year.csv'))\ndisplay(pd.read_csv(out / 'intraday_information_gate.csv'))\ndisplay(pd.read_csv(out / 'paired_bootstrap.csv'))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'segment_diagnostics.csv'))\nchecks = pd.read_csv(out / 'validation_checks.csv')\nassert checks.passed.all()\ndisplay(checks)")])
    n.metadata.kernelspec={"display_name":"Auction local","name":"auction-local","language":"python"}
    nbformat.write(n,ROOT / "notebooks/19_intraday_1530_update.ipynb")
    con.close()
    progress("Intraday comparison, verification and results notebook complete")
    print(decision.to_string(index=False),flush=True)


if __name__=="__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        run()
