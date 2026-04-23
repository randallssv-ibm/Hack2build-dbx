# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Hack2Build — Prophet Cashflow Forecasting Model
# MAGIC **Experiment:** `2490562988127812`
# MAGIC
# MAGIC Trains one Prophet model per material (weekly grain, multiplicative seasonality).
# MAGIC DE temperature + anomaly used as external regressors for the heat-wave demand signal.

# COMMAND ----------

# MAGIC %pip install mlflow prophet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Imports & experiment

# COMMAND ----------

import mlflow
import mlflow.prophet
import pandas as pd
import numpy as np
from prophet import Prophet
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
mlflow.set_experiment(experiment_id="2490562988127812")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Parameters

# COMMAND ----------

FEATURE_TABLE      = "h2b_dbx_salesorder.featuredatasets.material_lvl"
WEATHER_TABLE      = "h2b_bdc_weather.weather.weather"
RESULT_CATALOG     = "h2b_dbx_salesorder"
RESULT_SCHEMA      = "forecasts"
RESULT_TABLE       = "forecast_salesorderquantity_material"

TRAINING_END       = "2025-12-31"
FORECAST_HORIZON   = 26          # weeks

# CM-MLFL-KM-VXX excluded — phased out 2025-06-01, no forecast needed
FORECAST_MATERIALS = ["TG11", "TG12", "FPP", "RTE", "CM-FL-V00"]

# DE stations carry the heat-wave signal; Tier A (91% revenue) is German market
WEATHER_COUNTRIES  = ["DE"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Load & aggregate training data (daily → weekly)

# COMMAND ----------

def floor_to_monday(s):
    """Floor a datetime Series to the Monday of its ISO week."""
    return s - pd.to_timedelta(s.dt.dayofweek, unit="D")


raw_pdf = (
    spark.table(FEATURE_TABLE)
    .withColumn("SalesOrderDate", F.to_timestamp("SalesOrderDate"))
    .filter(F.col("SalesOrderDate") <= TRAINING_END)
    .filter(F.col("Material").isin(FORECAST_MATERIALS))
    .select("SalesOrderDate", "Material", "MaterialGroup", "Sum_OrderQuantity")
    .toPandas()
)
raw_pdf["SalesOrderDate"] = pd.to_datetime(raw_pdf["SalesOrderDate"])
raw_pdf["week_start"]     = floor_to_monday(raw_pdf["SalesOrderDate"])

history_pdf = (
    raw_pdf
    .groupby(["week_start", "Material", "MaterialGroup"], as_index=False)
    .agg(Sum_OrderQuantity=("Sum_OrderQuantity", "sum"))
    .rename(columns={"week_start": "SalesOrderDate"})
)

print(f"Training rows  : {len(history_pdf):,}  (weekly)")
print(f"Date range     : {history_pdf['SalesOrderDate'].min().date()} → {history_pdf['SalesOrderDate'].max().date()}")
print(f"Weeks/material : ~{len(history_pdf) // history_pdf['Material'].nunique()}")
print(f"Materials      : {sorted(history_pdf['Material'].unique())}")
display(history_pdf.groupby("Material")["Sum_OrderQuantity"].describe().round(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Weather regressors
# MAGIC > **Known issue \#1** — current grain is monthly. Weekly weather upgrade pending.

# COMMAND ----------

weather_pdf = spark.table(WEATHER_TABLE).toPandas()

de_monthly = (
    weather_pdf[weather_pdf["country"].isin(WEATHER_COUNTRIES)]
    .groupby(["year", "month"], as_index=False)
    .agg(avg_temp_c=("avg_temp_c", "mean"), temp_anomaly_c=("temp_anomaly_c", "mean"))
)
de_monthly["month_start"] = pd.to_datetime(
    de_monthly["year"].astype(str) + "-" +
    de_monthly["month"].astype(str).str.zfill(2) + "-01"
)
weather_monthly = de_monthly[["month_start", "avg_temp_c", "temp_anomaly_c"]].copy()

print(f"Weather monthly: {len(weather_monthly)} rows  "
      f"{weather_monthly['month_start'].min().date()} → {weather_monthly['month_start'].max().date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Prophet helpers

# COMMAND ----------

def _merge_weather(df, ds_col, weather_monthly):
    """Join monthly DE weather onto df by year-month.
    Robust to any grain in df — no exact-date dependency.
    Unmatched months fall back to 10 °C / 0 anomaly.
    """
    tmp = df.copy()
    tmp["_ym"] = pd.to_datetime(tmp[ds_col]).dt.to_period("M").dt.to_timestamp()
    out = tmp.merge(
        weather_monthly.rename(columns={"month_start": "_ym"}),
        on="_ym", how="left"
    ).drop(columns=["_ym"])
    out[["avg_temp_c", "temp_anomaly_c"]] = (
        out[["avg_temp_c", "temp_anomaly_c"]].ffill().bfill().fillna(10.0)
    )
    return out


def train_prophet(material, mat_df, weather_monthly, forecast_horizon):
    """Train one Prophet model, log to MLflow, return forecast-window rows."""
    mat_group = mat_df["MaterialGroup"].iloc[0]

    train = (
        mat_df[["SalesOrderDate", "Sum_OrderQuantity"]]
        .rename(columns={"SalesOrderDate": "ds", "Sum_OrderQuantity": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    train = _merge_weather(train, "ds", weather_monthly)

    with mlflow.start_run(run_name=f"prophet_{material}"):
        mlflow.log_params({
            "material":               material,
            "material_group":         mat_group,
            "training_rows":          len(train),
            "training_end":           str(train["ds"].max().date()),
            "forecast_horizon_weeks": forecast_horizon,
            "seasonality_mode":       "multiplicative",
            "weather_regressors":     "avg_temp_c, temp_anomaly_c",
        })

        m = Prophet(
            yearly_seasonality = True,
            weekly_seasonality = False,
            daily_seasonality  = False,
            seasonality_mode   = "multiplicative",
            interval_width     = 0.80,
        )
        m.add_regressor("avg_temp_c")
        m.add_regressor("temp_anomaly_c")
        m.fit(train)

        future   = m.make_future_dataframe(periods=forecast_horizon, freq="W-MON")
        future   = _merge_weather(future, "ds", weather_monthly)
        forecast = m.predict(future)

        # Fit metrics on training window
        fit  = forecast[forecast["ds"].isin(train["ds"])][["ds", "yhat"]].merge(
                   train[["ds", "y"]], on="ds")
        mae  = float(np.mean(np.abs(fit["y"] - fit["yhat"])))
        mape = float(np.mean(np.abs((fit["y"] - fit["yhat"]) / fit["y"].clip(lower=1))) * 100)
        mlflow.log_metrics({"train_mae": round(mae, 2), "train_mape_pct": round(mape, 2)})

        input_example = train[["ds", "avg_temp_c", "temp_anomaly_c"]].head(5)
        mlflow.prophet.log_model(m, name=f"prophet_{material}",
                                 input_example=input_example)

        print(f"  {material:<20s}  train_rows={len(train):>3}  "
              f"MAE={mae:,.0f}  MAPE={mape:.1f}%")

        cutoff = train["ds"].max()
        out = forecast[forecast["ds"] > cutoff][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        out["Material"]      = material
        out["MaterialGroup"] = mat_group
        return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Train — one Prophet model per material

# COMMAND ----------

all_forecasts = []
print("Training Prophet models...\n")

for material in FORECAST_MATERIALS:
    mat_df = history_pdf[history_pdf["Material"] == material].copy()
    if len(mat_df) < 10:
        print(f"  {material}: skipping — only {len(mat_df)} rows")
        continue
    try:
        fc = train_prophet(material, mat_df, weather_monthly, FORECAST_HORIZON)
        all_forecasts.append(fc)
    except Exception as e:
        import traceback
        print(f"  ERROR in {material}: {e}")
        traceback.print_exc()

forecast_pdf = pd.concat(all_forecasts, ignore_index=True)
print(f"\nForecast rows: {len(forecast_pdf):,}  "
      f"({forecast_pdf['ds'].min().date()} → {forecast_pdf['ds'].max().date()})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Join actuals & save results

# COMMAND ----------

# Read actuals directly from raw tables — material_lvl only covers training window
# date_trunc('WEEK') in Spark truncates to Monday (ISO standard)
actuals_spark = (
    spark.table("h2b_dbx_salesorder.salesorder.salesorderitem")
    .join(
        spark.table("h2b_dbx_salesorder.salesorder.salesorder")
             .select("SalesOrder", F.to_timestamp("SalesOrderDate").alias("SalesOrderDate")),
        on="SalesOrder"
    )
    .filter(F.col("SalesOrderDate") > TRAINING_END)
    .filter(F.col("Material").isin(FORECAST_MATERIALS))
    .withColumn("SalesOrderDate",
                F.date_trunc("WEEK", F.col("SalesOrderDate")).cast("date"))
    .groupBy("SalesOrderDate", "MaterialGroup", "Material")
    .agg(F.sum("OrderQuantity").alias("actual_qty"))
)

forecast_spark = spark.createDataFrame(
    forecast_pdf.rename(columns={
        "ds":          "SalesOrderDate",
        "yhat":        "predicted_qty",
        "yhat_lower":  "predicted_qty_lower",
        "yhat_upper":  "predicted_qty_upper",
    })
)

results = (
    forecast_spark
    .join(actuals_spark, on=["SalesOrderDate", "Material", "MaterialGroup"], how="left")
    .withColumn("gap_actual_vs_forecast",
                F.col("actual_qty") - F.col("predicted_qty"))
    .withColumn("pct_error",
                F.when(F.col("actual_qty").isNotNull(),
                       F.round(F.abs(F.col("gap_actual_vs_forecast"))
                               / F.col("actual_qty") * 100, 2)))
    .orderBy("Material", "SalesOrderDate")
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {RESULT_CATALOG}.{RESULT_SCHEMA}")
(
    results.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{RESULT_CATALOG}.{RESULT_SCHEMA}.{RESULT_TABLE}")
)

print(f"Saved {results.count():,} rows → {RESULT_CATALOG}.{RESULT_SCHEMA}.{RESULT_TABLE}")
display(results)
