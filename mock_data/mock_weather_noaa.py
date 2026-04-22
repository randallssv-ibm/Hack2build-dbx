# =============================================================================
# Mock Weather Data Generator — WeatherNOAA
# Seed: 42  |  5 regions × 42 months = 210 rows
# Coverage: 2023-01-01 → 2026-06-30  (36-month training + 6-month forecast)
# Target:   hack2build.bronze.weather_noaa
# =============================================================================

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, BooleanType,
)

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

_REGIONS = [
    {"station_id": "NOAA_DE_BY",  "station_name": "Bavaria",               "country": "DE", "region": "BY"},
    {"station_id": "NOAA_DE_NW",  "station_name": "North Rhine-Westphalia", "country": "DE", "region": "NW"},
    {"station_id": "NOAA_FR_IDF", "station_name": "Ile-de-France",          "country": "FR", "region": "IDF"},
    {"station_id": "NOAA_ES_MD",  "station_name": "Madrid",                 "country": "ES", "region": "MD"},
    {"station_id": "NOAA_US_CA",  "station_name": "California",             "country": "US", "region": "CA"},
]

# Monthly temperature anchors (Jan / Apr / Jul / Oct) in °C
_ANCHOR_TEMPS = {
    "DE_BY":  {1: -2, 4: 9,  7: 19, 10: 11},
    "DE_NW":  {1:  2, 4: 10, 7: 20, 10: 12},
    "FR_IDF": {1:  4, 4: 12, 7: 23, 10: 14},
    "ES_MD":  {1:  6, 4: 15, 7: 29, 10: 17},
    "US_CA":  {1: 13, 4: 17, 7: 27, 10: 21},
}

_WEATHER_SCHEMA = StructType([
    StructField("station_id",        StringType(),  False),
    StructField("station_name",      StringType(),  False),
    StructField("country",           StringType(),  False),
    StructField("region",            StringType(),  False),
    StructField("year",              IntegerType(), False),
    StructField("month",             IntegerType(), False),
    StructField("avg_temp_c",        DoubleType(),  False),
    StructField("temp_anomaly_c",    DoubleType(),  False),
    StructField("precipitation_mm",  DoubleType(),  False),
    StructField("extreme_heat_flag", BooleanType(), False),
])

_TARGET_TABLE = "h2b_bdc_weather.weather.weather"

# ---------------------------------------------------------------------------
# Sinusoidal baseline helpers
# ---------------------------------------------------------------------------

def _fit_sinusoid(anchor_dict: dict) -> np.ndarray:
    """
    Least-squares fit of a single-harmonic sinusoid to monthly anchor temps:
        T(m) = a0 + a1·cos(2π(m-1)/12) + b1·sin(2π(m-1)/12)
    Returns coefficients [a0, a1, b1].
    """
    ms = np.array(list(anchor_dict.keys()), dtype=float)
    vs = np.array(list(anchor_dict.values()), dtype=float)
    theta = 2.0 * np.pi * (ms - 1.0) / 12.0
    A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    coeffs, _, _, _ = np.linalg.lstsq(A, vs, rcond=None)
    return coeffs


def _baseline(coeffs: np.ndarray, month: int) -> float:
    theta = 2.0 * np.pi * (month - 1) / 12.0
    return float(coeffs[0] + coeffs[1] * np.cos(theta) + coeffs[2] * np.sin(theta))


# Pre-compute sinusoid coefficients at import time
_SINUSOID_COEFFS: dict[str, np.ndarray] = {
    k: _fit_sinusoid(v) for k, v in _ANCHOR_TEMPS.items()
}

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_weather_table(
    reference_date: str = "2023-01-01",
    months: int = 42,
) -> dict:
    """
    Generate synthetic NOAA-style monthly weather data for 5 regions.

    Parameters
    ----------
    reference_date : str
        ISO date string for the first month in the series (day is ignored).
    months : int
        Number of calendar months to generate.  Default 42 covers 2023-01 → 2026-06.

    Returns
    -------
    dict
        {'WeatherNOAA': pyspark.sql.DataFrame}  (also persisted to Delta in UC)

    Random draws per row (always 4, regardless of heat-wave branch):
        1. avg_temp noise      — Normal(0, 0.8)
        2. temp_anomaly_c      — Normal(0, 0.6)  |  Uniform heat-wave range
        3. precipitation base  — Uniform(season range)
        4. extreme precip roll — Uniform(0, 1)
    """
    spark = SparkSession.builder.getOrCreate()
    rng   = np.random.default_rng(seed=42)

    # Month-start timestamps for the full series
    date_index = pd.date_range(start=reference_date, periods=months, freq="MS")

    rows: list[tuple] = []

    # Outer loop: regions  |  Inner loop: months  →  deterministic draw order
    for region_info in _REGIONS:
        rkey    = f"{region_info['country']}_{region_info['region']}"
        coeffs  = _SINUSOID_COEFFS[rkey]
        jul_baseline = _baseline(coeffs, 7)

        for step, ts in enumerate(date_index):
            year  = int(ts.year)
            month = int(ts.month)

            # ── avg_temp_c ──────────────────────────────────────────────────
            base     = _baseline(coeffs, month)
            trend    = 0.04 * step                          # +0.04°C / month warming
            noise    = rng.normal(0.0, 0.8)                 # draw 1
            avg_temp = base + trend + noise

            # ── temp_anomaly_c & extreme_heat_flag ──────────────────────────
            # Heat-wave signal: DE_BY / DE_NW, Jul–Aug 2023 and 2024 only
            is_heat_wave = (
                rkey in ("DE_BY", "DE_NW")
                and month in (7, 8)
                and year in (2023, 2024)
            )

            if is_heat_wave:
                if year == 2023:
                    anomaly = rng.uniform(2.5, 3.5)         # draw 2 — 2023 event
                else:
                    anomaly = rng.uniform(3.0, 4.0)         # draw 2 — 2024 (stronger)
                extreme_heat = bool(avg_temp > (jul_baseline + 3.0))
            else:
                anomaly      = rng.normal(0.0, 0.6)         # draw 2 — background
                extreme_heat = False

            # ── precipitation_mm ────────────────────────────────────────────
            if month in (12, 1, 2):
                precip = rng.uniform(40.0,  80.0)           # draw 3 — Winter
            elif month in (3, 4, 5):
                precip = rng.uniform(50.0,  90.0)           # draw 3 — Spring
            elif month in (6, 7, 8):
                precip = rng.uniform(20.0,  60.0)           # draw 3 — Summer
            else:
                precip = rng.uniform(50.0, 100.0)           # draw 3 — Autumn

            if rng.random() < 0.10:                         # draw 4 — extreme event
                precip *= 2.5

            rows.append((
                region_info["station_id"],
                region_info["station_name"],
                region_info["country"],
                region_info["region"],
                year,
                month,
                round(float(avg_temp),  2),
                round(float(anomaly),   2),
                round(float(precip),    1),
                extreme_heat,
            ))

    # ── Pandas DataFrame ────────────────────────────────────────────────────
    col_names = [f.name for f in _WEATHER_SCHEMA.fields]
    pdf = pd.DataFrame(rows, columns=col_names)

    pdf = pdf.astype({
        "year":              "int32",
        "month":             "int32",
        "avg_temp_c":        "float64",
        "temp_anomaly_c":    "float64",
        "precipitation_mm":  "float64",
        "extreme_heat_flag": "bool",
    })

    # ── Spark DataFrame ─────────────────────────────────────────────────────
    df = spark.createDataFrame(pdf, schema=_WEATHER_SCHEMA)

    # ── Persist to Unity Catalog Delta table ────────────────────────────────
    (
        df.write
          .format("delta")
          .mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(_TARGET_TABLE)
    )

    print(f"[WeatherNOAA] {df.count()} rows written to {_TARGET_TABLE}")

    return {"WeatherNOAA": df}


# ---------------------------------------------------------------------------
# Entry point (run directly in a Databricks notebook or job)
# ---------------------------------------------------------------------------

result = generate_weather_table(reference_date="2023-01-01", months=42)
display(result["WeatherNOAA"])
