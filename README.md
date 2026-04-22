# Hack2Build — SAP BDC Cashflow Forecasting PoC

## Executive Summary

This proof of concept demonstrates how a beverage distributor can predict cashflow 6 months ahead by connecting customer order behaviour to weather signals. Using SAP BDC as the source of truth for master data, we generate a realistic synthetic transaction history, train a forecasting model, and compare its predictions against known actuals — giving a clear, quantified view of forecast accuracy before any production investment is made.

**Business question:** Can we predict monthly cashflow with enough accuracy to replace reactive cash management with a forward-looking plan?

**Expected outcome:** A model that outperforms the naive seasonal baseline on a 6-month held-out test window, with results presented as predicted vs. actual cashflow per period and product.

---

## Business Briefing

### What we are forecasting

Cashflow for a beverage distributor is driven primarily by when customers pay for orders. Payment timing depends on the customer relationship (credit terms, dunning history), the volume of goods ordered, and the price of those goods. Volume, in turn, is influenced by the season and — critically for beverages — by temperature. Hot summers drive outsized demand for cooling drinks.

This PoC encodes that logic end-to-end: from a heat wave in Germany to a spike in sales orders, through to cash hitting the account weeks later.

### The demand signal

Three customer tiers cover the full range of the distributor's book:

| Tier | Profile | Revenue share |
|------|---------|---------------|
| A | 3 large accounts, high volume, fast payers | ~91% |
| B | 4 mid-size accounts, moderate volume | ~9% |
| C | 3 small accounts, low volume, slower payers | <1% |

The synthetic history runs from January 2023 through June 2026 — 36 months of training data followed by a 6-month test window that mirrors the live forecast horizon.

### The weather signal

Two consecutive above-average summers (Germany, 2023 and 2024) create a strong, unambiguous demand signal. A normal 2025 summer validates that the model does not over-forecast when the heat wave is absent. This three-year pattern gives the model clear evidence to learn from before it is asked to predict 2026.

### How performance is measured

Model predictions for January–June 2026 are compared directly against the generated actuals for the same period. The benchmark is the naive seasonal baseline — if the model cannot beat a simple year-over-year repeat, it adds no value. Results are published to a dedicated results catalog as predicted vs. actual cashflow per period and material, with delta and relative error.

### Current pipeline status

| Step | Deliverable | Status |
|------|------------|--------|
| S1 | Synthetic weather data (5 regions, 42 months) | Done |
| S2 | Sales orders + order items (7,000 orders, 42 months) | Done |
| S3 | Billing documents + billing document items | Pending |
| S4 | Cashflow records | Pending |
| ML | Feature engineering, model training, forecast | Pending |

---

## Technical Briefing

### Repository structure

```
mock_data_spec.md          ← full generation spec (source of truth)
mock_data/
  Weather Mock.ipynb       ← S1: NOAA weather (done)
  SalesOrder mock.ipynb    ← S2: sales orders + items (done)
```

---

### Data generation overview

#### Timeline

| Window | Period | Months | Purpose |
|--------|--------|--------|---------|
| Training | 2023-01-01 → 2025-12-31 | 36 | Model training |
| Test | 2026-01-01 → 2026-06-30 | 6 | Model evaluation |
| Weather | 2023-01-01 → 2026-06-30 | 42 | Covers both windows |

The test window shares the same timeframe as the forecast — generated actuals for Jan–Jun 2026 are used to evaluate model performance against the naive seasonal baseline.

---

#### Catalog conventions

| Data type | Catalog pattern | Example |
|-----------|----------------|---------|
| BDC source — read-only | `h2b_bdc_{entity}.{entity}.{table}` | `h2b_bdc_customer.customer.customer` |
| Generated weather (mimics BDC source) | `h2b_bdc_weather.weather.weather` | — |
| Generated mock transactions (DBX side) | `h2b_dbx_{entity}.{entity}.{table}` | `h2b_dbx_salesorder.salesorder.salesorder` |
| ML feature datasets + models | `h2b_dbx_{entity}.featuredatasets / mlmodels / forecasts` | — |
| Forecast results + actuals comparison | `h2b_dbx_resultset.resultset.{table}` | — |

Table names contain no underscores (`salesorderitem`, `billingdocument`, etc.).

---

#### BDC → DBX data cloning

Before populating any DBX catalog with mock data, the schema structure is cloned from its BDC counterpart using the Databricks SDK. This ensures the DBX tables inherit the correct column definitions from the BDC Delta Share.

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# 1. Discover schemas and tables from the BDC catalog (Delta Share)
schemas = [s.name for s in w.schemas.list(catalog_name="h2b_bdc_{entity}")
           if s.name != "information_schema"]
tables  = [(t.schema_name, t.name)
           for s in schemas
           for t in w.tables.list(catalog_name="h2b_bdc_{entity}", schema_name=s)]

# 2. Mirror structure into the DBX catalog
for schema, table in tables:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS h2b_dbx_{{entity}}.{schema}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS h2b_dbx_{{entity}}.{schema}.{table}
        AS SELECT * FROM h2b_bdc_{{entity}}.{schema}.{table}
    """)

# 3. Truncate — ready to receive mock data
for schema, table in tables:
    spark.sql(f"TRUNCATE TABLE h2b_dbx_{{entity}}.{schema}.{table}")
```

This clone-then-truncate pattern means:
- The DBX catalog has schema parity with the BDC source at generation time
- Mock data is inserted into correctly typed tables
- The BDC source is never written to

---

#### Volume targets

| Table | Catalog path | ~Rows |
|-------|-------------|-------|
| `weather` | `h2b_bdc_weather.weather.weather` | 210 |
| `salesorder` | `h2b_dbx_salesorder.salesorder.salesorder` | ~7,000 |
| `salesorderitem` | `h2b_dbx_salesorder.salesorder.salesorderitem` | ~17,500 |
| `billingdocument` | `h2b_dbx_billingdocument.billingdocument.billingdocument` | ~4,851 |
| `billingdocumentitem` | `h2b_dbx_billingdocument.billingdocument.billingdocumentitem` | ~12,600 |
| `cashflow` | `h2b_dbx_cashflow.cashflow.cashflow` | ~3,969 |

---

### How the data is generated

#### S1 — WeatherNOAA

Five regional stations covering the customer base geography:

| Station | Country | Region |
|---------|---------|--------|
| NOAA_DE_BY | DE | BY |
| NOAA_DE_NW | DE | NW |
| NOAA_FR_IDF | FR | IDF |
| NOAA_ES_MD | ES | MD |
| NOAA_US_CA | US | CA |

**Temperature** follows a single-harmonic sinusoidal baseline fitted to four monthly anchors (Jan / Apr / Jul / Oct), plus a warming trend of +0.04 °C per month, plus Normal(0, 0.8) noise.

**Heat wave signal** — the primary ML feature:

| Year | Months | Regions | Anomaly |
|------|--------|---------|---------|
| 2023 | Jul–Aug | DE/BY + DE/NW | Uniform(2.5, 3.5) °C |
| 2024 | Jul–Aug | DE/BY + DE/NW | Uniform(3.0, 4.0) °C — stronger |
| 2025 | all | all | Normal(0, 0.6) °C — normal baseline |

Two consecutive above-baseline summers give the model a strong, unambiguous signal. The 2025 normal summer validates it doesn't over-forecast when the heat wave is absent.

**Precipitation** uses seasonal Uniform ranges (Winter 40–80 mm, Spring 50–90 mm, Summer 20–60 mm, Autumn 50–100 mm) with a 10% chance of an extreme month (×2.5 multiplier).

---

#### S2 — SalesOrder + SalesOrderItem

**7,000 orders** distributed across 42 months (~167/month average). Order density follows a seasonal weight so more orders fall in peak months naturally.

##### Customer tiers

| Tier | Customers | Order weight | Qty range (CS) |
|------|-----------|-------------|----------------|
| A | 10100006, 10100002, 12200001 | 6× | 200–800 |
| B | 10100012, 10186001–3 | 2× | 40–180 |
| C | EWM10-CU01–03 | 1× | 2–15 |

Revenue split: A ≈ 91% · B ≈ 9% · C < 0.1%. Tier map is hardcoded — not derived from `CustomerABCClassification`.

##### Order quantity formula

Each item quantity is the product of four factors:

```
OrderQty = max(1, round(base_qty × seasonal_weight × weather_mult × lognormal(0, 0.10)))
```

1. **base_qty** — Uniform draw per tier (A: 200–800, B: 40–180, C: 2–15)
2. **seasonal_weight** — peaks in Jul–Aug and Oct–Nov, dips in Jan–Feb; L001 has the full swing, L004 half, P001 nearly flat; grows +3% YoY
3. **weather_mult** — temperature ramp above 22 °C up to +80% at 27 °C, with an additional bonus for positive anomalies (heat waves); capped at 3×
4. **lognormal noise** — σ=0.10, kept low to preserve the weather signal

##### Materials

| Material | Group | Seasonal profile | Price EUR/CS | Status |
|----------|-------|-----------------|--------------|--------|
| TG11, TG12 | L001 | summer-peak | 8–18 | active |
| FPP, RTE | P001 | flat ±5% | 5–12 | active |
| CM-FL-V00 | L004 | mild-summer | 15–35 | active |
| CM-MLFL-KM-VXX | L004 | mild-summer | 15–35 | phase-out from 2025-06-01 |

Base prices are drawn once at seed=42 and reused for every order. Seasonal surges are applied on top: +5% Jul–Aug (non-P001), and +5–10% Sep–Dec for two designated SKUs (one P001, CM-MLFL-KM-VXX).

##### Order date distribution

Creation dates are allocated month-by-month using:
```
weight(month, year) = 1.0 + seasonal_offset(month) × (1 + 0.03 × (year − 2023))
```
where seasonal_offset is +0.40 Jul–Aug, +0.20 Oct–Nov, −0.20 Jan–Feb, 0 otherwise. Within each month a random day is drawn uniformly.

---

#### S3 — BillingDocument + BillingDocumentItem

One billing document is created per **eligible** sales order — those with `OverallSDProcessStatus ∈ {B, C}` and no rejection (`OverallSDDocumentRejectionSts ≠ C`). That filter passes ~63% of orders (~4,410 of 7,000).

**10% of billed orders** get a cancellation pair: the original F2 document is marked `BillingDocumentIsCancelled=True`, and a new S1 document is created with negated amounts.

Billing document date = `CreationDate + randint(3, 10)` days — this is the DSO clock start.

---

#### S4 — CashFlow

**CashFlow** — one record per non-cancelled F2 billing document (~3,969 rows). Payment amount is the billing total discounted by a tier-based collection rate and a dunning-level multiplier. Posting date is derived from:

```
PostingDate = BillingDocumentDate
            + base_DSO(tier)        ← Normal(30,5) A / Normal(45,8) B / Normal(65,20) C
            + dunning_adjustment    ← 0d / +10d / +25d / +50d by level
            + noise                 ← Normal(0, 3)
            capped at 180 days, minimum 1 day after BillingDocumentDate
```

---

### ML catalog structure

All ML artifacts live inside the same DBX entity catalog (`h2b_dbx_{entity}`) alongside the transactional data, organised into dedicated schemas.

```
h2b_dbx_salesorder/
├── salesorder/              ← transactional mock data
│   ├── salesorder
│   └── salesorderitem
│
├── featuredatasets/         ← engineered feature tables
│   ├── train_agg            ← 2023-01-01 → 2025-12-31  used to train models
│   ├── complete_agg         ← 2023-01-01 → 2026-06-30  train + test window combined
│   │                           use this to compare predicted vs actual after the fact
│   └── (material-level cuts, e.g. salesorders_tg11, salesorders_fpp …)
│
├── forecasts/               ← model output tables
│   ├── forecast_salesorder_matlvl_weekly
│   ├── forecast_salesorderquantity_material
│   └── forecast_salesorderquantity_material_withid
│
└── mlmodels/                ← registered MLflow models
    ├── forecast_model_{run_id}
    └── forecast_salesorder_matlvl_weekly
```

#### Feature dataset purpose

| Table | Rows cover | Purpose |
|-------|-----------|---------|
| `train_agg` | 2023-01 → 2025-12 (36 months) | Input to model training — no future leakage |
| `complete_agg` | 2023-01 → 2026-06 (42 months) | Full series including test window; used post-prediction to compare forecast vs actuals |

The split is purely by date — there are no flag columns to distinguish training from test rows. Consumers filter on the date range they need.

---

### Forecast results and actuals comparison

The `h2b_dbx_resultset` catalog is the outcome layer of the pipeline. It holds the side-by-side comparison of model predictions against generated actuals for the Jan–Jun 2026 test window.

```
h2b_dbx_resultset/
└── resultset/
    └── forecast_results     ← predicted vs actual per period/material
```

**`forecast_results`** joins:
- Model output from `h2b_dbx_salesorder.forecasts.*`
- Actuals from `complete_agg` (filtered to 2026-01 → 2026-06)

Expected columns include forecast value, actual value, delta, and relative error — giving a direct read on model performance vs the naive seasonal baseline.

---

### Relational integrity

```
WeatherNOAA ──(lookup)──► SalesOrderItem
                           join: (country, region, year, month)
                           default: (10.0°C, 0.0) if no match

SalesOrder ──1:N──► SalesOrderItem
                    every order has exactly 2 or 3 items
                    no orphan items, no orders without items

SalesOrder ──0..1:1──► BillingDocument (F2)
                       only ~63% of orders are eligible for billing
                       eligible orders get exactly 1 F2 document

BillingDocument (F2) ──0..1:1──► BillingDocument (S1, cancellation)
                                  10% of F2 docs get a paired cancellation

BillingDocument ──1:N──► BillingDocumentItem
                          mirrors the parent SalesOrderItems (2–3 items)
                          S1 cancellations carry negated amounts

BillingDocument (F2, not cancelled) ──1:1──► CashFlow
                                             ~90% of F2 docs produce a cash flow record

```

#### Eligibility cascade

```
7,000 SalesOrders
  × 70%  (OverallSDProcessStatus B or C)
  × 90%  (not rejected)
= ~4,410  F2 BillingDocuments
  × 10%  → ~441 paired S1 cancellation documents
  → ~4,851 total BillingDocuments

~4,410 F2 docs
  × 90%  (not cancelled)
= ~3,969 CashFlow records
```

---

### Key assumptions

| Assumption | Value / Rule |
|------------|-------------|
| Random seed | 42 (all notebooks) |
| Fixed base prices | Drawn once at seed=42, reused for every order |
| YoY order growth | +3% per year from 2023 baseline |
| Warming trend | +0.04 °C per month from 2023-01-01 |
| Phase-out | CM-MLFL-KM-VXX: zero weight from 2025-06-01 |
| FX rate | USD/EUR = 0.92, fixed |
| Tax rate | 19% on EUR orders, 0% on USD |
| DSO cap | 180 days maximum |
| Weather default | (10.0 °C, 0.0 anomaly) for unmapped regions |
| Tier map | Hardcoded — do not use CustomerABCClassification |
| Training window | 36 months (2023-01-01 → 2025-12-31) |
| Test window | 6 months (2026-01-01 → 2026-06-30) — same as forecast horizon |
