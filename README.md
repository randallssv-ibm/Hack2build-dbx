# Hack2Build — Databricks PoC
**SAP BDC Cashflow Forecasting · Synthetic Data Pipeline**

Databricks notebooks that generate a fully synthetic beverage-retail dataset for training and evaluating a cashflow forecasting ML model. Real master data (customers, products) is read from existing BDC catalogs and is never regenerated.

---

## Repository structure

```
mock_data_spec.md          ← full generation spec (source of truth)
mock_data/
  Weather Mock.ipynb       ← S1: NOAA weather (done)
  SalesOrder mock.ipynb    ← S2: sales orders + items (done)
```

---

## Data generation overview

### Timeline

| Window | Period | Months | Purpose |
|--------|--------|--------|---------|
| Training | 2023-01-01 → 2025-12-31 | 36 | Model training |
| Test | 2026-01-01 → 2026-06-30 | 6 | Model evaluation |
| Weather | 2023-01-01 → 2026-06-30 | 42 | Covers both windows |

The test window shares the same timeframe as the forecast — generated actuals for Jan–Jun 2026 are used to evaluate model performance against the naive seasonal baseline.

---

### Catalog conventions

| Data type | Catalog pattern | Example |
|-----------|----------------|---------|
| BDC source — read-only | `h2b_bdc_{entity}.{entity}.{table}` | `h2b_bdc_customer.customer.customer` |
| Generated weather (mimics BDC source) | `h2b_bdc_weather.weather.weather` | — |
| Generated mock transactions (DBX side) | `h2b_dbx_{entity}.{entity}.{table}` | `h2b_dbx_salesorder.salesorder.salesorder` |

Table names contain no underscores (`salesorderitem`, `billingdocument`, etc.).

---

### Volume targets

| Table | Catalog path | ~Rows |
|-------|-------------|-------|
| `weather` | `h2b_bdc_weather.weather.weather` | 210 |
| `salesorder` | `h2b_dbx_salesorder.salesorder.salesorder` | ~7,000 |
| `salesorderitem` | `h2b_dbx_salesorder.salesorder.salesorderitem` | ~17,500 |
| `billingdocument` | `h2b_dbx_billingdocument.billingdocument.billingdocument` | ~4,851 |
| `billingdocumentitem` | `h2b_dbx_billingdocument.billingdocument.billingdocumentitem` | ~12,600 |
| `cashflow` | `h2b_dbx_cashflow.cashflow.cashflow` | ~3,969 |
| `cashflowforecast` | `h2b_dbx_cashflow.cashflow.cashflowforecast` | ~181 |

---

## How the data is generated

### S1 — WeatherNOAA

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

### S2 — SalesOrder + SalesOrderItem

**7,000 orders** distributed across 42 months (~167/month average). Order density follows a seasonal weight so more orders fall in peak months naturally.

#### Customer tiers

| Tier | Customers | Order weight | Qty range (CS) |
|------|-----------|-------------|----------------|
| A | 10100006, 10100002, 12200001 | 6× | 200–800 |
| B | 10100012, 10186001–3 | 2× | 40–180 |
| C | EWM10-CU01–03 | 1× | 2–15 |

Revenue split: A ≈ 91% · B ≈ 9% · C < 0.1%. Tier map is hardcoded — not derived from `CustomerABCClassification`.

#### Order quantity formula

Each item quantity is the product of four factors:

```
OrderQty = max(1, round(base_qty × seasonal_weight × weather_mult × lognormal(0, 0.10)))
```

1. **base_qty** — Uniform draw per tier (A: 200–800, B: 40–180, C: 2–15)
2. **seasonal_weight** — peaks in Jul–Aug and Oct–Nov, dips in Jan–Feb; L001 has the full swing, L004 half, P001 nearly flat; grows +3% YoY
3. **weather_mult** — temperature ramp above 22 °C up to +80% at 27 °C, with an additional bonus for positive anomalies (heat waves); capped at 3×
4. **lognormal noise** — σ=0.10, kept low to preserve the weather signal

#### Materials

| Material | Group | Seasonal profile | Price EUR/CS | Status |
|----------|-------|-----------------|--------------|--------|
| TG11, TG12 | L001 | summer-peak | 8–18 | active |
| FPP, RTE | P001 | flat ±5% | 5–12 | active |
| CM-FL-V00 | L004 | mild-summer | 15–35 | active |
| CM-MLFL-KM-VXX | L004 | mild-summer | 15–35 | phase-out from 2025-06-01 |

Base prices are drawn once at seed=42 and reused for every order. Seasonal surges are applied on top: +5% Jul–Aug (non-P001), and +5–10% Sep–Dec for two designated SKUs (one P001, CM-MLFL-KM-VXX).

#### Order date distribution

Creation dates are allocated month-by-month using:
```
weight(month, year) = 1.0 + seasonal_offset(month) × (1 + 0.03 × (year − 2023))
```
where seasonal_offset is +0.40 Jul–Aug, +0.20 Oct–Nov, −0.20 Jan–Feb, 0 otherwise. Within each month a random day is drawn uniformly.

---

### S3 — BillingDocument + BillingDocumentItem

One billing document is created per **eligible** sales order — those with `OverallSDProcessStatus ∈ {B, C}` and no rejection (`OverallSDDocumentRejectionSts ≠ C`). That filter passes ~63% of orders (~4,410 of 7,000).

**10% of billed orders** get a cancellation pair: the original F2 document is marked `BillingDocumentIsCancelled=True`, and a new S1 document is created with negated amounts.

Billing document date = `CreationDate + randint(3, 10)` days — this is the DSO clock start.

---

### S4 — CashFlow + CashFlowForecast

**CashFlow** — one record per non-cancelled F2 billing document (~3,969 rows). Payment amount is the billing total discounted by a tier-based collection rate and a dunning-level multiplier. Posting date is derived from:

```
PostingDate = BillingDocumentDate
            + base_DSO(tier)        ← Normal(30,5) A / Normal(45,8) B / Normal(65,20) C
            + dunning_adjustment    ← 0d / +10d / +25d / +50d by level
            + noise                 ← Normal(0, 3)
            capped at 180 days, minimum 1 day after BillingDocumentDate
```

**CashFlowForecast** — naive seasonal baseline for Jan–Jun 2026 (~181 rows, one per calendar day). Built from the ISO-week average of historical cash flows, scaled by a 4% growth assumption and Uniform(0.92, 1.08) noise. Intentionally imperfect — the ML model is expected to beat it.

---

## Relational integrity

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

CashFlowForecast ── standalone (no FK)
                    one row per calendar day, Jan–Jun 2026
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

## Key assumptions

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
