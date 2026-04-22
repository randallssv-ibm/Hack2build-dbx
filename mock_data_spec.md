# Hack2Build — Mock Data Specification
## SAP BDC Cashflow Forecasting PoC · Data Generation Guide

---

## 1. Overview

Synthetic beverage-retail dataset for a cashflow forecasting ML model.  
Real customer and product data live in `h2b_bdc_*` catalogs and are **read-only inputs** — do not regenerate them.

**Catalog conventions:**

| Type | Pattern | Example |
|------|---------|---------|
| BDC source data (read-only) | `h2b_bdc_{entity}.{entity}.{table}` | `h2b_bdc_customer.customer.customer` |
| Weather mock (mimics BDC source) | `h2b_bdc_weather.weather.weather` | — |
| Generated mock data (DBX side) | `h2b_dbx_{entity}.{entity}.{table}` | `h2b_dbx_salesorder.salesorder.salesorder` |

> Table names use no underscores: `salesorder`, `salesorderitem`, `billingdocument`, etc.

### Generation order (strict — never generate a child before its parent)

| Step | Table(s) | Depends on |
|------|----------|------------|
| S1 | WeatherNOAA | — |
| S2 | SalesOrder + SalesOrderItem | real product, real customer, WeatherNOAA |
| S3 | BillingDocument + BillingDocumentItem | S2 |
| S4 | CashFlow + CashFlowForecast | S3 + real customer_dunning |

---

## 2. Fixed constants

| Parameter | Value |
|-----------|-------|
| Company code | `CC01` |
| Sales org | `VKORG1` |
| Distribution channels | `10` (direct) · `20` (wholesale) |
| Plants | `PLANT1` Germany · `PLANT2` Netherlands |
| Default currency (domestic) | `EUR` |
| Export currency | `USD` · FX rate `0.92` fixed |
| Random seed | `42` |

---

## 3. Timeline

| Window | Start | End | Months |
|--------|-------|-----|--------|
| Training | `2023-01-01` | `2025-12-31` | 36 |
| Test | `2026-01-01` | `2026-06-30` | 6 |
| Forecast | `2026-01-01` | `2026-06-30` | 6 |
| Weather | `2023-01-01` | `2026-06-30` | 42 |

**ML assumption:** 36 months of historical transactional data (2023–2025) train the cashflow forecasting model. The 6-month test window (Jan–Jun 2026) shares the same timeframe as the forecast — generated actuals for that period are used to evaluate model performance against the naive seasonal baseline produced in S4.

---

## 4. Real data — read, do not mock

```python
# Products — filter to these 6 only
spark.table("h2b_bdc_product.product.product")
    .filter(col("Product").isin([
        'TG11','TG12','FPP','RTE','CM-FL-V00','CM-MLFL-KM-VXX']))

# Customer sales area — filter to these 10 only
spark.table("h2b_bdc_customer.customer.customersalesarea")
    .filter(col("Customer").isin([
        '10100006','10100002','12200001',
        '10100012','10186001','10186002','10186003',
        'EWM10-CU01','EWM10-CU02','EWM10-CU03']))
# if multiple rows per customer take DistributionChannel='10' as default

# Customer geo
spark.table("h2b_bdc_customer.customer.customer")
    .filter(col("Customer").isin([...same 10...]))

# Customer dunning (used in S4)
spark.table("h2b_bdc_customer.customer.customerdunning")
    .filter(col("Customer").isin([...same 10...]))
# default DunningLevel=0 if customer not found
```

---

## 5. Materials

| Material | Group | Seasonal profile | Price EUR/CS | Status |
|----------|-------|-----------------|--------------|--------|
| `TG11` | L001 | summer-peak | 8–18 | active |
| `TG12` | L001 | summer-peak | 8–18 | active |
| `FPP` | P001 | flat ±5% | 5–12 | active |
| `RTE` | P001 | flat ±5% | 5–12 | active |
| `CM-FL-V00` | L004 | mild-summer | 15–35 | active |
| `CM-MLFL-KM-VXX` | L004 | mild-summer | 15–35 | phase-out — 0 orders from `2025-06-01` |

**Seasonal weight per group:**
- L001 (TG11, TG12): +40% Jul–Aug · +20% Oct–Nov · −20% Jan–Feb
- L004 (CM-FL-V00, CM-MLFL-KM-VXX): half the L001 swing (+20% Jul–Aug · +10% Oct–Nov · −10% Jan–Feb)
- P001 (FPP, RTE): 5% of L001 swing (nearly flat)

**Variable prices:**
- +5% Jul–Aug for all groups except P001 (summer peak)
- +5–10% Sep–Dec for 2 designated SKUs only: 1 randomly picked from P001 + CM-MLFL-KM-VXX (drawn once at seed=42, reused every order)

**Fixed base prices** — drawn once with seed=42, reused for every order:

| Material(s) | Range |
|-------------|-------|
| TG11, TG12 | Uniform(8.0, 18.0) EUR |
| FPP, RTE | Uniform(5.0, 12.0) EUR |
| CM-FL-V00, CM-MLFL-KM-VXX | Uniform(15.0, 35.0) EUR |

---

## 6. Customers

| Customer | Tier | Order weight | Qty range (CS) | DSO | Collection rate |
|----------|------|-------------|----------------|-----|-----------------|
| `10100006` | A | 6× | 200–800 | Normal(30, 5) | 0.96–0.99 |
| `10100002` | A | 6× | 200–800 | Normal(30, 5) | 0.96–0.99 |
| `12200001` | A | 6× | 200–800 | Normal(30, 5) | 0.96–0.99 |
| `10100012` | B | 2× | 40–180 | Normal(45, 8) | 0.92–0.97 |
| `10186001` | B | 2× | 40–180 | Normal(45, 8) | 0.92–0.97 |
| `10186002` | B | 2× | 40–180 | Normal(45, 8) | 0.92–0.97 |
| `10186003` | B | 2× | 40–180 | Normal(45, 8) | 0.92–0.97 |
| `EWM10-CU01` | C | 1× | 2–15 | Normal(65, 20) | 0.80–0.92 |
| `EWM10-CU02` | C | 1× | 2–15 | Normal(65, 20) | 0.80–0.92 |
| `EWM10-CU03` | C | 1× | 2–15 | Normal(65, 20) | 0.80–0.92 |

> Tier map is hardcoded — do not rely on `CustomerABCClassification`.

**Revenue distribution:** A ≈ 91% (~30% each) · B ≈ 9% (~2.2% each) · C < 0.1% each

**Dunning level weights by tier:**

| Level | A-tier | B-tier | C-tier |
|-------|--------|--------|--------|
| 0 | 95% | 75% | 40% |
| 1 | 4% | 18% | 30% |
| 2 | 1% | 6% | 22% |
| 3 | 0% | 1% | 8% |

**Dunning adjustments on DSO:** Level 0 +0d · Level 1 +10d · Level 2 +25d · Level 3 +50d  
**Dunning multiplier on collection rate:** Level 0 ×1.00 · Level 1 ×0.97 · Level 2 ×0.93 · Level 3 ×0.85

---

## 7. Order quantity formula

```python
OrderQuantity = max(1, round(
    base_qty(tier)
    × seasonal_weight(month, year, material_group)
    × weather_mult(avg_temp_c, temp_anomaly_c, material_group)
    × lognormal(0, 0.10)          # noise — σ kept low to preserve weather signal
))

# base_qty
A-tier: Uniform(200, 800)
B-tier: Uniform(40,  180)
C-tier: Uniform(2,   15)

# seasonal_weight
base = 1.0
if material_group in ['L001','L004']:
    base += 0.40 if month in [7,8] else 0
    base += 0.20 if month in [10,11] else 0
    base -= 0.20 if month in [1,2] else 0
if material_group == 'L004':
    base = 1.0 + (base - 1.0) * 0.50     # L004 half the swing of L001
if material_group == 'P001':
    base = 1.0 + (base - 1.0) * 0.05     # P001 nearly flat
base *= (1 + 0.03 * (year - 2023))        # YoY growth

# weather_mult
# Lookup: h2b_bdc_weather.weather.weather for (customer.Country, customer.Region, year, month)
# Default (10.0°C, 0.0 anomaly) if region not found
mult = 1.0
if avg_temp_c > 22:
    mult += 0.50 * min((avg_temp_c - 22) / 5, 1.0)   # ramp +0% → +50% from 22°C to 27°C
if avg_temp_c > 25:
    mult += 0.30                                        # extra boost above 25°C
mult += 0.20 * max(0, temp_anomaly_c)                  # positive anomaly bonus only
mult = min(mult, 3.0)                                  # hard cap at 3×
if material_group == 'P001':
    mult = 1.0 + (mult - 1.0) * 0.30                  # powder less temp-sensitive
```

---

## 8. Weather heat waves (PoC signal design)

Two consecutive above-baseline summers give the model a strong, unambiguous weather signal.
The 2025 normal summer validates the model doesn't over-forecast in the absence of a heat wave.

| Year | Months | Regions | Anomaly | Order uplift |
|------|--------|---------|---------|--------------|
| 2023 | Jul–Aug | DE/BY + DE/NW | Uniform(2.5, 3.5)°C | ~×1.56 |
| 2024 | Jul–Aug | DE/BY + DE/NW | Uniform(3.0, 4.0)°C | ~×1.70 |
| 2025 | all | all | Normal(0, 0.6)°C | ~×1.00 (normal) |

---

## 9. Volume targets

| Table | Catalog path | ~Rows |
|-------|-------------|-------|
| `weather` | `h2b_bdc_weather.weather.weather` | 210 |
| `salesorder` | `h2b_dbx_salesorder.salesorder.salesorder` | ~5,000 |
| `salesorderitem` | `h2b_dbx_salesorder.salesorder.salesorderitem` | ~12,500 |
| `billingdocument` | `h2b_dbx_billingdocument.billingdocument.billingdocument` | ~3,500 |
| `billingdocumentitem` | `h2b_dbx_billingdocument.billingdocument.billingdocumentitem` | ~9,000 |
| `cashflow` | `h2b_dbx_cashflow.cashflow.cashflow` | ~3,150 |
| `cashflowforecast` | `h2b_dbx_cashflow.cashflow.cashflowforecast` | ~181 |

---

## 10. Prompts

Run each prompt in a separate Databricks notebook in the order shown.

---

### Prompt 1 — WeatherNOAA

```
Generate `generate_weather_table(reference_date='2023-01-01', months=42)`
returning {'WeatherNOAA': df}. Use seed=42.

Covers: 2023-01-01 to 2026-06-30 (36 months training + 6 months forecast).
Output: 5 regions × 42 months = 210 rows.

Columns: station_id, station_name, country, region, year, month,
         avg_temp_c, temp_anomaly_c, precipitation_mm, extreme_heat_flag

Regions — must match h2b_bdc_customer.customer.customer Country + Region exactly:
  NOAA_DE_BY  → country='DE', region='BY'
  NOAA_DE_NW  → country='DE', region='NW'
  NOAA_FR_IDF → country='FR', region='IDF'
  NOAA_ES_MD  → country='ES', region='MD'
  NOAA_US_CA  → country='US', region='CA'

avg_temp_c:
  Sinusoidal baseline per region from these monthly anchors (interpolate):
    DE_BY:  Jan=-2  Apr=9   Jul=19  Oct=11
    DE_NW:  Jan=2   Apr=10  Jul=20  Oct=12
    FR_IDF: Jan=4   Apr=12  Jul=23  Oct=14
    ES_MD:  Jan=6   Apr=15  Jul=29  Oct=17
    US_CA:  Jan=13  Apr=17  Jul=27  Oct=21
  Add warming trend: +0.04°C per month from reference_date
  Add monthly noise: Normal(0, 0.8)

temp_anomaly_c and extreme_heat_flag:
  Default: Normal(0, 0.6) / False for all months and regions.

  HEAT WAVE 2023 — DE_BY and DE_NW only, months 7 and 8:
    temp_anomaly_c = Uniform(2.5, 3.5)
    extreme_heat_flag = True if avg_temp_c > (region Jul baseline + 3.0)

  HEAT WAVE 2024 — DE_BY and DE_NW only, months 7 and 8 (stronger than 2023):
    temp_anomaly_c = Uniform(3.0, 4.0)
    extreme_heat_flag = True if avg_temp_c > (region Jul baseline + 3.0)

precipitation_mm:
  Winter (Dec–Feb): Uniform(40, 80)
  Spring (Mar–May): Uniform(50, 90)
  Summer (Jun–Aug): Uniform(20, 60)
  Autumn (Sep–Nov): Uniform(50, 100)
  10% chance of extreme month: multiply by 2.5

Join key to orders:
  customer.Country + customer.Region + year(CreationDate) + month(CreationDate)

Write to h2b_bdc_weather.weather.weather as Delta table.
```

---

### Prompt 2 — SalesOrder

```
Generate `generate_sales_order_tables(reference_date='2023-01-01', months=36)`
returning {'SalesOrder': df1, 'SalesOrderItem': df2}. Use seed=42.

READ AT THE START (do not mock these):

  products_df = spark.table("h2b_bdc_product.product.product") \
      .filter(col("Product").isin([
          'TG11','TG12','FPP','RTE','CM-FL-V00','CM-MLFL-KM-VXX']))
  # columns needed: Product, ProductGroup, ProductHierarchy,
  #                 CrossPlantStatus, ProductExternalID

  customer_area_df = spark.table("h2b_bdc_customer.customer.customersalesarea") \
      .filter(col("Customer").isin([
          '10100006','10100002','12200001','10100012',
          '10186001','10186002','10186003',
          'EWM10-CU01','EWM10-CU02','EWM10-CU03']))
  # columns: Customer, DistributionChannel, CustomerPaymentTerms,
  #          IncotermsClassification, Currency, CustomerGroup
  # if multiple rows per customer take DistributionChannel='10' as default

  customer_geo_df = spark.table("h2b_bdc_customer.customer.customer") \
      .filter(col("Customer").isin([...same 10...]))
  # columns: Customer, Country, Region

  weather_df = spark.table("h2b_bdc_weather.weather.weather")
  # Build lookup dict: {(country, region, year, month): (avg_temp_c, temp_anomaly_c)}

HARDCODED TIER MAP (use this — do not rely on CustomerABCClassification):
  A: ['10100006', '10100002', '12200001']
  B: ['10100012', '10186001', '10186002', '10186003']
  C: ['EWM10-CU01', 'EWM10-CU02', 'EWM10-CU03']

FIXED PRICE MAP — draw once with seed=42, reuse for every order:
  TG11, TG12:          Uniform(8.0,  18.0) EUR
  FPP,  RTE:           Uniform(5.0,  12.0) EUR
  CM-FL-V00:           Uniform(15.0, 35.0) EUR
  CM-MLFL-KM-VXX:      Uniform(15.0, 35.0) EUR

--- DataFrame 1: SalesOrder (~5,000 rows) ---
Columns: SalesOrder, SalesOrderType, CreationDate, SalesOrderDate, SoldToParty,
         SalesOrganization, DistributionChannel, OrganizationDivision, SalesGroup,
         SalesOffice, TransactionCurrency, TotalNetAmount, CustomerPaymentTerms,
         RequestedDeliveryDate, IncotermsClassification, DeliveryBlockReason,
         HeaderBillingBlockReason, OverallSDProcessStatus, OverallSDDocumentRejectionSts,
         FiscalYear, FiscalPeriod, BillingCompanyCode, PaymentMethod,
         AdditionalCustomerGroup1

Rules:
- SalesOrder: 'SO-000001' sequential
- SalesOrderType: 'OR' 90% / 'RE' 10%
- CreationDate: distribute across 2023-01-01 to 2025-12-31 weighted by
    1.0 + seasonal_offset(month) * (1 + 0.03*(year-2023))
  where seasonal_offset: +0.40 Jul-Aug · +0.20 Oct-Nov · -0.20 Jan-Feb · 0 otherwise
- SalesOrderDate: same as CreationDate
- SoldToParty: sample from 10 customers — A=6× B=2× C=1×
- SalesOrganization: 'VKORG1'
- DistributionChannel: from customer_area_df, default '10'
- OrganizationDivision: '01'
- SalesGroup / SalesOffice: 'SG01' / 'SO01'
- TransactionCurrency: from customer_area_df.Currency
- TotalNetAmount: back-filled from SalesOrderItem after generation
- CustomerPaymentTerms: from customer_area_df
- RequestedDeliveryDate: CreationDate + randint(5, 21) days
- IncotermsClassification: from customer_area_df
- DeliveryBlockReason: 3% = 'ZL' / rest = ''
- HeaderBillingBlockReason: 2% = 'Z1' / rest = ''
- OverallSDProcessStatus: 'A' 30% / 'B' 50% / 'C' 20%
- OverallSDDocumentRejectionSts: '' 90% / 'C' 10%
- FiscalYear: str(CreationDate.year)
- FiscalPeriod: str(CreationDate.month).zfill(2)
- BillingCompanyCode: 'CC01'
- PaymentMethod: 'T' 70% / 'D' 20% / 'C' 10%
- AdditionalCustomerGroup1: from customer_area_df.CustomerGroup, else 'KG01'

--- DataFrame 2: SalesOrderItem (~12,500 rows, 2–3 items per order) ---
Columns: SalesOrder, SalesOrderItem, Material, Product, Plant, MaterialGroup,
         ProductHierarchyNode, OrderQuantity, OrderQuantityUnit,
         RequestedQuantity, RequestedQuantityUnit, NetPriceAmount, NetAmount,
         TaxAmount, SalesOrderItemCategory, SalesDocumentRjcnReason,
         DeliveryStatus, BillingBlockStatus, DeliveryPriority,
         InternationalArticleNumber, SalesOrganization, DistributionChannel,
         SoldToParty, PayerParty

Rules:
- SalesOrderItem: '000010', '000020', '000030' per order
- Material = Product: sample 2–3 from the 6 materials without replacement per order
    CM-MLFL-KM-VXX: weight=0 for CreationDate >= 2025-06-01
    Material sampling weights by month:
      L001 (TG11, TG12): weight +40% in Jul-Aug
      L004 (CM-FL-V00, CM-MLFL-KM-VXX): weight +20% in Jul-Aug
      P001 (FPP, RTE): constant weight
- Plant: 'PLANT1' if DistributionChannel='10' / 'PLANT2' if '20'
- MaterialGroup: from products_df.ProductGroup
- ProductHierarchyNode: from products_df.ProductHierarchy
- OrderQuantity (integer, minimum 1): see section 7
- OrderQuantityUnit: 'CS'
- RequestedQuantity: max(OrderQuantity+1, round(OrderQuantity * Uniform(1.05, 1.15)))
- RequestedQuantityUnit: 'CS'
- NetPriceAmount: from FIXED PRICE MAP (with seasonal surges applied)
- NetAmount: OrderQuantity × NetPriceAmount
- TaxAmount: NetAmount × 0.19 if Currency='EUR' / 0.0 if 'USD'
- SalesOrderItemCategory: 'TAN' 95% / 'TANN' 5%
- SalesDocumentRjcnReason: 'Z1' if parent OverallSDDocumentRejectionSts='C' else ''
- DeliveryStatus: 'A' 30% / 'B' 40% / 'C' 30%
- BillingBlockStatus: '' 95% / 'B' 5%
- DeliveryPriority: '1' A-tier / '5' B-tier / '9' C-tier
- InternationalArticleNumber: from products_df.ProductExternalID
- SalesOrganization, DistributionChannel, SoldToParty: carry from order header
- PayerParty: same as SoldToParty

After all items are generated:
  back-fill SalesOrder.TotalNetAmount = SUM(SalesOrderItem.NetAmount) per SalesOrder

Write to:
  h2b_dbx_salesorder.salesorder.salesorder
  h2b_dbx_salesorder.salesorder.salesorderitem
```

---

### Prompt 3 — BillingDocument

```
Generate `generate_billing_tables(sales_order_df, sales_order_item_df)`
returning {'BillingDocument': df1, 'BillingDocumentItem': df2}. Use seed=42.

READ AT THE START:
  sales_order_df      = spark.table("h2b_dbx_salesorder.salesorder.salesorder")
  sales_order_item_df = spark.table("h2b_dbx_salesorder.salesorder.salesorderitem")

Source orders: SalesOrders where OverallSDProcessStatus IN ('B','C')
               AND OverallSDDocumentRejectionSts != 'C'

--- DataFrame 1: BillingDocument (~3,500 rows) ---
Columns: BillingDocument, BillingDocumentType, BillingDocumentCategory, CreationDate,
         BillingDocumentDate, SalesOrganization, TotalNetAmount, TotalTaxAmount,
         TransactionCurrency, CustomerPaymentTerms, PayerParty, SoldToParty,
         CompanyCode, IncotermsClassification, PaymentMethod,
         BillingDocumentIsCancelled, CancelledBillingDocument,
         AccountingPostingStatus, FiscalYear, FiscalPeriod

Rules:
- BillingDocument: 'BD-000001' sequential
- BillingDocumentType: 'F2' (standard) / 'S1' (cancellation pair)
- BillingDocumentCategory: 'M'
- BillingDocumentDate: SalesOrder.CreationDate + randint(3, 10) days
  This is the DSO clock start — must be strictly after CreationDate
- CreationDate: same as BillingDocumentDate
- TotalNetAmount: from SalesOrder.TotalNetAmount
- TotalTaxAmount: SUM(SalesOrderItem.TaxAmount) for items of that order
- All org/customer/currency fields: carry from SalesOrder
- CompanyCode: 'CC01'
- FiscalYear: str(BillingDocumentDate.year)
- FiscalPeriod: str(BillingDocumentDate.month).zfill(2)

Cancellation logic — applied to 10% of source orders:
  Original document:
    BillingDocumentIsCancelled = True
    AccountingPostingStatus    = 'A'
    CancelledBillingDocument   = None
  Paired cancellation document (new sequential BD number):
    BillingDocumentType        = 'S1'
    BillingDocumentIsCancelled = False
    CancelledBillingDocument   = original BD number
    AccountingPostingStatus    = 'C'
    TotalNetAmount             = -1 × original amount
    TotalTaxAmount             = -1 × original tax
Remaining 90%:
    BillingDocumentIsCancelled = False
    AccountingPostingStatus    = 'C'
    CancelledBillingDocument   = None

--- DataFrame 2: BillingDocumentItem (~9,000 rows) ---
Columns: BillingDocument, BillingDocumentItem, Material, Product, Plant,
         MaterialGroup, ProductHierarchyNode, NetAmount, GrossAmount, TaxAmount,
         EligibleAmountForCashDiscount, SalesDocumentItemCategory, SalesOrganization

Rules:
- Join SalesOrderItem to BillingDocument via SalesOrder key
- BillingDocumentItem: '000010', '000020', '000030' per document
- NetAmount, TaxAmount: carry from SalesOrderItem
- GrossAmount: NetAmount + TaxAmount
- EligibleAmountForCashDiscount: NetAmount
- All product/plant/org fields: carry from SalesOrderItem
- Cancellation documents (BillingDocumentType='S1'):
    NetAmount, GrossAmount, TaxAmount = negated values of originals

Write to:
  h2b_dbx_billingdocument.billingdocument.billingdocument
  h2b_dbx_billingdocument.billingdocument.billingdocumentitem
```

---

### Prompt 4 — CashFlow

```
Generate `generate_cashflow_tables(billing_df, forecast_months=6)`
returning {'CashFlow': df1, 'CashFlowForecast': df2}. Use seed=42.

READ AT THE START:
  billing_df = spark.table("h2b_dbx_billingdocument.billingdocument.billingdocument")

  customer_dunning_df = spark.table("h2b_bdc_customer.customer.customerdunning") \
      .filter(col("Customer").isin([
          '10100006','10100002','12200001','10100012',
          '10186001','10186002','10186003',
          'EWM10-CU01','EWM10-CU02','EWM10-CU03']))
  # columns: Customer, DunningLevel (default 0 if not found)

TIER MAP (hardcoded — do not rely on CustomerABCClassification):
  A: ['10100006', '10100002', '12200001']
  B: ['10100012', '10186001', '10186002', '10186003']
  C: ['EWM10-CU01', 'EWM10-CU02', 'EWM10-CU03']

--- DataFrame 1: CashFlow (~3,150 rows) ---
Columns: CashFlowID, CshFlwValdtyStrtDteTmeVal, CompanyCode, TransactionDate,
         PostingDate, TransactionCurrency, AmountInTransactionCurrency,
         CompanyCodeCurrency, AmountInCompanyCodeCurrency, GlobalCurrency,
         AmountInGlobalCurrency, BankAccountInternalID

Source: BillingDocument where BillingDocumentIsCancelled=False AND BillingDocumentType='F2'

PostingDate derivation:
  1. tier      = TIER MAP lookup for BillingDocument.PayerParty
  2. dso_level = customer_dunning_df.DunningLevel for this customer (default 0)
  3. base_dso  = max(1, round(Normal(μ, σ))):
                   A → Normal(30,  5)
                   B → Normal(45,  8)
                   C → Normal(65, 20)
  4. dunning_adj:  Level 0 → +0d  Level 1 → +10d  Level 2 → +25d  Level 3 → +50d
  5. noise         = round(Normal(0, 3))
  6. PostingDate   = BillingDocumentDate + min(base_dso + dunning_adj + noise, 180) days
                     minimum 1 day after BillingDocumentDate

AmountInTransactionCurrency:
  base_rate:    A → Uniform(0.96, 0.99)
                B → Uniform(0.92, 0.97)
                C → Uniform(0.80, 0.92)
  dunning_mult: Level 0 → ×1.00  Level 1 → ×0.97  Level 2 → ×0.93  Level 3 → ×0.85
  Amount = BillingDocument.TotalNetAmount × base_rate × dunning_mult

Currency fields:
  TransactionCurrency:          from BillingDocument.TransactionCurrency
  CompanyCodeCurrency:          'EUR'
  AmountInCompanyCodeCurrency:  EUR → same as Amount / USD → Amount × 0.92
  GlobalCurrency:               'EUR'
  AmountInGlobalCurrency:       same as AmountInCompanyCodeCurrency
  BankAccountInternalID:        'BANK001' (EUR) / 'BANK002' (USD)

Other fields:
  CashFlowID:                   'CF-000001' sequential
  CshFlwValdtyStrtDteTmeVal:    int(PostingDate.timestamp()) × 10_000_000 as Decimal(21,7)
  TransactionDate:              same as PostingDate
  CompanyCode:                  'CC01'

--- DataFrame 2: CashFlowForecast (~181 rows — 2026-01-01 to 2026-06-30) ---
Columns: same as CashFlow minus BankAccountInternalID

Generation — naive seasonal baseline (intentionally imperfect, ML model must beat this):
  Step 1: weekly_avg = mean(CashFlow.AmountInCompanyCodeCurrency)
          grouped by ISO week-of-year, over PostingDates in 2023–2025
  Step 2: overall_mean = mean(weekly_avg values)
  Step 3: for each day d in 2026-01-01 to 2026-06-30:
            seasonal_index = weekly_avg[iso_week(d)] / overall_mean
            forecast = (overall_mean / 7) × seasonal_index × 1.04 × Uniform(0.92, 1.08)
  Step 4:
    All currency amounts = forecast (EUR only)
    CashFlowID:                'FCF-000001' sequential
    CshFlwValdtyStrtDteTmeVal: same encoding
    TransactionDate = PostingDate, CompanyCode = 'CC01', TransactionCurrency = 'EUR'

Write to:
  h2b_dbx_cashflow.cashflow.cashflow
  h2b_dbx_cashflow.cashflow.cashflowforecast
```

---

### Final assembly prompt

```
Write `main()` that orchestrates all 4 generation functions in strict order.
Use seed=42 everywhere.

STEP 1 — create catalogs and schemas as needed:
  spark.sql("CREATE CATALOG IF NOT EXISTS h2b_bdc_weather")
  spark.sql("CREATE SCHEMA  IF NOT EXISTS h2b_bdc_weather.weather")
  spark.sql("CREATE CATALOG IF NOT EXISTS h2b_dbx_salesorder")
  spark.sql("CREATE SCHEMA  IF NOT EXISTS h2b_dbx_salesorder.salesorder")
  spark.sql("CREATE CATALOG IF NOT EXISTS h2b_dbx_billingdocument")
  spark.sql("CREATE SCHEMA  IF NOT EXISTS h2b_dbx_billingdocument.billingdocument")
  spark.sql("CREATE CATALOG IF NOT EXISTS h2b_dbx_cashflow")
  spark.sql("CREATE SCHEMA  IF NOT EXISTS h2b_dbx_cashflow.cashflow")

STEP 2 — call generation functions in strict order:
  weather   = generate_weather_table(reference_date='2023-01-01', months=42)
  orders    = generate_sales_order_tables(reference_date='2023-01-01', months=36)
  billing   = generate_billing_tables(orders['SalesOrder'], orders['SalesOrderItem'])
  cashflows = generate_cashflow_tables(billing['BillingDocument'], forecast_months=6)

STEP 3 — confirm each table exists and print row count + 3-row sample:
  TABLE_MAP = {
      'h2b_bdc_weather.weather.weather':                              weather['WeatherNOAA'],
      'h2b_dbx_salesorder.salesorder.salesorder':                    orders['SalesOrder'],
      'h2b_dbx_salesorder.salesorder.salesorderitem':                orders['SalesOrderItem'],
      'h2b_dbx_billingdocument.billingdocument.billingdocument':     billing['BillingDocument'],
      'h2b_dbx_billingdocument.billingdocument.billingdocumentitem': billing['BillingDocumentItem'],
      'h2b_dbx_cashflow.cashflow.cashflow':                          cashflows['CashFlow'],
      'h2b_dbx_cashflow.cashflow.cashflowforecast':                  cashflows['CashFlowForecast'],
  }

STEP 4 — integrity validation (print all results):

  a. SalesOrder.SoldToParty not in known 10 customers                  → expect 0
  b. SalesOrderItem.Material not in
     ['TG11','TG12','FPP','RTE','CM-FL-V00','CM-MLFL-KM-VXX']         → expect 0
  c. SalesOrderItem with Material='CM-MLFL-KM-VXX'
     AND parent CreationDate >= '2025-06-01'                           → expect 0
  d. BillingDocument.PayerParty not in known 10 customers              → expect 0
  e. CashFlow.PostingDate < BillingDocument.BillingDocumentDate        → expect 0
  f. WeatherNOAA: mean temp_anomaly_c where year=2023, month IN (7,8),
     region IN ('BY','NW')                                             → expect > 2.0°C
  g. WeatherNOAA: mean temp_anomaly_c where year=2024, month IN (7,8),
     region IN ('BY','NW')                                             → expect > 2.5°C
  h. SalesOrderItem: mean(OrderQuantity) for Jul-Aug 2023 vs Jul-Aug 2025
     (L001+L004 materials only)                                        → expect 2023 ~40-70% higher
```
