# 📦 Fresh Fruit Import Elasticities – Calibration Dataset  
**Dynamic SARIMAX, ARDL, and Armington Models for U.S. Import Demand**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.11-3776AB.svg?logo=python)
![Data Frequency](https://img.shields.io/badge/Frequency-Monthly-orange)
![Status](https://img.shields.io/badge/Status-Research%20Dataset-success)
![Last Updated](https://img.shields.io/badge/Updated-2025-lightgrey)

This repository provides the **calibration dataset** used to estimate the econometric models in:

**_“U.S. Fresh Fruit Import Elasticities: Dynamic Estimates and Insights on Tariff and Export Strategies”_**

The dataset consolidates monthly import quantities, CIF values, unit prices, climate variables, and derived indicators for major exporters supplying fresh fruit to the U.S.

---

## 📁 Dataset Location

**File:**  
`Base_de_Calibracion_q_en_ton_por_usd_utf8.csv`  

This CSV is the **core input** used to estimate the dynamic models (SARIMAX, ARDL, and structural Armington-logit) featured in the manuscript.

---

## 🌎 Dataset Overview

- **Time span:** January 2010 – February 2025  
- **Frequency:** Monthly  
- **Exporting countries:**  
  - Brazil  
  - Canada  
  - Chile  
  - Costa Rica  
  - Guatemala  
  - Mexico  
  - Peru  

The dataset integrates:

- U.S. **import quantities**  
- **CIF import values** (USD)  
- **Unit prices**  
- **Climate variables** (monthly average temperature, TAVG)  
- **Relative-price indices**  
- **Competitor price index**  
- Derived variables for econometric modeling (logs, shares, etc.)

---

## 📊 Column Structure

| Column | Description |
|--------|-------------|
| `date` | Year–Month (YYYY-MM) |
| `country` | Exporting country |
| `quantity_ton` | Import quantity (tons) |
| `value_usd` | CIF value (USD) |
| `unit_value` | CIF unit price (USD/ton) |
| `log_q` | Log of quantity |
| `log_p` | Log of CIF price |
| `tavg` | Average temperature at origin |
| `relative_price` | Relative-price index (M2 specification for Mexico) |
| `competitor_index` | Aggregate competitor-price measure |
| `share` | Import share (for Armington-logit models) |

---

## 🧮 Econometric Uses

This dataset supports the estimation of:

### ✔️ Own-price elasticities  
via SARIMAX and ARDL specifications.

### ✔️ Cross-price elasticities  
via competitor price index.

### ✔️ Structural Armington elasticities  
via logit share models.

### ✔️ Policy simulations  
including tariff shocks, climate shocks, and cost changes.

---

## 📘 Example: Running a SARIMAX Model in Python

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("Base_de_Calibracion_q_en_ton_por_usd_utf8.csv")

model = SARIMAX(
    df["log_q"],
    exog=df[["log_p", "tavg"]],
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 12)
)

results = model.fit()
print(results.summary())
