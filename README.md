US Price Index Time Series Analysis

A comprehensive time series project exploring the dynamics of the US Price Index (monthly data). The analysis covers statistical time-series modeling, long-memory diagnostics, and deep learning forecasting using LSTM and GRU.

📌 Project Overview

This project performs an end-to-end study of price index behavior, focusing on:

Detecting changing variability using rolling statistics.

Assessing stationarity, seasonality, and long-term memory.

Applying both classical time-series models (SARIMA, ARFIMA) and deep learning models (LSTM/GRU).

Conducting a rigorous diagnostic check of model assumptions.

Comparing forecast performance across models.

The dataset consists of monthly US Price Index values.

🔍 Exploratory Data Analysis (EDA)
✅ 1. Variability Analysis

Computed rolling mean and rolling standard deviation.

Identified potential periods of changing volatility in the series.

✅ 2. Stationarity Check

Tools used:

ADF test

KPSS test

Visual inspection of rolling statistics

Finding:
The series showed non-stationary behavior.

✅ 3. Seasonality Detection

Seasonal patterns observed through:

Seasonal decomposition

Monthly autocorrelation patterns

Spectral analysis

✅ 4. Long-Term Memory

Two methods were applied:

• Hurst Exponent

Estimated > 0.5, indicating persistent long-memory behavior.

• GPH Log-Periodogram Regression

Confirmed fractional differencing requirement.

⚙️ Data Transformation
✅ Seasonal Differencing

Applied D = 1 for monthly seasonality (lag 12).

✅ Fractional Differencing for Long Memory

Estimated fractional differencing parameter d using:

GPH estimator

Hurst-based inference

📈 Modeling Approach
1. ARFIMA (Fractionally Integrated Model)
✅ Specification:

Fractional differencing d applied.

Seasonal AR and MA terms included for monthly seasonality.

The model successfully captured long-memory + seasonality structure.

✅ Diagnostics:

Residual ACF + PACF

Ljung-Box test

No significant autocorrelation left in residuals

Reasonable information criteria values

2. SARIMA (Short Memory Model)
✅ Model Used:
SARIMA(p, d=1, q)(P, D=1, Q)m


d = 1 for trend

D = 1 for seasonal structure

Seasonal periodicity m = 12 (monthly)

✅ SARIMA Assumptions Verified:

Residuals ≈ white noise

No serial correlation (ACF/PACF of residuals)

Homoscedasticity checked via:

ACF of squared residuals

ARCH test

Finding:
SARIMA handled seasonality but struggled due to long-memory behavior.

🤖 Deep Learning Models

The series showed persistent, non-linear long-term dependencies, making it suitable for neural sequence models.

✅ Models Implemented:

LSTM

GRU

✅ Why DL Was Considered?

Non-linear patterns

Long-memory and extended dependencies

SARIMA/ARFIMA could not fully capture complex temporal structure

✅ Results:

LSTM performed best, achieving strong predictive accuracy

Captured both short-term noise and long-range structure

GRU performed reasonably but slightly weaker than LSTM

📊 Forecast Comparison

Classical models (SARIMA, ARFIMA) provided interpretable structure.

LSTM delivered superior predictive accuracy, especially for longer horizons.

📁 Repository Structure
├── data/
│   └── us_price_index.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ARFIMA_Modeling.ipynb
│   ├── 03_SARIMA_Modeling.ipynb
│   └── 04_LSTM_GRU_Forecasting.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models_arima.py
│   ├── models_lstm.py
│   └── evaluation.py
└── README.md

🧪 Methods & Libraries Used
Statistical Modeling

statsmodels

arch

pmdarima

Deep Learning

tensorflow / keras

Sequence modeling layers (LSTM, GRU)

General

pandas

numpy

matplotlib

seaborn

✅ Conclusions

The US Price Index series displays trend, seasonality, and long-memory persistence.

ARFIMA is well-suited for long-memory modeling but less effective for complex non-linearities.

SARIMA works for short-memory seasonal components but cannot capture long-term persistence well.

LSTM outperformed all statistical models in forecasting accuracy.

🚀 Future Work

Try ARFIMA + GARCH to capture volatility clustering

Explore Transformer-based forecasting models

Compare with Prophet, ETS, and N-BEATS

Hyperparameter optimization for LSTM/GRU
