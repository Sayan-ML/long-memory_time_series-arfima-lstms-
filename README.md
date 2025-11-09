US Price Index Time Series Analysis










A comprehensive time series project exploring the dynamics of the US Price Index (monthly data). The project includes statistical modeling, long-memory diagnostics, and deep learning forecasting using LSTM and GRU.

📌 Project Overview

This project performs an end-to-end study of US price index behavior, focusing on:

Detecting changing variability using rolling statistics

Assessing stationarity, seasonality, and long-term memory

Applying both classical time-series models (SARIMA, ARFIMA) and deep learning (LSTM/GRU)

Conducting rigorous diagnostic checks of model assumptions

Comparing performance across all models

Dataset: Monthly US Price Index values.

🔍 Exploratory Data Analysis (EDA)
✅ 1. Variability Analysis

Computed rolling mean and rolling standard deviation

Identified periods of changing volatility

✅ 2. Stationarity Check

Methods used:

ADF Test

KPSS Test

Rolling statistics

Finding: The series is non-stationary.

✅ 3. Seasonality Detection

Seasonality was identified via:

Seasonal decomposition

Monthly ACF/PACF structure

Spectral analysis

✅ 4. Long-Term Memory Detection

Two techniques were applied:

• Hurst Exponent

Result: H > 0.5 indicating persistent long-memory behavior

• GPH (Geweke–Porter–Hudak) Log-Periodogram Regression

Result: Suggests the need for fractional differencing (d)

⚙️ Data Transformation
✅ Seasonal Differencing

Applied D = 1 for monthly seasonality (lag 12)

✅ Fractional Differencing

Estimated fractional differencing parameter d using:

GPH estimator

Hurst-based inference

📈 Modeling Approach
1. ARFIMA (Fractionally Integrated Model)
✅ Specification

Applied fractional differencing (d)

Included seasonal AR and MA terms

Captured long-memory + seasonality effectively

✅ Diagnostics

Residual ACF/PACF

Ljung-Box test

No significant residual autocorrelation

2. SARIMA (Short Memory Model)
✅ Model Used
SARIMA(p, d=1, q)(P, D=1, Q)m


d = 1 for trend

D = 1 for seasonality

m = 12 for monthly periodicity

✅ Assumptions Checked

Residuals ≈ white noise

No serial correlation

Homoscedasticity validated using:

ACF of squared residuals

ARCH test

Finding:
SARIMA handled seasonality but struggled with long-memory behavior.

🤖 Deep Learning Models

The dataset exhibited persistent, nonlinear long-memory patterns, making it suitable for deep sequence models.

✅ Models Implemented

LSTM

GRU

✅ Why DL Was Used

Captures non-linear dynamics

Memorizes long-term dependencies

Overcomes limitations of linear statistical models

✅ Results

LSTM outperformed all models

Captured both short-term variations and long-range structure

GRU performed reasonably but weaker than LSTM

📊 Forecast Comparison
Model	Handles Seasonality	Handles Long Memory	Handles Non-Linearity	Performance
SARIMA	✅	❌	❌	Moderate
ARFIMA	✅	✅	❌	Good
LSTM	✅	✅	✅	⭐ Best
GRU	✅	✅	✅	Good
📁 Repository Structure
├── data/
│   └── us_price_index.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ARFIMA_Modeling.ipynb
│   ├── 03_SARIMA_Modeling.ipynb
│   └── 04_LSTM_GRU_Forecasting.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models_arima.py
│   ├── models_lstm.py
│   └── evaluation.py
│
└── README.md

🧪 Methods & Libraries Used
Statistical Modeling

statsmodels

pmdarima

arch

Deep Learning

tensorflow

keras

General

pandas

numpy

matplotlib

seaborn

✅ Conclusions

The US Price Index shows trend, seasonality, and long-memory persistence

ARFIMA effectively captured long-memory structure

SARIMA handled seasonal dynamics but not long-range memory

LSTM achieved the most accurate forecasts

🚀 Future Work

Combine ARFIMA + GARCH to model long memory + volatility

Try Transformer-based forecasting

Compare against Prophet, ETS, N-BEATS

Hyperparameter tuning for LSTM/GRU
