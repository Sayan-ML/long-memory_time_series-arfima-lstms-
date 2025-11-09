# Time Series Analysis of US Price Index

A comprehensive time series analysis project examining monthly US Price Index data using classical statistical methods and modern deep learning approaches.

## 📊 Project Overview

This project performs an in-depth analysis of US Price Index monthly data, exploring temporal patterns, stationarity, seasonality, and long-term memory characteristics. The analysis progresses from exploratory data analysis through traditional ARFIMA/SARIMA modeling to advanced LSTM/GRU neural network architectures.

## 🎯 Objectives

- Analyze temporal characteristics of US Price Index data
- Identify and quantify seasonality patterns
- Assess stationarity and long-term memory properties
- Compare classical time series models (ARFIMA, SARIMA) with deep learning approaches (LSTM, GRU)
- Build robust forecasting models accounting for non-linearity and long-term dependencies

## 📁 Dataset

- **Source**: US Price Index Monthly Data
- **Type**: Time Series
- **Characteristics**: Long-term memory, seasonal patterns, non-linear behavior

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)

#### Variability Analysis
- **Rolling Standard Deviation**: Assessed changing variability over time to detect heteroscedasticity and volatility clustering

#### Seasonality Detection
- Comprehensive visual and statistical analysis of seasonal patterns
- Seasonal decomposition to isolate trend, seasonal, and residual components
- Identification of periodic cycles in the data

### 2. Stationarity Testing

Evaluated the stationarity properties of the time series:
- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Visual inspection of rolling statistics

### 3. Long-Term Memory Analysis

#### Hurst Coefficient
- Calculated Hurst exponent to measure long-term memory
- **H > 0.5**: Indicates persistent long-term memory (positive autocorrelation)
- **H = 0.5**: Random walk behavior
- **H < 0.5**: Anti-persistent behavior

#### GPH Filter (Geweke-Porter-Hudak)
- Semi-parametric estimation of the fractional differencing parameter *d*
- Validated presence of long-term dependencies in the series

### 4. Data Preprocessing

#### Seasonal Differencing
- Applied seasonal differencing to remove seasonal patterns
- Prepared data for ARFIMA and SARIMA modeling

### 5. Classical Time Series Models

#### ARFIMA Model (AutoRegressive Fractionally Integrated Moving Average)
- **Model Structure**: ARFIMA(p, d, q) × (P, D, Q)s
- Incorporated seasonal AR and MA terms
- Captured long-term memory through fractional differencing parameter *d*
- Suitable for data with long-range dependence

#### SARIMA Model (Seasonal ARIMA)
- **Model Structure**: SARIMA(p, d, q) × (P, D, Q)s
- **Differencing**: d=1 (non-seasonal), D=1 (seasonal)
- Short-term memory alternative to ARFIMA
- Appropriate for stationary data with seasonal patterns

### 6. Model Diagnostics

#### Residual Analysis
- **Normality**: Q-Q plots, Shapiro-Wilk test
- **Autocorrelation**: ACF/PACF plots of residuals
- **White Noise**: Ljung-Box test

#### Heteroscedasticity Testing
- **Visual Analysis**: ACF of squared residuals to detect volatility clustering
- **ARCH Test**: Engle's test for autoregressive conditional heteroscedasticity
- Assessed whether variance is constant over time

### 7. Deep Learning Models

Given the identified characteristics (long-term memory, non-linearity), neural network architectures were employed:

#### LSTM (Long Short-Term Memory)
- **Architecture**: Specialized RNN with memory cells
- **Advantages**: 
  - Captures long-term dependencies effectively
  - Handles non-linear patterns
  - Mitigates vanishing gradient problem
- **Performance**: Achieved superior results due to long-term memory capability

#### GRU (Gated Recurrent Unit)
- **Architecture**: Simplified variant of LSTM
- **Advantages**:
  - Fewer parameters than LSTM
  - Faster training
  - Comparable performance for many time series tasks

## 📈 Results

### Model Comparison

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **ARFIMA** | Captures long-term memory, handles fractional integration | Assumes linearity, complex parameter estimation |
| **SARIMA** | Well-suited for seasonal data, interpretable | Short-term memory only, linear assumptions |
| **LSTM** | ✅ **Best Performance** - Handles non-linearity and long-term dependencies | Requires more data, less interpretable |
| **GRU** | Good balance of performance and efficiency | Slightly less powerful than LSTM for complex patterns |

### Key Findings

1. **Long-Term Memory**: Hurst coefficient and GPH filter confirmed significant long-term dependence
2. **Seasonality**: Strong seasonal patterns identified and modeled
3. **Non-Linearity**: Evidence of non-linear dynamics in the data
4. **Heteroscedasticity**: ARCH effects detected, indicating time-varying volatility
5. **LSTM Performance**: Superior forecasting accuracy due to ability to capture long-term memory and non-linear patterns

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Statistical Libraries**: 
  - `statsmodels`: ARFIMA, SARIMA, statistical tests
  - `arch`: ARCH/GARCH testing
  - `scipy`: Statistical analysis
- **Deep Learning**: 
  - `TensorFlow`/`Keras` or `PyTorch`: LSTM/GRU implementation
- **Data Processing**: 
  - `pandas`: Data manipulation
  - `numpy`: Numerical computations
- **Visualization**: 
  - `matplotlib`: Plotting
  - `seaborn`: Statistical visualizations

## 📊 Project Structure

```
├── data/
│   └── us_price_index.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_stationarity_tests.ipynb
│   ├── 03_long_term_memory.ipynb
│   ├── 04_arfima_model.ipynb
│   ├── 05_sarima_model.ipynb
│   └── 06_lstm_gru_models.ipynb
├── models/
│   ├── arfima_model.pkl
│   ├── sarima_model.pkl
│   ├── lstm_model.h5
│   └── gru_model.h5
├── results/
│   ├── plots/
│   └── evaluation_metrics.csv
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/us-price-index-analysis.git
cd us-price-index-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
# Example: Running LSTM model
from models.lstm_model import LSTMForecaster

# Load and preprocess data
forecaster = LSTMForecaster()
forecaster.load_data('data/us_price_index.csv')
forecaster.preprocess()

# Train model
forecaster.train(epochs=100, batch_size=32)

# Make predictions
predictions = forecaster.predict(steps=12)
```

## 📊 Key Insights

1. **Long-Term Memory Dominates**: The Hurst coefficient indicated strong long-term dependencies, making ARFIMA and LSTM more suitable than simple SARIMA
2. **Seasonality Matters**: Seasonal differencing and seasonal terms significantly improved model performance
3. **Non-Linear Patterns**: LSTM outperformed linear models, confirming non-linear dynamics
4. **Volatility Clustering**: ARCH effects suggest periods of high and low volatility cluster together

## 🔮 Future Work

- Implement hybrid ARFIMA-LSTM models
- Explore GARCH models for volatility forecasting
- Add attention mechanisms to LSTM architecture
- Perform multivariate analysis with economic indicators
- Deploy forecasting model as REST API

## 📚 References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*
- Geweke, J., & Porter-Hudak, S. (1983). The estimation and application of long memory time series models
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
- Hurst, H. E. (1951). Long-term storage capacity of reservoirs

## 👤 Author

Your Name - [GitHub Profile](https://github.com/Sayan-ML)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/us-price-index-analysis/issues).

---

**Note**: This project demonstrates advanced time series analysis techniques combining classical econometric methods with modern deep learning approaches.
