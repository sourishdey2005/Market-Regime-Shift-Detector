# 🎯 Market Regime Shift Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

&gt; **Professional-grade market regime detection using Hidden Markov Models and multi-factor macro analysis**

This project implements a sophisticated quantitative finance tool used by hedge funds to identify when markets shift between Bull, Bear, and Sideways regimes. Unlike simple technical indicators, this uses machine learning (Gaussian HMM) to detect latent states in market data.

![Demo](https://img.shields.io/badge/Live-Demo-green)

## 🚀 Features

- **Real-time Data**: Fetches live data from Yahoo Finance (SPY, VIX, Treasury yields, Commodities, Gold, USD)
- **AI-Powered Detection**: Gaussian Hidden Markov Model with 12 macro/technical features
- **Regime Classification**: Automatically labels Bull 🟢, Bear 🔴, and Sideways 🟡 markets
- **Confidence Scoring**: Statistical confidence based on model likelihood
- **Interactive UI**: Beautiful Streamlit dashboard with Plotly visualizations
- **Regime Timeline**: Visual history of regime changes with exact dates
- **Feature Importance**: Shows which indicators drive current regime detection

## 📊 Data Sources

| Ticker | Description | Usage |
|--------|-------------|-------|
| SPY | S&P 500 ETF | Primary market proxy |
| ^VIX | Volatility Index | Fear gauge, regime volatility |
| ^TNX | 10-Year Treasury Yield | Interest rate environment |
| DBC | Commodities Index | Inflation proxy |
| GLD | Gold ETF | Safe haven flows |
| UUP | US Dollar Index | Currency strength |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/market-regime-detector.git
cd market-regime-detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py    