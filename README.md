
# Credit-Informed Equity Strategy Using CDS and the Merton Model

This project implements a cross-asset equity trading strategy informed by credit market signals using Credit Default Swap (CDS) spreads. We use a robust version of the Merton (1974) model to detect mispricing in CDS spreads and translate those signals into equity trades. The system includes data ingestion, modeling, signal generation, backtesting, and performance visualization.

---

## Overview

- **Goal:** Exploit mispricing between equity and CDS markets to generate alpha
- **Framework:** Merton model + firm fundamentals + volatility adjustment
- **Strategy:** Long equity when CDS is overpriced; short equity when underpriced
- **Backtested On:** 8 U.S. large-cap stocks from 2019–2024

---

## Theoretical Background

This project is inspired by the classic work:

> Merton, R. C. (1974). *On the pricing of corporate debt: The risk structure of interest rates*. Journal of Finance, 29(2), 449–470.

It also builds on empirical insights from:

> Buus, I. & Nielsen, C. R. J. (2009). *The Relationship Between Equity Prices and Credit Default Swap Spreads*. Copenhagen Business School.

---

## How It Works

### 1. **Data Loader (`data_loader.py`)**
- Downloads adjusted equity prices via `yfinance`
- Loads daily risk-free rate and computes excess returns
- Downloads firm fundamentals (e.g., debt, ROA, interest coverage)

### 2. **Strategy Logic (`main.py`)**
- Calculates model-implied CDS spreads using an enhanced Merton model
- Detects mispricing between model and market CDS spreads
- Converts mispricing into long/short equity signals
- Runs backtests with realistic assumptions
- Plots performance against buy-and-hold and risk-free benchmarks

---

## Signal Definition

- **Buy Signal:** Market CDS spread > model-implied spread (after volatility adjustment)
- **Sell Signal:** Market CDS spread < model-implied spread
- **No Trade:** Mispricing within threshold

Additional filters:
- Volatility-scaled threshold
- Z-score filter and CDS momentum confirmation

---

## Installation

```bash
git clone https://github.com/jennlangley/mf728-project
cd mf728-project
pip install -r requirements.txt
```

---

## Requirements

- Python 3.9+
- pandas, numpy, yfinance, matplotlib, scipy

---

## References

- Merton, R. C. (1974). *On the pricing of corporate debt*. Journal of Finance.
- Buus, I., & Nielsen, C. R. J. (2009). *The Relationship Between Equity Prices and CDS Spreads*. CBS Thesis.

---

## Authors

- [Jennifer Langley](https://github.com/jennlangley)
- [Alexa Whitesell](https://github.com/alexawhitesell)

---
