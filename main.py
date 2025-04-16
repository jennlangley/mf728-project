import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

from data.main import (
    download_data,
    load_risk_free_rate,
    merge_data_with_rf,
    calculate_excess_returns,
    get_firm_fundamentals
)

# ------------------------
# Constants
# ------------------------
TICKERS = ['JPM','XOM', 'GM', 'T', 'PFE', 'IBM', 'GS', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
THRESHOLD = 0.005  # ~50bps in decimal form
T = 1  # 1 year time horizon

# ------------------------
# File Paths
# ------------------------
FF_CSV_PATH = "./data/ff_factors_daily.csv"
CDS_CSV_PATH = "./data/cds.csv"
FIRM_FUNDAMENTALS_PATH = "./data/firm_fundamentals.csv"
EXCESS_RETURNS_PATH = "./data/excess_returns.csv"

# ------------------------
# Merton Model Function
# ------------------------
def merton_model(E, sigma_E, D, r, T):
    V = E + D
    sigma_V = sigma_E * E / V
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    default_prob = norm.cdf(-d2)
    model_spread = -np.log(1 - default_prob) / T

    return {
        "Asset Value": V,
        "Asset Volatility": sigma_V,
        "Distance to Default": d2,
        "Default Probability": default_prob,
        "Model-Implied Spread": model_spread
    }

# ------------------------
# Mispricing Detector
# ------------------------
def detect_mispricing(model_spread, market_spread, threshold):
    diff = market_spread - model_spread
    if diff > threshold:
        return "Short CDS"
    elif diff < -threshold:
        return "Long CDS"
    else:
        return "No Trade"

# ------------------------
# Backtesting
# ------------------------
def generate_position_series(signal_df):
    signal_map = {"Short CDS": -1, "Long CDS": 1, "No Trade": 0}
    return signal_df["Signal"].map(signal_map)

def backtest_strategy(spread_series, position_series):
    spread_returns = spread_series.pct_change().shift(-1)
    strat_returns = spread_returns * position_series.shift(1)
    cumulative_returns = (1 + strat_returns).cumprod()
    return cumulative_returns, strat_returns

# ------------------------
# Main Analysis
# ------------------------

data = download_data(TICKERS, start_date=START_DATE, end_date=END_DATE)
risk_free_data = load_risk_free_rate(FF_CSV_PATH)
merged_data = merge_data_with_rf(data, risk_free_data)
excess_returns = calculate_excess_returns(merged_data)
excess_returns.to_csv(EXCESS_RETURNS_PATH)

fundamentals_df = get_firm_fundamentals(TICKERS, FIRM_FUNDAMENTALS_PATH)
debt_dict = fundamentals_df["Debt"].to_dict()
market_cap_dict = fundamentals_df["MarketCap"].to_dict()

cds_df = pd.read_csv(CDS_CSV_PATH, parse_dates=["Date"])

strategy_log = []

for ticker in TICKERS:
    print(f"Processing: {ticker}")
    try:
        returns = excess_returns[ticker].dropna()
        sigma_E = returns.std() * np.sqrt(252)
        date = returns.index[-1]
        r = risk_free_data.loc[:date].iloc[-1]

        E = market_cap_dict.get(ticker, None)
        D = debt_dict.get(ticker, 0)
        if E is None:
            print(f"Missing market cap for {ticker}, skipping.")
            continue

        merton = merton_model(E, sigma_E, D, r, T)
        sub_df = cds_df[cds_df["Ticker"] == ticker].set_index("Date")
        sub_df = sub_df.loc[:date]
        if sub_df.empty:
            print(f"No CDS data for {ticker} before {date}")
            continue

        market_spread = sub_df.iloc[-1]["Spread"] / 10000  # Convert from bps to decimal
        signal = detect_mispricing(merton["Model-Implied Spread"], market_spread, THRESHOLD)

        strategy_log.append({
            "Ticker": ticker,
            "Date": date,
            "Asset Value": E,
            "Model Spread": merton["Model-Implied Spread"],
            "Market Spread": market_spread,
            "Mispricing": market_spread - merton["Model-Implied Spread"],
            "Signal": signal
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

signal_df = pd.DataFrame(strategy_log)
signal_df.to_csv("./data/signals.csv", index=False)

for ticker in TICKERS:
    cds_prices = cds_df[cds_df["Ticker"] == ticker].set_index("Date")["Spread"]
    signals = signal_df[signal_df["Ticker"] == ticker].set_index("Date")

    if len(signals) < 2:
        print(f"Not enough signals to backtest for {ticker}")
        continue

    positions = generate_position_series(signals)
    cumulative, strat_returns = backtest_strategy(cds_prices, positions)

    plt.figure()
    plt.plot(cumulative, label=ticker)
    plt.title(f"Backtested Cumulative Returns: {ticker} CDS")
    plt.grid(True)
    plt.legend()
    plt.show()


# ad var, cgar, ,drawdown, hows it compare to the benchmark for presentation.
# stocks don't matter much. objective criteria to select stocks.
