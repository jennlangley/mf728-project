import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

from data.data_loader import (
    download_data,
    load_risk_free_rate,
    merge_data_with_rf,
    calculate_excess_returns,
    get_firm_fundamentals
)

# ------------------------
# Constants
# ------------------------
TICKERS = ['JPM','BAC', 'GS', 'IBM', 'F', 'XOM', 'GM', 'T']
START_DATE = "2019-01-01"
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
# Helper Functions
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

def detect_mispricing(model_spread, market_spread, threshold):
    diff = market_spread - model_spread
    if diff > threshold:
        return "Short CDS"
    elif diff < -threshold:
        return "Long CDS"
    else:
        return "No Trade"

def generate_position_series(signal_df):
    signal_map = {"Short CDS": -1, "Long CDS": 1, "No Trade": 0}
    return signal_df["Signal"].map(signal_map)

def backtest_strategy(spread_series, position_series):
    spread_returns = spread_series.pct_change().shift(-1)
    strat_returns = spread_returns * position_series.shift(1)
    cumulative_returns = (1 + strat_returns).cumprod()
    return cumulative_returns, strat_returns

def calculate_cagr(cumulative_returns):
    cumulative_returns = cumulative_returns.dropna()
    if cumulative_returns.empty:
        return float("nan")
    total_days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
    years = total_days / 252  # market days in a year
    final_value = cumulative_returns.iloc[-1]
    initial_value = cumulative_returns.iloc[0]
    if initial_value == 0:
        return float("nan")
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_var(returns, confidence=0.95):
    return np.percentile(returns.dropna(), (1 - confidence) * 100)

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

cds_df = pd.read_csv(CDS_CSV_PATH, parse_dates=['Date'], index_col='Date')
cds_df.index = cds_df.index.tz_localize(None)
strategy_log = []
window = 252

# Load CDX data
cdx_df = pd.read_csv("./data/cdx.csv", parse_dates=["Date"])
cdx_df.set_index("Date", inplace=True)
cdx_df.index = cdx_df.index.tz_localize(None)
cdx_ig = cdx_df["CDX.IG"].dropna()

# Convert to returns and cumulative
cdx_ig_returns = cdx_ig.pct_change().shift(-1)
cdx_ig_cumulative = (1 + cdx_ig_returns).cumprod()


for ticker in TICKERS:
    print(f"Processing: {ticker}")
    try:
        returns = excess_returns[ticker].dropna()
        E = market_cap_dict.get(ticker, None)
        D = debt_dict.get(ticker, 0)
        if E is None:
            print(f"Missing market cap for {ticker}, skipping.")
            continue

        for i in range(window, len(returns)):
            date = returns.index[i]
            sigma_E = returns.iloc[i-window:i].std() * np.sqrt(252)
            r_slice = risk_free_data.loc[:date]
            if r_slice.empty:
                continue
            r = r_slice.iloc[-1]

            merton = merton_model(E, sigma_E, D, r, T)

            try:
                market_spread = cds_df.loc[date, ticker] / 10000
            except KeyError:
                continue

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


# Plot comparison of strategy vs benchmark
for ticker in TICKERS:
    ticker_signals = signal_df[signal_df["Ticker"] == ticker].copy()
    if ticker_signals.shape[0] < 2:
        continue

    ticker_signals["Date"] = pd.to_datetime(ticker_signals["Date"])
    ticker_signals.set_index("Date", inplace=True)

    cds_series = cds_df[ticker].loc[ticker_signals.index.min():ticker_signals.index.max()] / 10000
    cds_series = cds_series.reindex(ticker_signals.index)

    positions = generate_position_series(ticker_signals)
    cumulative, strat_returns = backtest_strategy(cds_series, positions)

    # Align benchmark
    cdx_subset = cdx_ig_cumulative.loc[ticker_signals.index.min():ticker_signals.index.max()]
    cdx_subset = cdx_subset.reindex(ticker_signals.index)

    # Performance Metrics
    cagr = calculate_cagr(cumulative)
    max_dd = calculate_max_drawdown(cumulative)
    var_95 = calculate_var(strat_returns, confidence=0.95)

    print(f"{ticker} - CAGR: {cagr:.2%}, Max DD: {max_dd:.2%}, VaR(95%): {var_95:.2%}")

    plt.figure()
    plt.plot(cumulative, label=f"{ticker} Strategy")
    plt.plot(cdx_subset / cdx_subset.iloc[0], label="CDX.IG Benchmark", linestyle='--')
    plt.title(f"{ticker} Strategy vs CDX.IG Benchmark")
    plt.legend()
    plt.grid(True)
    plt.show()
