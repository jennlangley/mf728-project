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
CDS_THRESHOLD = 0.005  # ~50bps in decimal form for CDS trading
EQUITY_THRESHOLD = 0.02  # 2% threshold for equity signals
T = 1  # 1 year time horizon
ZSCORE_WINDOW = 20  # Window for calculating z-scores
MOMENTUM_WINDOW = 10  # Window for calculating momentum

# ------------------------
# File Paths
# ------------------------
FF_CSV_PATH = "./data/ff_factors_daily.csv"
CDS_CSV_PATH = "./data/cds.csv"
CDX_CSV_PATH = "./data/cdx.csv"  # CDX index data
FIRM_FUNDAMENTALS_PATH = "./data/firm_fundamentals.csv"
EXCESS_RETURNS_PATH = "./data/excess_returns.csv"

# ------------------------
# Enhanced Credit Risk Modeling
# ------------------------
def enhanced_merton_model(E, sigma_E, D, r, T, liquidity_factor=1.0):
    """
    Enhanced Merton model that incorporates liquidity factors
    and provides more robust estimates of default probabilities
    """
    V = E + D
    sigma_V = sigma_E * E / V

    # Adding a mean-reverting adjustment
    # Helps with more realistic long-term default probabilities
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)

    # Calculate default probability
    default_prob = norm.cdf(-d2)

    # Incorporate liquidity premium into the model-implied spread
    model_spread = (-np.log(1 - default_prob) / T) * liquidity_factor

    return {
        "Asset Value": V,
        "Asset Volatility": sigma_V,
        "Distance to Default": d2,
        "Default Probability": default_prob,
        "Model-Implied Spread": model_spread
    }

# ------------------------
# Signal Generation Functions
# ------------------------
def detect_cds_mispricing(model_spread, market_spread, threshold, z_score=None):
    """
    Enhanced mispricing detection that incorporates both absolute levels
    and statistical measures (z-scores)
    """
    diff = market_spread - model_spread

    # Base signal on absolute difference
    if diff > threshold:
        signal = "Short CDS"
    elif diff < -threshold:
        signal = "Long CDS"
    else:
        signal = "No Trade"

    # If z-score is provided, use it to strengthen or weaken signal
    if z_score is not None:
        if abs(z_score) < 1.0 and signal != "No Trade":
            signal = "Weak " + signal
        elif abs(z_score) > 2.0:
            if signal == "No Trade" and z_score > 2.0:
                signal = "Weak Short CDS"
            elif signal == "No Trade" and z_score < -2.0:
                signal = "Weak Long CDS"

    return signal

def translate_cds_to_equity_signal(cds_signal):
    """
    Translate CDS trading signals to equity trading signals
    based on the cross-asset relationship
    """
    signal_map = {
        "Short CDS": "Buy Equity",  # When shorting CDS (credit improving), buy stock
        "Weak Short CDS": "Weak Buy Equity",
        "Long CDS": "Sell Equity",  # When going long CDS (credit deteriorating), sell stock
        "Weak Long CDS": "Weak Sell Equity",
        "No Trade": "No Trade"
    }
    return signal_map.get(cds_signal, "No Trade")

def calculate_cds_basis(ticker_spread, index_spread):
    """
    Calculate the basis between individual CDS and the index
    """
    if index_spread == 0:
        return 0
    return (ticker_spread / index_spread) - 1

def calculate_zscore(series, window=ZSCORE_WINDOW):
    """
    Calculate the z-score of the latest value relative to the window
    """
    if len(series) < window:
        return 0
    mean = series[-window:].mean()
    std = series[-window:].std()
    if std == 0:
        return 0
    return (series.iloc[-1] - mean) / std

def calculate_momentum(series, window=MOMENTUM_WINDOW):
    """
    Calculate price momentum
    """
    if len(series) < window:
        return 0
    return (series.iloc[-1] / series.iloc[-window] - 1)

def generate_position_series(signal_df, signal_column="Equity Signal"):
    """
    Convert signals to position sizes
    """
    signal_map = {
        "Buy Equity": 1,
        "Weak Buy Equity": 0.5,
        "Sell Equity": -1,
        "Weak Sell Equity": -0.5,
        "No Trade": 0
    }
    return signal_df[signal_column].map(signal_map)

# ------------------------
# Backtest and Performance Functions
# ------------------------
def backtest_equity_strategy(equity_series, position_series):
    """
    Backtest the equity trading strategy
    """
    equity_returns = equity_series.pct_change().shift(-1)
    strat_returns = equity_returns * position_series.shift(1)
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

def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_ret = returns - risk_free_rate
    return excess_ret.mean() / excess_ret.std() * np.sqrt(252)

# ------------------------
# Main Analysis
# ------------------------
def main():
    # Load data
    data = download_data(TICKERS, start_date=START_DATE, end_date=END_DATE)
    risk_free_data = load_risk_free_rate(FF_CSV_PATH)
    merged_data = merge_data_with_rf(data, risk_free_data)
    excess_returns = calculate_excess_returns(merged_data)
    excess_returns.to_csv(EXCESS_RETURNS_PATH)

    # Load firm fundamentals
    fundamentals_df = get_firm_fundamentals(TICKERS, FIRM_FUNDAMENTALS_PATH)
    debt_dict = fundamentals_df["Debt"].to_dict()
    market_cap_dict = fundamentals_df["MarketCap"].to_dict()

    # Additional fundamental factors for enhanced model
    if "ROA" in fundamentals_df.columns:
        roa_dict = fundamentals_df["ROA"].to_dict()
    else:
        roa_dict = {ticker: 0.05 for ticker in TICKERS}  # Default if not available

    if "InterestCoverage" in fundamentals_df.columns:
        interest_coverage_dict = fundamentals_df["InterestCoverage"].to_dict()
    else:
        interest_coverage_dict = {ticker: 5.0 for ticker in TICKERS}  # Default if not available

    # Load CDS data
    cds_df = pd.read_csv(CDS_CSV_PATH, parse_dates=['Date'], index_col='Date')
    cds_df.index = cds_df.index.tz_localize(None)

    # Load CDX index data
    try:
        cdx_df = pd.read_csv(CDX_CSV_PATH, parse_dates=["Date"])
        cdx_df.set_index("Date", inplace=True)
        cdx_df.index = cdx_df.index.tz_localize(None)
        cdx_ig = cdx_df["CDX.IG"].dropna()
        has_cdx = True
    except (FileNotFoundError, KeyError):
        print("CDX index data not found or invalid. Proceeding without index comparison.")
        has_cdx = False
        cdx_ig = pd.Series(dtype=float)

    # Portfolio and strategy tracking
    strategy_log = []
    window = 252  # Rolling window for volatility calculation

    # Calculate liquidity adjustment factors based on interest coverage and ROA

    sector_dict = {
        "JPM": "Financial Services",  # JPMorgan Chase
        "BAC": "Financial Services",  # Bank of America
        "GS": "Financial Services",   # Goldman Sachs
        "IBM": "Technology",          # International Business Machines
        "F": "Consumer Cyclical",     # Ford Motor Company
        "XOM": "Energy",              # Exxon Mobil
        "GM": "Consumer Cyclical",    # General Motors
        "T": "Communication Services"  # AT&T
    }

    def calculate_liquidity_factor(ticker):
        ic = interest_coverage_dict.get(ticker, 5.0)
        roa = roa_dict.get(ticker, 0.05)
        sector = sector_dict.get(ticker, "Unknown")

        # Base liquidity factor calculation
        liq_factor = 1.0

        # Interest coverage component
        if ic is not None and ic > 0:
            liq_factor += (0.3 / ic)
        else:
            liq_factor += 0.2

        # ROA component
        if roa is not None:
            liq_factor -= (roa * 1.5)

        # Sector-specific adjustments
        sector_adjustments = {
            "Financial Services": 0.1,    # Banks typically have higher liquidity premium in CDS
            "Energy": -0.05,              # Energy companies' CDS often trade differently
            "Technology": -0.03,
            "Consumer Cyclical": 0.02,    # Automotive companies
            "Communication Services": 0.04 # Telecom
        }
        liq_factor += sector_adjustments.get(sector, 0)

        # Bound between reasonable limits
        return max(0.8, min(1.5, liq_factor))

    for ticker in TICKERS:
        print(f"Processing: {ticker}")
        try:
            # Prepare data
            returns = excess_returns[ticker].dropna()
            prices = data[ticker].dropna()
            E = market_cap_dict.get(ticker, None)
            D = debt_dict.get(ticker, 0)

            if E is None:
                print(f"Missing market cap for {ticker}, skipping.")
                continue

            # Liquidity factor specific to this company
            liquidity_factor = calculate_liquidity_factor(ticker)
            print(f"{ticker} liquidity factor: {liquidity_factor:.2f}")

            # Time series of mispricing and signals
            mispricing_series = []

            for i in range(window, len(returns)):
                date = returns.index[i]
                sigma_E = returns.iloc[i-window:i].std() * np.sqrt(252)
                r_slice = risk_free_data.loc[:date]
                if r_slice.empty:
                    continue
                r = r_slice.iloc[-1]

                # Enhanced Merton model with liquidity adjustment
                merton = enhanced_merton_model(E, sigma_E, D, r, T, liquidity_factor)

                try:
                    market_spread = cds_df.loc[date, ticker] / 10000

                    # Calculate CDS basis if CDX data is available
                    if has_cdx and date in cdx_ig.index:
                        index_spread = cdx_ig.loc[date] / 10000
                        basis = calculate_cds_basis(market_spread, index_spread)
                    else:
                        basis = 0
                        index_spread = 0

                    # Store mispricing for z-score calculation
                    mispricing = market_spread - merton["Model-Implied Spread"]
                    mispricing_series.append(mispricing)

                    # Calculate z-score if enough history is available
                    if len(mispricing_series) >= ZSCORE_WINDOW:
                        z_score = calculate_zscore(pd.Series(mispricing_series))
                    else:
                        z_score = 0

                    # Calculate momentum in spreads
                    cds_momentum = 0
                    if ticker in cds_df.columns and len(cds_df[ticker][:date]) >= MOMENTUM_WINDOW:
                        recent_cds = cds_df[ticker][:date].iloc[-MOMENTUM_WINDOW:]
                        if not recent_cds.empty:
                            cds_momentum = calculate_momentum(recent_cds)

                    # Generate CDS signal
                    cds_signal = detect_cds_mispricing(
                        merton["Model-Implied Spread"],
                        market_spread,
                        CDS_THRESHOLD,
                        z_score
                    )

                    equity_signal = translate_cds_to_equity_signal(cds_signal)

                    # Store in strategy log
                    strategy_log.append({
                        "Ticker": ticker,
                        "Date": date,
                        "Asset Value": E,
                        "Model Spread": merton["Model-Implied Spread"],
                        "Market Spread": market_spread,
                        "Mispricing": mispricing,
                        "Z-Score": z_score,
                        "CDS-Index Basis": basis,
                        "CDS Momentum": cds_momentum,
                        "CDS Signal": cds_signal,
                        "Equity Signal": equity_signal
                    })

                except KeyError:
                    continue

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Convert to DataFrame and save signals
    signal_df = pd.DataFrame(strategy_log)
    signal_df.to_csv("./data/enhanced_signals.csv", index=False)


    portfolio_returns = {}
    # Performance analysis
    print("\n--- Performance Analysis ---")
    for ticker in TICKERS:
        ticker_signals = signal_df[signal_df["Ticker"] == ticker].copy()
        if ticker_signals.shape[0] < 2:
            continue

        ticker_signals["Date"] = pd.to_datetime(ticker_signals["Date"])
        ticker_signals.set_index("Date", inplace=True)

        # Use equity prices for backtest
        equity_series = data[ticker].loc[ticker_signals.index.min():ticker_signals.index.max()]
        equity_series = equity_series.reindex(ticker_signals.index)
        positions = generate_position_series(ticker_signals, "Equity Signal")
        cumulative, strat_returns = backtest_equity_strategy(equity_series, positions)
        portfolio_returns[ticker] = strat_returns

        # Calculate statistics
        if not cumulative.empty and not strat_returns.empty:
            cagr = calculate_cagr(cumulative)
            max_dd = calculate_max_drawdown(cumulative)
            var_95 = calculate_var(strat_returns, confidence=0.95)

            # Sharpe ratio
            avg_rf = risk_free_data.loc[strat_returns.index].mean()
            sharpe = calculate_sharpe_ratio(strat_returns, avg_rf)

            # Additional metrics
            volatility = strat_returns.std() * np.sqrt(252)
            hit_rate = len(strat_returns[strat_returns > 0]) / len(strat_returns.dropna())

            print(f"{ticker} - CAGR: {cagr:.2%}, Max DD: {max_dd:.2%}, VaR(95%): {var_95:.2%}, Sharpe: {sharpe:.2f}")

            # Buy-and-hold
            bh_returns = (equity_series / equity_series.iloc[0])
            plt.plot(bh_returns, label=f"{ticker} Buy & Hold", linestyle='--', color='green')
            plt.plot(cumulative, label=f"{ticker} Strategy", color='blue')

            # Risk-free benchmark
            rf_subset = risk_free_data.loc[ticker_signals.index]
            rf_cumulative = (1 + rf_subset).cumprod()
            plt.plot(rf_cumulative, label="Risk-Free Benchmark", linestyle=':', color='orange')

            # Metrics box
            metrics_text = "\n".join([
                f"CAGR: {cagr:.2%}",
                f"Volatility: {volatility:.2%}",
                f"Sharpe: {sharpe:.2f}",
                f"Max DD: {max_dd:.2%}",
                f"VaR (95%): {var_95:.2%}",
                f"Hit Rate: {hit_rate:.2%}"
            ])
            plt.annotate(metrics_text, xy=(0.02, 0.5), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

            plt.title(f"{ticker} CDS-Based Equity Trading Strategy")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./data/figures/{ticker}_strategy_performance.png")
            plt.close()

    # -----------------------------
    # Equal-weighted portfolio analysis
    # -----------------------------
    print("\n--- Equal-Weighted Portfolio Analysis ---")

    # Strategy portfolio
    portfolio_df = pd.DataFrame(portfolio_returns).fillna(0)
    equal_weighted_returns = portfolio_df.mean(axis=1)
    equal_weighted_cumulative = (1 + equal_weighted_returns).cumprod()

    # Buy-and-hold portfolio
    buy_hold_prices = pd.DataFrame({
        ticker: data[ticker].loc[portfolio_df.index.min():portfolio_df.index.max()]
        for ticker in TICKERS if ticker in data
    }).reindex(portfolio_df.index)

    buy_hold_returns = buy_hold_prices.pct_change()
    buy_hold_equal_weighted = buy_hold_returns.mean(axis=1)
    buy_hold_cumulative = (1 + buy_hold_equal_weighted).cumprod()

    # Risk-free benchmark
    rf_portfolio = risk_free_data.loc[portfolio_df.index]
    rf_cumulative_portfolio = (1 + rf_portfolio).cumprod()

    # Performance Metrics
    portfolio_metrics = {
        "CAGR": calculate_cagr(equal_weighted_cumulative),
        "Annualized Volatility": equal_weighted_returns.std() * np.sqrt(252),
        "Sharpe Ratio": calculate_sharpe_ratio(equal_weighted_returns, rf_portfolio.mean()),
        "Max Drawdown": calculate_max_drawdown(equal_weighted_cumulative),
        "VaR (95%)": calculate_var(equal_weighted_returns),
        "Hit Rate": len(equal_weighted_returns[equal_weighted_returns > 0]) / len(equal_weighted_returns.dropna())
    }

    print(f"Equal-Weighted Portfolio - CAGR: {portfolio_metrics['CAGR']:.2%}, Max DD: {portfolio_metrics['Max Drawdown']:.2%}, VaR(95%): {portfolio_metrics['VaR (95%)']:.2%}, Sharpe: {portfolio_metrics['Sharpe Ratio']:.2f}")


    plt.figure(figsize=(12, 8))
    plt.plot(equal_weighted_cumulative, label="Strategy Portfolio", linewidth=2)
    plt.plot(buy_hold_cumulative, label="Buy & Hold Portfolio", linestyle='--')
    plt.plot(rf_cumulative_portfolio, label="Risk-Free", linestyle=':')

    # Metrics box
    metrics_text = "\n".join([
        f"CAGR: {portfolio_metrics['CAGR']:.2%}",
        f"Volatility: {portfolio_metrics['Annualized Volatility']:.2%}",
        f"Sharpe: {portfolio_metrics['Sharpe Ratio']:.2f}",
        f"Max DD: {portfolio_metrics['Max Drawdown']:.2%}",
        f"VaR (95%): {portfolio_metrics['VaR (95%)']:.2%}",
        f"Hit Rate: {portfolio_metrics['Hit Rate']:.2%}"
    ])
    plt.annotate(metrics_text, xy=(0.02, 0.5), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    plt.title("Equal-Weighted CDS Strategy vs Buy & Hold vs Risk-Free")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./data/figures/portfolio_equal_weighted_performance.png")

if __name__ == "__main__":
    main()
