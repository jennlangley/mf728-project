import pandas as pd
import yfinance as yf
import os

# Download data for selected assets
import os

def download_data(tickers, start_date, end_date, cache_path='./data/asset_data.csv'):
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
        data.index = data.index.tz_localize(None)
        return data
    else:
        try:
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
            data.index = data.index.tz_localize(None)
            data = data.dropna(axis=0)
            data.to_csv(cache_path)
            print(f"Downloaded data saved to {cache_path}")
            return data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()

def load_risk_free_rate(file_path):
    ff_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    ff_data.index = ff_data.index.tz_localize(None)
    risk_free_df = ff_data['RF'] / 100
    return risk_free_df

def merge_data_with_rf(asset_data, risk_free_data):
    merged_data = asset_data.merge(risk_free_data, left_index=True, right_index=True, how='inner')
    return merged_data


def calculate_excess_returns(merged_data):
    returns = merged_data.drop(columns='RF').pct_change().dropna()
    excess_returns = returns.sub(merged_data['RF'], axis=0)
    return excess_returns

def get_firm_fundamentals(tickers, output_path):
    firm_data = {}
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        debt = info.get("totalDebt", 0) or 0
        market_cap = info.get("marketCap", None)
        firm_data[ticker] = {
            "Debt": debt,
            "MarketCap": market_cap,
            "DebtToEquity": debt / market_cap if market_cap else None
        }

    df = pd.DataFrame.from_dict(firm_data, orient="index")
    df.index.name = "Ticker"
    df.to_csv(output_path)
    print(f"Firm fundamentals saved to {output_path}")
    return df

def main():
    cds = pd.read_csv('./data/cds.csv')
    tickers = ['JPM','BAC', 'GS', 'IBM', 'F', 'XOM', 'GM', 'T']
    # 'F' bank of america, citi group, RTX
    data = download_data(tickers, start_date='2019-01-01', end_date='2024-12-31')
    risk_free_data = load_risk_free_rate('./data/ff_factors_daily.csv')
    merged_data = merge_data_with_rf(data, risk_free_data)
    excess_returns = calculate_excess_returns(merged_data).dropna()[1:]  # Drop the first row with NaN values
    excess_returns.to_csv('./data/excess_returns.csv')
    print(excess_returns.head())
    print("Excess returns calculated and saved to ./data/excess_returns.csv")
    print(data.head())

    firm_fundamentals = get_firm_fundamentals(tickers, './data/firm_fundamentals.csv')
    print("Firm fundamentals preview:")
    print(firm_fundamentals.head())

if __name__ == "__main__":
    main()
