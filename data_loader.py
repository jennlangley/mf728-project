import pandas as pd
import yfinance as yf
import os

# Download data for selected assets
def download_data(tickers, start_date, end_date, cache_path='./data/asset_data.csv'):
    if os.path.exists(cache_path):
        print(f"Loading asset data from {cache_path}")
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
    """
    Enhanced function to collect more comprehensive firm fundamentals
    critical for credit risk modeling and liquidity factor calculation
    """
    firm_data = {}

    for ticker in tickers:
        print(f"Downloading fundamentals for {ticker}...")
        try:
            # Get ticker info
            stock = yf.Ticker(ticker)
            info = stock.info

            # Basic market data
            market_cap = info.get("marketCap", None)
            current_price = info.get("currentPrice", info.get("previousClose", None))

            # Balance sheet items
            try:
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty:
                    total_assets = balance_sheet.loc["Total Assets"].iloc[0]
                    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in balance_sheet.index else None
                    long_term_debt = balance_sheet.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in balance_sheet.index else 0
                    short_term_debt = balance_sheet.loc["Current Debt"].iloc[0] if "Current Debt" in balance_sheet.index else 0
                else:
                    total_assets = info.get("totalAssets", 0)
                    total_liabilities = info.get("totalDebt", 0) + info.get("totalCurrentLiabilities", 0)
                    long_term_debt = info.get("longTermDebt", 0)
                    short_term_debt = info.get("shortTermDebt", 0)
            except Exception as e:
                print(f"Error getting balance sheet for {ticker}: {e}")
                total_assets = info.get("totalAssets", 0)
                total_liabilities = info.get("totalDebt", 0) + info.get("totalCurrentLiabilities", 0)
                long_term_debt = info.get("longTermDebt", 0)
                short_term_debt = info.get("shortTermDebt", 0)

            total_debt = long_term_debt + short_term_debt

            # Income statement items
            try:
                income_stmt = stock.income_stmt
                if not income_stmt.empty:
                    ebit = income_stmt.loc["EBIT"].iloc[0] if "EBIT" in income_stmt.index else None
                    net_income = income_stmt.loc["Net Income"].iloc[0] if "Net Income" in income_stmt.index else None
                    interest_expense = income_stmt.loc["Interest Expense"].iloc[0] if "Interest Expense" in income_stmt.index else None
                    revenue = income_stmt.loc["Total Revenue"].iloc[0] if "Total Revenue" in income_stmt.index else None
                else:
                    ebit = info.get("ebitda", 0) - info.get("totalDepreciationAndAmortization", 0)
                    net_income = info.get("netIncomeToCommon", 0)
                    interest_expense = None  # Not directly available in info
                    revenue = info.get("totalRevenue", 0)
            except Exception as e:
                print(f"Error getting income statement for {ticker}: {e}")
                ebit = info.get("ebitda", 0) - info.get("totalDepreciationAndAmortization", 0)
                net_income = info.get("netIncomeToCommon", 0)
                interest_expense = None
                revenue = info.get("totalRevenue", 0)

            # Calculate key financial ratios
            if market_cap and market_cap > 0:
                debt_to_equity = total_debt / market_cap
            else:
                debt_to_equity = None

            if total_assets and total_assets > 0:
                roa = (net_income / total_assets) if net_income is not None else None
                leverage = total_assets / (total_assets - total_liabilities) if total_liabilities is not None else None
            else:
                roa = None
                leverage = None

            if interest_expense and interest_expense != 0 and ebit is not None:
                interest_coverage = ebit / abs(interest_expense)
            else:
                interest_coverage = None

            # Determine sector using industry info
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            # Create data dictionary
            firm_data[ticker] = {
                "MarketCap": market_cap,
                "CurrentPrice": current_price,
                "TotalAssets": total_assets,
                "TotalLiabilities": total_liabilities,
                "LongTermDebt": long_term_debt,
                "ShortTermDebt": short_term_debt,
                "Debt": total_debt,
                "EBIT": ebit,
                "Revenue": revenue,
                "NetIncome": net_income,
                "InterestExpense": interest_expense,
                "DebtToEquity": debt_to_equity,
                "ROA": roa,
                "Leverage": leverage,
                "InterestCoverage": interest_coverage,
                "Sector": sector,
                "Industry": industry
            }

            print(f"Successfully downloaded fundamentals for {ticker}")

        except Exception as e:
            print(f"Error processing fundamentals for {ticker}: {e}")
            firm_data[ticker] = {
                "MarketCap": None,
                "Debt": None,
                "DebtToEquity": None,
                "ROA": None,
                "InterestCoverage": None,
                "Sector": "Unknown"
            }

    df = pd.DataFrame.from_dict(firm_data, orient="index")
    df.index.name = "Ticker"
    df.to_csv(output_path)
    print(f"Enhanced firm fundamentals saved to {output_path}")
    return df

def main():
    cds = pd.read_csv('./data/cds.csv')
    tickers = ['JPM','BAC', 'GS', 'IBM', 'F', 'XOM', 'GM', 'T']
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
