import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Download historical price data
csv_file = '/Users/ethanloo/Desktop/Coding Projects/Portfolio Optimization/data.csv'
tickers   = ['AAPL','AMZN','GOOG','META', 'MSFT', 'NVDA', 'TSLA']
end_date  = datetime.today().strftime('%Y-%m-%d')

data = yf.download(tickers, interval= '1mo', start='2020-01-31', end='2024-12-31')['Adj Close']
data.to_csv(csv_file)
