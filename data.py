import yfinance as yf
import csv
import numpy as np

tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'NVDA', 'AMZN', 'META']

d = yf.download(tickers, start='2020-01-01', end='2025-01-01', auto_adjust=False)['Adj Close']
monthly_d = d.resample('M').last()
monthly_d.to_csv("data.csv")

log_return = np.log(monthly_d / monthly_d.shift(1))
log_return_matrix = log_return.dropna().to_numpy()

print(log_return)

