import yfinance as yf
import csv
import numpy as np

tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'NVDA', 'AMZN', 'META']


d = yf.download(tickers, start='2024-01-01', end='2025-01-01', auto_adjust=False)['Adj Close']
monthly_d = d.resample('D').last()
monthly_d.to_csv("data.csv")
log_return = np.log(monthly_d / monthly_d.shift(1))
log_return = log_return.dropna()
mu = log_return.mean()
mu_annualized = mu * 256
sigma = log_return.cov()
sigma_annual = sigma * 256
