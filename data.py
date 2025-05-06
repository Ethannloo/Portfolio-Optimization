import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime, timedelta

# Download or load historical price data
csv_file = '/Users/ethanloo/Desktop/Coding Projects/Portfolio Optimization/data.csv'
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
end_date = datetime.today()
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start='2023-01-01', end= end_date)
    adj_close_df[ticker] = data['Adj Close']
    adj_close_df.to_csv(csv_file)
    


# Compute daily log returns
daily_prices = data.dropna()
log_returns = np.log(daily_prices / daily_prices.shift(1)).dropna()

# Estimate expected returns and covariance matrix
mu = log_returns.mean()  # Expected daily returns
Sigma = log_returns.cov()  # Daily covariance matrix

# Sharpe ratio optimization (maximize excess return/volatility)
mu_vec = mu.values
Sigma_mat = Sigma.values
r_f = 0.0001  # Assume 0.01% daily risk-free rate
n = len(mu_vec)

# Optimization variable
w = cp.Variable(n)

# Objective: maximize (mu.T @ w - r_f), subject to variance = 1
objective = cp.Maximize(mu_vec @ w - r_f)
constraints = [
    cp.quad_form(w, Sigma_mat) == 1,
    cp.sum(w) == 1,
    w >= 0
]

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Rescale weights to get real-world portfolio (variance normalization step)
weights_raw = w.value
scaling_factor = 1 / np.sqrt(weights_raw.T @ Sigma_mat @ weights_raw)
weights_scaled = weights_raw * scaling_factor

# Print results
print("\nOptimal Sharpe Ratio Portfolio Weights:")
for ticker, weight in zip(mu.index, weights_scaled):
    print(f"{ticker}: {weight:.4f}")

# Portfolio metrics
expected_return = mu_vec @ weights_scaled
portfolio_std = np.sqrt(weights_scaled.T @ Sigma_mat @ weights_scaled)
sharpe_ratio = (expected_return - r_f) / portfolio_std

print(f"\nExpected Daily Return: {expected_return:.4f}")
print(f"Portfolio Std Dev: {portfolio_std:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
