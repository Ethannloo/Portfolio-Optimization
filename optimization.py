import pandas as pd
import numpy as np
import cvxpy as cp

# Define annual risk-free rate
risk_free_rate = 0.05

# Load monthly price data from CSV (one row per month-end)
csv_file = 'data.csv'
df = pd.read_csv(csv_file, index_col=0, parse_dates=True).sort_index()

# Compute monthly log-returns
returns = np.log(df / df.shift(1)).dropna()

# Annualize expected returns and covariance
mu = returns.mean() * 12            # annualized returns vector
Sigma = returns.cov() * 12          # annualized covariance matrix

mu_vec = mu.values
Sigma_mat = Sigma.values
tickers = mu.index.tolist()
n = len(mu_vec)

# Set up CVXPY optimization: minimize variance for unit excess return
w = cp.Variable(n)
objective = cp.Minimize(cp.quad_form(w, Sigma_mat))
constraints = [
    mu_vec @ w - risk_free_rate == 1,  # target 1 unit excess return
    w >= 0                 # long-only
]
prob = cp.Problem(objective, constraints)
prob.solve()

# Check solver status
if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    raise RuntimeError(f"Optimization failed: {prob.status}")

# Normalize weights to sum to 1 (fully invested)
w_tilde = w.value
w_opt = w_tilde / np.sum(w_tilde)

# Print optimal weights and metrics
print("Optimal portfolio weights (max Sharpe):")
for ticker, weight in zip(tickers, w_opt):
    print(f"{ticker}: {weight:.4f}")

# Portfolio metrics
exp_return = mu_vec @ w_opt
volatility = np.sqrt(w_opt.T @ Sigma_mat @ w_opt)
sharpe = (exp_return - risk_free_rate) / volatility

print(f"\nExpected annual return: {exp_return:.4f}")
print(f"Annualized volatility: {volatility:.4f}")
print(f"Sharpe ratio: {sharpe:.4f}")
