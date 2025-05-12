#!/usr/bin/env python3
"""
tangency_portfolio.py

Compute the long-only maximum‐Sharpe portfolio given monthly price data.

1 Load monthly price data from CSV
2 Compute monthly log-returns
3 Annualize expected returns and covariance matrix
4 Solve a unit‐excess‐return, minimum‐variance QP with long‐only constraint
5 Normalize the solution to sum to 1
6 Report portfolio weights and metrics (expected return, volatility, Sharpe)
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from visual import *


def load_price_data(csv_file: str) -> pd.DataFrame:
    """
    Load historical price data from a CSV file.
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute continuously compounded log returns from price data.
    """
    # Shift prices to compute period‐over‐period ratios, then take log.
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns


def annualize_moments(returns: pd.DataFrame, periods_per_year: int = 12):
    """
    Annualize mean returns and covariance matrix.
    """
    mu = returns.mean() * periods_per_year
    Sigma = returns.cov() * periods_per_year
    return mu, Sigma


def solve_tangency_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float
) -> np.ndarray:
    """
    Solve the convex QP for the unit‐excess‐return minimum‐variance portfolio.
    """
    n = mu.shape[0]
    w = cp.Variable(n)

    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # Constraints: unit excess return & long-only
    constraints = [
        mu @ w - risk_free_rate == 1,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"Solver failed: {prob.status}")

    return w.value


def normalize_weights(w_raw: np.ndarray) -> np.ndarray:
    """
    Scale a weight vector so that its elements sum to 1.
    """
    total = np.sum(w_raw)
    if total <= 0:
        raise ValueError("Sum of raw weights must be positive for normalization.")
    return w_raw / total


def portfolio_metrics(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float
):
    """
    Compute key portfolio metrics: expected return, volatility, and Sharpe ratio.
    """
    exp_return = mu @ w
    volatility = np.sqrt(w.T @ Sigma @ w)
    sharpe = (exp_return - risk_free_rate) / volatility
    return exp_return, volatility, sharpe


def main():
    # configurable inputs
    CSV_FILE = 'data.csv'       # CSV with monthly prices
    RISK_FREE_RATE = 0.05       # 5% annual risk-free

    # Load data
    prices = load_price_data(CSV_FILE)

    # Compute log-returns
    returns = compute_log_returns(prices)

    # Annualize expected returns & covariance
    mu_series, Sigma_df = annualize_moments(returns, periods_per_year=12)

    # Prepare raw arrays for the optimizer
    mu_vec = mu_series.values
    Sigma_mat = Sigma_df.values
    tickers = mu_series.index.tolist()

    # Solve for raw weights
    w_raw = solve_tangency_portfolio(mu_vec, Sigma_mat, RISK_FREE_RATE)

    # Normalize to a fully-invested portfolio
    w_opt = normalize_weights(w_raw)

    # Compute portfolio metrics
    exp_ret, vol, sr = portfolio_metrics(w_opt, mu_vec, Sigma_mat, RISK_FREE_RATE)

    print("\nOptimal tangency portfolio (long-only):")
    for tckr, weight in zip(tickers, w_opt):
        print(f"  {tckr:>5s}: {weight:>6.2%}")
    print(f"\nExpected annual return: {exp_ret:.2%}")
    print(f"Annualized volatility:   {vol:.2%}")
    print(f"Sharpe ratio:            {sr:.4f}")

        # 8. Plot efficient frontier
    plot_efficient_frontier(mu_series, Sigma_df, RISK_FREE_RATE, num_points=100)



if __name__ == "__main__":
    main()

