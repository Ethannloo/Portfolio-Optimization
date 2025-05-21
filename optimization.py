"""

Compute the long-only maximum‐Sharpe portfolio given monthly price data,
using an external ML model to predict next-month expected returns.

1  Load monthly price data from CSV
2  Compute monthly log-returns
3  Predict next-month returns via ml_model.predict_next_mu
4  Compute historical covariance matrix
5  Solve a unit excess return, minimum variance QP with long-only constraint
6  Normalize and report metrics
7  Plot efficient frontier
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from ml_model import predict_next_mu


def load_price_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def solve_tangency_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float
) -> np.ndarray:
    n = mu.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [mu @ w - risk_free_rate == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"Solver failed: {prob.status}")
    return w.value


def normalize_weights(w_raw: np.ndarray) -> np.ndarray:
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
    exp_return = mu @ w
    volatility = np.sqrt(w.T @ Sigma @ w)
    sharpe = (exp_return - risk_free_rate) / volatility
    return exp_return, volatility, sharpe


def main():
    CSV_FILE = 'data.csv'
    RISK_FREE_RATE = 0.05

    # Load prices and returns
    prices = load_price_data(CSV_FILE)
    returns = compute_log_returns(prices)

    # Predict expected returns using the ML model
    mu_series = predict_next_mu(returns)
    tickers = mu_series.index.tolist()
    mu_vec = mu_series.values

    # Calculate historical covariance matrix (annualized)
    Sigma_df = returns.cov() * 12
    Sigma_mat = Sigma_df.values

    # Solve optimization
    w_raw = solve_tangency_portfolio(mu_vec, Sigma_mat, RISK_FREE_RATE)
    w_opt = normalize_weights(w_raw)

    # Compute and display metrics
    exp_ret, vol, sr = portfolio_metrics(w_opt, mu_vec, Sigma_mat, RISK_FREE_RATE)
    print("\nOptimal tangency portfolio (using ML mu):")
    for t, wt in zip(tickers, w_opt):
        print(f"  {t:>5s}: {wt:>6.2%}")
    print(f"\nExpected annual return: {exp_ret:.2%}")
    print(f"Annualized volatility:   {vol:.2%}")
    print(f"Sharpe ratio:            {sr:.4f}")

    # Plot efficient frontier using historical means for comparison
    mu_hist = returns.mean() * 12
    # import plotting here to avoid circular dependency
    from visual import plot_efficient_frontier
    plot_efficient_frontier(mu_hist, Sigma_df, RISK_FREE_RATE, num_points=100)

if __name__ == "__main__":
    main()



