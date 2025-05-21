import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd
from optimization import *

def compute_efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    num_points: int = 100
):
    """
    Compute the efficient frontier for a long-only portfolio.
    """
    n = mu.shape[0]
    # choose target returns uniformly between min and max expected returns
    rets = np.linspace(mu.min(), mu.max(), num_points)
    vols = []

    for R_target in rets:
        w = cp.Variable(n)
        # objective: minimize variance
        obj = cp.Minimize(cp.quad_form(w, Sigma))
        # constraints: target return, fully invested, long-only
        cons = [
            mu @ w == R_target,
            cp.sum(w) == 1,
            w >= 0
        ]
        prob = cp.Problem(obj, cons)
        prob.solve(warm_start=True)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            vols.append(np.nan)
        else:
            w_opt = w.value
            vols.append(np.sqrt(w_opt.T @ Sigma @ w_opt))

    return rets, np.array(vols)

def plot_efficient_frontier(
    mu_series: pd.Series,
    Sigma_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
    num_points: int = 100
):
    """
    Compute and plot the efficient frontier along with the tangency portfolio.

    Args:
        mu_series: Series of annualized expected returns.
        Sigma_df:  DataFrame of annualized covariance matrix.
        risk_free_rate: Annual risk-free rate (decimal).
        num_points: Number of points on the efficient frontier.
    """
    mu_vec = mu_series.values
    Sigma_mat = Sigma_df.values

    # 1. Efficient frontier
    rets, vols = compute_efficient_frontier(mu_vec, Sigma_mat, num_points)

    # 2. Tangency portfolio (unit‐excess version, then normalize)
    w_tang_raw = solve_tangency_portfolio(mu_vec, Sigma_mat, risk_free_rate)
    w_tang = normalize_weights(w_tang_raw)
    ret_t, vol_t, sharpe_t = portfolio_metrics(w_tang, mu_vec, Sigma_mat, risk_free_rate)

    # 3. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(vols, rets, label="Efficient frontier")
    plt.scatter([vol_t], [ret_t], c='r', marker='*', s=100, label=f"Tangency (SR={sharpe_t:.2f})")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier + Tangency Portfolio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

