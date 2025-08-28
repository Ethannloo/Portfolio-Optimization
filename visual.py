import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp

from optimization import solve_tangency_portfolio, normalize_weights, portfolio_metrics

def compute_efficient_frontier(mu: np.ndarray,
                               Sigma: np.ndarray,
                               num_points: int = 100):
    """
    Compute the efficient frontier for a long-only portfolio.
    Returns (rets, vols).
    """
    n = mu.shape[0]
    rets = np.linspace(mu.min(), mu.max(), num_points)
    vols = []

    for R_target in rets:
        w = cp.Variable(n, nonneg=True)
        obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
        cons = [mu @ w == R_target, cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            vols.append(np.nan)
        else:
            w_opt = w.value
            vols.append(np.sqrt(w_opt.T @ Sigma @ w_opt))

    return rets, np.array(vols)


def plot_efficient_frontier(mu_series: pd.Series,
                            Sigma_df: pd.DataFrame,
                            risk_free_rate: float = 0.05,
                            num_points: int = 100,
                            mc_results: tuple[np.ndarray, np.ndarray] | None = None):
    """
    Plot efficient frontier + tangency portfolio + optional MC scatter.

    mc_results: optional tuple (rets, vols) from Monte Carlo portfolios.
    """
    mu_vec = mu_series.values if isinstance(mu_series, pd.Series) else np.asarray(mu_series)
    Sigma_mat = Sigma_df.values if isinstance(Sigma_df, pd.DataFrame) else np.asarray(Sigma_df)

    # 1. Efficient frontier
    rets, vols = compute_efficient_frontier(mu_vec, Sigma_mat, num_points)

    # 2. Tangency portfolio
    w_tang_raw = solve_tangency_portfolio(mu_vec, Sigma_mat, risk_free_rate)
    w_tang = normalize_weights(w_tang_raw)
    ret_t, vol_t, sharpe_t = portfolio_metrics(w_tang, mu_vec, Sigma_mat, risk_free_rate)

    # 3. Plot
    plt.figure(figsize=(8, 6))

    # Efficient frontier curve
    plt.plot(vols, rets, 'b-', label="Efficient frontier")

    # Monte Carlo scatter (if provided)
    if mc_results is not None:
        mc_rets, mc_vols = mc_results
        plt.scatter(mc_vols, mc_rets, c='lightgrey', s=10, alpha=0.5, label="MC portfolios")

    # Tangency portfolio
    plt.scatter([vol_t], [ret_t], c='r', marker='*', s=140,
                label=f"Tangency (SR={sharpe_t:.2f})")

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier + Tangency + Monte Carlo Portfolios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
