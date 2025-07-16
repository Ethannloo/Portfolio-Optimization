"""
Long-only maximum-Sharpe portfolio built from *daily* price data,
with expected returns predicted by `ml_model.predict_next_mu`.

Pipeline
--------
1. Load daily adjusted-close prices.
2. Compute daily log returns.
3. Use ML model to forecast next-day annualized μ.
4. Compute historical Σ (annualized).
5. Solve unit-excess-return, minimum-variance QP (long-only).
6. Normalise, print weights & metrics.
7. Plot efficient frontier for comparison.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp

from ml_model import predict_next_mu


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_price_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ----------------------------------------------------------------------
# Optimisation helpers
# ----------------------------------------------------------------------
def solve_tangency_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float
) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    objective    = cp.Minimize(cp.quad_form(w, Sigma))
    constraints  = [mu @ w - risk_free_rate == 1, w >= 0]
    cp.Problem(objective, constraints).solve()
    if w.value is None:
        raise RuntimeError("QP solver failed to converge.")
    return w.value


def normalize_weights(w_raw: np.ndarray) -> np.ndarray:
    total = w_raw.sum()
    if total <= 0:
        raise ValueError("Sum of weights must be positive.")
    return w_raw / total


def portfolio_metrics(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float
) -> tuple[float, float, float]:
    exp_ret  = mu @ w
    sigma    = np.sqrt(w.T @ Sigma @ w)
    sharpe   = (exp_ret - risk_free_rate) / sigma
    return exp_ret, sigma, sharpe


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
def main() -> None:
    CSV_FILE         = "adj_close_prices.csv"   # daily prices
    RISK_FREE_RATE   = 0.05                     # annual
    PERIODS_PER_YEAR = 252                      # trading days
    LAGS             = 5
    ROLL_WINDOWS     = [5, 21, 63]

    # 1-2. prices → returns
    prices  = load_price_data(CSV_FILE)
    returns = compute_log_returns(prices)

    # 3. ML-predicted annualised μ
    mu_series = predict_next_mu(
        returns,
        periods_per_year=PERIODS_PER_YEAR,
        lag_nums=LAGS,
        rolling_windows=ROLL_WINDOWS,
        model_path="ml_model_daily.pkl"
    )
    tickers  = mu_series.index
    mu_vec   = mu_series.values

    # 4. historical Σ (annualised)
    Sigma_df  = returns.cov() * PERIODS_PER_YEAR
    Sigma_mat = Sigma_df.values

    # 5-6. optimisation
    w_raw = solve_tangency_portfolio(mu_vec, Sigma_mat, RISK_FREE_RATE)
    w_opt = normalize_weights(w_raw)

    # 6. metrics
    exp_r, vol, sr = portfolio_metrics(w_opt, mu_vec, Sigma_mat, RISK_FREE_RATE)

    print("\nOptimal long-only tangency portfolio:")
    for t, w in zip(tickers, w_opt):
        print(f"  {t:>6}: {w:6.2%}")
    print(f"\nExpected annual return : {exp_r:6.2%}")
    print(f"Annualised volatility  : {vol:6.2%}")
    print(f"Sharpe ratio           : {sr:6.4f}")

    # 7. plot efficient frontier (optional visual module)
    try:
        from visual import plot_efficient_frontier
        mu_hist = returns.mean() * PERIODS_PER_YEAR
        plot_efficient_frontier(mu_hist, Sigma_df, RISK_FREE_RATE, num_points=100)
    except ModuleNotFoundError:
        print("\n[visual.plot_efficient_frontier] not found – skipping plot.")


if __name__ == "__main__":   # pragma: no cover
    main()
