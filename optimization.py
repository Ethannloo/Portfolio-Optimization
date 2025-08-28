"""
Tangency + Monte Carlo (mean-variance) with MC cloud plot.

Requirements:
  pip install numpy pandas cvxpy matplotlib xgboost scikit-learn

Files expected in the same folder:
  - adj_close_prices.csv  (date index in col 0; columns = tickers AdjClose)
  - ml_model.py           (contains predict_next_mu)
  - ml_model_daily.pkl    (optional; created if missing)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

from ml_model import predict_next_mu


# ---------------------------
# I/O + transforms
# ---------------------------
def load_price_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df.sort_index()

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------
# Core mean-variance helpers
# ---------------------------
def solve_tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, risk_free_rate: float) -> np.ndarray:
    """
    Unit-excess-return minimum-variance:
      min w'Σw  s.t.  (mu - rf*1)' w = 1,  w >= 0
    Then normalize to sum to 1 for a fully-invested long-only tangency portfolio.
    """
    n = len(mu)
    w = cp.Variable(n)
    mu_ex = mu - risk_free_rate
    objective   = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [mu_ex @ w == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
    except Exception:
        prob.solve(solver=cp.ECOS, warm_start=True)
    if w.value is None:
        raise RuntimeError("QP solver failed to converge.")
    return w.value

def normalize_weights(w_raw: np.ndarray) -> np.ndarray:
    s = float(w_raw.sum())
    if s <= 0:
        raise ValueError("Sum of weights must be positive.")
    return w_raw / s

def portfolio_metrics(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float) -> tuple[float, float, float]:
    exp_ret = float(mu @ w)
    quad    = float(w.T @ Sigma @ w)
    vol     = float(np.sqrt(max(quad, 0.0)))
    sharpe  = (exp_ret - rf) / vol if vol > 0 else np.nan
    return exp_ret, vol, sharpe


# ---------------------------
# Monte Carlo over weights (Dirichlet)
# ---------------------------
def mc_search_mv(mu: np.ndarray,
                 Sigma: np.ndarray,
                 rf: float = 0.05,
                 n_samples: int = 50_000,
                 w_max: float | None = None,
                 seed: int | None = 42) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """
    Monte Carlo search for best-by-Sharpe long-only, fully-invested weights.
    Samples weights from Dirichlet (uniform over simplex), evaluates analytically.
    Returns (best_w, metrics_dict, all_exp_rets, all_vols) for plotting the cloud.
    """
    rng = np.random.default_rng(seed)
    k = len(mu)

    # 1) sample long-only, sum-to-1 weights
    W = rng.dirichlet(alpha=np.ones(k), size=n_samples)  # (n_samples, k)

    # optional weight cap (filtering)
    if w_max is not None:
        mask = (W <= (w_max + 1e-12)).all(axis=1)
        if not np.any(mask):
            raise ValueError("w_max too tight for sampled weights.")
        W = W[mask]
        # renormalize (should already sum to 1, but keep it safe)
        W = W / W.sum(axis=1, keepdims=True)

    # 2) analytic MV scoring (numerically safe)
    exp_ret = W @ mu
    quad    = np.einsum('ij,jk,ik->i', W, Sigma, W)
    vol     = np.sqrt(np.clip(quad, 0.0, None))
    sharpe  = (exp_ret - rf) / np.where(vol > 0, vol, np.nan)

    # 3) pick the best-by-Sharpe
    i = int(np.nanargmax(sharpe))
    best = W[i]
    metrics = dict(exp=float(exp_ret[i]), vol=float(vol[i]), sr=float(sharpe[i]))

    return best, metrics, exp_ret, vol


# ---------------------------
# Efficient frontier + plot
# ---------------------------
def compute_efficient_frontier(mu: np.ndarray,
                               Sigma: np.ndarray,
                               num_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute long-only efficient frontier by sweeping target return.
    Returns (rets, vols).
    """
    n = len(mu)
    rets = np.linspace(mu.min(), mu.max(), num_points)
    vols = np.full_like(rets, np.nan, dtype=float)

    for idx, R_target in enumerate(rets):
        w = cp.Variable(n, nonneg=True)
        obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
        cons = [mu @ w == R_target, cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            prob.solve(solver=cp.ECOS, warm_start=True)
        if w.value is not None:
            w_opt = w.value
            vols[idx] = float(np.sqrt(max(w_opt.T @ Sigma @ w_opt, 0.0)))
    return rets, vols

def plot_frontier_tangency_mc(mu: np.ndarray,
                              Sigma: np.ndarray,
                              rf: float,
                              w_tang: np.ndarray,
                              mc_rets: np.ndarray | None,
                              mc_vols: np.ndarray | None) -> None:
    """
    Single figure: MC cloud (if provided), efficient frontier, tangency star.
    """
    rets, vols = compute_efficient_frontier(mu, Sigma, num_points=300)
    ret_t, vol_t, sr_t = portfolio_metrics(w_tang, mu, Sigma, rf)

    plt.figure(figsize=(9, 6))

    # MC cloud first (underlay)
    if mc_rets is not None and mc_vols is not None and len(mc_rets) > 0:
        plt.scatter(mc_vols, mc_rets, s=10, alpha=0.35, label="MC portfolios")

    # Efficient frontier curve
    plt.plot(vols, rets, label="Efficient frontier")

    # Tangency point
    plt.scatter([vol_t], [ret_t], marker="*", s=160, label=f"Tangency (SR={sr_t:.2f})")

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier + Tangency + Monte Carlo Portfolios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    # ---- Config ----
    CSV_FILE         = "adj_close_prices.csv"   # daily prices (AdjClose)
    RISK_FREE_RATE   = 0.05                     # annualized
    PERIODS_PER_YEAR = 252                      # trading days/year
    LAGS             = 5
    ROLL_WINDOWS     = [5, 21, 63]
    N_SAMPLES        = 100_000                  # MC sample size
    W_MAX_CAP        = 0.25                     # None for no cap

    # 1–2) prices → returns
    prices  = load_price_data(CSV_FILE)
    returns = compute_log_returns(prices)

    # 3) ML μ (annualized)
    mu_series = predict_next_mu(
        returns,
        periods_per_year=PERIODS_PER_YEAR,
        lag_nums=LAGS,
        rolling_windows=ROLL_WINDOWS,
        model_path="ml_model_daily.pkl"
    )
    tickers  = mu_series.index.tolist()
    mu_vec   = mu_series.values

    # 4) Σ (annualized), aligned to μ order
    Sigma_df  = returns.cov() * PERIODS_PER_YEAR
    Sigma_mat = Sigma_df.loc[tickers, tickers].values

    # 5–6) Tangency (QP)
    w_raw = solve_tangency_portfolio(mu_vec, Sigma_mat, RISK_FREE_RATE)
    w_qp  = normalize_weights(w_raw)
    exp_r_qp, vol_qp, sr_qp = portfolio_metrics(w_qp, mu_vec, Sigma_mat, RISK_FREE_RATE)

    print("\nOptimal long-only tangency portfolio (QP):")
    for t, w in zip(tickers, w_qp):
        if w > 1e-8:
            print(f"  {t:>6}: {w:6.2%}")
    print(f"\nExpected annual return : {exp_r_qp:6.2%}")
    print(f"Annualised volatility  : {vol_qp:6.2%}")
    print(f"Sharpe ratio           : {sr_qp:6.4f}")

    # 7–8) Monte Carlo search + cloud
    print("\n--- Monte Carlo search (long-only, sum=1) ---")
    w_mc, mc, mc_rets, mc_vols = mc_search_mv(
        mu_vec, Sigma_mat,
        rf=RISK_FREE_RATE,
        n_samples=N_SAMPLES,
        w_max=W_MAX_CAP,
        seed=42
    )
    print(f"MC points: {len(mc_rets)} | NaN rets: {np.isnan(mc_rets).sum()} | NaN vols: {np.isnan(mc_vols).sum()}")

    print("\nMonte Carlo best-by-Sharpe portfolio:")
    for t, w in sorted(zip(tickers, w_mc), key=lambda x: -x[1]):
        if w > 1e-8:
            print(f"  {t:>6}: {w:6.2%}")
    print(f"\nExpected annual return : {mc['exp']:6.2%}")
    print(f"Annualised volatility  : {mc['vol']:6.2%}")
    print(f"Sharpe ratio           : {mc['sr']:6.4f}")

    print("\nComparison:")
    print(f"  QP Tangency Sharpe : {sr_qp:6.4f}")
    print(f"  MC Best Sharpe     : {mc['sr']:6.4f}")

    # Plot frontier + MC cloud + tangency star (uses ML μ for the curve)
    plot_frontier_tangency_mc(mu_vec, Sigma_mat, RISK_FREE_RATE, w_qp, mc_rets, mc_vols)


if __name__ == "__main__":   # pragma: no cover
    main()
