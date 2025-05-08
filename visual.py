import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def plot_efficient_frontier(mu_vec: np.ndarray,
                            Sigma_mat: np.ndarray,
                            num_points: int = 50):
    """
    Solves and plots the efficient frontier for long-only portfolios.
    """
    n = len(mu_vec)
    targets = np.linspace(mu_vec.min(), mu_vec.max(), num_points)
    vols, rets = [], []

    for target in targets:
        w = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, Sigma_mat))
        cons = [mu_vec @ w == target, w >= 0]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.SCS, verbose=False)
        w_opt = w.value
        vols.append(np.sqrt(w_opt.T @ Sigma_mat @ w_opt))
        rets.append(mu_vec @ w_opt)

    plt.figure()
    plt.plot(vols, rets, linestyle='--', linewidth=2)
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title('Efficient Frontier')
    plt.grid(True)
    plt.show()
