import pandas as pd
import numpy as np
import cvxpy as cp

#Load & index
df = pd.read_csv('/Users/ethanloo/Desktop/Coding Projects/Portfolio Optimization/data.csv',
                  index_col=0, parse_dates=True)

#Compute returns (assume monthly)
ret = np.log(df / df.shift(1)).dropna()
mu = ret.mean() * 12             # annualized
Sigma = ret.cov() * 12           # annualized
mu_vec, Sigma_mat = mu.values, Sigma.values
n = len(mu_vec)

# Sharpe‐ratio portfolio via min‐variance @ unit excess return
w = cp.Variable(n)
objective = cp.Minimize(cp.quad_form(w, Sigma_mat))
constraints = [
    mu_vec @ w - r_f == 1,       # target 1% excess return
    w >= 0
]
prob = cp.Problem(objective, constraints)
prob.solve()

if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    raise RuntimeError("Optimization failed")

w_tilde = w.value
w_opt   = w_tilde / np.sum(w_tilde)  # fully invested, long‐only

print("Weights:", dict(zip(mu.index, w_opt.round(4))))






