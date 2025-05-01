from data import *
import numpy as np
import cvxpy as cp

exp_returns = mu.values
cov_matrix = sigma.values

num = len(exp_returns)
target = .01

w = cp.Variable(num)
portfolio_variance = cp.quad_form(w, cov_matrix)
objective = cp.Minimize(portfolio_variance)


contraints = [cp.sum(w)==1, exp_returns @ w >= target, w >= 0]

problem = cp.Problem(objective, contraints)
problem.solve()


def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights)*12
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)






