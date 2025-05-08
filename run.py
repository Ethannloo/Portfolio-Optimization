from optimization import optimize_portfolio
from visual import plot_efficient_frontier, plot_weights_bar

w_opt, mu_vec, Sigma_mat, tickers = optimize_portfolio('data.csv')
plot_efficient_frontier(mu_vec, Sigma_mat)
plot_weights_bar(tickers, w_opt)
