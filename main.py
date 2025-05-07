import numpy as np
import matplotlib.pyplot as plt


n = len(mu_vec)
num_sims = 5000
vols_sim = np.zeros(num_sims)
rets_sim = np.zeros(num_sims)

for i in range(num_sims):
    w_rand = np.random.rand(n)
    w_rand /= w_rand.sum()           
    rets_sim[i] = mu_vec @ w_rand
    vols_sim[i] = np.sqrt(w_rand.T @ Sigma_mat @ w_rand)

plt.figure()
plt.scatter(vols_sim, rets_sim, alpha=0.2)
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.title("Monte Carlo Portfolios")
plt.grid(True)

plt.plot(vols, rets, linestyle='--', linewidth=2)


opt_vol = np.sqrt(w_opt.T @ Sigma_mat @ w_opt)
opt_ret = mu_vec @ w_opt
plt.scatter([opt_vol], [opt_ret], marker='*', s=200)

plt.show()

plt.figure()
plt.bar(mu.index, w_opt)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Weight")
plt.title("Optimal Portfolio Weights")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
