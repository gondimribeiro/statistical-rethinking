import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 50
W = 3
T = 3

grid = np.linspace(0, 1, n)

prior = np.concatenate([np.zeros(int(n / 2)), 2*np.ones(int(n / 2))])

likelihood = ss.binom(T, grid).pmf(W)
unstd_posterior = likelihood * prior
posterior = unstd_posterior / unstd_posterior.sum()

plt.scatter(grid, posterior, s=50, alpha=0.5)
plt.show()
