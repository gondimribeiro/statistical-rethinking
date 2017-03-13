import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 20
grid = np.linspace(0, 1, n)
prior = np.ones(n)
likelihood = ss.binom(9, grid).pmf(6)
unstd_posterior = likelihood * prior
posterior = unstd_posterior / unstd_posterior.sum()

plt.scatter(grid, posterior, s=50, alpha=0.5)
plt.show()
