import numpy as np
import matplotlib.pyplot as plt


def lwr(x, X, y, tau):
    W = np.diag(np.exp(-np.sum((X - x) ** 2, axis=1) / (2 * tau**2)))
    theta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return x @ theta


np.random.seed(42)

X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

Xb = np.c_[np.ones(len(X)), X]
xt = np.linspace(0, 2 * np.pi, 200)
xtb = np.c_[np.ones(len(xt)), xt]

tau = 0.5
yp = np.array([lwr(x, Xb, y, tau) for x in xtb])

plt.scatter(X, y, c="r", label="Data")
plt.plot(xt, yp, label=f"LWR (τ={tau})")
plt.legend()
plt.grid()
plt.show()
