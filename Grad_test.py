import numpy as np
import matplotlib.pyplot as plt
from SoftMax import softmax_loss
from sgd import sgd
from data_utils import load_data


Yt, Ct, Yv, Cv = load_data("PeaksData.mat")

X = np.vstack([Yt, np.ones((1, Yt.shape[1]))])   # adding bias term
n = X.shape[0]   # 3
l = Ct.shape[0]  # 5


def gradient_test(F, X, C, W, num_iters=10, eps0=1.0):
    """
    F: הפונקציה שבודקים, softmax_loss, שמחזירה (loss, grad)
    """
    f0, g0 = F(X, C, W)
    D = np.random.randn(W.shape[0], W.shape[1])                      # like W
    D = D / np.linalg.norm(D)     # מנרמלים, כדי ש-‖D‖ = 1 (התנאי ‖d‖ = O(1) בשקף)

    zero_order = []      # (5)
    first_order = []     # (6)
    eps = eps0
    for i in range(num_iters):
        f_eps, _ = F(X, C, W + eps * D)
        zero_order.append(abs(f_eps - f0))                              # (5)
        first_order.append(abs(f_eps - f0 - eps * np.sum(g0 * D)))      # (6)
        eps = eps / 2                           
    return zero_order, first_order


W = np.random.randn(n, l)
zero, first = gradient_test(softmax_loss, X, Ct, W)
for i in range(len(zero) - 1):
    print(zero[i] / zero[i+1], first[i] / first[i+1])


np.random.seed(0)
W = np.random.randn(n, l)
zero, first = gradient_test(softmax_loss, X, Ct, W)


## plotting

np.random.seed(0)
W = np.random.randn(n, l)
eps0 = 1.0
num_iters = 10
zero, first = gradient_test(softmax_loss, X, Ct, W, num_iters=num_iters, eps0=eps0)

eps_values = [eps0 * 0.5**i for i in range(num_iters)]

plt.figure(figsize=(7, 5))
plt.loglog(eps_values, zero, 'o-', label=r'$|F(W+\epsilon d) - F(W)|$  — expected $O(\epsilon)$')
plt.loglog(eps_values, first, 's-', label=r'$|F(W+\epsilon d) - F(W) - \epsilon\, d^T \nabla F|$  — expected $O(\epsilon^2)$')
plt.gca().invert_xaxis()          # ε קטן משמאל לימין, כמו התקדמות האיטרציות
plt.xlabel(r'$\epsilon$')
plt.ylabel('error')
plt.title('Gradient test — softmax loss w.r.t. W')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("grad_test_softmax.png", dpi=150)
plt.show()

