import numpy as np
import matplotlib.pyplot as plt
from SoftMax import softmax_loss, add_bias
from data_utils import load_data


def gradient_test(F, X, C, W, num_iters=10, eps0=1.0):
    """
    F: הפונקציה שבודקים, softmax_loss, שמחזירה (loss, grad)
    """
    f0, g0 = F(X, C, W)
    D = np.random.randn(*W.shape)    # like W used unpack symbol will work for more
    #matrices of more dimensions then 2d
    D = D / np.linalg.norm(D)     # מנרמלים, כדי ש-‖D‖ = 1 (התנאי ‖d‖ = O(1) בשקף)

    zero_order = []      # (5)
    first_order = []     # (6)
    eps = eps0
    for i in range(num_iters):
        f_eps, _ = F(X, C, W + eps * D)
        zero_order.append(abs(f_eps - f0))                              # (5)
        first_order.append(abs(f_eps - f0 - eps * np.sum(g0 * D)))      # (6)
        eps = eps / 2.0                           
    return zero_order, first_order


if __name__ == "__main__":
    np.random.seed(0)
    Yt, Ct, _, _ = load_data("PeaksData.mat")
    X = add_bias(Yt)
    n, l = X.shape[0], Ct.shape[0]
    W = np.random.randn(n, l)

    eps0 = 1.0
    num_iters = 10
    zero, first = gradient_test(softmax_loss, X, Ct, W, num_iters=num_iters, eps0=eps0)

    print("Zero-order ratios  (expected ~2.0):", [round(zero[i] / zero[i + 1], 3) for i in range(len(zero) - 1)])
    print("First-order ratios (expected ~4.0):", [round(first[i] / first[i + 1], 3) for i in range(len(first) - 1)])

    eps_values = [eps0 * (0.5 ** i) for i in range(num_iters)]

    plt.figure(figsize=(7, 5))
    plt.loglog(eps_values, zero, "o-", label=r"$|F(W+\epsilon D) - F(W)|$ — expected $O(\epsilon)$")
    plt.loglog(eps_values, first, "s-", label=r"$|F(W+\epsilon D) - F(W) - \epsilon\langle\nabla F, D\rangle|$ — expected $O(\epsilon^2)$")
    plt.gca().invert_xaxis()
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Error")
    plt.title("Gradient Test — Softmax Loss wrt W")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("grad_test_softmax.png", dpi=150)
    plt.show()