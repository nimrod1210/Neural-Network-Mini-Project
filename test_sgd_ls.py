import numpy as np
import matplotlib.pyplot as plt
from SoftMax import add_bias
from Grad_test import gradient_test
from sgd import sgd

def least_squares_loss(X: np.ndarray, C: np.ndarray, W: np.ndarray):
    """
    Mean-squared Frobenius regression objective and analytical gradient:
        F(X, C, W) = (1 / 2m) * ||X^T W - C^T||_F^2

    Args:
        X (np.ndarray): shape (n, m) bias-augmented input matrix (n features, m samples)
        C (np.ndarray): shape (l, m) continuous target matrix (l outputs, m samples)
        W (np.ndarray): shape (n, l) weight matrix (including absorbed bias row)
    Returns:
        loss (float): scalar meanes squared loss
        grad np.ndarray, shape (n, l) : gradient dF/dW = (1 / m) * X @ (X^T W - C^T)
    """
    m = X.shape[1]
    res = X.T @ W - C.T
    loss = 0.5 * np.sum(res ** 2) / m
    grad = (X @ res) / m  
    return float(loss), grad

if __name__ == "__main__":
    # Fix seed for reproducible synthetic data and convergence curve
    np.random.seed(42)
    d, l, m = 5, 3, 500

    # Build synthetic linear regression dataset C = W_true^T X + noise
    Y = np.random.randn(d, m)               # (5, 500) raw features
    X = add_bias(Y)                         # (6, 500) append row of ones for bias
    W_true = np.random.randn(d + 1, l)      # (6, 3) ground-truth weights + bias
    noise = 0.01 * np.random.randn(l, m)    # small Gaussian observation noise
    C = (X.T @ W_true).T + noise            # (3, 500) noisy target outputs

    # Verify least-squares analytical gradient before running optimization
    # Expected 1st-order Taylor error ratios ~ 4.0 when halving epsilon
    W0 = np.zeros((d + 1, l))
    _, first_ls = gradient_test(least_squares_loss, X, C, np.random.randn(d + 1, l), num_iters=6)
    print(
        "LS first-order ratios (expected ~4.0):",
        [round(first_ls[i] / first_ls[i + 1], 3) for i in range(len(first_ls) - 1)],
    )

    # Minimize least-squares objective with mini-batch SGD to recover W_true
    init_loss, _ = least_squares_loss(X, C, W0)
    W_est, history = sgd(
        least_squares_loss,
        X,
        C,
        W0,
        lr=0.1,
        batch_size=32,
        num_epochs=80,
        seed=42,
    )

    # Compare final loss against noise floor (~0.5 * l * sigma^2 = 1.5e-4)
    # and check relative Frobenius recovery error against W_true
    final_loss = history["train_loss"][-1]
    rel_err = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)

    print(f"Initial loss: {init_loss:.6f} -> Final loss: {final_loss:.6e}")
    print(f"Relative weight error ||W_est - W_true|| / ||W_true||: {rel_err:.6f}")

    # Plot per-epoch loss on a log y-axis to verify linear convergence rate
    plt.figure(figsize=(7, 5))
    plt.semilogy(range(1, len(history["train_loss"]) + 1), history["train_loss"],"o-",markersize=3,)
    plt.xlabel("Epoch")
    plt.ylabel("Least Squares Loss (log scale)")
    plt.title("SGD Convergence on Synthetic Least Squares")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("sgd_least_squares.png", dpi=150)
    plt.show()