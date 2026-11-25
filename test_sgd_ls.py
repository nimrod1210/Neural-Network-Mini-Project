import numpy as np
import matplotlib.pyplot as plt
from SoftMax import add_bias
from Grad_test import gradient_test
from sgd import sgd

def least_squares_loss(X: np.ndarray, C: np.ndarray, W: np.ndarray):
    """
    F(X, C, W) = (1 / 2m) * ||X^T W - C^T||_F^2
    X: (n, m), C: (l, m), W: (n, l)
    """
    m = X.shape[1]
    res = X.T @ W - C.T
    loss = 0.5 * np.sum(res ** 2) / m
    grad = (X @ res) / m  
    return float(loss), grad