import numpy as np
from data_utils import load_data


def add_bias(Y: np.ndarray) -> np.ndarray:
    """
    Append a row of ones to Y of shape (d, m) -> (d + 1, m) to absorb bias into W
    """
    return np.vstack([Y,np.ones((1,Y.shape[1]),dtype=Y.dtype)])

def softmax_loss(X, C, W):
    m = X.shape[1]
    Z = X.T @ W        # (m, n) @ (n, l) = (m, l)
    Z = Z - np.max(Z, axis=1, keepdims=True)  # DeppLearning.pdf - 1.2 - Remark 2 - Computing extreme exponentials “safely”
    ez = np.exp(Z)
    s=np.sum(ez, axis=1, keepdims=True)
    P = ez / s      # softmax
    log_P = Z - np.log(s) # softmax log
    loss = -np.sum(C.T * log_P) / m        # safe log
    grad = X @ (P - C.T) / m                              # softmax gradient
    return loss, grad

def softmax_accuracy(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> float:
    """success is defined by any percentage if success"""
    scores = X.T @ W
    preds = np.argmax(scores, axis=1)
    labels = np.argmax(C, axis=0)
    return float(np.mean(preds == labels) * 100.0)

if __name__ == "__main__":
    Yt, Ct, Yv, Cv = load_data("PeaksData.mat")
    X = np.vstack([Yt, np.ones((1, Yt.shape[1]))])   # adding bias term
    n = X.shape[0]
    l = Ct.shape[0]
    print("Zero weights loss:",softmax_loss(X, Ct, np.zeros((n, l))))
    print("Random weights loss:",softmax_loss(X, Ct, np.random.randn(n, l)))