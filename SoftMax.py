import scipy.io as sio
import numpy as np
from data_utils import load_data
from sgd import sgd


Yt, Ct, Yv, Cv = load_data("PeaksData.mat")
X = np.vstack([Yt, np.ones((1, Yt.shape[1]))])   # adding bias term
n = X.shape[0]
l = Ct.shape[0]

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

# print(softmax_loss(X, Ct, np.zeros((n, l))))
# print(softmax_loss(X, Ct, np.random.randn(n, l)))