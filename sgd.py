import numpy as np

def sgd(F, X, C, W0, lr, batch_size, num_epochs):
    W = W0.copy()
    for _ in range(num_epochs):
        # Shuffle the data
        indices = np.random.permutation(X.shape[1])
        X_shuffled = X[:, indices]
        C_shuffled = C[:, indices]

        # Perform SGD on each batch
        for i in range(0, X.shape[1], batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            C_batch = C_shuffled[:, i:i+batch_size]
            loss, grad = F(X_batch, C_batch, W)
            W -= lr * grad
    return W