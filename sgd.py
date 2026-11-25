import numpy as np

def sgd(F,X,C,W0,lr,batch_size,num_epochs,Xv=None,Cv=None,acc_fn=None,subsample_size=None,lr_decay=1.0,seed=None,):
    
    rng = np.random.default_rng(seed)
    W = W0.copy().astype(float)
    m = X.shape[1]
    current_lr = lr

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
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
        #added here to track training metrics
        if subsample_size is not None and subsample_size < m:
            sub_idx = rng.choice(m, size=subsample_size, replace=False)
            X_eval, C_eval = X[:, sub_idx], C[:, sub_idx]
        else:
            X_eval, C_eval = X, C

        t_loss, _ = F(X_eval, C_eval, W)
        history["train_loss"].append(t_loss)
        if acc_fn is not None:
            history["train_acc"].append(acc_fn(X_eval, C_eval, W))

        # Track validation metrics if provided
        if Xv is not None and Cv is not None:
            mv = Xv.shape[1]
            if subsample_size is not None and subsample_size < mv:
                v_idx = rng.choice(mv, size=subsample_size, replace=False)
                Xv_eval, Cv_eval = Xv[:, v_idx], Cv[:, v_idx]
            else:
                Xv_eval, Cv_eval = Xv, Cv

            v_loss, _ = F(Xv_eval, Cv_eval, W)
            history["val_loss"].append(v_loss)
            if acc_fn is not None:
                history["val_acc"].append(acc_fn(Xv_eval, Cv_eval, W))

        current_lr *= lr_decay
    return W, history