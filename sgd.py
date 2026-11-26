import numpy as np

def sgd(F,X,C,W0,lr,batch_size,num_epochs,Xv=None,Cv=None,acc_fn=None,subsample_size=None,lr_decay=1.0,seed=None,):
    """_summary_
    Mini Batch Stochastic Gradient Decent optimiser
    This function minimizes a generic objective F(X_batch, C_batch, W) -> (loss, grad) over
    mini-batches of columns from X and C. With options for
    1) validation logging 
    2) random sampaling 
    3) exponential LR decay
    Args:
        F (callable): function with structure F(X_batch, C_batch, W) -> (loss, grad) 
        we wish to minimize
        X (np.ndarray , shape(n,m)):training input matrix n features including bias row, m samples
        C (np.ndarray , shape(l,m)): training target matrix l classes/outputs, m samples
        W0 (np.ndarray , shape(n,l)): initial weight matrix
        lr (float): initial learning rate
        batch_size (int): number of samples per mini-batch gradient update
        num_epochs (int): number of full passes over the training set
        Xv (np.ndarray, optional): validation inputs (n, m_v) and targets (l, m_v). Defaults to None.
        Cv (np.ndarray, optional): validation inputs (n, m_v) and targets (l, m_v). Defaults to None.
        acc_fn (callable, optional): metric function acc_fn(X_eval, C_eval, W) -> float returning accuracy (%) Defaults to None.
        subsample_size (int, optional): when set and set tosmaller than dataset size, evaluates epoch loss/accuracy on a
        random subset of this size to avoid full-dataset passes every epoch. Defaults to None.
        lr_decay (float, default=1.0): learnign decay factor mulitpliy the learning rate after each epoch. Defaults to 1.0.
        seed (int, optional): ranodm seed for reprodicable results. Defaults to None.

    Returns:
        W : np.ndarray, shape (n, l) , learend weight matrix after num_epochs number of epochs
        history : dict[str, list[float]] , Per-epoch logs for 'train_loss', 'val_loss', 'train_acc', and 'val_acc'
    """
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