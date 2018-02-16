import numpy as np

def compute_dual(alpha, Y, W, trace, Lambda):
    total_alpha = 0.
    for a, y in zip(alpha, Y):
        total_alpha = total_alpha + np.mean(-1 * a.reshape(-1)* y.reshape(-1))

    den = trace #np.trace(np.matmul(W, np.matmul(Omega, W.T)))
    dual_obj = -Lambda / 2 * den - total_alpha
    return dual_obj[0]

def compute_primal(X, Y, W, trace, Lambda):
    """
    X = list of m, xi = n-by-d
    W = d-by-m
    """
    total_loss = 0.
    for x,y,w in zip(X,Y,W.T):
        p = np.matmul(x, w[:,None]).reshape(-1)
        preds = (y*p).reshape(-1)
        total_loss = total_loss + np.mean(np.clip(1-preds, 0, np.inf))

    den = trace #np.trace(np.matmul(W, np.matmul(Omega, W.T)))
    primal_obj = total_loss + Lambda / 2*den
    return primal_obj

def compute_rmse(X,Y,W,opts):
    m = len(X)
    obj = opts["obj"]

    Y_hat = []
    for i in xrange(m):
        if obj == "R":
            y = np.matmul(X[i], W[:,i]).reshape(-1)
        else:
            y = np.sign(np.matmul(X[i], W[:,i]).reshape(-1))
        Y_hat.append(y)

    if opts["ave"]:
        all_errs = np.zeros(m, 1)
        for i in xrange(m):
            if obj == "R":
                e = np.sqrt(np.mean(np.square(Y[i] - Y_hat[i])))
            else:
                e = np.mean(Y[i] != Y_hat[i])
            all_errs.append(e)
        err = np.mean(all_errs)
    else:
        Y = np.hstack(np.ravel(Y))
        Y_hat = np.hstack(np.ravel(Y_hat))

        if obj == "R":
            err = np.sqrt(np.mean(np.square(Y-Y_hat)))
        else:
            err = np.mean(Y != Y_hat)
    return err
