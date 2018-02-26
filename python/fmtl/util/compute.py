import numpy as np
import scipy.linalg as sl

def compute_dual(alpha, Y, W, trace, Lambda):
    total_alpha = 0.
    for a, y in zip(alpha, Y):
        y = np.ravel(y)
        total_alpha = total_alpha + np.mean(-1 * a.reshape(-1)* y)

    #trace #np.trace(np.matmul(W, np.matmul(Omega, W.T)))
    dual_obj = -Lambda / 2 * trace - total_alpha
    return dual_obj

def compute_primal(X, Y, W, trace, Lambda):
    """
    X = list of m, xi = n-by-d
    W = d-by-m
    """
    total_loss = 0.
    total_norm = 0.
    for x,y,w in zip(X,Y,W.T):
        y = np.ravel(y)
        p = np.matmul(x, w[:,None]).reshape(-1)
        preds = (y*p).reshape(-1)
        total_loss = total_loss + np.mean(np.clip(1-preds, 0, np.inf))
        i = (y-p)**2
        total_norm += np.mean(i)
    #trace #np.trace(np.matmul(W, np.matmul(Omega, W.T)))
    primal_obj = total_loss + Lambda / 2 * trace
    return primal_obj

def compute_rmse(X,Y,W,opts):
    m = W.shape[1]
    obj = opts["obj"]

    Y_hat = []
    for i in xrange(m):
        if obj == "R":
            y = np.matmul(X[i], W[:,i]).reshape(-1)
        else:
            y = np.sign(np.matmul(X[i], W[:,i]).reshape(-1))
        Y_hat.append(y)

    if opts["avg"]:
        all_errs = []
        for i in xrange(m):
            if obj == "R":
                e = np.sqrt(np.mean(np.square(Y[i] - Y_hat[i])))
            else:
                e = np.mean(Y[i] != Y_hat[i])
            all_errs.append(e)
        err = np.mean(all_errs)
    else:
        Y = np.hstack(np.concatenate(Y))
        Y_hat = np.hstack(np.ravel(Y_hat))

        if obj == "R":
            err = np.sqrt(np.mean(np.square(Y-Y_hat)))
        else:
            err = np.mean(Y != Y_hat)
    return err

def sqrtm_utri_inplace(T):
    M = T.copy()
    row = M.shape[0]
    for jj in xrange(row):
        #print "diag: {}".format(M[jj,jj])
        M[jj,jj] = np.sqrt(M[jj,jj])
        for i in reversed(xrange(jj)):
            M[i,jj] /= (M[i,i] + M[jj,jj])
            k = np.arange(i)
            M[k,jj] -= M[k,i]*M[i,jj]
    M = np.real(M)
    return M

def sqrtm_utri_inplace_(T):
    M = np.zeros(T.shape, T.dtype)
    row = M.shape[0]
    for jj in xrange(row):
        M[jj,jj] = np.sqrt(T[jj,jj])
        for i in reversed(xrange(jj)):
            k = np.arange(i,jj)
            s = np.dot(M[i,k], M[k,jj])
            M[i,jj] = (T[i,jj]-s)/(M[i,i]+M[jj,jj])
    return M

def sqrtm(M):
    T,Z = sl.schur(M)
    T,Z = sl.rsf2csf(T,Z)
    C = sqrtm_utri_inplace_(T)
    sqm = np.real(np.matmul(Z,np.matmul(C, Z.T)))
    return sqm
