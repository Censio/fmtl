import numpy as np
import scipy.linalg as slinalg
from fmtl.util import compute

def run_mocha(Xtrain, Ytrain, Xtest, Ytest, Lambda, opts):
    m = len(Xtrain)
    d = Xtrain[0].shape[1]
    rho = 1.0

    W = np.zeros((d,m))
    Sigma = np.eye(m)*1./m
    Omega = np.linalg.inv(Sigma)
    trace = np.trace(np.matmul(W, np.matmul(Omega, W.T)))

    alpha = []
    totaln = 0
    n = []
    for i in xrange(m):
        n.append(len(Ytrain[i]))
        totaln = totaln + n[i]
        alpha.append(np.zeros((int(n[i]), 1)))

    if opts["w_update"]:
        rmse = np.zeros((opts["mocha_inner_iters"],1))
        dual_objs = np.zeros((opts["mocha_inner_iters"],1))
        primal_objs = np.zeros((opts["mocha_inner_iters"],1))
    else:
        rmse = np.zeros((opts["mocha_outer_iters"],1))
        dual_objs = np.zeros((opts["mocha_outer_iters"],1))
        primal_objs = np.zeros((opts["mocha_outer_iters"],1))

    for h in xrange(opts["mocha_outer_iters"]):
        if not opts["w_update"]:
            curr_err = compute.compute_rmse(Xtest, Ytest, W, opts)
            rmse[h] = curr_err
            primal_objs[h] = compute.compute_primal(Xtrain, Ytrain, W, trace, Lambda)
            dual_objs[h] = compute.compute_dual(alpha, Ytrain, W, trace, Lambda)

        W = updateW(W, (Xtrain, Xtest), (Ytrain, Ytest), Sigma, alpha, n, Lambda, trace, rho, opts)
        Sigma, trace, rho = updateOmega(W)

    return rmse, primal_objs, dual_objs

def updateOmega(W):
    A = check_W(W)
    sqm = slinalg.sqrtm(A)
    Sigma = sqm / np.trace(sqm)
    trace = get_trace(Sigma, W)
    rho = np.max(np.sum(np.abs(Sigma), 1) / np.diag(Sigma))
    return Sigma, trace, rho

def updateW(W, X, Y, Sigma, alpha, n, Lambda, trace, rho, opts):
    Xtrain, Xtest = X
    Ytrain, Ytest = Y
    m = Sigma.shape[0]

    rmse = []
    primal_objs = []
    dual_objs = []

    for hh in xrange(int(opts["mocha_inner_iters"])):
        np.random.seed(hh*1000)
        if opts["sys_het"]:
            sys_iters = (opts["top"] - opts["bottom"]) * np.random.rand(m,1) + opts["bottom"]

        if opts["w_update"]:
            rmse[hh] = compute.compute_rmse(Xtest, Ytest, W, opts)
            primal_objs[hh] = compute.compute_primal(Xtrain, Ytrain, W, trace, Lambda)
            dual_objs[hh] = compute.compute_dual(alpha, Ytrain, W, trace, Lambda)

        deltaW, deltaB = get_delta(W, Xtrain, Ytrain, Sigma, alpha, n, Lambda, rho, opts, hh)
        W = _updateW(W, deltaB, Sigma, Lambda)
    return W

def get_delta(W, X, Y, Sigma, alpha, n, Lambda, rho, opts, sys_iters=None):
    """
    X: Xtrain
    Y: Ytrain
    """
    d,m = W.shape
    deltaW = np.zeros((d,m))
    deltaB = np.zeros((d,m))

    for t in xrange(m):
        tperm = np.random.choice(n[t], n[t], replace=False)
        alpha_t = alpha[t]
        curr_sig = Sigma[t,t]

        if opts["sys_het"]:
            local_iters = n[t]*sys_iters[t]
        else:
            local_iters = n[t]*opts["mocha_sdca_frac"]

        for s in xrange(int(local_iters)):
            idx = tperm[np.mod(s, n[t])]
            alpha_old = alpha_t[idx]
            curr_y = Y[t][idx]
            curr_x = X[t][idx]

            new_alpha, new_delta_b, new_delta_w = _update_delta(
                n=n[t], Lambda=Lambda, rho=rho,
                current_x=curr_x, current_y=curr_y,
                w=W[:,t], delta_w=deltaW[:,t],
                delta_b=deltaB[:,t],
                alpha_old=alpha_old,
                current_sig=curr_sig
            )
            alpha_t[idx] = new_alpha
            deltaW[:, t] = new_delta_w
            deltaB[:, t] = new_delta_b
            alpha[t] = alpha_t

    return deltaW, deltaB

def _updateW(W, deltaB, Sigma, Lambda):
    m = W.shape[1]
    for t in xrange(m):
        delta = (deltaB * Sigma[t] * (1./Lambda)).sum(1)
        W[:,t] = W[:,t] + delta
    return W

def _update_delta(n, Lambda, rho,
            current_x, current_y,
            w, delta_w,
            delta_b,
            alpha_old,
            current_sig):
    current_x = np.atleast_2d(current_x)
    w = w[:,None]
    delta_w = delta_w[:,None]
    delta_b = delta_b[:,None]
    update = current_y * np.matmul(
        current_x, (w + rho*delta_w))
    grad = Lambda * n * (1.-update) / \
           (current_sig*rho*np.matmul(current_x, current_x.T)) + \
            (alpha_old*current_y)
    grad = np.real(grad)
    new_alpha = current_y*np.max([0, np.min([1, grad])])
    new_delta_w = delta_w + \
                  current_sig * (new_alpha - alpha_old)*current_x.T /\
                  (Lambda*n)
    new_delta_b = delta_b + (new_alpha - alpha_old)*current_x.T / n
    new_alpha = np.real(new_alpha)
    new_delta_b = np.real(new_delta_b)
    new_delta_w = np.real(new_delta_w)
    return new_alpha, new_delta_b.reshape(-1), new_delta_w.reshape(-1)

def check_W(W):
    A = np.matmul(W.T, W)
    d, V = np.linalg.eig(A)
    if np.any(d < 0):
        d[d<=1e-7] = 1e-7
        D = np.diag(d)
        A = np.matmul(V, np.matmul(D, V.T))
    A = np.real(A)
    return A

def get_trace(Sigma, W):
    """
    equation
        Omega^{-1}W.T = Y
    solve for Y by
        W.T = OmegaY
    """
    Omegainv_Wtrans = slinalg.solve(Sigma, W.T )
    trace_ = np.trace(np.matmul(W, Omegainv_Wtrans))
    return trace_
