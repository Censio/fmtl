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

        if False:
            print "running {}".format(h)
            print "rmse {}".format(rmse[h])
            print "norm: {}".format(np.linalg.norm(W))
        Sigma, trace, rho = updateOmega(W)

    return rmse, primal_objs, dual_objs, W, Sigma

def updateOmega(W):
    ## attempt1:
    ## sqrt(diagonal) of svd to get sqrt matrix
    # d,v = check_W(W)
    # sqm = np.matmul(v,np.matmul(np.sqrt(d), v.T))

    ## attempt2:
    ## use scipy sqrtm
    A = check_W(W)
    sqm = slinalg.sqrtm(A)

    ## attemp3:
    ## use Octave sqrtm
    # A = check_W(W)
    # sqm = compute.sqrtm(A)

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
        else:
            sys_iters = None

        if opts["w_update"]:
            rmse[hh] = compute.compute_rmse(Xtest, Ytest, W, opts)
            primal_objs[hh] = compute.compute_primal(Xtrain, Ytrain, W, trace, Lambda)
            dual_objs[hh] = compute.compute_dual(alpha, Ytrain, W, trace, Lambda)

        deltaW, deltaB, alpha = get_delta(
            W = W,
            X = Xtrain,
            Y = Ytrain,
            Sigma = Sigma,
            alpha = alpha,
            n = n,
            Lambda = Lambda,
            rho = rho,
            opts = opts,
            sys_iters = sys_iters)

        W = _updateW(W, deltaB, Sigma, Lambda)
    return W

def run_local_delta(w, x, y, n, a, S, Lambda, rho, opts, sys_iters=None):
    tperm = np.random.choice(n, n, replace=False,
                             # p=sys_iters ## for testing purpose
    )
    if opts["sys_het"]:
        local_iters = n*sys_iters
    else:
        local_iters = n*opts["mocha_sdca_frac"]

    delta_w = np.zeros(w.shape)
    delta_b = np.zeros(w.shape)
    alpha = np.zeros(a.shape)
    for s in xrange(int(local_iters)):
        idx = tperm[np.mod(s, n)]
        alpha_old = a[idx]
        curr_y = y[idx]
        curr_x = x[idx]
        new_alpha, new_delta_b, new_delta_w = _update_delta(
            n=n, Lambda=Lambda, rho=rho,
            current_x=curr_x, current_y=curr_y,
            w=w, delta_w=delta_w,
            delta_b=delta_b,
            alpha_old=alpha_old,
            current_sig=S
        )
        alpha[idx] = new_alpha
        delta_w = new_delta_w
        delta_b = new_delta_b
    return alpha, delta_w, delta_b

def get_delta(W, X, Y, Sigma, alpha, n, Lambda, rho, opts, sys_iters=None):
    d,m = W.shape

    deltaW = []
    deltaB = []
    for t in xrange(m):
        new_a, new_delta_w, new_delta_b = run_local_delta(w=W[:,t],
                                                          x=X[t],
                                                          y=Y[t],
                                                          n=n[t],
                                                          a=alpha[t],
                                                          S=Sigma[t,t],
                                                          Lambda=Lambda,
                                                          rho=rho, opts=opts, sys_iters=sys_iters)
        alpha[t] = new_a
        deltaW.append(new_delta_w)
        deltaB.append(new_delta_b)

    deltaW = np.vstack(deltaW).T
    deltaB = np.vstack(deltaB).T
    return deltaW, deltaB, alpha

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
        D = np.real(np.diag(d))
        V = np.real(V)
        A = np.matmul(V, np.matmul(D, V.T))
    return A

def get_trace(Sigma, W):
    """
    numerically stable inversion
    equation
        Omega^{-1}W.T = Y
    solve for Y by
        W.T = OmegaY
    """
    ## direct inversion
    # Omega = np.linalg.inv(Sigma)
    # Omegainv_Wtrans = np.matmul(Omega, W.T)

    ## numerically stable inversion
    Omega =  Omegainv_Wtrans = slinalg.solve(Sigma, W.T )
    trace_ = np.real(np.trace(np.matmul(W, Omegainv_Wtrans)))
    return trace_
