import numpy as np


def split_data(X, Y, ratio, seed=None):
    np.random.seed(seed)

    cvxtr = []
    cvxte = []
    cvytr = []
    cvyte = []
    for x,y in zip(X, Y):
        n = len(y)
        i = np.arange(n)
        te_idx = np.random.choice(i, int(n*ratio), replace=False)
        tr_idx = np.setxor1d(i, te_idx)

        cvxtr.append(x[tr_idx])
        cvxte.append(x[te_idx])
        cvytr.append(y[tr_idx])
        cvyte.append(y[te_idx])

    return cvxtr, cvxte, cvytr, cvyte

def cross_validation_once(X, Y, lambda_range, cv_fold, func, opts, seed=None):
    """
    X, Y: list of array
    """
    np.random.seed(seed)

    task_num = len(X)
    ratio = 1./cv_fold
    perf_vec = np.zeros(len(lambda_range))

    for cv in xrange(cv_fold):
        cv_Xtr, cv_Xte, cv_Ytr, cv_Yte = split_data(X, Y, ratio, seed)

        for i, l in enumerate(lambda_range):
            curr_rmse, primal_obj, dual_obj = func(
                cv_Xtr,
                cv_Ytr,
                cv_Xte,
                cv_Yte,
                l, opts)
            perf_vec[i] += curr_rmse

    idx = np.argmin(perf_vec)
    best_lambda = lambda_range[idx]
    return best_lambda
