import numpy as np


def simple_svm(X,Y, Lambda, opts):
    n, d = X.shape
    w = np.zeros((d, 1))
    alpha = np.zeros((n,1))
    primal_old = 0.

    for t in np.arange(opts["max_sdca_iters"]):
        for i in np.arange(n):
            alpha_old = alpha[i]
            curr_x = X[i]
            curr_y = Y[i]

            grad = Lambda * n * (1.0 - (curr_y*np.matmul(curr_x, w))) /\
                   np.matmul(curr_x, curr_x[:, None]) + (alpha_old * curr_y)

            new_alpha = curr_y * np.max([0., np.min([1., grad])])
            w = w + ((new_alpha - alpha_old)* curr_x[:,None] * \
                (1./(Lambda*n))
            )
            alpha[i] = new_alpha

        preds = Y*np.matmul(X, w)
        primal_new = (1./n)*np.mean(np.clip(1.-preds, 0., np.inf)) + \
                     Lambda*np.matmul(w.T, w)

        if np.abs(primal_old-primal_new) < opts["tol"]:
            break
        primal_old = primal_new

    return w
