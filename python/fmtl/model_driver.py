
import os
import numpy as np
from scipy.io import loadmat

from fmtl.opt import run
from fmtl.util import cross_val, baselines

data = loadmat(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/small.mat")
))

X = []
Y = []
for x,y in zip(data["X"][0], data["Y"][0]):
    X.append(x)
    Y.append(y)


ntrials = 1 # number of trials to run
training_percent = 0.75 # percentage of data for training
opts = {}

opts["obj"]='C' # classification
opts["avg"] = 1 # compute average error across tasks

lambda_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10]

for i in xrange(10):
    SEED = i
    Xtrain, Xtest, Ytrain, Ytest = cross_val.split_data(X, Y, training_percent, SEED)

    opts["type"] = "local"
    opts["max_sdca_iters"] = 500
    opts["tol"] = 1e-5

    opts["mocha_outer_iters"] = 10
    opts["mocha_inner_iters"] = 100
    opts["mocha_sdca_frac"] = 0.5
    opts["mocha_w_update"] = 0
    opts["sys_het"] = 0
    opts["w_update"] = 0

    if False:
        global_lambda = cross_val.cross_validation_once(Xtrain,
                                                   Ytrain,
                                                   lambda_range=lambda_range,
                                                   cv_fold=5,
                                                   opts=opts,
                                                   func=baselines.baselines,
                                                   seed=0
    )
    else:
        global_lambda = 0.0001
        local_lambda = 1e-05 # cost: 0.2700291690897007
    print "lambda {}".format(global_lambda)
    err = baselines.baselines(Xtrain, Ytrain, Xtest, Ytest, global_lambda, opts)
    print "local error: {}".format(err)

    print "==="

    if False:
        mocha_lambda = cross_val.cross_validation_once(Xtrain,
                                                   Ytrain,
                                                   lambda_range=lambda_range,
                                                   cv_fold=5,
                                                   opts=opts,
                                                   func=run.run_mocha,
                                                   seed=0
    )
    else:
        mocha_lambda = 1.0000e-05
    print "lambda {}".format(mocha_lambda)

    rmse_mocha, primal_mocha, dual_mocha, W, Sigma = run.run_mocha(Xtrain,
                                                   Ytrain,
                                                   Xtest,
                                                   Ytest,
                                                   mocha_lambda,
                                                   opts)
    print "mocha error:{}".format(rmse_mocha)

    rmse_mocha, primal_mocha, dual_mocha, W, Sigma = run.run_mocha(Xtrain[:-1],
                                                   Ytrain[:-1],
                                                   Xtest[:-1],
                                                   Ytest[:-1],
                                                   mocha_lambda,
                                                   opts)
    print "mocha error:{}".format(rmse_mocha)

    m = len(Xtrain)
    d = W.shape[0]
    tr = np.trace(Sigma)
    s = 0.05
    SSigma = np.eye(m)*s
    SSigma[:-1, :-1] = (1-s)*Sigma
    import ipdb; ipdb.set_trace()
    W = np.hstack((W, np.zeros((d,1))))
    opts["mocha_outer_iters"] = 20
    rmse_mocha, primal_mocha, dual_mocha, W, Sigma = run.run_mocha(Xtrain,
                                                                   Ytrain,
                                                                   Xtest,
                                                                   Ytest,
                                                                   mocha_lambda,
                                                                   opts,
                                                                   W, SSigma)
    print "mocha error:{}".format(rmse_mocha)
