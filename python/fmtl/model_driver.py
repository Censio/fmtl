
from fmtl.opt import run
from fmtl.util import cross_val
from scipy.io import loadmat

X, Y

SEED = 0
ntrials = 1 # number of trials to run
training_percent = 0.75 # percentage of data for training
opts = {}

opts["obj"]='C' # classification
opts["avg"] = 1 # compute average error across tasks

lambda_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10]

for _ in xrange(ntrials):
    Xtrain, Xtest, Ytrain, Ytest = cross_val.split_data(X, Y, training_percent, SEED)

    opts["mocha_outer_iters"] = 10
    opts["mocha_inner_iters"] = 100
    opts["mocha_sdca_frac"] = 0.5
    opts["mocha_w_update"] = 0
    opts["sys_het"] = 0
    mocha_lambda = cross_val.cross_validation_once(Xtrain,
                                  Ytrain,
                                  run.run_mocha,
                                  opts,
                                  lambda_range,
                                  5
    )
    rmse_mocha, primal_mocha, dual_mocha = run.run_mocha(Xtrain,
                                                   Ytrain,
                                                   Xtest,
                                                   Ytest,
                                                   mocha_lambda,
                                                   opts)
