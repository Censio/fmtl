
import fmtl.opt.run as run
import mock
import unittest
import numpy as np
from scipy.io import loadmat


class Test(unittest.TestCase):

    def setUp(self):
        self.DIR = "/Users/shokoryu/Packages/fmtl/"


    def test_get_delta(self):
        data = loadmat(self.DIR+"index/index1-1.mat")
        Xtrain = [i[0] for i in data["Xtrain"]]
        Xtest = [i[0] for i in data["Xtest"]]
        Ytrain = [i[0] for i in data["Ytrain"]]
        Ytest = [i[0] for i in data["Ytest"]]
        Lambda = 1.0000e-05
        opts = {
            "obj":"C",
            "avg":1,
            "mocha_outer_iters": 10,
            "mocha_inner_iters": 100,
            "mocha_sdca_frac": 0.50000,
            "max_sdca_iters": 500,
            "w_update": 0,
            "sys_het": 0,
            "tol": 1.e-5,
            "type": "global",
        }

        m = 29
        rho = 1.
        W = np.zeros((9,m))
        Sigma = np.eye(m) * (1./m)
        Omega = np.linalg.inv(Sigma)
        N = [len(i) for i in Ytrain]
        alpha = [np.zeros(n) for n in N]

        def get_index(a, b, replace, p):
            idx = data["III"]
            i = idx[idx[:,0]==p+1,-1].astype(int)-1
            return i

        @mock.patch("fmtl.opt.run.np.random.choice")
        def test_local_delta(W, X, Y, N, alpha, Sigma, Lambda, rho, opts, sys_iters, mock_np):
            mock_np.side_effect = get_index
            return run.run_local_delta(W, X, Y, N, alpha, Sigma, Lambda, rho, opts, sys_iters)

        t = 0
        new_a, new_delta_w, new_delta_b = test_local_delta(W[:,t],
                                                          Xtrain[t],
                                                          Ytrain[t],
                                                          N[t],
                                                          alpha[t],
                                                          Sigma[t,t],
                                                          Lambda,
                                                          rho, opts, t)
        self.assertTrue(np.isclose(new_a, data["alpha"][0][0].reshape(-1)).all())
        self.assertTrue(np.isclose(new_delta_w, data["deltaW"][:,0]).all())
        self.assertTrue(np.isclose(new_delta_b, data["deltaB"][:,0]).all())

        @mock.patch("fmtl.opt.run.np.random.choice")
        def test_delta(W, X, Y, N, alpha, Sigma, Lambda, rho, opts, mock_np):
            mock_np.side_effect = get_index
            return run.get_delta(W, X, Y, Sigma, alpha, N, Lambda, rho, opts)

        out_deltaW, out_deltaB, alpha = test_delta(W, Xtrain, Ytrain, N, alpha, Sigma, Lambda, rho, opts)

        self.assertTrue(np.isclose(out_deltaW, data["deltaW"]).all())
        self.assertTrue(np.isclose(out_deltaB, data["deltaB"]).all())

        opts["mocha_inner_iters"] = 1
        @mock.patch("fmtl.opt.run.np.random.choice")
        def test_updateW(W, Xtrain, Xtest, Ytrain, Ytest, Sigma, alpha, N, Lambda, trace, rho, opts, mock_np):
            mock_np.side_effect = get_index
            W = run.updateW(W, (Xtrain, Xtest), (Ytrain, Ytest), Sigma, alpha, N, Lambda, trace, rho, opts)
            return W

        trace = np.trace(np.matmul(W, np.matmul(Omega, W.T)))
        W = test_updateW(W, Xtrain, Xtest, Ytrain, Ytest, Sigma, alpha, N, Lambda, trace, rho, opts,)
        self.assertTrue(np.isclose(W, data["W"]).all())
