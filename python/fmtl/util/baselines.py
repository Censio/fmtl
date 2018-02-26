
import numpy as np

from fmtl.opt.simple_svm import simple_svm

def baselines(Xtrain, Ytrain, Xtest, Ytest, Lambda, opts):
    m = len(Xtrain)
    d = Xtrain[0].shape[1]

    if opts["type"] == "global":
        allX = np.vstack(Xtrain)
        allXtest = np.vstack(Xtest)
        allY = np.vstack(Ytrain)
        allYtest = np.vstack(Ytest)

        if opts["obj"] == "C":
            w = simple_svm(allX, allY, Lambda, opts)
            if opts["avg"]:
                errs = np.zeros((m,1))
                for i in np.arange(m):
                    predvals = np.sign(np.matmul(Xtest[i], w))
                    errs[i] = np.mean(predvals != Ytest[i])

                err = np.mean(errs)

    elif opts["type"] == "local":
        yhat = []
        for x,y,xtest in zip(Xtrain, Ytrain, Xtest):
            w = simple_svm(x,y,Lambda,opts)
            y = np.sign(np.matmul(xtest, w))
            yhat.append(y)

        if opts["avg"]:
            err = []
            for i in xrange(m):
                err.append(np.mean(Ytest[i]!=yhat[i]))
            err = np.mean(err)
        else:
            y = np.hstack(np.concatenate(yhat))
            ytest = np.hstack(np.concatenate(Ytest))
            err = np.mean(ytest!=y)
    return err, w
