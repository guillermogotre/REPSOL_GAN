import numpy as np
import pandas as pd

class MEnsamble:
    def __init__(self, Model, n, ws, bins, **kwparams):
        self.n = kwparams.get('n', n)
        self.Model = kwparams.get('Model', Model)
        self.ms = None
        self.idxs = None
        self.__updateN__(**kwparams)
        self.ws = kwparams.get('ws', ws)
        self.bins = kwparams.get('bins', bins)

    def __updateN__(self, **kwparams):
        self.ms = [self.Model(**kwparams) for i in range(self.n)]
        self.idxs = [[] for i in range(self.n)]

    def splitDataset(self, y):
        bres = [(y > (self.bins[i] - 1e-10)) & (y <= self.bins[i + 1]) for i in range(len(bins) - 1)]
        for i in range(self.n):
            res = []
            for j, w in enumerate(self.ws):
                res.append(np.random.choice(
                    np.where(bres[j])[0],
                    np.min([w, bres[j].sum()]),
                    replace=False))
            #                 print(bres[j].sum(),w,len(res[-1]))
            self.idxs[i] = np.hstack(res)

    def fit(self, X, y, splitDataset=True):
        self.splitDataset(y)
        for m, idx in zip(self.ms, self.idxs):
            m.fit(X[idx], y[idx])

    def predict(self, X):
        yp = [m.predict(X) for m in self.ms]
        return np.median(yp, axis=0)

    def set_params(self, **params):
        m_keys = {}
        for p in params:
            assert not (hasattr(self, p) and hasattr(self.ms[0], p))
            if hasattr(self, p):
                self.__setattr__(p, params[p])
                if p == 'n':
                    self.__updateN__()
            else:
                m_keys[p] = params[p]
        for m in self.ms:
            m.set_params(**m_keys)


class Model_Wrapper(object):
    def __init__(self, model, columns, X_scaler, Y_scaler):
        self.model = model
        self.columns = columns

        self.X_scaler = X_scaler

        self.Y_scaler = Y_scaler

    def predict(self, X):
        X = X[self.columns]

        Xs = self.X_scaler.transform(X)
        yp = self.model.predict(Xs)

        ys = self.Y_scaler.transform(yp)
        y = pd.DataFrame(ys)
        return y


class NoScale:
    def transform(self, x, *args, **kwargs):
        return x