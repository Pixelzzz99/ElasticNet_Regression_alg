import numpy as np
import pandas as pd

class ElasticNetRegressor(object):
    def __init__(self, n_epoch=100, batch_size=100, alpha=100., delta=100.0, l1_coef=100.0, l2_coef=100.0):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.alpha = alpha
        self.delta = delta
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.n_epoch):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = np.zeros(self.n)
        for j in range(self.n):
            if(self.weights[j] > 0):
                dW[j] = self._weight_bigger(j, Y_pred)
            else:
                dW[j] = self._weight_less(j, Y_pred)

        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        self.weights = self.weights - self.alpha * dW
        self.b = self.b - self.alpha * db
        return self

    def _weight_bigger(self, index, y_pred):
        dW_temp = (-(2 * (self.X[:, index]).dot(self.Y - y_pred)) +
            self.l1_coef + 2 * self.l2_coef * self.weights[index])/self.m
        return dW_temp

    def _weight_less(self, index, y_pred):
        dW_temp = (- (2 * (self.X[:, index]).dot(self.Y - y_pred)) -
           self.l1_coef + 2 * self.l2_coef * self.weights[index])/self.m
        return dW_temp

    def predict(self, X):
        return X.dot(self.weights) + self.b

    def score(self, y_gt, y_pred):
        u = ((y_gt - y_pred)**2).sum()
        v = ((y_gt - y_gt.mean())**2).sum()
        return (1-u/v)

    class ElasticNetIterator:
        def __init__(self, limit):
            self.limit = limit
            self.counter = 0

        def __new__(self):
            return self
        
        def __next__(self):
            if self.counter < self.limit:
                self.counter += 1
                return 1
            else: raise StopIteration
