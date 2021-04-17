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
        self.X = X
        self.Y = Y
        batches_for_x = ElasticNetIterator(X, self.batch_size)
        batches_for_y = ElasticNetIterator(Y, self.batch_size)
        self.m, self.n = X.shape
        
        self.weights = np.zeros(self.n)
        self.b = 0
        
        for epoch in range(self.n_epoch):
            for batch_x, batch_y in zip(batches_for_x, batches_for_y):
                self.update_weights(batch_x, batch_y)
        return self

    def update_weights(self, batch_x, batch_y):
        x = np.array(batch_x)
        y = np.array(batch_y)
        m, n = np.shape(x)
        Y_pred = self.predict(x)
        dW = np.zeros(n)

        for j in range(n):
            if(self.weights[j] > 0):
                dW[j] = self._weight_bigger(j, Y_pred)
            else:
                dW[j] = self._weight_less(j, Y_pred)

        db = - 2 * np.sum(y - Y_pred) / m
        self.weights = self.weights - self.alpha * dW
        self.b = self.b - self.alpha * db
        self.alpha = self.alpha / self.delta
        return self

    def _weight_bigger(self, index, y_pred):
        dW_temp = (-(2 * (x[:, index]).dot(y - y_pred)) +
                   self.l1_coef + 2 * self.l2_coef * self.weights[index])/m
        return dW_temp

    def _weight_less(self, index, y_pred):
        dW_temp = (- (2 * (x[:, index]).dot(y - y_pred)) -
                   self.l1_coef + 2 * self.l2_coef * self.weights[index])/m
        return dW_temp

    def predict(self, X):
        return X.dot(self.weights) + self.b

    def score(self, y_gt, y_pred):
        u = ((y_gt - y_pred)**2).sum()
        v = ((y_gt - y_gt.mean())**2).sum()
        return (1-u/v)


class ElasticNetIterator:
    def __init__(self, limit, batch_size):
        self.data = np.array(limit)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        starter = 0
        finisher = self.batch_size
        pointer = self.batch_size**2
        if pointer < np.size(self.data):
            batch = self.data[starter:finisher:, starter:finisher:].copy()
            starter += finisher
            finisher += self.batch_size
            pointer += self.batch_size**2
            return batch
        else:
            raise StopIteration
