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
        self.weights = 0
        self.b = 0

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        batches_for_x = ElasticNetIterator(X, self.batch_size)
        batches_for_y = ElasticNetIterator(Y, self.batch_size)
        self.m, self.n = X.shape

        self.weights = np.zeros(self.n)
        

        for epoch in range(self.n_epoch):
            for batch_x, batch_y in zip(batches_for_x, batches_for_y):
                self.update_weights(batch_x, batch_y)
        self.alpha = self.alpha / self.delta   
        return self
        

    def update_weights(self, batch_x, batch_y):
        
        x_batch = np.array(batch_x)
        y_batch = np.array(batch_y)

        m_row, n_column = np.shape(x_batch)

        Y_pred = self.predict(x_batch)

        dW = np.zeros(self.n)

        for j in range(n_column):
            if self.weights[j] > 0:
                dW[j] = (-(2 * x_batch[:, j].dot(y_batch - Y_pred)) +
                         self.l1_coef + 2 * self.l2_coef * self.weights[j]) / m_row
            else:
                dW[j] = (- (2 * x_batch[:, j].dot(y_batch - Y_pred)) -
                           self.l1_coef + 2 * self.l2_coef * self.weights[j]) / m_row

        db = - 2 * np.sum(y_batch - Y_pred) / m_row
        self.weights = self.weights - self.alpha * dW
        self.b = self.b - self.alpha * db
        
        return self
        

    def predict(self, X):
        return X.dot(self.weights) + self.b
        

    def score(self, y_gt, y_pred):
        u = ((y_gt - y_pred)**2).sum()
        v = ((y_gt - y_gt.mean())**2).sum()
        return (1-u/v)


class ElasticNetIterator:
    def __init__(self, limit, batch_size):
        self.data = limit.copy()
        self.batch_size = batch_size
        self.starter = 0
        self.finisher = self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.finisher < self.data.shape[0]:
            batch = self.data[self.starter : self.finisher]
            self.starter += self.batch_size
            self.finisher += self.batch_size
            return batch
        else:
            raise StopIteration
            


'''
    def _weight_bigger(self, x, y, m, index, y_pred):
        dW_temp = (-(2 * (x[:, index]).dot(y - y_pred)) +
           self.l1_coef + 2 * self.l2_coef * self.weights[index])/m
        return dW_temp

    def _weight_less(self, x, y, m, index, y_pred):
        dW_temp = (- (2 * (x[:, index]).dot(y - y_pred)) -
            self.l1_coef + 2 * self.l2_coef * self.weights[index])/m
        return dW_temp
'''
