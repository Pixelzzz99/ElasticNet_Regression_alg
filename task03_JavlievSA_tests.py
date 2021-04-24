import task03_JavlievSA_model as EN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn import datasets 
import matplotlib.pyplot as plt

class ElasticNetRegressorsTester:
    def __init__(self):
        self.test = 0

    @classmethod
    def test1_boston_test(cls):
        dataset = datasets.load_boston()
        Y=dataset.target
        X=dataset.data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        
        print("X Shape: ", X.shape)
        print("Y Shape: ", Y.shape)
        print("X_Train Shape: ", x_train.shape)
        print("X_Test Shape: ", x_test.shape)
        print("Y_Train Shape: ", y_train.shape)
        print("Y_Test Shape: ", y_test.shape)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)       

        l1 = 0
        l2 = 0
        best_result_l1=0
        best_result_l2=0
        print("Find best score with best values")
        score_best = 0
        alpha = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        best_alpha = alpha[0]
        for k in alpha:
            for i in range(20):
                for j in range(20):
                    mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=k, delta=2, batch_size=10, l1_coef=l1, l2_coef=l2)
                    mySGDElastic.fit(x_train, y_train)
                    test = mySGDElastic.predict(x_test)
                    score = mySGDElastic.score(y_test, test)
                    if score_best < score < 1:
                        best_result_l1 = l1
                        best_result_l2 = l2
                        best_alpha = k
                        score_best = score
                    l2+=1
                l1+=1
                l2=0
        print("Best alpha: ", best_alpha)
        print("Best l1 value: ", best_result_l1) 
        print("Best l2 value: ", best_result_l2)
        print("Best score with this values: ", score_best)

        regr = ElasticNet(random_state=0)
        regr.fit(x_train, y_train)
        y_sklearn_test = regr.predict(x_test)

        mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=best_alpha, delta=2, batch_size=10, l1_coef=best_result_l1, l2_coef=best_result_l2)
        mySGDElastic.fit(x_train, y_train)

        y_my_test = mySGDElastic.predict(x_test)
        score1 = mySGDElastic.score(y_test, y_my_test)
        score2 = mySGDElastic.score(y_test, y_sklearn_test)
        score3 = mySGDElastic.score(y_sklearn_test, y_my_test)
        print("My score in relation to tests: ", score1)
        print("Sklearn score in relation to tests: ", score2)
        print("My score in relation to result sklearn: ", score3)

        plt.scatter(y_test, y_my_test, color="blue", label="my prediction")
        plt.scatter(y_test, y_sklearn_test, color="red", label="sklearn prediction")
        plt.xlabel("Salary y test")
        plt.ylabel("Salary y prediction")
        plt.legend()
        plt.grid(True)
        plt.show()
        return score1, score2, score3
        
    @classmethod
    def test2_diabetes_test(cls):
        dataset = datasets.load_diabetes()
        Y=dataset.target
        X=dataset.data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        
        print("X Shape: ", X.shape)
        print("Y Shape: ", Y.shape)
        print("X_Train Shape: ", x_train.shape)
        print("X_Test Shape: ", x_test.shape)
        print("Y_Train Shape: ", y_train.shape)
        print("Y_Test Shape: ", y_test.shape)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)       

        l1 = 0
        l2 = 0
        best_result_l1=0
        best_result_l2=0
        print("Find best score with best values")
        score_best = 0
        alpha = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        best_alpha = alpha[0]
        for k in alpha:
            for i in range(20):
                for j in range(20):
                    mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=k, delta=2, batch_size=10, l1_coef=l1, l2_coef=l2)
                    mySGDElastic.fit(x_train, y_train)
                    test = mySGDElastic.predict(x_test)
                    score = mySGDElastic.score(y_test, test)
                    if score_best < score < 1:
                        best_result_l1 = l1
                        best_result_l2 = l2
                        best_alpha = k
                        score_best = score
                    l2+=1
                l1+=1
                l2=0
        print("Best alpha: ", best_alpha)
        print("Best l1 value: ", best_result_l1) 
        print("Best l2 value: ", best_result_l2)
        print("Best score with this values: ", score_best)

        regr = ElasticNet(random_state=0)
        regr.fit(x_train, y_train)
        y_sklearn_test = regr.predict(x_test)


        mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=best_alpha, delta=2, batch_size=10, l1_coef=best_result_l1, l2_coef=best_result_l2)
        mySGDElastic.fit(x_train, y_train)
        y_my_test = mySGDElastic.predict(x_test)
        score1 = mySGDElastic.score(y_test, y_my_test)
        score2 = mySGDElastic.score(y_test, y_sklearn_test)
        score3 = mySGDElastic.score(y_sklearn_test, y_my_test)
        print("My score in relation to tests: ", score1)
        print("Sklearn score in relation to tests: ", score2)
        print("My score in relation to result sklearn: ", score3)
        plt.scatter(y_test, y_my_test, color="blue", label="my prediction")
        plt.scatter(y_test, y_sklearn_test, color="red", label="sklearn prediction")
        plt.xlabel("Salary y test")
        plt.ylabel("Salary y prediction")
        plt.legend()
        plt.grid(True)
        plt.show()
        return score1, score2, score3

    @classmethod
    def test3_iris_test(cls):
        dataset = datasets.load_iris()
        Y=dataset.target
        X=dataset.data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        
        print("X Shape: ", X.shape)
        print("Y Shape: ", Y.shape)
        print("X_Train Shape: ", x_train.shape)
        print("X_Test Shape: ", x_test.shape)
        print("Y_Train Shape: ", y_train.shape)
        print("Y_Test Shape: ", y_test.shape)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)       

        l1 = 0
        l2 = 0
        best_result_l1=0
        best_result_l2=0
        print("Find best score with best values")
        score_best = 0
        alpha = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        best_alpha = alpha[0]
        for k in alpha:
            for i in range(20):
                for j in range(20):
                    mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=k, delta=2, batch_size=10, l1_coef=l1, l2_coef=l2)
                    mySGDElastic.fit(x_train, y_train)
                    test = mySGDElastic.predict(x_test)
                    score = mySGDElastic.score(y_test, test)
                    if score_best < score < 1:
                        best_result_l1 = l1
                        best_result_l2 = l2
                        best_alpha = k
                        score_best = score
                    l2+=1
                l1+=1
                l2=0
        print("Best alpha: ", best_alpha)
        print("Best l1 value: ", best_result_l1) 
        print("Best l2 value: ", best_result_l2)
        print("Best score with this values: ", score_best)

        regr = ElasticNet(random_state=0)
        regr.fit(x_train, y_train)
        y_sklearn_test = regr.predict(x_test)


        mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=best_alpha, delta=2, batch_size=10, l1_coef=best_result_l1, l2_coef=best_result_l2)
        mySGDElastic.fit(x_train, y_train)
        y_my_test = mySGDElastic.predict(x_test)
        score1 = mySGDElastic.score(y_test, y_my_test)
        score2 = mySGDElastic.score(y_test, y_sklearn_test)
        score3 = mySGDElastic.score(y_sklearn_test, y_my_test)
        print("My score in relation to tests: ", score1)
        print("Sklearn score in relation to tests: ", score2)
        print("My score in relation to result sklearn: ", score3)
        plt.scatter(y_test, y_my_test, color="blue", label="my prediction")
        plt.scatter(y_test, y_sklearn_test, color="red", label="sklearn prediction")
        plt.xlabel("Salary y test")
        plt.ylabel("Salary y prediction")
        plt.legend()
        plt.grid(True)
        plt.show()
        return score1, score2, score3

    @classmethod
    def test4_digits_test(cls):
        dataset = datasets.load_digits()
        Y=dataset.target
        X=dataset.data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        
        print("X Shape: ", X.shape)
        print("Y Shape: ", Y.shape)
        print("X_Train Shape: ", x_train.shape)
        print("X_Test Shape: ", x_test.shape)
        print("Y_Train Shape: ", y_train.shape)
        print("Y_Test Shape: ", y_test.shape)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)       

        l1 = 0
        l2 = 0
        best_result_l1=0
        best_result_l2=0
        print("Find best score with best values")
        score_best = 0
        alpha = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        best_alpha = alpha[0]
        for k in alpha:
            for i in range(20):
                for j in range(20):
                    mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=k, delta=2, batch_size=10, l1_coef=l1, l2_coef=l2)
                    mySGDElastic.fit(x_train, y_train)
                    test = mySGDElastic.predict(x_test)
                    score = mySGDElastic.score(y_test, test)
                    if score_best < score < 1:
                        best_result_l1 = l1
                        best_result_l2 = l2
                        best_alpha = k
                        score_best = score
                    l2+=1
                l1+=1
                l2=0
        print("Best alpha: ", best_alpha)
        print("Best l1 value: ", best_result_l1) 
        print("Best l2 value: ", best_result_l2)
        print("Best score with this values: ", score_best)

        regr = ElasticNet(random_state=0)
        regr.fit(x_train, y_train)
        y_sklearn_test = regr.predict(x_test)


        mySGDElastic = EN.ElasticNetRegressor(n_epoch=20, alpha=best_alpha, delta=2, batch_size=10, l1_coef=best_result_l1, l2_coef=best_result_l2)
        mySGDElastic.fit(x_train, y_train)
        y_my_test = mySGDElastic.predict(x_test)
        score1 = mySGDElastic.score(y_test, y_my_test)
        score2 = mySGDElastic.score(y_test, y_sklearn_test)
        score3 = mySGDElastic.score(y_sklearn_test, y_my_test)
        print("My score in relation to tests: ", score1)
        print("Sklearn score in relation to tests: ", score2)
        print("My score in relation to result sklearn: ", score3)
        plt.scatter(y_test, y_my_test, color="blue", label="my prediction")
        plt.scatter(y_test, y_sklearn_test, color="red", label="sklearn prediction")
        plt.xlabel("Salary y test")
        plt.ylabel("Salary y prediction")
        plt.legend()
        plt.grid(True)
        plt.show()
        return score1, score2, score3
