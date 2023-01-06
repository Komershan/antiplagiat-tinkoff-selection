from math import exp
from random import uniform
from primitives import MyMatrix


def sigmoid(x: float):
    return 1 / (1 + exp(-1 * x))


def sigmoid(x: MyMatrix):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[(i, j)] = sigmoid(x[(i, j)])

    return x


class MyLogisticRegression:

    # features -- двумерный массив размерами [количество объектов, количество признаков]
    # targets -- одномерный массив размерами [количество объектов]

    def __init__(self, l2_coef=0.1):
        self.w = None
        self.l2_coef = l2_coef

    def fit(self, X, y, epochs=10, eps=0.1):  # метод в котором мы учим модель

        X = MyMatrix.scalars(X.shape[0], 1, 1).concatenate(X)

        if self.w == None:
            self.w = MyMatrix([[uniform(-10000, 10000)] for index in range(X[0].size)])

        for epochs_passed in range(epochs):
            grads = self.get_gradient(X, y, self.predict_proba(X))
            self.w -= grads * MyMatrix([[eps]])

    def predict_proba(self, X):
        return sigmoid(X * self.w)

    def get_gradient(self, X, y, predictions):

        grad_basic = X.transpose() * (predictions - y)
        # grad_l2 = self.w * MyMatrix([[2 * self.l2_coef]])

        return grad_basic  # + grad_l2
