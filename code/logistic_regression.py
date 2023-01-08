from math import exp, log
from random import uniform, shuffle
from primitives import MyMatrix, MyVector


def sigmoid(x: float):
    return 1 / (1 + exp(min(-1 * x, 100)))


def sigmoid_matrix(x: MyMatrix):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[(i, j)] = sigmoid(x[(i, j)])

    return x


class MyLogisticRegression:

    # features -- двумерный массив размерами [количество объектов, количество признаков]
    # targets -- одномерный массив размерами [количество объектов]

    def __init__(self, l2_coef=0.0):
        self.w = None
        self.l2_coef = l2_coef

    def fit(self, X, y, epochs=1000, eps=0.001):  # метод в котором мы учим модель

        X = MyMatrix.scalars(X.shape[0], 1, 1).concatenate(X)

        if self.w == None:
            self.w = MyMatrix([[uniform(-100, 100)] for index in range(X.shape[1])])

        for epochs_passed in range(epochs):
            grads = self.get_gradient(X, y, self.predict_proba(X))
            self.w -= grads * MyMatrix([[eps]])

    def predict_proba(self, X):
        return sigmoid_matrix(X * self.w)

    def predict_proba_external(self, X):
        X = MyMatrix.scalars(X.shape[0], 1, 1).concatenate(X)
        return sigmoid_matrix(X * self.w)

    def get_gradient(self, X, y, predictions):

        grad_basic = X.transpose() * (predictions - y)
        # grad_l2 = self.w * MyMatrix([[2 * self.l2_coef]])

        return grad_basic  # + grad_l2

    def count_loss(y, predictions):
        result = 0
        for index in range(y.shape[0]):
            result += y[(index, 0)] * log(predictions[(index, 0)]) + (1 - y[(index, 0)]) * log(predictions[(index, 0)])

        return result

def cross_validation(self, X: list[list], y, cv_split): #все же решил написать кросс-валидацию, пусть тут останется

    test_pairs = zip(X, y)
    shuffle(test_pairs)

    batch_size = (len(test_pairs) + cv_split - 1) // cv_split

    batches = [[] for count in range(cv_split)]

    for batch_index in range(cv_split):
        for object_index in range(batch_size * batch_index, min(len(test_pairs), (batch_index + 1) * batch_size)):
            batches[batch_index].append(test_pairs[object_index])

    min_loss = 0
    result_model = MyLogisticRegression()

    for index_test in range(cv_split):
        X_train = [], y_train = []
        X_test = [], y_test = []

        for index_batch in range(cv_split):
            if index_batch != index_test:
                X_train.append(
                    [batches[batch_index][element_index][0] for element_index in range(len(batches[batch_index]))])
                y_train.append(
                    [[batches[batch_index][element_index][1]] for element_index in range(len(batches[batch_index]))])
            else:
                X_test.append(
                    [batches[batch_index][element_index][0] for element_index in range(len(batches[batch_index]))])
                y_test.append(
                    [[batches[batch_index][element_index][1]] for element_index in range(len(batches[batch_index]))])

        model = MyLogisticRegression()
        model.fit(MyMatrix(X_train), MyMatrix(y_train))
        predictions = MyLogisticRegression.predict_proba(X_test)

        if index_test == 0 or min_loss >= MyLogisticRegression.count_loss(y, predictions):
            min_loss = MyLogisticRegression.count_loss(y, predictions)
            result_model = model

    return result_model, min_loss
