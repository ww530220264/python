import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

from AI._3_浅层神经网络.planar_utils import get_planar_dataset, plot_decision_boundary

np.random.seed(1)
##################    测试单神经元模型预测效果---start    #################
X, Y = get_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), marker="p", s=40, cmap=plt.cm.Spectral)
plt.show()

shape_X = X.shape  # (2,400)
shape_Y = Y.shape  # (1,400)
m = Y.shape[1]  # 400

clf = lm.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())
LR_predict = clf.predict(X.T)
print("预测准确度是：" + str(float((np.dot(Y, LR_predict) + np.dot(1 - Y, 1 - LR_predict)) / float(Y.size) * 100)) + '%')
print("预测准确度是：" + str(clf.score(X.T, Y.T.ravel())) + '%')

plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())


##################    测试单神经元模型预测效果---end    #################

def sigmod(z):
    return 1 / (1 + np.exp(-z))


'''
初始化参数
n_x：输入层神经元个数
n_h：隐藏层神经元个数
n_y：输出层神经元个数
'''


def initial_params(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params


'''
前向传播
'''


def propagation_forward(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmod(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return A2, cache


'''
前向传播之后计算成本
'''


def get_cost(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m

    return cost


'''
反向传播
计算各个参数的偏导数以用来梯度下降
'''


def propagation_backward(params, cache, X, Y):
    m = X.shape[1]

    W1 = params["W1"]
    W2 = params["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }

    return grads


'''
根据学习率进行梯度下降，更新参数
'''


def update_params(params, grads, learning_rate=1.2):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return params


'''
模型类
'''


class Model(object):
    def __init__(self, X, Y, n_h, iterations, print_cost):
        self.X = X
        self.Y = Y
        self.n_h = n_h
        self.iterations = iterations
        self.print_cost = print_cost

    '''
    训练模型->得到参数
    '''

    def model(self):
        np.random.seed(3)
        n_x = self.X.shape[0]
        n_y = self.Y.shape[0]

        params = initial_params(n_x, self.n_h, n_y)

        for i in range(self.iterations):
            A2, cache = propagation_forward(X, params)
            cost = get_cost(A2, Y)
            grads = propagation_backward(params, cache, X, Y)
            params = update_params(params, grads, learning_rate=0.1)

            if self.print_cost and i % 1000 == 0:
                print("在训练%i次后，成本是： %f" % (i, cost))

        self.params = params

    '''
    根据得到的模型预测数据
    '''

    def predict(self, X, print_result=False):
        A2, cache = propagation_forward(X, self.params)
        predicition = np.round(A2)
        if print_result:
            print("预测准确率是：" + str(
                float((np.dot(Y, predicition.T) + np.dot(1 - Y, 1 - predicition.T)) / Y.size * 100)) + "%")

        return predicition

    '''
    可视化预测结果
    '''

    def plot(self):
        plot_decision_boundary(lambda x: self.predict(x.T, print_result=False), X, Y.ravel())


model = Model(X, Y, 4, 10000, True)
model.model()
model.predict(X, print_result=True)
model.plot()
