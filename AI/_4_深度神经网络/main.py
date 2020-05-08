import numpy as np
import matplotlib.pyplot as plt

from AI._4_深度神经网络.utils import sigmoid, relu, relu_backward, sigmoid_backward, load_data

plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(1)


def initial_params(layer_dims):
    np.random.seed(1)
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))
        assert params["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert params["b" + str(l)].shape == (layer_dims[l], 1)

    return params


# params = initial_params(list((5,4,3)))
# print(params)


def propagate_forward_linear(A, W, b):
    Z = np.dot(W, A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    linear_cache = (A, W, b)

    return Z, linear_cache


def propagate_active_forward(A, W, b, activation):
    Z, linear_cache = propagate_forward_linear(A, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    assert A.shape == (W.shape[0], A.shape[1])

    cache = (linear_cache, Z)

    return A, cache


def propagate_forward_L(X, params):
    caches = []
    A = X

    L = len(params) // 2

    for l in range(1, L):
        A, cache = propagate_active_forward(
            A,
            params["W" + str(l)],
            params["b" + str(l)],
            activation="relu"
        )
        caches.append(cache)

    AL, cache = propagate_active_forward(
        A,
        params["W" + str(L)],
        params["b" + str(L)],
        "sigmoid"
    )
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


def get_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)
    assert cost.shape == ()

    return cost


def propagate_backward_linear(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def propagate_active_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    return propagate_backward_linear(dZ, linear_cache)


def propagation_backward_L(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = propagate_active_backward(dAL, current_cache,
                                                                                                     "sigmoid")

    for c in reversed(range(1, L)):
        grads["dA" + str(c - 1)], grads["dW" + str(c)], grads["db" + str(c)] = propagate_active_backward(
            grads["dA" + str(c)], caches[c - 1], "relu")

    return grads


def update_params(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return params


class Model(object):

    def __init__(self, X, Y, layer_dims, learning_rate, iterations, print_cost=False):
        self.X = X
        self.Y = Y
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.print_cost = print_cost

    def model(self):
        np.random.seed(1)
        costs = []

        params = initial_params(self.layer_dims)

        for i in range(0, self.iterations):
            AL, caches = propagate_forward_L(self.X, params)
            cost = get_cost(AL, self.Y)
            costs.append(cost)
            grads = propagation_backward_L(AL, self.Y, caches)
            params = update_params(params, grads, self.learning_rate)
            if self.print_cost and i % 100 == 0:
                print("训练[%i]次后的成本为 %f" % (i, cost))

        self.params = params

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate = " + str(self.learning_rate))
        plt.show()

        return params

    def predict(self, X):
        m = X.shape[1]
        n = len(self.params) // 2
        p = np.zeros((1, m))

        probas, caches = propagate_forward_L(X, self.params)
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        return p


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, classes = load_data()
    train_x = train_x.reshape((train_x.shape[0], -1)).T
    test_x = test_x.reshape((test_x.shape[0], -1)).T

    train_x = train_x / 255
    test_x = test_x / 255

    layer_dims = [12288, 20, 7, 5, 1]
    model = Model(train_x, train_y, layer_dims, 0.0075, 2000, True)
    model.model()
    prediction = model.predict(train_x)
    print("预测准确率为：" + str(np.sum((prediction == train_y)) / train_y.shape[1]))
    prediction = model.predict(test_x)
    print("预测准确率为：" + str(np.sum((prediction == test_y)) / test_y.shape[1]))

    for index in range(0,50):
        plt.imshow(test_x[:, index].reshape((64, 64, 3)))
        plt.show()
        print("图片实际标签: " + str(test_y[0, index]) + " 图片预测标签: " + str(prediction[0, index]) + ", 这是一个'" + classes[
            np.squeeze(test_y[:, index])].decode("utf-8") + "' 图片")
