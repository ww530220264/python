import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io

epsilon = 1e-5


def load_2D_dataset():
    data = scipy.io.loadmat("./datasets/data.mat")
    train_X = data["X"].T
    train_Y = data["y"].T
    test_X = data["Xval"].T
    test_Y = data["yval"].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), cmap=plt.cm.Spectral)
    plt.show()

    return train_X, train_Y, test_X, test_Y


def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    for l in (1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def propagation_forward_linear(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache


def propagation_forward_active(A, W, b, activation):
    Z, linear_cache = propagation_forward_linear(A, W, b)
    cache = (Z, linear_cache)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    return A, cache


def propagation_forward_L(A, params):
    L = len(params) // 2
    caches = []
    for l in (0, L):
        A, cache = propagation_forward_active(A, params["W" + str(l)], params["b" + str(l)], "relu")
        caches.append(cache)

    A, cache = propagation_forward_active(A, params["W" + str(L)], params["b" + str(L)], "sigmoid")
    caches.append(cache)

    return A, caches


def get_cost(A, Y):
    m = Y.shpe[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A + epsilon)) + np.multiply(1 - Y, np.log(1 - A + epsilon)))
    cost = np.squeeze(cost)
    return cost


def propagation_backward_sigmoid(dA, Z):
    A = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ


def propagation_backward_relu(dA, Z):
    dA = np.array(dA, copy=True)
    dA[Z <= 0] = 0
    return dA


def propatation_backward_linear(dZ, linear_cache):
    A_prev, W, b = linear_cache
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)

    return dA_prev, dW, db


def propagation_backward_activation(dA, cache, activation):
    Z, linear_cache = cache

    if activation == "sigmoid":
        dZ = propagation_backward_sigmoid(dA, Z)
    elif activation == "relu":
        dZ = propagation_backward_relu(dA, Z)

    dA_prev, dW, db = propatation_backward_linear(dZ, linear_cache)
    return dA_prev, dW, db


def propagation_backward_L(Y, dA, caches):
    grads = {}
    L = len(caches)
    #   最后一层是sigmod激活函数
    cache = caches[-1]
    dZ = sigmoid(cache[0]) - Y
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = propatation_backward_linear(dZ, cache[-1])

    for l in range(L - 1, 0):
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = propagation_backward_activation(
            grads["dA" + str(l)], caches[l], "relu")
