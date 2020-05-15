import numpy as np
import matplotlib.pyplot as plt
import scipy.io

epsilon = 1e-5


def load_2D_dataset():
    data = scipy.io.loadmat("./datasets/data.mat")
    train_X = data["X"].T
    train_Y = data["y"].T
    test_X = data["Xval"].T
    test_Y = data["yval"].T

    # plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), cmap=plt.cm.Spectral)
    # plt.show()

    return train_X, train_Y, test_X, test_Y


def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
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


def propagation_forward_active(A, W, b, activation, keep_prop=1):
    Z, linear_cache = propagation_forward_linear(A, W, b)
    cache = (Z, linear_cache)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    if keep_prop != 1 and activation != "sigmoid":
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prop
        A = A * D
        A = A / keep_prop
        cache = (cache[0], cache[1], D)
    return A, cache


def propagation_forward_L(A, params, keep_prop=1):
    L = len(params) // 2
    caches = []
    for l in range(1, L):
        A, cache = propagation_forward_active(A, params["W" + str(l)], params["b" + str(l)], "relu", keep_prop)
        caches.append(cache)

    A, cache = propagation_forward_active(A, params["W" + str(L)], params["b" + str(L)], "sigmoid", keep_prop)
    caches.append(cache)

    return A, caches


def get_cost(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A + epsilon)) + np.multiply(1 - Y, np.log(1 - A + epsilon)))
    cost = np.squeeze(cost)
    return cost


def get_cost_with_regularization(A, Y, params, λ):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A + epsilon)) + np.multiply(1 - Y, np.log(1 - A + epsilon)))
    L = len(params) // 2
    regularization = 0
    for l in range(1, L + 1):
        regularization += np.sum(np.square(params["W" + str(l)]))
    cost += λ * regularization / (2 * m)

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
    m = A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    return dA_prev, dW, db


def propagation_backward_activation(dA, cache, activation):
    if len(cache) == 3:
        Z, linear_cache,D = cache
    else:

        Z, linear_cache = cache
    if activation == "sigmoid":
        dZ = propagation_backward_sigmoid(dA, Z)
    elif activation == "relu":
        dZ = propagation_backward_relu(dA, Z)

    dA_prev, dW, db = propatation_backward_linear(dZ, linear_cache)
    return dA_prev, dW, db


def propagation_backward_L(Y, A, caches, λ, params, keep_prop):
    grads = {}
    L = len(caches)
    #   最后一层是sigmod激活函数
    cache = caches[-1]
    dZ = sigmoid(cache[0]) - Y
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = propatation_backward_linear(dZ, cache[1])
    if λ != 0:
        grads["dW" + str(L)] += λ * params["W" + str(L)] / L
    if keep_prop != 1 and L > 1:
        grads["dA" + str(L - 1)] = grads["dA" + str(L - 1)] * caches[L - 2][2] / keep_prop

    for l in reversed(range(1, L)):
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = propagation_backward_activation(
            grads["dA" + str(l)], caches[l - 1], "relu")
        if λ != 0:
            grads["dW" + str(l)] += λ * params["W" + str(l)] / L
        if keep_prop != 1 and l > 1:
            grads["dA" + str(l - 1)] = grads["dA" + str(l - 1)] * caches[l - 2][2] / keep_prop

    return grads


def update_params(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return params


def plt_boundary(model, X, Y):
    x_min, x_max = X[0, :].min(), X[0, :].max()
    y_min, y_max = X[1, :].min(), X[1, :].max()
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral)
    plt.show()
