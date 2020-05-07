import numpy as np
import matplotlib.pyplot as plt

from AI._4_深度神经网络.utils import sigmod, relu

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


params = initial_params(list((3, 4, 5, 1)))
print(params)


def propagate_forward_single(A, W, b):
    Z = np.dot(W, A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)

    return Z, cache


def propagate_active_forward(A, W, b, activation):
    Z, cache = propagate_forward_single(A, W, b)
    if activation == "sigmod":
        A = sigmod(Z)
    elif activation == "relu":
        A = relu(Z)

    assert A.shape == (W.shape[0], A.shape[1])

    cache = (cache, Z)

    return A, cache


def propagate_forward_all(X, params):
    caches = []
    A = X

    L = len(params) // 2

    for l in range(1, L):
        A, cache = propagate_active_forward(
            A,
            params["W" + str(l)],
            params["b"] + str(l),
            activation="relu"
        )
        caches.append(cache)

    AL, cache = propagate_active_forward(
        A,
        params["W" + str(L)],
        params["b" + str(L)],
        "sigmod"
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


def propagate_backward_single(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shpe[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape
