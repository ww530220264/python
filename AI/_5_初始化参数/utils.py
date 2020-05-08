import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt


def initial_params_zeros(layer_dims):
    L = len(layer_dims)
    params = {}
    for l in range(1, L):
        params["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def initial_params_random(layer_dims):
    np.random.seed(3)
    L = len(layer_dims)
    params = {}
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 10
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def initial_params_he(layer_dims):
    np.random.seed(3)
    L = len(layer_dims)
    params = {}
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(2 / layer_dims[l - 1])
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def load_datasets():
    np.random.seed(1)
    train_x, train_y = sklearn.datasets.make_circles(n_samples=500, noise=.05)
    np.random.seed(2)
    test_x, test_y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    train_x = train_x.T
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_x = test_x.T
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y
