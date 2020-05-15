import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def load_2D_dataset():
    data = scipy.io.loadmat("./datasets/data.mat")
    train_X = data["X"].T
    train_Y = data["y"].T
    test_X = data["Xval"].T
    test_Y = data["yval"].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), cmap=plt.cm.Spectral)
    plt.show()

    return train_X, train_Y, test_X, test_Y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu(z):
    return np.maximum(0, z)


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


'''
数据加载
'''


def load_data():
    #   训练数据
    train_data = h5py.File("./datasets/train_catvnoncat.h5", "r")
    #   (209,64,64,3)
    train_x = np.array(train_data["train_set_x"][:])
    #   (209,1)
    train_y = np.array(train_data["train_set_y"][:])
    #   测试数据
    test_data = h5py.File("./datasets/test_catvnoncat.h5", "r")
    #   (50,64,64,3)
    test_x = np.array(test_data["test_set_x"][:])
    #   (50,1)
    test_y = np.array(test_data["test_set_y"][:])
    #   (209,1) -> (1,209)
    train_y = train_y.reshape((1, train_y.shape[0]))
    #   (50,1) -> (1,50)
    test_y = test_y.reshape((1, test_y.shape[0]))
    #   (2,)
    classes = np.array(train_data["list_classes"][:])

    return train_x, train_y, test_x, test_y, classes
