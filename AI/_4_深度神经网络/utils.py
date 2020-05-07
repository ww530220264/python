import numpy as np


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)
