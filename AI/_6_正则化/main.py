import numpy as np
import matplotlib.pyplot as plt

from AI._6_正则化.utils import load_2D_dataset, init_params, propagation_forward_L, get_cost

plt.rcParams["figure.figsize"] = (7.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

train_X, train_Y, test_X, test_Y = load_2D_dataset()


def model(X, Y, learning_rate=0.3, iterations=2000, print_cost=True, lambd=0, keep_prop=1):
    grads = {}
    costs = []
    m = X.shape[1]

    layer_dims = [X.shape[0], 20, 3, 1]

    params = init_params(layer_dims)
    costs = []
    for i in range(0, iterations):
        A, caches = propagation_forward_L(X, params)
        cost = get_cost(A, Y)
        costs.append(cost)
        if print_cost and i > 0 and i % 100 == 0:
            print("迭代第[%i]次的损失为%f" % (i, cost))
