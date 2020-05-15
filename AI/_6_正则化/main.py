import numpy as np
import matplotlib.pyplot as plt

from AI._6_正则化.utils import load_2D_dataset, init_params, propagation_forward_L, get_cost, propagation_backward_L, \
    update_params, plt_boundary, get_cost_with_regularization

plt.rcParams["figure.figsize"] = (7.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

train_X, train_Y, test_X, test_Y = load_2D_dataset()


def model(X, Y, learning_rate=0.1, iterations=10001, print_cost=True, λ=0, keep_prop=1):
    costs = []
    layer_dims = [X.shape[0], 120, 120, 120, 120, 3, 1]  # 严重过拟合
    # layer_dims = [X.shape[0], 20, 3, 1]  # 严重过拟合

    params = init_params(layer_dims)
    for i in range(0, iterations):
        A, caches = propagation_forward_L(X, params, keep_prop)
        if λ == 0:
            cost = get_cost(A, Y)
        else:
            cost = get_cost_with_regularization(A, Y, params, λ)
        costs.append(cost)
        grads = propagation_backward_L(Y, A, caches, λ, params, keep_prop)
        params = update_params(params, grads, learning_rate)
        if print_cost and i > 0 and i % (iterations//10) == 0:
            print("迭代第[%i]次的损失为%f" % (i, cost))

    plt.plot(np.squeeze(costs))
    plt.ylabel("costs")
    plt.xlabel("iterations per hundred")
    plt.title("cost if iterations")
    plt.show()

    return params


params = model(train_X, train_Y)


def predict(X, Y, params):
    Y_precidtion, caches = propagation_forward_L(X, params)
    Y_precidtion = np.round(Y_precidtion)
    if Y is not None:
        print("预测准确率是：" + str((np.sum(np.dot(Y, Y_precidtion.T)) + np.sum(np.dot(1 - Y, 1 - Y_precidtion.T))) /
                              Y.ravel().shape[0] * 100) + "%")
    return Y_precidtion


predict(train_X, train_Y, params)
predict(test_X, test_Y, params)
plt_boundary(lambda x: predict(x.T, None, params), train_X, train_Y)
plt_boundary(lambda x: predict(x.T, None, params), test_X, test_Y)

params = model(train_X, train_Y, λ=0.001)


def predict(X, Y, params):
    Y_precidtion, caches = propagation_forward_L(X, params)
    Y_precidtion = np.round(Y_precidtion)
    if Y is not None:
        print("预测准确率是：" + str((np.sum(np.dot(Y, Y_precidtion.T)) + np.sum(np.dot(1 - Y, 1 - Y_precidtion.T))) /
                              Y.ravel().shape[0] * 100) + "%")
    return Y_precidtion


predict(train_X, train_Y, params)
predict(test_X, test_Y, params)
plt_boundary(lambda x: predict(x.T, None, params), train_X, train_Y)
plt_boundary(lambda x: predict(x.T, None, params), test_X, test_Y)

params = model(train_X, train_Y, λ=0, keep_prop=0.3)


def predict(X, Y, params):
    Y_precidtion, caches = propagation_forward_L(X, params)
    Y_precidtion = np.round(Y_precidtion)
    if Y is not None:
        print("预测准确率是：" + str((np.sum(np.dot(Y, Y_precidtion.T)) + np.sum(np.dot(1 - Y, 1 - Y_precidtion.T))) /
                              Y.ravel().shape[0] * 100) + "%")
    return Y_precidtion


predict(train_X, train_Y, params)
predict(test_X, test_Y, params)
plt_boundary(lambda x: predict(x.T, None, params), train_X, train_Y)
plt_boundary(lambda x: predict(x.T, None, params), test_X, test_Y)
