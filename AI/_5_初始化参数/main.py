import matplotlib.pyplot as plt

from AI._3_浅层神经网络.planar_utils import plot_decision_boundary
from AI._4_深度神经网络.main import propagate_backward_L, get_cost, propagate_forward_L, update_params
from AI._5_初始化参数.utils import load_datasets, initial_params_he, initial_params_random, initial_params_zeros


class Model(object):

    def __init__(self, X, Y, layer_dims, iterations, learning_rate, print_cost, initial_algorithm):
        self.X = X
        self.Y = Y
        layer_dims.insert(0, X.shape[0])
        self.layer_dims = layer_dims
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.initial_algorithm = initial_algorithm

    def train(self):
        costs = []

        if self.initial_algorithm == "zeros":
            self.params = initial_params_zeros(self.layer_dims)
        elif self.initial_algorithm == "random":
            self.params = initial_params_random(self.layer_dims)
        elif self.initial_algorithm == "he":
            self.params = initial_params_he(self.layer_dims)

        for i in range(0, self.iterations):
            AL, caches = propagate_forward_L(self.X, self.params)
            cost = get_cost(AL, self.Y)
            grads = propagate_backward_L(AL, self.Y, caches)
            params = update_params(self.params, grads, self.learning_rate)
            self.params = params

            if self.print_cost and i > 0 and i % 1000 == 0:
                costs.append(cost)
                print("Cost after iteration {}： {}".format(i, cost))

        plt.plot(costs)
        plt.xlabel("iterations (per thousands)")
        plt.ylabel("costs")
        plt.title("Learning rate = " + str(self.learning_rate))
        plt.show()

    def predict(self, X):
        AL, caches = propagate_forward_L(X.T,self.params)

        return AL


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_datasets()
    model = Model(
        train_x,
        train_y,
        layer_dims=[10, 5, 1],
        iterations=15000,
        learning_rate=0.01,
        print_cost=True,
        initial_algorithm="zeros"
    )
    model.train()
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x:model.predict(x),train_x,train_y.ravel())
    model = Model(
        train_x,
        train_y,
        layer_dims=[10, 5, 1],
        iterations=15000,
        learning_rate=0.01,
        print_cost=True,
        initial_algorithm="random"
    )
    model.train()
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: model.predict(x), train_x, train_y.ravel())
    model = Model(
        train_x,
        train_y,
        layer_dims=[10, 5, 1],
        iterations=15000,
        learning_rate=0.01,
        print_cost=True,
        initial_algorithm="he"
    )
    model.train()
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: model.predict(x), train_x, train_y.ravel())
