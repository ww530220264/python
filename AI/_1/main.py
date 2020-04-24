import numpy as np
import h5py
import matplotlib.pyplot as plt

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

'''
sigmod激活函数:
    经过sigmod算法计算后的值在[0,1]范围内
'''
def sigmod(z):
    s = 1 / (1 + np.exp(-z))
    return s

'''
初始化w和b的值:
    dim是特征的个数,一个特征对应一个权重参数
'''
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

'''
一次前向传播和反向传播
'''
def propagate(w, b, X, Y):
    #   样本个数
    m = X.shape[1]
    '''
    前向传播
    '''
    A = sigmod(np.dot(w.T, X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    '''
    反向传播,求w和b的偏导/梯度
    '''
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost

'''
迭代优化:
    不断的对参数w和b进行梯度下降,从而减少成本
'''
def optimize(w, b, train_x, train_y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        #   一次前向传播
        grads, cost = propagate(w, b, train_x, train_y)
        dw = grads["dw"]
        db = grads["db"]
        #   进行梯度下降,更新参数,使其越来越优化,使成本越来越小
        w -= learning_rate * dw
        b -= learning_rate * db
        #   记录成本,后续画图使用
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("优化%i次后成本是: %f" % (i, cost))
    params = {
        "w": w,
        "b": b
    }
    return params, costs


'''
预测:
    使用迭代之后的w和b参数来对册数数据进行预测
'''
def predict(w, b, test_x):
    m = test_x.shape[1] #   数据个数
    y_prediction = np.zeros((1, m)) #   用来存放预测结果
    A = sigmod(np.dot(w.T, test_x) + b)
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            y_prediction[0, i] = 1
    return y_prediction

'''
神经网络模型:
    train_x:    训练数据
    train_y:    训练数据对应标签
    test_x:     测试数据
    test_y:     测试数据对应标签
    num_iterations=2000:    迭代次数
    learning_rate=0.5:  学习率
    print_cost=False:   每百次是否打印损失/成本
'''
def model(train_x, train_y, test_x, test_y, num_iterations=2000, learning_rate=0.5, print_cost=False):
    #   初始化参数
    w, b = initialize_with_zeros(train_x.shape[0])
    #   迭代优化参数和减少成本
    params, costs = optimize(w, b, train_x, train_y, num_iterations, learning_rate, True)
    w = params["w"]
    b = params["b"]
    #   使用优化后的参数预测数据
    y_prediction_train = predict(w, b, train_x)
    y_prediction_test = predict(w, b, test_x)

    print("对训练图片的预测准确率为: {}%".format(100 - np.mean(np.abs(y_prediction_train - train_y)) * 100))
    print("对测试图片的预测准确率为: {}%".format(100 - np.mean(np.abs(y_prediction_test - test_y)) * 100))

    d = {
        "costs": costs,
        "y_predict_train": y_prediction_train,
        "y_predict_test": y_prediction_test,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    return d


'''
加载训练数据和测试数据
'''
train_x, train_y, test_x, test_y, classes = load_data()
print("展示一张图片")
index = 30
plt.imshow(train_x[index])
plt.show()
print("图片的标签为: " + str(train_y[:, index]) + ", 这是一个'" + classes[np.squeeze(train_y[:, index])].decode("utf-8") + "' 图片")
print("=================")
#   训练样本个数  209
m_train = train_x.shape[0]
#   测试样本个数  50
m_test = test_x.shape[0]
#   图片的宽高   64
num_px = test_x.shape[1]

print("便于矩阵运算扁平化和转置")
#   (209,64,64,3) -> (209,64 * 64 * 3) -> (64 * 64 * 3,209)
train_x = train_x.reshape((train_x.shape[0], -1)).T
#   (50,64,64,3) -> (50,64 * 64 * 3) -> (64 * 64 * 3,50)
test_x = test_x.reshape((test_x.shape[0], -1)).T
print("标准化,像素值除以255,使所有的值都在[0,1]范围")
train_x = train_x / 255
test_x = test_x / 255

d = model(train_x, train_y, test_x, test_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

index = 8
plt.imshow(test_x[:, index].reshape((num_px, num_px, 3)))
plt.show()
print("图片实际标签: " + str(test_y[0, index]) + " 图片预测标签: " + str(d["y_predict_test"][0, index]))

costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("学习率为: " + str(i) + "时")
    models[str(i)] = model(train_x, train_y, test_x, test_y, num_iterations=1500, learning_rate=i, print_cost=False)
    print("\n----------------------------------------------------\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()
