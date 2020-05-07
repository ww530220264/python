import matplotlib.pyplot as plt
import numpy as np

'''
生成玫瑰花数据集
'''
def get_planar_dataset():
    np.random.seed(1)
    m = 400 # 样本个数
    N = int(m / 2)
    D = 2 # 特征数量
    X = np.zeros((m, D)) # 0初始化数据
    Y = np.zeros((m, 1), dtype="uint8") # 初始化数据标签
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        '''
        在start和end中间选取n个均匀间隔的点
        np.random.rand(N)：返回一组符合0~1均匀分布的随机样本值
        在二维平面上生成数据的坐标，可以看做一个极坐标
        t代表角度，r代表由角度生成的半径，也就是玫瑰花瓣的长度
        利用参数方程，将极坐标转换为直角坐标
        j相当于坐标的颜色（0,1）
        '''
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.rand(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.rand(N) * 0.2
        # 使用转换后的坐标批量替换初始化的坐标数据
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y


X, Y = get_planar_dataset()
'''
散点图
c:颜色
cmap=plt.cm.Spectral：光谱类型
alpha：透明度
marker：散点的形状
cmap：colormap
'''
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, alpha=1, marker="*", cmap=plt.cm.Spectral)
plt.show()


def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # 步长
    '''
    生成网格点坐标矩阵
    xx：所有网格点的横坐标
    yy：所有网格点的纵坐标
    '''
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # np.c_[xx.ravel(), yy.ravel]：获取所有的网格点坐标矩阵
    # model: lambda x:clf.predict(x)
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    '''
    绘画不同分类的边界线
    xx：点的x坐标
    yy：点的y坐标
    Z：对应点的权重值
    '''
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
    plt.show()
