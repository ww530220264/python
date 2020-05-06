import copy
import math
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    irisCsv = pd.read_csv("./datasets/iris.csv", header=1)
    dataset = np.array(irisCsv).tolist()
    labels = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return dataset, labels


'''
出现次数最多的标签值
'''


def max_class(classes):
    return max(classes, key=labels.count)


def max_class_by_dataset(dataset):
    classes = [vector[-1] for vector in dataset]
    return max(classes, key=labels.count)


'''
使用基尼系数选择最佳分裂特征和特征值
float("inf"): 正无穷
'''


def choose_feature_cart(dataset):
    min_gini = float("inf")  # 最小基尼系数
    nums_data = len(dataset)  # 样本数据个数
    nums_feature = len(dataset[0]) - 1  # 特征个数
    feature_index = 0  # 最佳分裂特征索引
    feature_point = None  # 最佳分裂特征对应的特征值

    for i in range(nums_feature):
        feature_i_set = [vector[i] for vector in dataset]  # 特征i的所有取值
        feature_i_set = list(set(feature_i_set))  # 特征i去重后的所有取值
        feature_i_gini = 0  # 特征i的基尼系数总和
        feature_i_min_gini = float("inf")  # 特征i的取每个特征值后剩余数据的基尼系数的最小值
        feature_i_point = None  # 特征i中最小的基尼系数对应的特征值
        for feature_value in feature_i_set:
            # 根据该特征和特征值切分数据，用来获取切分后数据的基尼系数
            nums_right, right, nums_left, left = split_dataset_cart(dataset, i, feature_value)
            p_right = nums_right / nums_data
            gini_right = get_gini(right)
            p_left = nums_left / nums_data
            gini_left = get_gini(left)
            feature_i_value_gini = p_right * gini_right + p_left * gini_left
            feature_i_gini += feature_i_value_gini
            # 在该特征下根据每个特征值分裂后的数据的基尼系数确定最佳分裂特征值，选择基尼系数小的特征值作为分裂特征值
            if feature_i_value_gini < feature_i_min_gini:
                feature_i_point = feature_value
        # 根据每个特征分裂后的基尼系数的总和的最小值确定最佳分裂特征
        if feature_i_gini < min_gini:
            min_gini = feature_i_gini
            feature_index = i
            feature_point = feature_i_point

    return feature_index, feature_point


'''
根据ID3或C45算法选择最佳分裂特征
float("-inf"): 负无穷
'''


def choose_feature(dataset, algorithm="ID3"):
    nums_data = len(dataset)
    nums_feature = len(dataset[0]) - 1
    entropy = get_entropy(dataset)
    max_information_gain = float("-inf")
    max_information_gain_ratio = float("-inf")
    feature_index = 0
    for i in range(nums_feature):
        feature_i_set = [vector[i] for vector in dataset]
        feature_i_set = set(feature_i_set)
        condition_entropy = 0  # 条件熵
        feature_entropy = 0  # 特征i的分裂信息度量（SplitInformation）

        for feature_point in feature_i_set:
            nums_sub_dataset, sub_dataset = split_dataset(dataset, i, feature_point)
            p_feature = nums_sub_dataset / nums_data
            feature_entropy -= p_feature * math.log(p_feature, 2)
            sub_entropy = get_entropy(sub_dataset)
            condition_entropy += p_feature * sub_entropy
        feature_information_gain = entropy - condition_entropy  # 信息增益

        if algorithm == "ID3" or feature_entropy == 0:
            if feature_information_gain > max_information_gain:
                max_information_gain = feature_information_gain
                feature_index = i
        elif algorithm == "C45":
            feature_information_gain_ratio = feature_information_gain / feature_entropy
            if feature_information_gain_ratio > max_information_gain_ratio:
                max_information_gain_ratio = feature_information_gain_ratio
                feature_index = i
        else:
            exit("algorithm should be ID3 or C45")

    return feature_index


def get_entropy(dataset):
    nums_data = len(dataset)
    count_labels = {}
    entropy = 0
    for vector in dataset:
        if vector[-1] not in count_labels:
            count_labels[vector[-1]] = 0
        count_labels[vector[-1]] += 1
    for key in count_labels:
        p = float(count_labels[key] / nums_data)
        entropy -= p * math.log(p, 2)

    return entropy


'''
获取数据集的基尼系数
'''


def get_gini(dataset):
    nums_data = len(dataset)
    count_labels = {}
    p_sum = 0
    for vector in dataset:
        if vector[-1] not in count_labels:
            count_labels[vector[-1]] = 0
        count_labels[vector[-1]] += 1
    for key in count_labels:
        p = float(count_labels[key] / nums_data)
        p_sum += p ** 2

    return 1 - p_sum


'''
根据 特征和特征值分裂数据集
'''


def split_dataset_cart(dataset, feature_index, feature_value):
    right = []
    left = []

    for vector in dataset:
        split = vector[:feature_index]
        split.extend(vector[feature_index + 1:])
        if vector[feature_index] >= feature_value:
            right.append(split)
        else:
            left.append(split)

    return len(right), right, len(left), left


'''
等值切分数据集
'''


def split_dataset(dataset, feature_index, feature_point):
    sub_dataset = []
    for vector in dataset:
        if vector[feature_index] == feature_point:
            split = vector[:feature_index]
            split.extend(vector[feature_index + 1:])
            sub_dataset.append(split)

    return len(sub_dataset), sub_dataset


class DecisionTree(object):

    def __init__(self, dataset, labels, method_name, algorithm="ID3"):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm
        self.fun = getattr(self, method_name)

    def main(self):
        labels = copy.deepcopy(self.labels)
        tree = self.fun(self.dataset, labels)

        return tree

    def create_tree(self, dataset, labels):
        classes = [vector[-1] for vector in dataset]
        if len(set(classes)) == 1:
            return classes[0]
        if len(dataset[0]) == 1:
            return max_class(classes)
        feature_index = choose_feature(dataset, algorithm=self.algorithm)
        feature_label = labels[feature_index]
        tree = {feature_label: {}}
        labels = copy.deepcopy(labels)
        del labels[feature_index]
        feature_value_list = list(set([vector[feature_index] for vector in dataset]))
        feature_value_list.sort()
        for feature_point in feature_value_list:
            nums_sub_dataset, sub_dataset = split_dataset(dataset, feature_index, feature_point)
            tree[feature_label][" == " + str(feature_point)] = self.create_tree(sub_dataset, labels)
        return tree

    '''
    创建CART决策树
    特点：和ID3一样，存在偏向细小切分，即存在过拟合问题，可对特别长的树枝进行剪枝操作来解决
    CART ：分类回归树，二叉树，使用基尼系数决定如何分裂
    '''

    def create_tree_cart(self, dataset, labels):
        classes = [vector[-1] for vector in dataset]
        if len(set(classes)) == 1:  # 如果只有一个分类，直接返回该分类
            return classes[0]
        if len(dataset[0]) == 1:  # 如果只有一个特征，直接返回分类出现次数最多的分类
            return max_class(classes)
        labels = copy.deepcopy(labels)
        feature_index, feature_point = choose_feature_cart(dataset)
        feature_label = labels[feature_index]
        tree = {feature_label: {}}
        del labels[feature_index]
        nums_right, right, nums_left, left = split_dataset_cart(dataset, feature_index, feature_point)
        if nums_right == 0 or nums_left == 0:
            tree[feature_label][" >= " + str(feature_point)] = max_class_by_dataset(dataset)
            tree[feature_label][" < " + str(feature_point)] = max_class_by_dataset(dataset)
        else:
            tree[feature_label][" >= " + str(feature_point)] = self.create_tree_cart(right, labels)
            tree[feature_label][" < " + str(feature_point)] = self.create_tree_cart(left, labels)

        return tree

    '''
    预测
    '''

    def predict(self, tree, vector):
        k = 0
        while type(tree).__name__ == "dict":
            for i1, j1 in tree.items():
                nums = 0
                for i2, j2 in j1.items():
                    nums += 1
                    feature_value = vector[self.labels.index(i1)]
                    if eval(str(feature_value) + str(i2)):
                        tree = j2
                        k += 1
                        break
                    if nums >= len(j1):
                        key_list = list(j1.keys())
                        key_list.sort()
                        re_compile = re.compile(r"[=|\s]+")
                        for i in range(len(key_list) - 1):
                            before = float(re.sub(re_compile, "", str(key_list[i])))
                            after = float(re.sub(re_compile, "", str(key_list[i + 1])))
                            if before < feature_value < after:
                                if feature_value - before < after - feature_value:
                                    tree = j1.get(key_list[i])
                                else:
                                    tree = j1.get(key_list[i + 1])
                            elif i + 1 == len(key_list) - 1:
                                tree = j1.get(key_list[i + 1])
        else:
            return tree


if __name__ == "__main__":
    dataset, labels = get_data()
    '''
    dataset: 待切分的数据集
    test_size: 测试集比例
    random_state: 随机种子
    '''
    x_train, x_test = train_test_split(dataset, test_size=0.25, random_state=None)
    '''
    构建决策树
    '''
    tree = DecisionTree(x_train, labels, "create_tree")
    decision_tree = tree.main()
    print(decision_tree)
    '''
    预测
    '''
    count_correct = 0  # 预测正确个数
    for vector in x_test:
        predict = tree.predict(decision_tree, vector)
        print("真实值： " + str(vector[-1]) + "\t预测值： " + str(predict))
        if predict == vector[-1]:
            count_correct += 1
    print("正确率： " + str(count_correct / len(x_test)))
