# encoding: utf-8

from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    kNN: k Nearest Neighbors
    :param in_x: vector to compare to existing dataset (1xN)
    :param data_set: size m data set of known vectors (NxM)
    :param labels: data set labels (1xM vector)
    :param k: number of neighbors to use for comparison (should be an odd number)
    :return: the most popular class label
    """

    data_set_size = data_set.shape[0]   # 维度(m, n)
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set    # tile构造data_set_size x 1 个copy
    sq_diff_mat = diff_mat**2   # 矩阵每个值求平方
    sq_distances = sq_diff_mat.sum(axis=1)  # 矩阵行求和
    distances = sq_distances**0.5   # 开根号
    sorted_dist_indicies = distances.argsort()  # 从小到大排序后的下标
    class_count = {}    # 从最近的k各结果统计各个label的出现次数
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)  # 次数排序
    return sorted_class_count[0][0]     # 分类结果


if __name__ == '__main__':
    group, labels = create_data_set()
    result = classify0([0, 0], group, labels, 3)
    print(result)
