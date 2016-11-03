# encoding: utf-8
from math import log
import operator


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


# 香农信息熵计算: 不确定性越大，熵值越大
def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# 按照给定特征划分数据集
def split_data_set(data_set, axis, value):
    """
    :param data_set: 待划分数据集
    :param axis: 待划分数据集的特征
    :param value: 需要返回的特征值
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


# 选择最好的数据集划分方式
def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1         # 特征个数，最后一列是label
    base_entropy = calc_shannon_ent(data_set)   # 原始香农熵
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):               # 遍历每个特征
        feat_list = [example[i] for example in data_set]    # 这个特征的所有值
        unique_vals = set(feat_list)             # 特征值得集合，为了去重
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy      # 信息增益
        if info_gain > best_info_gain:              # 找出最大信息增益
            best_info_gain = info_gain
            best_feature = i
    return best_feature                      # 返回最大信息增益的特征


# 选举叶子节点的分类
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建决策树
def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]  # 分类列表
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]                        # 所有分类都是一样, 停止
    if len(data_set[0]) == 1:                       # 遍历完所有特征， 停止
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)  # 选择最好的特征划分数据集
    best_feat_label = labels[best_feat]                 # 最好特征的标签
    my_tree = {best_feat_label: {}}                     # 返回结果
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]  # 选出特征的所有值
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]          # 复制
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


# 使用决策树分类
def classify(input_tree, feat_labels, test_vec):
    """
    :param input_tree: 决策树
    :param feat_labels: 特征标签
    :param test_vec: 测试特征值
    :return: 分类
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)   # 标签字符串转换成索引
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, feat_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label


if __name__ == '__main__':
    my_dat, labels = create_data_set()
    # print(calc_shannon_ent(my_dat))
    # print(split_data_set(my_dat, 0, 1))
    # print(split_data_set(my_dat, 0, 0))
    # print(choose_best_feature_to_split(my_dat))
    print(create_tree(my_dat, labels))
