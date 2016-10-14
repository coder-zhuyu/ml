# encoding: utf-8

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


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


# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())           # 文件行数
    return_mat = zeros((number_of_lines, 3))        # 以0填充的矩阵, number_of_lines x 3
    class_label_vector = []                         # 分类标签向量
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()                                     # 去掉回车
        list_from_line = line.split('\t')                       # 分割
        return_mat[index, :] = list_from_line[0:3]              # 数据存到特征矩阵的每一行
        class_label_vector.append(int(list_from_line[-1]))      # 标签
        index += 1
    return return_mat, class_label_vector


# 特征值归一化
def auto_norm(data_set):
    min_vals = data_set.min(0)      # 每列取一个最小值
    max_vals = data_set.max(0)      # 每列取一个最大值
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))       # 矩阵每个元素减去最小值
    norm_data_set = norm_data_set/tile(ranges, (m, 1))      # 矩阵每个元素除以max-min
    return norm_data_set, ranges, min_vals


# 分类器针对约会网站的测试代码
def dating_class_test():
    ho_ratio = 0.10     # 测试数据比例
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')      # 读取数据
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)     # 特征值归一化
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)     # 测试数据条数
    error_count = 0.0
    for i in range(num_test_vecs):
        # 分类
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 3)

        print("the classifier came back with: %d, the real answer is: %d" %
              (classifier_result, dating_labels[i]))
        # 分类错误
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print(error_count)
    print("the total error rate is: %f" % (error_count/float(num_test_vecs)))


# 约会网站预测函数
def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input("percentage of time spent playing video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr-min_vals)/ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person: ", result_list[classifier_result - 1])


# 手写识别系统
# 32x32的二进制图像矩阵转换为1x1024的向量
def img2vector(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32*i+j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')           # load the training set
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]     # take off .txt
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)
    test_filelist = listdir('testDigits')        # iterate through the test set
    error_count = 0.0
    m_test = len(test_filelist)
    for i in range(m_test):
        file_name_str = test_filelist[i]
        file_str = file_name_str.split('.')[0]     # take off .txt
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count/float(m_test)))

if __name__ == '__main__':
    '''
    group, labels = create_data_set()
    result = classify0([0, 0], group, labels, 3)
    print(result)
    '''

    '''
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    # print(dating_data_mat)
    # print(dating_labels)
    '''

    '''
    # 散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)   # 添加子图1 x 1, 第1个
    # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])        # 矩阵第二、三列数据
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
               15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plt.show()
    '''

    '''
    # 归一化
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    print(norm_mat)
    print(ranges)
    print(min_vals)
    '''

    # dating_class_test()
    # classify_person()

    handwriting_class_test()
