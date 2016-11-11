# encoding: utf-8

from numpy import *


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0/(1+exp(-in_x))


# 逻辑回归梯度上升算法
def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)               # convert to NumPy matrix
    label_mat = mat(class_labels).transpose()    # convert to NumPy matrix
    m, n = shape(data_matrix)
    alpha = 0.001
    # 迭代次数
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):                     # heavy on matrix operations
        h = sigmoid(data_matrix*weights)            # matrix mult 整个矩阵
        error = (label_mat - h)                     # vector subtraction
        weights = weights + alpha * data_matrix.transpose() * error  # matrix mult
    return weights


# 拟合直线
def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stoc_grad_ascent0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)           # initialize to all ones
    for i in range(m):          # 遍历每个样本
        h = sigmoid(sum(data_matrix[i]*weights))
        error = class_labels[i] - h
        weights += alpha*error*data_matrix[i]
    return weights


# 随机梯度上升算法
def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)                       # initialize to all ones
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # apha decreases with iteration, does not
            rand_index = int(random.uniform(0, len(data_index)))     # go to 0 because of the constant
            h = sigmoid(sum(data_matrix[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 从疝气病症预测马的死亡率
def colic_test():
    fr_train = open('horseColicTraining.txt')       # 训练数据
    fr_test = open('horseColicTest.txt')            # 测试数据
    training_set = []
    training_labels = []
    # 读取训练数据
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 计算回归系数
    train_weights = stoc_grad_ascent1(array(training_set), training_labels, 1000)
    # 在测试数据上验证
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = (float(error_count)/num_test_vec)
    print "the error rate of this test is: %f" % error_rate
    return error_rate


# 多次测试取平均值
def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print "after %d iterations the average error rate is: %f" % (num_tests, error_sum/float(num_tests))


if __name__ == '__main__':
    '''
    data_arr, label_mat = load_data_set()
    # weights = grad_ascent(data_arr, label_mat)
    # weights = stoc_grad_ascent0(array(data_arr), label_mat)
    weights = stoc_grad_ascent1(array(data_arr), label_mat)
    print(weights)
    plot_best_fit(weights)
    '''
    multi_test()
