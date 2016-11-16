# encoding: utf-8


from numpy import *


def load_simp_data():
    dat_mat = matrix([[1.,  2.1],
                      [2.,  1.1],
                      [1.3,  1.],
                      [1.,  1.],
                      [2.,  1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    """
    just classify the data
    :param data_matrix: 数据矩阵
    :param dimen: 矩阵的每一列下标
    :param thresh_val: 阈值
    :param thresh_ineq: lt gt
    :return: 分类结果
    """
    ret_array = ones((shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


# 单层决策树生成函数
def build_stump(data_arr, class_labels, D):
    """
    单层决策树生成函数
    :param data_arr: 数据
    :param class_labels: 分类
    :param D: 给定权重向量
    :return: 字典, 错误率, 类别估计值
    """
    data_matrix = mat(data_arr)
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_clas_est = mat(zeros((m, 1)))
    min_error = inf     # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max-range_min)/num_steps
        for j in range(-1,int(num_steps)+1):    # loop over all range in current dimension
            for inequal in ['lt', 'gt']:        # go over less than and greater than
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = mat(ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T*err_arr     # calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \
                # (i, thresh_val, inequal, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_clas_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_clas_est


# 基于单层决策树的AdaBoost训练过程
def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    weak_class_arr = []     # 多个弱分类器
    m = shape(data_arr)[0]
    D = mat(ones((m, 1))/m)   # init D to all equal
    agg_class_est = mat(zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)   # build Stump
        # print "D:", D.T
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16))) # calc alpha, throw in max(error,eps) to account for error=0
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)                  # store Stump Params in Array
        # print "classEst: ", class_est.T
        expon = multiply(-1*alpha*mat(class_labels).T, class_est)    # exponent for D calc, getting messy
        D = multiply(D, exp(expon))                                  # Calc New D for next iteration
        D = D/D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        agg_class_est += alpha*class_est
        # print "aggClassEst: ", agg_class_est.T
        agg_errors = multiply(sign(agg_class_est) != mat(class_labels).T, ones((m, 1)))
        error_rate = agg_errors.sum()/m
        # print "total error: ", error_rate
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


# AdaBoost分类函数
def ada_classify(dat_to_class, classifier_arr):
    data_matrix = mat(dat_to_class)    # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(data_matrix)[0]
    agg_class_est = mat(zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])  # call stump classify
        agg_class_est += classifier_arr[i]['alpha']*class_est
        # print agg_class_est
    return sign(agg_class_est)


if __name__ == '__main__':
    dat_mat, class_labels = load_simp_data()
    # print(dat_mat[:, 0])
    '''
    D = mat(ones((5, 1))/5)
    best_stump, min_error, best_clas_est = build_stump(dat_mat, class_labels, D)
    print(best_stump, min_error, best_clas_est)

    print(mat(class_labels).T)
    print(multiply(mat(class_labels).T, best_clas_est))
    '''
    weak_class_arr, agg_class_est = ada_boost_train_ds(dat_mat, class_labels, 9)
    # print(weak_class_arr)
    print(ada_classify([0, 0], weak_class_arr))
    print(ada_classify([[5, 5], [0, 0]], weak_class_arr))
