# encoding: utf-8

from numpy import *


def load_data_set(file_name):  # general function to parse tab -delimited floats
    data_mat = []  # assume last column is target value
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = map(float, cur_line)  # map all elements to float()
        data_mat.append(flt_line)
    return data_mat


def bin_split_data_set(data_set, feature, value):
    try:
        mat0 = data_set[nonzero(data_set[:, feature] > value)[0], :][0]
    except Exception, e:
        mat0 = mat([[]])
    try:
        mat1 = data_set[nonzero(data_set[:, feature] <= value)[0], :][0]
    except Exception, e:
        mat1 = mat([[]])
    return mat0, mat1


def reg_leaf(data_set):  # returns the value used for each leaf
    return mean(data_set[:, -1])


def reg_err(data_set):
    return var(data_set[:, -1]) * shape(data_set)[0]


def linear_solve(data_set):  # helper function used in two places
    m, n = shape(data_set)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = data_set[:, 0:n - 1]
    Y = data_set[:, -1]  # and strip out Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def model_leaf(data_set):  # create linear model and return coeficients
    ws, X, Y = linear_solve(data_set)
    return ws


def model_err(data_set):
    ws, X, Y = linear_solve(data_set)
    y_hat = X * ws
    return sum(power(Y - y_hat, 2))


# 回归树的切分函数
def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    tol_s = ops[0]   # 容许的误差下降值
    tol_n = ops[1]   # 切分的最少样本数
    # if all the target variables are the same value: quit and return value
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leaf_type(data_set)
    m, n = shape(data_set)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    s = err_type(data_set)
    best_s = inf
    best_index = 0
    best_value = 0
    for feat_index in range(n - 1):
        for split_val in set(data_set[:, feat_index].T.tolist()[0]):
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val)
            if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (s - best_s) < tol_s:
        return None, leaf_type(data_set)  # exit cond 2
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)
    if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):  # exit cond 3
        return None, leaf_type(data_set)
    return best_index, best_value  # returns the best feature to split on and the value used for that split


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)  # choose the best split
    if feat == None:
        return val  # if the splitting hit a stop condition return val
    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val
    l_set, r_set = bin_split_data_set(data_set, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 回归树剪枝
def prune(tree, test_data):
    if shape(test_data)[0] == 0:
        return get_mean(tree)  # if we have no test data collapse the tree
    if is_tree(tree['right']) or is_tree(tree['left']):  # if the branches are not trees try to prune them
        l_set, r_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
    # if they are now both leafs, see if we can merge them
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_set, r_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
        error_no_merge = sum(power(l_set[:, -1] - tree['left'], 2)) + \
                       sum(power(r_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = sum(power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print "merging"
            return tree_mean
        else:
            return tree
    else:
        return tree


def reg_tree_eval(model, in_dat):
    return float(model)


def model_tree_eval(model, in_dat):
    n = shape(in_dat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = in_dat
    return float(X * model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree, in_data)
    if in_data[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_fore_cast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)
    else:
        if is_tree(tree['right']):
            return tree_fore_cast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    m = len(test_data)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = tree_fore_cast(tree, mat(test_data[i]), model_eval)
    return yHat


if __name__ == '__main__':
    my_data = load_data_set('ex00.txt')
    my_mat = mat(my_data)
    tree = create_tree(my_mat)
    print(tree)
