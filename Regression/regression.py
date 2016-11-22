# encoding: utf-8

from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name):      # general function to parse tab -delimited floats
    num_feat = len(open(file_name).readline().split('\t')) - 1    # get number of fields
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr =[]
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


# 标准线性回归
def stand_regres(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T*x_mat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (x_mat.T*y_mat)
    return ws


# 局部加权线性回归
def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye(m))
    for j in range(m):                        # next 2 lines create weights matrix
        diff_mat = test_point - x_mat[j, :]     #
        weights[j, j] = exp(diff_mat*diff_mat.T/(-2.0*k**2))        # gauss
    xTx = x_mat.T * (weights * x_mat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat_arr):     # yArr and yHatArr both need to be arrays
    return ((y_arr-y_hat_arr)**2).sum()


# 缩减系数--岭回归
def ridge_regres(x_mat, y_mat, lam=0.2):
    xTx = x_mat.T*x_mat
    denom = xTx + eye(shape(x_mat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (x_mat.T*y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean     # to eliminate X0 take mean off of Y
    # regularize X's
    x_means = mean(x_mat, 0)   # calc mean then subtract it off
    x_var = var(x_mat, 0)      # calc variance of Xi then divide by it
    x_mat = (x_mat - x_means)/x_var
    num_test_pts = 30
    w_mat = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regres(x_mat, y_mat, exp(i-10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):   # regularize by columns
    in_mat = x_mat.copy()
    in_means = mean(in_mat, 0)   # calc mean then subtract it off 按列取平均值
    in_var = var(in_mat, 0)      # calc variance of Xi then divide by it
    in_mat = (in_mat - in_means)/in_var
    return in_mat


# 前向逐步回归--贪心算法
def stage_wise(x_arr, y_arr, eps=0.01, numIt=100):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean  # can also regularize ys but will get smaller coef
    x_mat = regularize(x_mat)
    m, n = shape(x_mat)
    return_mat = zeros((numIt, n))  # testing code remove
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(numIt):
        print ws.T
        lowest_error = inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


if __name__ == '__main__':
    # x_arr, y_arr = load_data_set('ex0.txt')
    '''
    ws = stand_regres(x_arr, y_arr)
    print(ws)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    y_hat = x_mat*ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])

    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy*ws
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()

    y_hat = x_mat*ws
    print(corrcoef(y_hat.T, y_mat))
    '''

    '''
    print(lwlr(x_arr[0], x_arr, y_arr, 1.0))
    print(lwlr(x_arr[0], x_arr, y_arr, 0.001))

    y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.01)
    x_mat = mat(x_arr)
    srt_ind = x_mat[:, 1].argsort(0)
    x_sort = x_mat[srt_ind][:, 0, :]    # sort

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[srt_ind])
    ax.scatter(x_mat[:, 1].flatten().A[0], mat(y_arr).T.flatten().A[0], s=2, c='red')
    plt.show()
    '''

    '''
    abx, aby = load_data_set('abalone.txt')
    y_hat01 = lwlr_test(abx[:99], abx[:99], aby[:99], 0.1)
    y_hat1 = lwlr_test(abx[:99], abx[:99], aby[:99], 1)
    y_hat10 = lwlr_test(abx[:99], abx[:99], aby[:99], 10)

    print(rss_error(aby[:99], y_hat01.T))
    print(rss_error(aby[:99], y_hat1.T))
    print(rss_error(aby[:99], y_hat10.T))

    y_hat01 = lwlr_test(abx[100:199], abx[:99], aby[:99], 0.1)
    y_hat1 = lwlr_test(abx[100:199], abx[:99], aby[:99], 1)
    y_hat10 = lwlr_test(abx[100:199], abx[:99], aby[:99], 10)

    print(rss_error(aby[100:199], y_hat01.T))
    print(rss_error(aby[100:199], y_hat1.T))
    print(rss_error(aby[100:199], y_hat10.T))

    ws = stand_regres(abx[0:99], aby[0:99])
    y_hat = mat(abx[100:199])*ws
    print(rss_error(aby[100:199], y_hat.T.A))
    '''

    '''
    abx, aby = load_data_set('abalone.txt')
    ridge_weights = ridge_test(abx, aby)
    print(ridge_weights)

    flg = plt.figure()
    ax = flg.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()
    '''

    x_arr, y_arr = load_data_set('abalone.txt')
    stage_wise(x_arr, y_arr, 0.01, 200)
