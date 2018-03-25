import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
from basicCase.Utils import AveragableTable, SnapshotListHolder, row_norms, generate_cov_matrix, timer
from joblib import Parallel, delayed
import multiprocessing


def compute_p_i(x_i, beta):
    exponent = np.dot(x_i, beta)
    exp_result = math.exp(exponent)
    return exp_result / (1 + exp_result)


def compute_derivative_f_i(x_i, y_i, beta):
    p_i = compute_p_i(x_i, beta)
    return p_i * x_i - y_i * x_i


def average_of_derivatives(matrix_x, vector_y, beta):
    avg_vec = compute_derivative_f_i(matrix_x[0], vector_y[0], beta)
    for i in range(1, len(matrix_x)):
        avg_vec += compute_derivative_f_i(matrix_x[i], vector_y[i], beta)
    return avg_vec/len(matrix_x)


def sag_step_size(max_squared_sum, is_lasso, fit_intercept, is_saga, samples_num):
    alpha = float(not is_lasso) / samples_num
    L = (0.25 * (max_squared_sum + int(fit_intercept)) + alpha)
    if is_saga:
        mun = min(2 * samples_num * alpha, L)
        step = 1. / (2 * L + mun)
    else:
        step = 1. / L
    return step


def svrg_step_size(max_squared_sum, is_lasso, fit_intercept, samples_num, m):
    alpha = float(not is_lasso) / samples_num
    L = (0.25 * (max_squared_sum + int(fit_intercept)) + alpha)
    return 1. / ( L * m )


def lasso_proxy_operator(beta, gamma, lam):
    if lam is None or beta is None or gamma is None:
        raise ValueError("Error! Not expected None values: beta = " + str(beta) + ", gamma = " + str(gamma)
                         + ", lambda = " + str(lam))
    return [ np.sign(x)*max((math.fabs(x) - gamma*lam), 0) for x in beta]


def evaluate_stop_condition(current_beta, beta_snapshot, current_iter, max_iter, treshold=0.0001):
    if current_iter >= max_iter:
        return False
    elif current_iter == 0:
        return True
    elif metrics.mean_squared_error(current_beta, beta_snapshot.get_snapshot()) > treshold:
        beta_snapshot.take_snapshot(current_beta)
        return True
    else:
        return False


def sag(matrix_x, vector_y, gamma=None, proximal_op=None, lam=None, divisor_param=None, fit_intercept=False, max_iter=100, treshold=0.0001):
    if divisor_param is None:
        divisor_param = len(matrix_x)

    if gamma is None:
        max_sq_sum = row_norms(matrix_x, True).max()
        gamma = sag_step_size(max_sq_sum, is_lasso=(proximal_op is not None), fit_intercept=fit_intercept,
                              is_saga=(divisor_param == 1), samples_num=len(matrix_x))

    n = len(matrix_x)
    beta = np.repeat(0, matrix_x.shape[1])
    average_table = AveragableTable([compute_derivative_f_i(matrix_x[i], vector_y[i], beta) for i in range(n)])

    estimated_beta = beta
    beta_snapshot = SnapshotListHolder(estimated_beta)
    current_iter = 0

    while evaluate_stop_condition(estimated_beta, beta_snapshot, current_iter, max_iter, treshold=treshold):
        for i in range(n):
            j = random.randrange(n)

            derivative_phi_j_old = average_table.replace_item(j, compute_derivative_f_i(matrix_x[j], vector_y[j],
                                                                                        estimated_beta))
            estimated_beta = estimated_beta - gamma * (
                (average_table.get_item(j) - derivative_phi_j_old)/divisor_param + average_table.average)
            # only lasso prox operator
            if proximal_op is not None:
                estimated_beta = lasso_proxy_operator(estimated_beta, gamma, lam)
        current_iter += 1

    return estimated_beta


def saga(matrix_x, vector_y, gamma=None, proximal_op=None, lam=None, fit_intercept=1):
    return sag(matrix_x, vector_y, gamma, proximal_op, lam, 1, fit_intercept)


# internal loop length should be O(n) usually few times bigger, m will be used to determine loop length
# defined as m*n. for convex case it is advised to use 2 and for non-convex 5
def svrg(matrix_x, vector_y, gamma=None, proximal_op=None, lam=None, fit_intercept=False, max_iter=100, m=2):
    n = len(matrix_x)

    if gamma is None:
        max_sq_sum = row_norms(matrix_x, True).max()
        gamma = svrg_step_size(max_sq_sum, is_lasso=(proximal_op is not None), fit_intercept=fit_intercept,
                               samples_num=len(matrix_x), m=m)

    beta = np.repeat(0, matrix_x.shape[1])
    estimated_beta, tmp_beta = beta, beta

    beta_snapshot = SnapshotListHolder(estimated_beta)
    current_iter = 0

    while evaluate_stop_condition(estimated_beta, beta_snapshot, current_iter, max_iter, treshold=0.0001*m):
        gradient_avg = average_of_derivatives(matrix_x, vector_y, estimated_beta)

        for t in range(m * n):
            j = random.randrange(n)

            tmp_beta_derivative = compute_derivative_f_i(matrix_x[j], vector_y[j], tmp_beta)
            estimated_beta_derivative = compute_derivative_f_i(matrix_x[j], vector_y[j], estimated_beta)

            tmp_beta = tmp_beta - gamma*(tmp_beta_derivative - estimated_beta_derivative + gradient_avg)

            if proximal_op is not None:
                tmp_beta = lasso_proxy_operator(tmp_beta, gamma, lam)

        estimated_beta = tmp_beta
        current_iter += 1

    return estimated_beta


# very slow
def bgd(matrix_x, vector_y, gamma=None, proximal_op=None, lam=None, fit_intercept=False, max_iter=100):
    n = len(matrix_x)

    if gamma is None:
        max_sq_sum = row_norms(matrix_x, True).max()
        gamma = svrg_step_size(max_sq_sum, is_lasso=(proximal_op is not None), fit_intercept=fit_intercept,
                               samples_num=len(matrix_x), m=1)

    estimated_beta = np.repeat(0, matrix_x.shape[1])

    beta_snapshot = SnapshotListHolder(estimated_beta)
    current_iter = 0

    while evaluate_stop_condition(estimated_beta, beta_snapshot, current_iter, max_iter):
        for i in range(n):
            gradient_avg = average_of_derivatives(matrix_x, vector_y, estimated_beta)
            estimated_beta = estimated_beta - gamma*gradient_avg
            if proximal_op is not None:
                estimated_beta = lasso_proxy_operator(estimated_beta, gamma, lam)

        current_iter += 1

    return estimated_beta


def sgd(matrix_x, vector_y, gamma=None, proximal_op=None, lam=None, fit_intercept=False, max_iter=100):
    n = len(matrix_x)

    if gamma is None:
        max_sq_sum = row_norms(matrix_x, True).max()
        gamma = svrg_step_size(max_sq_sum, is_lasso=(proximal_op is not None), fit_intercept=fit_intercept,
                               samples_num=len(matrix_x), m=1)

    estimated_beta = np.repeat(0, matrix_x.shape[1])
    index_list = list(range(len(matrix_x)))

    beta_snapshot = SnapshotListHolder(estimated_beta)
    current_iter = 0

    while evaluate_stop_condition(estimated_beta, beta_snapshot, current_iter, max_iter):
        np.random.shuffle(index_list)
        for index in index_list:
            single_gradient = compute_derivative_f_i(matrix_x[index], vector_y[index], estimated_beta)
            estimated_beta = estimated_beta - gamma*single_gradient
            if proximal_op is not None:
                estimated_beta = lasso_proxy_operator(estimated_beta, gamma, lam)
        current_iter += 1
    return estimated_beta


def MSE(method, beta, matrix_x, vector_y, gamma=None, proximal_op=None, lam=None,  fit_intercept=False):
    if method=="saga":
        mse= metrics.mean_squared_error(beta, saga(matrix_x, vector_y, gamma, proximal_op, lam,  fit_intercept))
    elif method=="sag":
        mse = metrics.mean_squared_error(beta, sag(matrix_x, vector_y,gamma,  proximal_op, lam, None, fit_intercept))
    elif method=="svrg":
        mse = metrics.mean_squared_error(beta, svrg(matrix_x, vector_y,gamma,  proximal_op, lam, fit_intercept))
    elif method=="sgd":
        mse = metrics.mean_squared_error(beta, sgd(matrix_x, vector_y, gamma, proximal_op, lam,  fit_intercept))
    elif method=="bgd":
        mse = metrics.mean_squared_error(beta, bgd(matrix_x, vector_y, gamma, proximal_op, lam,  fit_intercept))
    else:
        mse=0
    return mse


def sym(method, beta, x, y, gamma, proximal_op, lam, fit_intercept, iter):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores - 1)(delayed(MSE)(method=method, beta=beta, matrix_x=x, vector_y=y, gamma=gamma,
                                                          proximal_op=proximal_op, lam=lam, fit_intercept=fit_intercept)
                                                          for i in range(iter))
    return np.mean(results)


def plot_mse(methods, d, n_vec, rho, beta, cov_type, repeat, lasso=0, fit_intercept=1, lam=0.0015):
    results = np.zeros((len(n_vec), len(methods)))
    k = 0
    for n in n_vec:
        mean_vec = np.repeat(0, d-1)
        cov_matrix = generate_cov_matrix(d-1, rho, cov_type)
        x = np.concatenate((np.ones((n, 1)), np.random.multivariate_normal(mean_vec, cov_matrix, n)), axis=1)
        np.random.seed(100)
        p = [compute_p_i(x[i], beta) for i in range(len(x))]
        y = np.random.binomial(np.ones((len(x),), dtype=int), p)
        if lasso == 1:
            prox = 1
        else:
            lam = None
            prox = None
        j = 0
        for m in methods:
            results[k, j] = sym(method=m, beta=beta, x=x, y=y, gamma=None, proximal_op=prox, lam=lam, fit_intercept=fit_intercept, iter=repeat)
            j = j+1
        k = k+1
    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.title('RHO = ' + str(rho) + ', ' + cov_type+ ", proxy=" + str(prox) + ", lam = " + str(lam))
    for i in range(len(methods)):
        plt.plot(n_vec, results[:,i], '.-', label=methods[i])
    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.5)
    plt.show()
    return 0

