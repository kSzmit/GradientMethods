import numpy as np
import math
import random
from basicCase.Utils import AveragableTable
from sklearn import linear_model


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


def lasso_proxy_operator(beta, gamma, lam):
    if lam is None or beta is None or gamma is None:
        raise ValueError("Error! Not expected None values: beta = " + str(beta) + ", gamma = " + str(gamma)
                         + ", lambda = " + str(lam))
    return [ np.sign(x)*max((math.fabs(x) - gamma*lam), 0) for x in beta]


def sag(matrix_x, vector_y, gamma, proximal_op=None, lam=None, divisor_param=None):
    if divisor_param is None:
        divisor_param = len(matrix_x)

    n = len(matrix_x)
    beta = np.repeat(0, matrix_x.shape[1])
    average_table = AveragableTable([compute_derivative_f_i(matrix_x[i], vector_y[i], beta) for i in range(n)])

    estimated_beta = beta
    for k in range(10000):
        j = random.randrange(n)

        derivative_phi_j_old = average_table.replace_item(j, compute_derivative_f_i(matrix_x[j], vector_y[j],
                                                                                    estimated_beta))
        estimated_beta = estimated_beta - gamma * (
            (average_table.get_item(j) - derivative_phi_j_old)/divisor_param + average_table.average)
        # only lasso prox operator
        if proximal_op is not None:
            estimated_beta = lasso_proxy_operator(beta, gamma, lam)

    return estimated_beta


def saga(matrix_x, vector_y, gamma, proximal_op=None, lam=None):
    return sag(matrix_x, vector_y, gamma, proximal_op, lam, 1)


def svrg(matrix_x, vector_y, gamma, proximal_op=None, lam=None):
    n = len(matrix_x)
    beta = np.repeat(0, matrix_x.shape[1])
    estimated_beta, tmp_beta = beta, beta

    for s in range(100):
        gradient_avg = average_of_derivatives(matrix_x, vector_y, estimated_beta)

        for t in range(100):
            j = random.randrange(n)

            tmp_beta_derivative = compute_derivative_f_i(matrix_x[j], vector_y[j], tmp_beta)
            estimated_beta_derivative = compute_derivative_f_i(matrix_x[j], vector_y[j], estimated_beta)

            tmp_beta = tmp_beta - gamma*(tmp_beta_derivative - estimated_beta_derivative + gradient_avg)

        # no idea if proxy should be applied in inner our outer loop
        if proximal_op is not None:
            tmp_beta = lasso_proxy_operator(beta, gamma, lam)

        estimated_beta = tmp_beta

    return estimated_beta

def batch_gradient_descent(matrix_x, vector_y, gamma):
    n = len(matrix_x)
    estimated_beta = np.repeat(0, matrix_x.shape[1])

    for k in range(1000):
        gradient_avg = average_of_derivatives(matrix_x, vector_y, estimated_beta)
        estimated_beta = estimated_beta - gamma*gradient_avg
    return estimated_beta

def stochastic_gradient_descent(matrix_x, vector_y, gamma):
    n = len(matrix_x)
    estimated_beta = np.repeat(0, matrix_x.shape[1])
    index_list = list(range(len(matrix_x)))
    for k in range(10):
        np.random.shuffle(index_list)
        for index in index_list:
            single_gradient = compute_derivative_f_i(matrix_x[index], vector_y[index], estimated_beta)
            estimated_beta = estimated_beta - gamma*single_gradient
    return estimated_beta