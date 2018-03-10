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


def sag(beta, matrix_x, vector_y, gamma, proximal_op=None, lam=None, divisor_param=None):
    if divisor_param is None:
        divisor_param = len(matrix_x)

    n = len(matrix_x)
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


def saga(beta, matrix_x, vector_y, gamma, proximal_op=None, lam=None):
    return sag(beta, matrix_x, vector_y, gamma, proximal_op, lam, 1)


def svrg(beta, matrix_x, vector_y, gamma, proximal_op=None, lam=None):
    n = len(matrix_x)
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


n1_mean = [0, 0]
n1_cov = [[1, 0], [0, 1]]
ile = 1000
x = np.random.multivariate_normal(n1_mean, n1_cov, ile)
beta = [1, 5]
p = [compute_p_i(x[i],beta) for i in range(len(x))]
y = np.random.binomial(np.ones((len(x),), dtype=int), p)

print("maine saga:",saga([1, 1], x, y, 1/10))
print("maine sag:",sag([1, 1], x, y, 1/10))
print("maine svrg:",svrg([1, 1], x, y, 1/10))
lr = linear_model.LogisticRegression(solver='sag')
fit = lr.fit(x,y)
print("not maine fit:", lr.coef_)
