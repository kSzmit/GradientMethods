import numpy as np
import math
import random
from sklearn import linear_model, metrics
from basicCase.Methods import compute_p_i, saga, sag, svrg, sgd, bgd, MSE, sym, plot_mse
from basicCase.Utils import generate_cov_matrix, timer
import matplotlib.pyplot as plt


### Exercise 1 ###

### Initial parameters ###

methods = np.array(["saga", "sag", "svrg", "sgd"])
d = 20
gamma = None
n_vec = np.array([200, 400, 600, 800, 1000])
iter = 10

rho = 0.3
cov_type = "id"

### Each plot must be executed independently (you have to comment another plots code)

### MSE plot ###

if __name__ == "__main__":
    print(plot_mse(methods=methods, d=d, n_vec=n_vec, rho=rho, cov_type=cov_type, repeat=iter))

### Execution time plot ###

# results = np.zeros((len(n_vec), len(methods)))
# k = 0
# for n in n_vec:
#     mean_vec = np.repeat(0, d)
#     cov_matrix = generate_cov_matrix(d, rho, cov_type)
#     x = np.random.multivariate_normal(mean_vec, cov_matrix, n)
#     beta = np.random.uniform(-2, 2, d)
#     p = [compute_p_i(x[i], beta) for i in range(len(x))]
#     y = np.random.binomial(np.ones((len(x),), dtype=int), p)
#     j = 0
#     for m in methods:
#         results[k, j] = timer(m+"(x, y, gamma)", "from __main__ import x, y, gamma;" +
#                                            "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                            "sgd, bgd; from basicCase.Utils import timer", repeat=iter)
#         j = j+1
#     k = k+1
# plt.xlabel('N')
# plt.ylabel('Time')
# plt.title('RHO = ' + str(rho))
# for i in range(len(methods)):
#     plt.plot(n_vec, results[:,i], '.-', label=methods[i])
# leg = plt.legend(loc='best', ncol=2)
# leg.get_frame().set_alpha(0.5)
# plt.show()


### Exercise 2 ###

### MSE plot ###

# if __name__ == "__main__":
#      print(plot_mse(methods=methods, d=d, n_vec=n_vec, rho=rho, cov_type=cov_type, lasso=1, repeat=iter))

### Execution time plot ###

# results = np.zeros((len(n_vec), len(methods)))
# k = 0
# for n in n_vec:
#     mean_vec = np.repeat(0, d)
#     cov_matrix = generate_cov_matrix(d, rho, cov_type)
#     x = np.random.multivariate_normal(mean_vec, cov_matrix, n)
#     beta = np.random.binomial(np.ones((d,), dtype=int), 0.25)
#     beta_vals = np.concatenate([np.array(np.random.uniform(-2, -1, math.floor(d / 2))), np.array(np.random.uniform(1, 2, math.ceil(d / 2)))])
#     beta = np.multiply(beta_vals, beta)
#     p = [compute_p_i(x[i], beta) for i in range(len(x))]
#     y = np.random.binomial(np.ones((len(x),), dtype=int), p)
#     lassocv = linear_model.LassoCV(fit_intercept=fit_intercept, alphas=np.arange(0.001, 0.05, 0.001))
#     lassocv.fit(x, y)
#     lam = lassocv.alpha_ / 10
#     j = 0
#     for m in methods:
#         results[k, j] = timer(m+"(x, y, proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                            "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                            "sgd, bgd; from basicCase.Utils import timer", repeat=iter)
#         j = j+1
#     k = k+1
# plt.xlabel('N')
# plt.ylabel('Time')
# plt.title('LASSO - RHO = ' + str(rho))
# for i in range(len(methods)):
#     plt.plot(n_vec, results[:,i], '.-', label=methods[i])
# leg = plt.legend(loc='best', ncol=2)
# leg.get_frame().set_alpha(0.5)
# plt.show()


### Exercise 2 - tests ###

# beta = np.random.binomial(np.ones((d,), dtype=int), 0.25)
# beta_vals = np.concatenate([np.array(np.random.uniform(-2, -1, math.floor(d/2))),np.array(np.random.uniform(1, 2, math.ceil(d/2)))])
# beta = np.multiply(beta_vals,beta)
# # beta = [-1.03495018,-0.,-0.,-0.,-0.,-0. ,-1.63825899,-1.75587161,-0.,-0.,0.,1.17102776 ,0.,	0.,	0.,	0.,	1.44381657,	0. ,0.,	0.	]
# p = [compute_p_i(x[i], beta) for i in range(len(x))]
# y = np.random.binomial(np.ones((len(x),), dtype=int), p)
# lassocv = linear_model.LassoCV( fit_intercept=False, alphas=np.arange(0.001,0.05, 0.001))
# lassocv.fit(x, y)
# lam = lassocv.alpha_
#
# if __name__ == "__main__":
#      print("--------------------------------------------------------")
#      print("EXERCISE 2")
#      print("--------------------------------------------------------")
#      print("PROJECT RESULTS - MSE")
#
#      print("10 iter for saga  proxy MSE: ", sym(method="saga", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
#      print("10 iter for sag  proxy MSE: ", sym(method="sag", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
#      print("10 iter for svrg  proxy MSE: ", sym(method="svrg", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
#      print("10 iter for sgd  proxy MSE: ", sym(method="sgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
#      #print("10 iter for bgd  proxy MSE: ", sym(method="bgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
#
#      print("--------------------------------------------------------")
#      print("EXERCISE 2 no PROXY")
#      print("--------------------------------------------------------")
#      print("PROJECT RESULTS - MSE")
#      print("10 iter for saga no proxy MSE: ", sym(method="saga", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
#      print("10 iter for sag no proxy MSE: ", sym(method="sag", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
#      print("10 iter for svrg no proxy MSE: ", sym(method="svrg", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
#      print("10 iter for sgd no proxy MSE: ", sym(method="sgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
#      #print("10 iter for bgd no proxy MSE: ", sym(method="bgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
#
#      print("--------------------------------------------------------")
#      print("PROJECT RESULTS - TIME")
#      print("--------------------------------------------------------")
#      print("SAGA:", timer("saga(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
#      print("SAG:", timer("sag(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
#      print("SVRG:", timer("svrg(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
#      print("SGD:", timer("stochastic_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
#      print("BGD:", timer("batch_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
