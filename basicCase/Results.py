import numpy as np
from sklearn import metrics
from sklearn import linear_model
from basicCase.Methods import compute_p_i, saga, sag, svrg, stochastic_gradient_descent, batch_gradient_descent, MSE, sym
from basicCase.Utils import generate_cov_matrix, timer
import math
import random

#Zadanie 1

d = 20
n = 1000
gamma = None
rho = 0.3

mean_vec = np.repeat(0, d)
cov_matrix_const = generate_cov_matrix(d, rho)
cov_matrix_autocorr = generate_cov_matrix(d, rho, "autocorrelation")
cov_matrix_id = np.eye(d)

x = np.random.multivariate_normal(mean_vec, cov_matrix_const, n)
beta = np.random.uniform(-2, 2, d)
p = [compute_p_i(x[i], beta) for i in range(len(x))]
y = np.random.binomial(np.ones((len(x),), dtype=int), p)

iter=10



# print("PROJECT RESULTS - TIME")
# print("--------------------------------------------------------")
# print("SAGA:", timer("saga(x, y, gamma)", "from __main__ import x, y, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("SAG:", timer("sag(x, y, gamma)", "from __main__ import x, y, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("SVRG:", timer("svrg(x, y, gamma)", "from __main__ import x, y, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("SGD:", timer("stochastic_gradient_descent(x, y, gamma)", "from __main__ import x, y, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("BGD:", timer("batch_gradient_descent(x, y, gamma)", "from __main__ import x, y, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +

if __name__ == "__main__":
     print("EXERCISE 1")
     print("--------------------------------------------------------")
     print("PROJECT RESULTS - MSE")

     print("10 iter for saga no proxy MSE: ", sym(method="saga", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     print("10 iter for sag no proxy MSE: ", sym(method="sag", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     print("10 iter for svrg no proxy MSE: ", sym(method="svrg", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     print("10 iter for sgd no proxy MSE: ", sym(method="sgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     #print("10 iter for bgd no proxy MSE: ", sym(method="bgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))





#Zadanie 2

beta = np.random.binomial(np.ones((d,), dtype=int), 0.25)
beta_vals= np.concatenate([np.array(np.random.uniform(-2, -1, math.floor(d/2))),np.array(np.random.uniform(1, 2, math.ceil(d/2)))])
beta=np.multiply(beta_vals,beta)
# beta=[-1.03495018,-0.,-0.,-0.,-0.,-0. ,-1.63825899,-1.75587161,-0.,-0.,0.,1.17102776 ,0.,	0.,	0.,	0.,	1.44381657,	0. ,0.,	0.	]
p = [compute_p_i(x[i], beta) for i in range(len(x))]
y = np.random.binomial(np.ones((len(x),), dtype=int), p)
lassocv = linear_model.LassoCV( fit_intercept=False, alphas=np.arange(0.001,0.05, 0.001))
lassocv.fit(x, y)
lam = lassocv.alpha_/10


if __name__ == "__main__":
     print("--------------------------------------------------------")
     print("EXERCISE 2")
     print("--------------------------------------------------------")
     print("PROJECT RESULTS - MSE")

     print("10 iter for saga  proxy MSE: ", sym(method="saga", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
     print("10 iter for sag  proxy MSE: ", sym(method="sag", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
     print("10 iter for svrg  proxy MSE: ", sym(method="svrg", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
     print("10 iter for sgd  proxy MSE: ", sym(method="sgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))
     #print("10 iter for bgd  proxy MSE: ", sym(method="bgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=1, lam=lam, fit_intercept=0, iter=iter))

     print("--------------------------------------------------------")
     print("EXERCISE 2 no PROXY")
     print("--------------------------------------------------------")
     print("PROJECT RESULTS - MSE")

     print("10 iter for saga no proxy MSE: ", sym(method="saga", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     print("10 iter for sag no proxy MSE: ", sym(method="sag", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     print("10 iter for svrg no proxy MSE: ", sym(method="svrg", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     print("10 iter for sgd no proxy MSE: ", sym(method="sgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))
     #print("10 iter for bgd no proxy MSE: ", sym(method="bgd", beta=beta, x=x, y=y, gamma=gamma, proximal_op=None, lam=None, fit_intercept=0, iter=iter))


#
# print("--------------------------------------------------------")
# print("PROJECT RESULTS - TIME")
# print("--------------------------------------------------------")
# print("SAGA:", timer("saga(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("SAG:", timer("sag(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("SVRG:", timer("svrg(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# print("SGD:", timer("stochastic_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
# # print("BGD:", timer("batch_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
# #                                            "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
# #                                            "stochastic_gradient_descent, batch_gradient_descent;" +
# #                                            "from basicCase.Utils import timer"), "sec.")
# print("--------------------------------------------------------")

