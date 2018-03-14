import numpy as np
from sklearn import metrics
from sklearn import linear_model
from basicCase.Methods import compute_p_i, saga, sag, svrg, stochastic_gradient_descent, batch_gradient_descent
from basicCase.Utils import generate_cov_matrix, timer

#Zadanie 1

d = 20
n = 500
gamma = None
rho = 0.3

mean_vec = np.repeat(0, d)
cov_matrix_const = generate_cov_matrix(d, rho)
cov_matrix_autocorr = generate_cov_matrix(d, rho, "autocorrelation")
cov_matrix_id = np.eye(d)

x = np.random.multivariate_normal(mean_vec, cov_matrix_const, n)
# beta = np.random.uniform(-2, 2, d)
# p = [compute_p_i(x[i], beta) for i in range(len(x))]
# y = np.random.binomial(np.ones((len(x),), dtype=int), p)
#
# print("EXERCISE 1")
# print("--------------------------------------------------------")
# print("PROJECT RESULTS - MSE")
# print("--------------------------------------------------------")
# print("SAGA:", metrics.mean_squared_error(beta, saga(x, y, gamma)))
# print("SAG:", metrics.mean_squared_error(beta, sag(x, y, gamma)))
# print("SVRG:", metrics.mean_squared_error(beta, svrg(x, y, gamma)))
# print("SGD:", metrics.mean_squared_error(beta, stochastic_gradient_descent(x, y, gamma)))
# print("BGD:", metrics.mean_squared_error(beta, batch_gradient_descent(x, y, gamma)))
# print("--------------------------------------------------------")
# print("IMPLEMENTED RESULTS - MSE")
# print("--------------------------------------------------------")
# lr_saga = linear_model.LogisticRegression(solver='saga')
# fit_saga = lr_saga.fit(x,y)
# print("SAGA:", metrics.mean_squared_error(beta, lr_saga.coef_[0]))
# lr_sag = linear_model.LogisticRegression(solver='sag')
# fit_sag = lr_sag.fit(x,y)
# print("SAG:", metrics.mean_squared_error(beta, lr_sag.coef_[0]))
# print("--------------------------------------------------------")
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
#                                           "from basicCase.Utils import timer"), "sec.")
# print("--------------------------------------------------------")
# print("IMPLEMENTED RESULTS - TIME")
# print("--------------------------------------------------------")
# print("SAGA:", timer("lr_saga = linear_model.LogisticRegression(solver='saga'); fit_saga = lr_saga.fit(x,y);" +
#                      "lr_saga.coef_", "from __main__ import x, y;" +
#                      "from sklearn import linear_model"), "sec.")
# print("SAG:", timer("lr_sag = linear_model.LogisticRegression(solver='sag'); fit_sag = lr_sag.fit(x,y);" +
#                      "lr_sag.coef_", "from __main__ import x, y;" +
#                      "from sklearn import linear_model"), "sec.")



#Zadanie 2


beta = np.random.binomial(np.ones((d,), dtype=int), 0.25)*4
print(beta)
p = [compute_p_i(x[i], beta) for i in range(len(x))]
y = np.random.binomial(np.ones((len(x),), dtype=int), p)

lassocv = linear_model.LassoCV()
lassocv.fit(x, y)
lam = lassocv.alpha_


print("EXERCISE 2")
print("lambda", lam )
print("--------------------------------------------------------")
print("PROJECT RESULTS - MSE")
print("--------------------------------------------------------")
print("SAGA:", metrics.mean_squared_error(beta, saga(x, y,proximal_op=1, lam=lam, gamma=gamma)))
print("SAG:", metrics.mean_squared_error(beta, sag(x, y,proximal_op=1, lam=lam, gamma=gamma)))
print("SVRG:", metrics.mean_squared_error(beta, svrg(x, y,proximal_op=1, lam=lam, gamma=gamma)))
print("SGD:", metrics.mean_squared_error(beta, stochastic_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)))
#print("BGD:", metrics.mean_squared_error(beta, batch_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)))
print("--------------------------------------------------------")
print("IMPLEMENTED RESULTS - MSE")
print("--------------------------------------------------------")
lr_saga = linear_model.LogisticRegression(solver='saga')
fit_saga = lr_saga.fit(x,y)
print("SAGA:", metrics.mean_squared_error(beta, lr_saga.coef_[0]))
lr_sag = linear_model.LogisticRegression(solver='sag')
fit_sag = lr_sag.fit(x,y)
print("SAG:", metrics.mean_squared_error(beta, lr_sag.coef_[0]))

print("--------------------------------------------------------")
print("PROJECT RESULTS - TIME")
print("--------------------------------------------------------")
print("SAGA:", timer("saga(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
                                          "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
                                          "stochastic_gradient_descent, batch_gradient_descent;" +
                                          "from basicCase.Utils import timer"), "sec.")
print("SAG:", timer("sag(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
                                          "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
                                          "stochastic_gradient_descent, batch_gradient_descent;" +
                                          "from basicCase.Utils import timer"), "sec.")
print("SVRG:", timer("svrg(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
                                          "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
                                          "stochastic_gradient_descent, batch_gradient_descent;" +
                                          "from basicCase.Utils import timer"), "sec.")
print("SGD:", timer("stochastic_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
                                          "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
                                          "stochastic_gradient_descent, batch_gradient_descent;" +
                                          "from basicCase.Utils import timer"), "sec.")
# print("BGD:", timer("batch_gradient_descent(x, y,proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
#                                           "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
#                                           "stochastic_gradient_descent, batch_gradient_descent;" +
#                                           "from basicCase.Utils import timer"), "sec.")
print("--------------------------------------------------------")
print("IMPLEMENTED RESULTS - TIME")
print("--------------------------------------------------------")
print("SAGA:", timer("lr_saga = linear_model.LogisticRegression(solver='saga'); fit_saga = lr_saga.fit(x,y);" +
                     "lr_saga.coef_", "from __main__ import x, y;" +
                     "from sklearn import linear_model"), "sec.")
print("SAG:", timer("lr_sag = linear_model.LogisticRegression(solver='sag'); fit_sag = lr_sag.fit(x,y);" +
                     "lr_sag.coef_", "from __main__ import x, y;" +
                     "from sklearn import linear_model"), "sec.")
