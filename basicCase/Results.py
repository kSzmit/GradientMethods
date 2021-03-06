import numpy as np
import math
import random
from sklearn import linear_model, metrics
from basicCase.Methods import compute_p_i, saga, sag, svrg, sgd, bgd, MSE, sym, plot_mse
from basicCase.Utils import generate_cov_matrix, timer, create_beta, seed_wrap_function
import matplotlib.pyplot as plt


### Exercise 1 ###

### Initial parameters ###

methods = np.array(["saga", "sag", "svrg", "sgd", "bgd"])
d = 20
beta = seed_wrap_function(create_beta, [1, d])
gamma = None
n_vec = np.array([200, 400, 600, 800, 1000])
iter = 10

### rho and cov parameters to test

rho = 0
cov_type = "id"

### Each plot must be executed independently (you have to comment another plots code)

### MSE plot ###

if __name__ == "__main__":
    print(plot_mse(methods=methods, d=d, n_vec=n_vec, rho=rho, beta=beta, cov_type=cov_type, repeat=iter))

### Execution time plot ###

    results = np.zeros((len(n_vec), len(methods)))
    k = 0
    for n in n_vec:
        mean_vec = np.repeat(0, d - 1)
        cov_matrix = generate_cov_matrix(d - 1, rho, cov_type)
        x = np.concatenate((np.ones((n, 1)), np.random.multivariate_normal(mean_vec, cov_matrix, n)), axis=1)
        np.random.seed(25)

        p = [compute_p_i(x[i], beta) for i in range(len(x))]
        y = np.random.binomial(np.ones((len(x),), dtype=int), p)
        j = 0
        for m in methods:
            results[k, j] = timer(m+"(x, y, gamma)", "from __main__ import x, y, gamma;" +
                                               "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
                                               "sgd, bgd; from basicCase.Utils import timer", repeat=iter)
            j = j+1
        k = k+1
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('RHO = ' + str(rho) + ", " + cov_type)
    for i in range(len(methods)):
        plt.plot(n_vec, results[:,i], '.-', label=methods[i])
    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.5)
    plt.show()





# ### Exercise 2 ###

# ### Initial parameters ###

beta = seed_wrap_function(create_beta, [2, d])

# ### MSE plot ###

##parameters to test
cov_type = "constant"
rho = 0.3

if_proxy=1
if cov_type=="id":
 rho=0
 lam=0.001
if (cov_type=="autocorrelation" and rho==0.3):
 lam=0.004
elif (cov_type=="constant" and rho==0.6):
 lam=0.001
elif (cov_type=="autocorrelation" and rho==0.6) or (cov_type=="autocorrelation" and rho==0.9):
 lam=0.0025
elif (cov_type=="constant" and rho==0.3) or (cov_type=="constant" and rho==0.9):
 lam=0.0005


if __name__ == "__main__":
 print(plot_mse(methods=methods, d=d, n_vec=n_vec, rho=rho, beta=beta, cov_type=cov_type, lasso=if_proxy, repeat=iter, lam=lam))





### Execution time plot ###

##parameters to test
cov_type = "constant"
rho = 0.3

if __name__ == "__main__":
    results = np.zeros((len(n_vec), len(methods)))
    k = 0
    if cov_type=="id":
        rho=0
    for n in n_vec:
        mean_vec = np.repeat(0, d-1)
        cov_matrix = generate_cov_matrix(d - 1, rho, cov_type)
        x = np.concatenate((np.ones((n, 1)), np.random.multivariate_normal(mean_vec, cov_matrix, n)), axis=1)
        p = [compute_p_i(x[i], beta) for i in range(len(x))]
        y = np.random.binomial(np.ones((len(x),), dtype=int), p)
        lam = 0.001
        j = 0
        for m in methods:
            results[k, j] = timer(m+"(x, y, proximal_op=1, lam=lam, gamma=gamma)", "from __main__ import x, y, lam, gamma;" +
                                               "from basicCase.Methods import compute_p_i, saga, sag, svrg," +
                                               "sgd; from basicCase.Utils import timer", repeat=iter)
            j = j+1
        k = k+1
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('LASSO - RHO = ' + str(rho)+ ", "+str(cov_type))
    for i in range(len(methods)):
        plt.plot(n_vec, results[:,i], '.-', label=methods[i])
    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.5)
    plt.show()



### Lambda symulation plot###

lam_vec = np.arange(0, 0.006, 0.0005)

if __name__ == "__main__":
    results = np.zeros((len(lam_vec), len(methods)))
    k=0
    for lam in lam_vec:
        n = 500
        mean_vec = np.repeat(0, d-1)
        cov_matrix = generate_cov_matrix(d-1, rho, cov_type)
        x = np.concatenate((np.ones((n, 1)), np.random.multivariate_normal(mean_vec, cov_matrix, n)), axis=1)
        np.random.seed(100)
        p = [compute_p_i(x[i], beta) for i in range(len(x))]
        y = np.random.binomial(np.ones((len(x),), dtype=int), p)
        j = 0
        for m in methods:
            results[k, j] = sym(method=m, beta=beta, x=x, y=y, gamma=None, proximal_op=1, lam=lam, fit_intercept=1, iter=iter)
            j = j+1
        k = k+1
    plt.xlabel('lam')
    plt.ylabel('MSE')
    plt.title('RHO = ' + str(rho) + ', ' + cov_type )
    for i in range(len(methods)):
        plt.plot(lam_vec, results[:,i], '.-', label=methods[i])
    leg = plt.legend(loc='best', ncol=2)
    leg.get_frame().set_alpha(0.5)
    plt.show()
