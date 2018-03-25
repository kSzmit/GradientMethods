import numpy as np
import timeit
import math

class AveragableTable:
    def __init__(self, matrix=None):
        self.__storage_table = []
        self.average = None
        if matrix is not None:
            for i in range(len(matrix)):
                if(self.average is None):
                    self.average = matrix[i]
                else:
                    self.average = self.average + matrix[i]
                self.__storage_table.append(matrix[i])

            self.average = self.average/len(matrix)


    # Replace item in storage table, adjust average and return previous value ###
    def replace_item(self, index, item):
        if index > len(self.__storage_table) - 1:
            raise IndexError("Error! Storage table size is " + str(len(self.__storage_table)) + ". Can't replace item on index " + str(index))
        else:
            self.average = self.average - self.__storage_table[index] / self.size() + item / self.size()
            replaced_item = self.__storage_table[index]
            self.__storage_table[index] = item
            return replaced_item


    def get_item(self, index):
        return self.__storage_table[index]


    # Add new element and modify average ###
    def append(self, item):
        self.average = self.average*self.size()/(self.size() + 1) + item/(self.size() + 1)
        self.__storage_table.append(item)


    def size(self):
        return len(self.__storage_table)

    def __str__(self):
        return "Storage table: " + str(self.__storage_table) + '\n' + "Average: " + str(self.average)


class SnapshotListHolder:
    def __init__(self, snapshot):
        self.__snapshot = list(snapshot)

    def take_snapshot(self, snapshot):
        self.__snapshot = list(snapshot)

    def get_snapshot(self):
        return self.__snapshot


def generate_cov_matrix(dim, rho = 0, correlation_type = "constant"):
    if correlation_type == "constant":
        return np.full((dim,dim),rho) + (1-rho)*np.eye(dim)
    elif correlation_type == "autocorrelation":
        return np.array( [ [ rho**abs(i-j) for j in range(dim)] for i in range(dim)] )
    elif correlation_type == "id":
        return np.eye(dim)
    else:
        raise ValueError("Error! Unknown correaltion type: " + correlation_type)


def timer(s1, s2, repeat):
    t = timeit.Timer(s1, s2)
    return round(t.timeit(repeat), 2)


def row_norms(X, squared=False):
    norms = np.einsum('ij,ij->i', X, X)
    if not squared:
        np.sqrt(norms, norms)
    return norms

def create_beta(zad, d):
    if zad==1:
        beta = np.concatenate(([1], np.random.uniform(-2, 2, d - 1)), axis=0)
    elif zad==2:
        beta = np.concatenate(([1], np.random.binomial(np.ones((d - 1,), dtype=int), 0.25)), axis=0)
        beta_vals = np.concatenate(([1], np.concatenate([np.array(np.random.uniform(-2, -1, math.floor((d - 1) / 2))),
                                                         np.array(np.random.uniform(1, 2, math.ceil((d - 1) / 2)))])),
                                   axis=0)
        beta = np.multiply(beta_vals, beta)
    return beta

def seed_wrap_function(func, args):
    np.random.seed(150)
    x=func(*args)
    np.random.seed(None)
    return x