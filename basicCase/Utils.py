import numpy as np

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


    # replace item in storage table, adjust average and return previous value
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


    # add new element and modify average
    def append(self, item):
        self.average = self.average*self.size()/(self.size() + 1) + item/(self.size() + 1)
        self.__storage_table.append(item)


    def size(self):
        return len(self.__storage_table)

    def __str__(self):
        return "Storage table: " + str(self.__storage_table) + '\n' + "Average: " + str(self.average)

def generate_cov_matrix(dim, rho = 0, correlation_type = "constant"):
    if correlation_type == "constant":
        return np.full((dim,dim),rho) + (1-rho)*np.eye(dim)
    elif correlation_type == "autocorrelation":
        return np.array( [ [ rho**abs(i-j) for j in range(dim)] for i in range(dim)] )
    else:
        raise ValueError("Error! Unknown correaltion type: " + correlation_type)

