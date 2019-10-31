###############
#
# Translate R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import math
import numpy as np
import pandas

print(1+1)
print(3-1)
print(3*4)
print(8/6)
print(2**10)

# valuable
x = 2

print(x+1)

# function
print(math.sqrt(4))

# vector
vector_1 = np.array([1,2,3,4,5])

print(vector_1)

print(np.arange(1, 11, 1))

# matrix
matrix_1 = np.array(
    [
        range(1,6,1),
        range(6,11,1)
    ]
)
print(matrix_1)

matrix_1_p = pandas.DataFrame(np.arange(1,11,1).reshape(2,5))
print(matrix_1_p)

matrix_1_p.index = ['Row1', 'Row2']
matrix_1_p.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']

print(matrix_1_p)

# array
array_1_1 = np.arange(1,16,1).reshape(3, 5, order = 'F')
array_1_2 = np.arange(16,31,1).reshape(3, 5, order = 'F')

print(array_1_1)
print(array_1_2)

# dataframe
data_frame_1 = pandas.DataFrame(
    {'col1': ['A', 'B', 'C', 'D', 'E'],
    'col2': [1,2,3,4,5]}
)

print(data_frame_1)

# list
list_1 = {
    'chara': ['A', 'B', 'C'],
    'matrix': matrix_1_p,
    'df': data_frame_1
}

print(list_1)

# data extraction
print(vector_1[0])
print(matrix_1_p.iloc[0, 1])
print(array_1_1[0][1])
print(matrix_1_p.iloc[0,:])
print(matrix_1_p.iloc[:, 0])
print(matrix_1_p.iloc[0, 1:4])

print(len(matrix_1_p.index))
print(len(matrix_1_p.columns))

print(matrix_1_p.index)
print(matrix_1_p.columns)

print(matrix_1_p.loc['Row1', 'Col1'])

print(data_frame_1.iloc[:, 1])

print(data_frame_1.iloc[:, 1][1])

print(list_1['chara'])

# time series

idx = pandas.date_range('2010-01-01', freq='BMS', periods=24)
print(idx)

ts = pandas.Series(np.arange(1,25,1), index=idx)
print(ts)

birds = pandas.read_csv('2-1-1-birds.csv')
print(birds)

# generating random variables
print(np.random.normal(0, 1))
print(np.random.normal(0, 1))

np.random.seed(1)
print(np.random.normal(0, 1))
np.random.seed(1)
print(np.random.normal(0, 1))

np.random.seed(1)
print(np.random.normal(0, 1))
print(np.random.normal(0, 1))
np.random.seed(1)
print(np.random.normal(0, 1))
print(np.random.normal(0, 1))

np.random.seed(1)
result_vec_1 = [np.random.normal(0,1) for i in range(0,3,1)]
print(result_vec_1)

mean_vec = [0, 10, -5]
np.random.seed(1)
result_vec_2 = [np.random.normal(mean_vec[i],1) for i in range(0,3,1)]
print(result_vec_2)
