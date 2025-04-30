import numpy as np
import sys
import time

arr1 = np.array([1,2,3,4,5])
# print(arr1)
#output = [1 2 3 4 5]

# print(type(arr1))
#output = <class 'numpy.ndarray'>

arr2 = np.array([[1,2,3], [4,5,6]])
# print(arr2)
#output = [[1 2 3]
#          [4 5 6]]

arr3 = np.zeros((2,3))
# print(arr3)
#Output - [[0. 0. 0.]
#          [0. 0. 0.]]

arr4 = np.ones((3,3))
# print(arr4)
#output - [[1. 1. 1.]
#          [1. 1. 1.]
#          [1. 1. 1.]]

arr5 = np.identity(5)
# print(arr5)
# Output - [[1. 0. 0. 0. 0.]
#           [0. 1. 0. 0. 0.]
#           [0. 0. 1. 0. 0.]
#           [0. 0. 0. 1. 0.]
#           [0. 0. 0. 0. 1.]]

arr6 = np.arange(10)
# print(arr6)
#output - [0 1 2 3 4 5 6 7 8 9]

arr7 = np.linspace(10,20,10)
#lower range, upper range and split
#linearly spaced - subtract the numbers and the difference will be the same
# print(arr7)
#Output -  [10.         11.11111111 12.22222222 13.33333333 14.44444444 15.55555556
#           16.66666667 17.77777778 18.88888889 20.        ]

arr8 = arr7.copy()
#print(arr8)
#Output -  [10.         11.11111111 12.22222222 13.33333333 14.44444444 15.55555556
#           16.66666667 17.77777778 18.88888889 20.        ]

'''
Numpy attributes and properties
'''

# print(arr1.shape)
# (5,)

arr9 = np.array([[[1,2], [2,3]], [[4,5], [5,6]]])
# print(arr9.shape)
# Output - (2, 2, 2)
# 2 rows, 2 columns and 2 matrices

# print(arr9.ndim)
#dimension of the array
# Output - 3

# print(arr9.size)
# 8

# print(arr9.itemsize)
# 8

# print(arr9.dtype)
# int64, float64, etc

'''
Why arrays are used in ML instead of lists:
1. saves space
2. saves time
'''

listA = range(100)
arr10 = np.arange(100)

# print(sys.getsizeof(87)*len(listA))
#2800 bytes
# print(arr10.itemsize*arr10.size)
#800 bytes

x = range(10000000)
y = range(10000000, 20000000)

start_time = time.time() #gives current time and storing it in start_time
c = [(x,y) for x,y in zip(x,y)] #taking one time from each variable and then zipping(adding) them

# print(time.time() - start_time) #duration
#output - 0.019310951232910156

a = np.arange(10000000)
b = np.arange(10000000, 20000000)

start_time = time.time() #gives current time and storing it in start_time
c = a + b

# print(time.time() - start_time)
# output - 0.006660938262939453

arr11 = np.arange(24).reshape(6,4)
print(arr11)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]
#  [16 17 18 19]
#  [20 21 22 23]]




