import numpy as np

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
print(arr9.shape)
# Output - (2, 2, 2)
# 2 rows, 2 columns and 2 matrices

# print(arr9.ndim)
#dimension of the array
# Output - 3

