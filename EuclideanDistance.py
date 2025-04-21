#Goal = I have 2 groups, k(black) and r(red). I have a new set of points(new_features) and I have 
#to predict which group it comes under, based on its k nearest neighbours. (small dataset)

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

#comment
# euclidean_dist = sqrt(((plot1[0] - plot2[0])**2) + ((plot1[1] - plot2[1])**2))

dataset = {'k': [[1,2], [2,3], [3,1]],
           'r': [[6,5], [7,7], [8,6]]
          }
new_features = [5,7]


def k_nearest_neighbors(data, predict, k=3): #data = dataset, predict = the new data point which I need to classify, 
                                                                                                    #k = neighbours
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    #here, there are 2 groups and k = 3. So 2 >= 3(false), so it will not send the warning. 

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_dist = sqrt(((features[0] - predict[0])**2) + ((features[1] - predict[1])**2))
            #this only works for 2D data. For 3D data:
            euclidean_dist_3d = np.sqrt(np.sum((np.array(features)-np.array(predict))**2)) 
            #or
            euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_dist_anyD, group])

    '''
    Tracing: 
    
    {'k': [[1,2], [2,3], [3,1]],
           'r': [[6,5], [7,7], [8,6]]
          }
    distances = []
    k = group 0, r = group 1
    for 0 in data (k is accessed):
        1. for features(1,2) in data[0], this means that the first (1,2) is accessed

        euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
        = np.linalg.norm(np.array(1,2)-np.array(5,7)) =>  np.linalg.norm(np.array(-4,-5))
        = sqrt(16 + 25) = sqrt(41) ≈ 6.4
        
        distances = [[6.4, 'k']]

        2. for features(2,3) in data[0], this means that  next (2,3) is accessed

        euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
        = np.linalg.norm(np.array(2,3)-np.array(5,7)) =>  np.linalg.norm(np.array(-3,-4))
        = sqrt(9 + 16) = sqrt(25) ≈ 5.0

        distances = [[6.4, 'k'], [5.0, 'k']]

        3. for features(3,1) in data[0], this means that  next (3,1) is accessed

        euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
        = np.linalg.norm(np.array(3,1)-np.array(5,7)) =>  np.linalg.norm(np.array(-2,-6))
        = sqrt(4 + 36) = sqrt(40) ≈ 6.32

        distances = [[6.4, 'k'], [5.0, 'k'], [6.32, 'k']]

        4. for 1 in data(r is accessed)

        5. for features(6,5) in data[1], this means that first (6,5) is accessed

        euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
        = np.linalg.norm(np.array(6,5)-np.array(5,7)) =>  np.linalg.norm(np.array(1,-2))
        = sqrt(1 + 4) = sqrt(2) ≈ 2.23

        distances = [[6.4, 'k'], [5.0, 'k'], [6.32, 'k'], [2.23, 'r']]

        6. for features(7,7) in data[1], this means that next (7,7) is accessed

        euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
        = np.linalg.norm(np.array(7,7)-np.array(5,7)) =>  np.linalg.norm(np.array(2,0))
        = sqrt(4 + 0) = sqrt(4) ≈ 2.0

        distances = [[6.4, 'k'], [5.0, 'k'], [6.32, 'k'], [2.23, 'r'], [2.0, 'r']]

        7. for features(8,6) in data[1], this means that next (8,6) is accessed

        euclidean_dist_anyD = np.linalg.norm(np.array(features)-np.array(predict))
        = np.linalg.norm(np.array(8,6)-np.array(5,7)) =>  np.linalg.norm(np.array(3,-1))
        = sqrt(9 + 1) = sqrt(10) ≈ 3.16

        distances = [[6.4, 'k'], [5.0, 'k'], [6.32, 'k'], [2.23, 'r'], [2.0, 'r'], [3.16, 'r']]
    '''

    votes = [i[1] for i in sorted(distances)[:k]]
    print(votes)
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    '''
    sorted(distances) = [[2.0, 'r'], [2.23, 'r'], [3.16, 'r'], [5.0, 'k'], [6.32, 'k'], [6.4, 'k']]
    for i in sorted(distances) but only till :3 ->
    this means only these three => [2.0, 'r'], [2.23, 'r'], [3.16, 'r']
    i[1] means the second index of the list, that is, the groups.
    so votes = ['r', 'r', 'r']

    Counter : This counts how many times each group label appears in votes. 
    Since votes = ['r', 'r', 'r'], this will output:
    => [('r', 3)]

    most_common(1) means that it will give the top 1 most common item. 

    [('r', 3)] ths is a tuple inside a list. So first we need to access the list, and then the items inside the tuple

    so ('r', 3) is the first item of the list, and hence, most_common(1)[0]
    next, we need only the group, so we take the first item of the tuple, and hence, it is most_common(1)[0][0]

    '''
        
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0], j[1], s = 100, color = i)

plt.scatter(new_features[0], new_features[1], color = result, s = 100)
plt.show()
