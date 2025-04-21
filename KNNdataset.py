import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
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


    votes = [i[1] for i in sorted(distances)[:k]]
    # print(votes)
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
        
    return vote_result


df = pd.read_csv('breast_cancer_data.csv')
df.replace('?', -99999, inplace=True)
df.drop('id', axis=1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
#initializing two dictionaries — one for training, one for testing.
#Each key (2 and 4) is a class label:
#2 = benign (non-cancerous)
#4 = malignant (cancerous)

train_data = full_data[:-int(test_size*len(full_data))] #first 80% of the data
test_data = full_data[-int(test_size*len(full_data)):] #rest

'''
suppose full_data = [
 [6.0, 5.0, 4.0, 4.0],
 [3.0, 2.0, 4.0, 2.0],
 [7.0, 7.0, 6.0, 4.0],
 [2.0, 3.0, 1.0, 2.0],
 [1.0, 2.0, 3.0, 2.0] 
]

train_data = full_data[:-1] -> everything except last

so train_data = [6.0, 5.0, 4.0, 4.0],
                [3.0, 2.0, 4.0, 2.0],
                [7.0, 7.0, 6.0, 4.0],
                [2.0, 3.0, 1.0, 2.0]

test_data = full_data[-1:] -> only the last row:
test_data = [1.0, 2.0, 3.0, 2.0] 
'''

for i in train_data:
    train_set[i[-1]].append(i[:-1])

'''
NOTE:
I am not indexing train_set by position (like 0, 1, 2), you're accessing it by key (2.0 or 4.0), 
using the label from train_data.
train_set = {2:[], 4:[]}
Tracing : 
i = 0 -> train_set[0[-1]].append(0[:-1]) 
      => train_set[0[-1]] is 4.0, which means, basically - i[-1] → 2.0 and i[:-1] → [3.0, 2.0, 4.0]. So:
                                                            train_set[2.0].append([3.0, 2.0, 4.0])
      we are appending [6.0, 5.0, 4.0, 4.0] all this except the last 4.0
      so, train_set = {2:[], 4:[[6.0, 5.0, 4.0]]}

i = 1 -> train_set[1[-1]].append(1[:-1])
      => train_set[1[-1]] is 2.0
      we are appending [3.0, 2.0, 4.0, 2.0] all this except the last 2.0
      so, train_set = {2:[3.0, 2.0, 4.0], 4:[[6.0, 5.0, 4.0]]}

i = 2 -> train_set[2[-1]].append(2[:-1])
      => train_set[2[-1]] is 4.0
      we are appending [7.0, 7.0, 6.0, 4.0] all this except the last 4.0
      so, train_set = {2:[[3.0, 2.0, 4.0]], 4:[[6.0, 5.0, 4.0], [7.0, 7.0, 6.0]]}

i = 3 -> train_set[3[-1]].append(3[:-1])
      => train_set[3[-1]] is 2.0
      we are appending [2.0, 3.0, 1.0, 2.0] all this except the last 2.0
      so, train_set = {2:[[3.0, 2.0, 4.0], [2.0, 3.0, 1.0]], 4:[[6.0, 5.0, 4.0], [7.0, 7.0, 6.0]]}
'''

for i in test_data:
    test_set[i[-1]].append(i[:-1])

'''
test_data = [1.0, 2.0, 3.0, 2.0] 
Tracing : 

i = 0 -> test_set[2.0].append(1.0, 2.0, 3.0)
test_set = {2:[1.0, 2.0, 3.0], 4:[]}
'''

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct = correct + 1
        total = total + 1

print('Accuracy: ', correct/total)

'''
Tracing :
test_set = {2:[1.0, 2.0, 3.0], 4:[]}
train_set = {2:[[3.0, 2.0, 4.0], [2.0, 3.0, 1.0]], 4:[[6.0, 5.0, 4.0], [7.0, 7.0, 6.0]]}

line 114 -> for loop:

for group in test_set: 
    this means group 2 in test_set, 
    data in test_set[2]:
    this means that [1.0, 2.0, 3.0] is data. 
    and then we are calling the k_nn function, which caclulates the distance between each of these with train_set:
    so, the euclidean distance between, 
    sqrt((3.0-1.0)**2 + (2.0-2.0)**2 + (4.0-3.0)**2) = sqrt(4 + 0 + 1) = sqrt(5) = 2.23
    distances = [2.23, 2]
    sqrt((2.0-1.0)**2 + (3.0-2.0)**2 + (1.0-3.0)**2) = sqrt(1 + 1 + 4) = sqrt(6) = 2.44
    distances = [[2.23, 2], [2.44, 2]]
    sqrt((6.0-1.0)**2 + (5.0-2.0)**2 + (4.0-3.0)**2) = sqrt(25 + 9 + 1) = sqrt(5) = 5.91
    distances = [[2.23, 2], [2.44, 2], [5.91, 4]]
    sqrt((7.0-1.0)**2 + (7.0-2.0)**2 + (6.0-3.0)**2) = sqrt(36 + 25 + 9) = sqrt(5) = 8.36
    distances = [[2.23, 2], [2.44, 2], [5.91, 4], [8.36, 4]]

    sorted distances = [[2.23, 2], [2.44, 2], [5.91, 4], [8.36, 4]]

    votes = [2, 2, 4, 4][:k]
    if k = 3
    votes = [2, 2 ,4]

    Counter = 2
'''
