from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5], dtype = np.float64)
# ys = np.array([2,3,5,4,6], dtype = np.float64)

def create_dataset(hm, variance, step=2, correlation=False): 
    #how many data points do we want to create here
    #how variable we want this dataset to be
    #how far on average, to step up the y value per point
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val = val + step
        elif correlation and correlation == 'neg':
            val = val - step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

'''
hm=5, variance=1, step=2, correlation='pos'

Start with val = 1
For each point:
Add a small random change (+/- variance)
Adjust val if correlation is 'pos' by adding step

Iteration breakdown (assume variance = 1):
i	val before	random noise	y	val after (correlation = 'pos')
0	    1	         +0	        1	        3
1	    3	         -1	        2	        5
2	    5	         +0	        5	        7
3	    7	         +1	        8	        9
4	    9	         -1	        8	        —

xs will be:
[0. 1. 2. 3. 4.]

ys will be something like:
[1. 2. 5. 8. 8.]
'''


def best_fit_slope_intercept(xs, ys):
    numerator = (mean(xs) * mean(ys)) - mean(xs * ys)
    denominator = (mean(xs) ** 2) - mean(xs ** 2)
    m = numerator / denominator

    b = mean(ys) - m*(mean(xs))

    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 10, 2, correlation='pos')

m, b = best_fit_slope_intercept(xs, ys)
regression_line = [(m*x) + b for x in xs]

# for x in xs: 
#     regression_line.append((m*x) + b)

# plt.plot(xs,ys) #straight lines

predict_x = 8
predict_y = m*predict_x + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys) #dots
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
