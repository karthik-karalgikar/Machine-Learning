from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5], dtype = np.float64)
ys = np.array([2,3,5,4,6], dtype = np.float64)

def best_fit_slope_intercept(xs, ys):
    numerator = (mean(xs) * mean(ys)) - mean(xs * ys)
    denominator = (mean(xs) ** 2) - mean(xs ** 2)
    m = numerator / denominator

    b = mean(ys) - m*(mean(xs))

    return m, b

m, b = best_fit_slope_intercept(xs, ys)
regression_line = [(m*x) + b for x in xs]

predict_x = 8
predict_y = m*predict_x + b

# line 18 can also be written as:
# for x in xs: 
#     regression_line.append((m*x) + b)

# plt.plot(xs,ys) #straight lines
plt.scatter(xs,ys) #dots
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
