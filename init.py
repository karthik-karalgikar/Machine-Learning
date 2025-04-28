import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import style
from perceptron import MyPerceptron

style.use('ggplot')

df = pd.read_csv('breast_cancer_data.csv')
df.replace('?', -99999, inplace=True)
df.drop('id', axis=1, inplace=True)

X = df.iloc[:,0:-1].values
#Selects all rows (:) and all columns except the last one (0:-1) and 
# .values converts the selected DataFrame part into a NumPy array
y = df.iloc[:, -1].values
#Selects all rows (:) and only the last column (-1)

# X = features used for predicting
# y = features that we are going to predict

y = np.where(y == 2, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# clf = Perceptron()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))
#0.9142857142857143

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

'''
StandardScaler is a tool from scikit-learn that standardizes your data.
It rescales the features so that:
The mean (average) of each feature becomes 0
The standard deviation of each feature becomes 1
'''



# print(X_train.shape)
# print(y_train.shape)

clf = MyPerceptron()
clf.fit(X_train, y_train)

'''
(559, 9)
(559,)
(559, 9)
(559,)
'''

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

X_vis = X_train[:, [0, 1]]  
y_vis = y_train

# Train again with only 2 features
clf_vis = MyPerceptron(learning_rate=0.01, n_iterations=1000)
clf_vis.fit(X_vis, y_vis)

# Plot
for label, color in zip(np.unique(y_vis), ['red', 'blue']):
    plt.scatter(X_vis[y_vis == label, 0], X_vis[y_vis == label, 1], 
                label=f"Class {label}", color=color)

# Decision boundary:
x_values = np.linspace(X_vis[:,0].min(), X_vis[:,0].max(), 100)
y_values = -(clf_vis.weights[0] * x_values + clf_vis.bias) / clf_vis.weights[1]

plt.plot(x_values, y_values, label="Decision Boundary", color="green")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary (Perceptron)')
plt.legend()
plt.show()