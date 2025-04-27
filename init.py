import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from matplotlib import style
from perceptron import MyPerceptron

style.use('fivethirtyeight')

df = pd.read_csv('breast_cancer_data.csv')
df.replace('?', -99999, inplace=True)
df.drop('id', axis=1, inplace=True)

X = df.iloc[:,0:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = Perceptron()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
#0.9142857142857143

X_vis = X_train[:, [0, 1]]  # first two features
y_vis = y_train

# Plot different classes with different colors
for label, color in zip(np.unique(y_vis), ['red', 'blue']):
    plt.scatter(X_vis[y_vis == label, 0], X_vis[y_vis == label, 1], 
                label=f"Class {label}", color=color)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Breast Cancer Data (2 features)')
plt.legend()
plt.show()