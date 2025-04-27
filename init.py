import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from perceptron import MyPerceptron

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