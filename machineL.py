#Predict the values of NASDAQ index ticker(IXIC) using price data 

import yfinance as yf #for financial data
import math 
import numpy as np #for array operations
from sklearn import preprocessing, svm # for data preprocessing and ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = yf.download("^IXIC", start="2022-01-01", end="2022-12-31", auto_adjust=False)  
#Downloading 1 year of NASDAQ index data using yfinance

df = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

'''
Adj Close: Adjusted closing price (takes into account dividends/splits)
High and Low: Daily high/low prices
Open: Opening price
Volume: Total traded volume
'''

df['HL_PCT'] = ((df['High'] - df['Adj Close']) / df['Adj Close']) * 100.0    #high - low percent
#HL_PCT: How much the price moved during the day

df['PCT_change'] = ((df['Adj Close'] - df['Open']) / df['Open']) * 100.0    # percent change

df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']] #these are the features

forecast_col = 'Adj Close'
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) #1% of the dataframe 
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
#this way, each row, the label column for each row will be adjusted close price 10 days into the future

df.dropna(inplace=True)
# print(df.head())

X = np.array(df.drop('label', axis=1)) #features
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#test_size = 0.2 says that we are going to use only 20% of the total data as testing data

#train_test_split(X, y, test_size=0.2) -> this is going to take all our features on our labels, remember the kind of order
#and it is going to shuffle them up keeping the X's and y's connected. 
# it is not going to shuffle them up to a point where you lose accuracy. 
#and then it outputs for X training and y training data and y testing data and x testing data

#clf = classifier
clf1 = LinearRegression()
clf1.fit(X_train, y_train)
accuracyLinearRegression = clf1.score(X_test, y_test)

#using a different classifier:
clf2 = svm.SVR(kernel='poly', C=1e3) # do not know what is kernel='linear', C=1e3
clf2.fit(X_train, y_train)
accuracySVM = clf2.score(X_test, y_test)

print(accuracyLinearRegression)
#output = 0.9139136427883758
#this means 91% accuracy on predicting what the price would be shifted 1% of the days. 
print(accuracySVM)
#output = 0.5571466702931369
#this means 55% accuracy on predicting what the price would be shifted 1% of the days. 
