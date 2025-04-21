#Predict the values of NASDAQ index ticker(IXIC) using price data 

import yfinance as yf #for financial data
import math, datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np #for array operations
from sklearn import preprocessing, svm # for data preprocessing and ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

style.use('ggplot')

df = yf.download("^IXIC", start="2022-01-01", end="2022-12-31", auto_adjust=False)  
#Downloading 1 year of NASDAQ index data using yfinance

df = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

#comment

'''
Adj Close: Adjusted closing price (takes into account dividends/splits)
High and Low: Daily high/low prices
Open: Opening price
Volume: Total traded volume
'''

df['HL_PCT'] = ((df['High'] - df['Adj Close']) / df['Adj Close']) * 100.0    #high - low percent
#HL_PCT: How much the price moved during the day

df['PCT_change'] = ((df['Adj Close'] - df['Open']) / df['Open']) * 100.0    # percent change
#PCT_change: How much the price changed from open to close

df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']] #these are the features
# only keeping the relevant features that I want to use for prediction.

#Let’s try to predict the Adj Close price n days into the future.
forecast_col = 'Adj Close'
df.fillna(-9999, inplace=True)

# forecast_out = int(math.ceil(0.01*len(df))) #1% of the dataframe 
#So basically, I have downloaded a 1 year dataset, so that means 252 trading days.
#-> math.ceil(0.01*252) = 2.52 -. ceil = 3
#so I am predicting only 3 days into the future.
#forecast_out = number of days to look into the future.

forecast_out = 30
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
#this way, each row, the label column for each row will be adjusted close price 10 days into the future
#.shift(-forecast_out) moves the target price up by n rows so that each row contains today’s features and 
# the price from the future as the label.

# print(df.head())

X = np.array(df.drop('label', axis=1)) #Take all columns except 'label', and make that my feature set (X)
'''
Say your dataframe looks like this:

Adj Close	HL_PCT	PCT_change	Volume	label (future Adj Close)
100	        2.0 	   -1.5	    30000	        105
105	        1.8	        0.2	    31000	        110
110	        2.5	        0.1	    32000	        115
115	        2.0	        0.3	    33000	        120
When you run:

X = np.array(df.drop('label', axis=1))
You get:

X = [[100, 2.0, -1.5, 30000],
     [105, 1.8, 0.2, 31000],
     [110, 2.5, 0.1, 32000],
     [115, 2.0, 0.3, 33000]]
And when you do:

y = np.array(df['label'])
You get:

y = [105, 110, 115, 120]
Now the model can learn a pattern between X and y.

'''
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #last 10% of the dataset
X = X[:-forecast_out] #first 90% of the dataset


df.dropna(inplace=True)
y = np.array(df['label'])
y = y[:-forecast_out]
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#test_size = 0.2 says that we are going to use only 20% of the total data as testing data and 80% as training data

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

# print(accuracyLinearRegression)
#output = 0.9139136427883758
#this means 91% accuracy on predicting what the price would be shifted 1% of the days. 
# print(accuracySVM)
#output = 0.5571466702931369
#this means 55% accuracy on predicting what the price would be shifted 1% of the days. 

forecast_set = clf1.predict(X_lately)

print(forecast_set, accuracyLinearRegression, forecast_out)
#output = [10404.97831258 10514.91299115 10438.3962626 ] 0.8821317302234541 3

#if forecaset_out = 30, then the output would be:
'''[11248.40735908 11019.46353685 11001.75153443 11090.65366003
 11195.855457   11009.39073312 11149.08831815 10958.32653977
 11815.68315899 11239.7625344  11368.71894316 11226.38746784
 11117.02805429 10999.92570748 11060.04256544 11131.39056453
 11081.58386302 11377.3148075  11278.84896403 11009.54229249
 10910.46446292 10790.01408755 10837.89091187 10970.54703378
 10702.63476035 10682.47994648 10630.73894515 10788.28633322
 10862.73052685 10686.0280336 ] 0.3295484974783027 30
 '''

# ========== Plotting the forecast with matplotlib ==========
# Add a new column 'Forecast' with NaNs
df['Forecast'] = np.nan
#initializing a new column called Forecast and filling it with NaN (Not a Number), which means missing values.
#This keeps the same number of rows in the dataframe but marks the forecast values to be filled later.

# Get the last date in the DataFrame
last_date = df.index[-1]
last_unix = last_date.timestamp()
one_day = 86400  # seconds in a day
next_unix = last_unix + one_day

# Append predicted prices into the 'Forecast' column
for pred in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [pred]

'''
For each predicted price in forecast_set:
Compute the next_date by converting the timestamp back to a datetime object.
Move to the next date by incrementing the timestamp.
Add a new row to the DataFrame at next_date with:
NaN for all columns except the last one (Forecast), which gets the predicted price.
This keeps the data clean: only forecast values are added in this future range, while others stay empty.
'''

'''
df.loc[] in a nutshell
df.loc[] is used to access or modify rows and columns in a pandas DataFrame by label (not position).

Think of df.loc[] like saying:
“Hey DataFrame, I want to look at or change the row that has this specific label (like a date).”

Example:
Let's say you have a DataFrame:

import pandas as pd
data = {
    'Price': [100, 105, 110],
    'Volume': [1000, 1100, 1200]
}
index = ['2023-01-01', '2023-01-02', '2023-01-03']
df = pd.DataFrame(data, index=pd.to_datetime(index))

So df looks like:
            Price  Volume
2023-01-01    100    1000
2023-01-02    105    1100
2023-01-03    110    1200

Now:

df.loc['2023-01-02']
Will give you:

Price     105
Volume    1100
Name: 2023-01-02, dtype: int64

And you can change values too:

df.loc['2023-01-02', 'Price'] = 999

In your loop:
This line:

df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [pred]
Means:
“At the row labeled next_date, set all columns to NaN except the last one ('Forecast') which should be pred.”

If next_date = 2023-01-04, and pred = 11200.0, you are doing:

df.loc['2023-01-04'] = [NaN, NaN, NaN, NaN, 11200.0]
'''

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
#Creates a figure (fig) and a single subplot (ax) with width 12 and height 6 inches. This is the canvas.
df['Adj Close'].plot(ax=ax, label='Actual Price')
#Plots the actual NASDAQ adjusted closing prices on that canvas.
df['Forecast'].plot(ax=ax, label='Forecasted Price')
#Plots the forecasted prices on the same graph using the same axes.
plt.legend(loc='upper left')
#Adds a legend so you can distinguish between actual and predicted prices.
plt.xlabel('Date')
#x axis is date
plt.ylabel('NASDAQ Price')
#y axis is price
plt.title('NASDAQ Index Forecast vs Actual')
#title of the graph
plt.grid(True)
#grid lines visible
plt.tight_layout()
#Automatically adjusts plot padding to prevent labels from getting cut off.
plt.show()
#displays the graph
