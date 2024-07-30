import pandas as pd
import pandas_datareader.data as reader
import datetime as dt
import statsmodels.api as sm
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

# time period and funds
end = dt.date(2023,12,31)
start = dt.date(end.year-5, end.month, end.day)
funds = ['VGT']
funds_returns = yf.download(funds, start=start, end=end)['Adj Close'].pct_change()

# resample to monthly period, much less data points
funds_returns = funds_returns.resample('M').agg(lambda x: (x+1).prod() - 1)
funds_returns = funds_returns[1:]

# load data from file
factors_df = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start,end)[0]
factors_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = factors_df[['Mkt-RF', 'SMB', 'HML', 'RF']]/100
factors_df = factors_df[1:]

# since we drop the first row, we need to do the same for the factors data
funds_returns.index = factors_df.index # setting the date of the two dataframe to be the same

# and then add the new column
factors_df['VGT_return'] = funds_returns
# lets add a column Ri - Rf for convenience
factors_df['excess_return'] = factors_df['VGT_return'] - factors_df['RF']

"""
# check linearity between factors and excess return

# lets plot the data
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.scatter(factors_df['Mkt-RF'], factors_df['excess_return'])
plt.title('Market excess return')
plt.subplot(2,2,2)
plt.scatter(factors_df['SMB'], factors_df['excess_return'])
plt.title('SMB excess return')
plt.subplot(2,2,3)
plt.scatter(factors_df['HML'], factors_df['excess_return'])
plt.title('HML excess return')
plt.subplot(2,2,4)
plt.scatter(factors_df['RF'], factors_df['excess_return'])
plt.title('RF excess return')
plt.show()

# lets check the correlations
factors_df[['Mkt-RF', 'SMB', 'HML', 'RF']].corr()
"""

# let's make the model
y = factors_df['excess_return']
X = factors_df[['Mkt-RF', 'SMB', 'HML']]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

