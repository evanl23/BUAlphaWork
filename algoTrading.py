import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spy_df = yf.download('SPY', period='20y', interval='1d')

DATA_SPLIT = 0.65
spy_df_train = spy_df.iloc[:int(len(spy_df) * DATA_SPLIT)]
spy_df_test = spy_df.iloc[int(len(spy_df) * DATA_SPLIT):]

def trial(df, long_window=50, short_window=9):
    '''given a df, long window length and short window length, return the cumulative returns'''
    # Calculate the fast and slow moving averages
    df['fast_mavg'] = df['Close'].rolling(window=short_window, min_periods=short_window, center=False).mean()
    df['slow_mavg'] = df['Close'].rolling(window=long_window, min_periods=long_window, center=False).mean()

    # Calculate crossover signal (fast greater than slow but not on previous bar)
    df['crossover_long'] = (df['fast_mavg'] > df['slow_mavg']) & (
        df['fast_mavg'].shift(1) <= df['slow_mavg'].shift(1))
    
    # calculate crossunders
    df['crossover_short'] = (df['fast_mavg'] < df['slow_mavg']) & (
        df['fast_mavg'].shift(1) >= df['slow_mavg'].shift(1))
    
    returns = []
    current_entry_price = 0
    # iterate over the dataframe rows and calculate returns
    for index, row in df.iterrows():
        if row['crossover_long']:
            current_entry_price = row['Close']
        elif row['crossover_short'] and current_entry_price != 0:
            returns.append((row['Close'] - current_entry_price) / current_entry_price)
            current_entry_price = 0

    # calculate the average returns
    average_returns = np.mean(returns)
    # calculate the standard deviation of returns
    std_returns = np.std(returns)

    # calculate annualized return 

    # print results
    print(f"Cumulative returns: {np.prod([1+i for i in returns]) - 1}")
    print(f"Number of profitable trades: {len([i for i in returns if i > 0])}")
    print(f"Number of unprofitable trades: {len([i for i in returns if i < 0])}")
    print(f"Number of trades: {len(returns)}")
    print(f"Average return: {average_returns}")
    
    # return cumulative returns
    return np.prod([1+i for i in returns]) - 1

def MACD(df):
    '''given a df, using the MACD formula, return the cumulative returns'''
    
    # calculate MACD and signal line
    df['MACD'] =  df['Close'].ewm(span=12, adjust=False, min_periods=1).mean() - df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

    # calculate when MACD moves above signal line (long)
    df['crossover'] = (df['MACD'] > df['signal']) & (df['MACD'].shift(1) <= df['signal'].shift(1))
    
    # calculate when MACD moves below signal line (short)
    df['crossunder'] = (df['MACD'] < df['signal']) & (df['MACD'].shift(1) >= df['signal'].shift(1))
    
    returns = []
    current_entry_price = 0
    # calculate returns
    for index, row in df.iterrows():
        if row['crossover']:
            current_entry_price = row['Close']
        elif row['crossunder'] and current_entry_price != 0:
            returns.append((row['Close'] - current_entry_price) / current_entry_price)
            current_entry_price = 0

    # calculate the average returns
    average_returns = np.mean(returns)
    # calculate the standard deviation of returns
    std_returns = np.std(returns)

    # calculate annualized return 

    # print results
    print(f"Cumulative returns: {np.prod([1+i for i in returns]) - 1}")
    print(f"Number of profitable trades: {len([i for i in returns if i > 0])}")
    print(f"Number of unprofitable trades: {len([i for i in returns if i < 0])}")
    print(f"Number of trades: {len(returns)}")
    print(f"Average return: {average_returns}")
    print(f"Std return: {std_returns}")
    
    # return cumulative return, plot, etc..   
    return np.prod([1+i for i in returns]) - 1 



def BollingerBands(df, period=20, std=2):
    '''given a df and optional period and standard deviation '''
    
    # create indicators
    # calculate SMA 
    df['SMA'] = df['Close'].rolling(window=period, min_periods=period, center=False).mean()
    df['std'] = df['Close'].rolling(period).std()
    # calculate bands
    df['upper'] = df['SMA'] + std*df['std']
    df['lower'] = df['SMA'] - std*df['std']

    # calulate bands tightening (sharp price movement)
    # calculate bands widening (increased volatility trend ending)
    # calculate price hugging 
    # calculate price moving out of bands

    # calculating price bouncing from bottom
    df['bounce_up'] = (df['SMA'] > df['lower']) & (df['SMA'].shift(1) <= df['lower'].shift(1))
    
    # calculating price bouncing from top 
    df['bounce_down'] = (df['SMA'] < df['upper']) & (df['SMA'].shift(1) >= df['upper'].shift(1))

    # calculate returns
    returns = []
    current_entry_price = 0
    for index, row in df.iterrows():
        if row['bounce_up']:
            current_entry_price = row['Close']
        elif row['bounce_down'] and current_entry_price != 0:
            returns.append((row['Close'] - current_entry_price) / current_entry_price)
            current_entry_price = 0

    # calculate the average returns
    average_returns = np.mean(returns)
    # calculate the standard deviation of returns
    std_returns = np.std(returns)

    # calculate annualized return 

    # print results
    print(f"Cumulative returns: {np.prod([1+i for i in returns]) - 1}")
    print(f"Number of profitable trades: {len([i for i in returns if i > 0])}")
    print(f"Number of unprofitable trades: {len([i for i in returns if i < 0])}")
    print(f"Number of trades: {len(returns)}")
    print(f"Average return: {average_returns}")
    
    # return cumulative return, plot, etc..   
    return np.prod([1+i for i in returns]) - 1


BollingerBands(spy_df_train)



"""
for long_window in [20, 50, 100, 200, 500]:
    for short_window in [10, 20, 50]:
        print(f"long_window: {long_window}, short_window: {short_window}, returns: {MACD(spy_df_train)}")


plt.figure(num=1, figsize=(10,3))
plt.subplot()
plt.plot(spy_df_test['SMA'], alpha=0.5)
plt.subplot()
plt.plot(spy_df_test['upper'], alpha=0.5)
plt.subplot()
plt.plot(spy_df_test['lower'], alpha=0.5)
plt.title("SPY returns and Bollinger")
plt.show() 
"""
