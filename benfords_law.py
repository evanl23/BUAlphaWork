import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.download("AAPL NFLX GOOGL AMZN NVDA PLTR", period="15y", interval='1d')["Close"]

apple = data["AAPL"]
first_dig_apple = [int(num // 10 ** (int(np.emath.log10(abs(num))))) for num in apple]
amazon = data["AMZN"]
first_dig_amazon = [int(num // 10 ** (int(np.emath.log10(abs(num))))) for num in amazon]
netflix = data["NFLX"]
first_dig_netflix = [int(num // 10 ** (int(np.emath.log10(abs(num))))) for num in netflix]
google = data["GOOGL"]
first_dig_google = [int(num // 10 ** (int(np.emath.log10(abs(num))))) for num in google]
nvidia = data["NVDA"]
first_dig_nvidia = [int(num // 10 ** (int(np.emath.log10(abs(num))))) for num in nvidia]
other = data["PLTR"]
first_dig_other = [int(num // 10 ** (int(np.emath.log10(abs(num))))) for num in nvidia]

plt.hist([first_dig_apple, first_dig_amazon, first_dig_netflix, first_dig_google, first_dig_nvidia, first_dig_other], label=["apple", "amazon", "netflix", "google", "nvidia", "PLTR"])

plt.ylabel("Frequency")
plt.xlabel("First digits")
plt.legend(loc='upper right')
plt.show()

