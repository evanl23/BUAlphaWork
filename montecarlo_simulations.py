import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf

"""
plt.hist(np.random.randint(1,7, size=100), bins=np.linspace(0.5, 6.5, 7), density=True)
plt.ylabel("Frequency/Probability")
plt.xlabel("Face of the Die")
plt.show()

N = 20000
simulations = range(100, N, 100) # increasing number of rolls by 100
outcomes = []

for rolls in simulations:
    average = np.mean(np.random.randint(1,7, size = rolls))
    outcomes = outcomes + [average]

np.mean(outcomes)
plt.hist(outcomes)
"""
# let's calculate the expected value and variance for each strategy
# FOR A SINGLE TRIAL
strategy_a_ev = 0.5 * 0.8 + 0.5 * 1.5
strategy_a_var = 0.5 * (0.8 - strategy_a_ev)**2 + 0.5 * (1.5 - strategy_a_ev)**2
print(f"Strategy A: EV = {strategy_a_ev}, Var = {strategy_a_var}")

strategy_b_ev = 0.9 * 0.95 + 0.1 * 2.95
strategy_b_var = 0.9 * (0.95 - strategy_b_ev)**2 + 0.1 * (2.95 - strategy_b_ev)**2
print(f"Strategy B: EV = {strategy_b_ev}, Var = {strategy_b_var}")
print()
#print('Note that this is just for a single execution of the strategy. This is important!')

# let's simulate trials
n_trials = 1000
n_days = 30
strategy_a_results = []
strategy_a_all_returns = []
strategy_b_results = []
strategy_b_all_returns = []

for i in range(n_trials):
    strategy_a_returns = [1]
    strategy_b_returns = [1]
    for j in range(n_days):
        strategy_a_returns.append(
            strategy_a_returns[-1] * np.random.choice([0.8, 1.5], p=[0.5, 0.5])) #changing buying power by changing probability and adding in new choice
        
        strategy_b_returns.append(
            strategy_b_returns[-1] * np.random.choice([0.95, 2.95], p=[0.9, 0.1])) #or use size= for number of times to execute strat B

    strategy_a_results.append(strategy_a_returns) #array for all of the results
    strategy_b_results.append(strategy_b_returns) #array for all of the results
    strategy_a_all_returns.append(strategy_a_returns[-1]) #array for the returns at the end of each 30 day period
    strategy_b_all_returns.append(strategy_b_returns[-1]) #array for the returns at the end of each 30 day period 

# calculate average final returns
strategy_a_avg = np.mean(strategy_a_all_returns)
strategy_b_avg = np.mean(strategy_b_all_returns)
strategy_a_median = np.median(strategy_a_all_returns)
strategy_b_median = np.median(strategy_b_all_returns)

# output results
print(f"Strategy A: Average final return = {strategy_a_avg}")
print(f"Strategy B: Average final return = {strategy_b_avg}")
    # median being less than average tells us that more returns are concentrated on the higher side 
print(f"Strategy A: Median return = {strategy_a_median}") 
print(f"Strategy B: Median return = {strategy_b_median}")
print()

# Plotting returns

x = strategy_a_all_returns
y = strategy_b_all_returns

bins = np.linspace(0, 100, 1000)

plt.hist(x, bins, alpha=0.5, label='Strategy A')
plt.hist(y, bins, alpha=0.5, label='Strategy B')
plt.legend(loc='upper right')
plt.show()

# Calculate Sharpe Ratio
standard_dev_A = np.std(strategy_a_all_returns)
standard_dev_B = np.std(strategy_b_all_returns)

print(f"Strategy A standard deviation = {standard_dev_A}")
print(f"Strategy A standard deviation = {standard_dev_B}")
print()

N = 255 #255 trading days in a year
risk_free = yf.Ticker("^TNX")
rf = risk_free.history(period='1d')["Close"].iloc[-1]

mean_A = strategy_a_avg * N - rf
sigma_A = standard_dev_A * np.sqrt(N)
Sharp_Ratio_A = mean_A / sigma_A

mean_B = strategy_b_avg * N - rf
sigma_B = standard_dev_B * np.sqrt(N)
Sharp_Ratio_B = mean_B / sigma_B

print(f"Sharpe Ratio for Strategy A = {Sharp_Ratio_A}")
print(f"Sharpe Ratio for Strategy B = {Sharp_Ratio_B}")
print()

# Calculate Sortino Ratio 
neg_returns_A = [x for x in strategy_a_all_returns if x < 1]
std_neg_A = np.std(neg_returns_A) * np.sqrt(N)

neg_returns_B = [x for x in strategy_b_all_returns if x < 1]
std_neg_B = np.std(neg_returns_B) * np.sqrt(N)

Sortino_Ratio_A = mean_A / std_neg_A
Sortino_Ratio_B = mean_B / std_neg_B

print(f"Sortino Ratio for Strategy A = {Sortino_Ratio_A}")
print(f"Sortino Ratio for Strategy B = {Sortino_Ratio_B}")
