import numpy as np

trials = 100
runs = 100
stay_percentage = []
switch_percentage = []
doors = ['a','b','c']
indices = [0,1,2] # create indices for the doors

for _ in range(trials):
    win_count_stay = 0
    win_count_switch = 0
    for i in range(runs):
        prize = np.random.choice(indices, p=[1/3, 1/3, 1/3])
        first_pick = np.random.choice(indices, p=[1/3, 1/3, 1/3])

        host_opens = np.random.choice(np.delete(indices, [prize, first_pick]))

        # switch
        switch = np.delete(doors, [first_pick, host_opens])
        if (switch[0] == doors[prize]):
            win_count_switch += 1

        # stay 
        if (doors[first_pick] == doors[prize]):
            win_count_stay += 1

    stay_percentage.append(win_count_stay/runs)
    switch_percentage.append(win_count_switch/runs)

mean_stay = np.mean(stay_percentage)
mean_switch = np.mean(switch_percentage)

print(f"Percentage of wins from switching: {mean_switch: .1%}")
print(f"Percentage of wins from staying: {mean_stay: .1%}")


""" tests:
print(f"first pick = {first_pick}")
print(f"prize = {prize}")
print(f"host opens = {host_opens}")
print(f"switch decision = {switch}")
print(f"win count stay = {win_count_stay}")
print(f"win count switch = {win_count_switch}")
"""
