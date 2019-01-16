#%%
# MACS 402 - Structural Analysis - Dr. Evans - Problem Set 2
# DATE: January 23rd, 2019
# AUTHOR: Adam Oppenheimer
import pandas as pd
import matplotlib.pyplot as plt

#Problem 1
CLAIMS = "Health Claims"
data = pd.read_csv("clms.txt", header=None, names=[CLAIMS])
key_vals = data.describe()

plot_data_1 = data/key_vals[0]
num_bins = 1000
observations, bin_cuts, patches = plt.hist(plot_data_1, num_bins)
plt.title("Histogram of Health Claims")
plt.ylabel("Percent of Observations")
plt.xlabel("Value of Monthly Health Expenditures")
plt.legend(loc="upper left")
plt.plot()

plot_data_2 = data[data <= 800]/key_vals[0]