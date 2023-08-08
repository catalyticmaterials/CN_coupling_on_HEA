import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from simulate_activity_functions import simulate_activity
import sys
sys.path.append('..')
from scripts import metals

np.random.seed(42)
composition = np.array([0.,0.,1.,0.,0.])
P_CO = 1
P_NO = 0.1
p_react = 1
n=50
time_interval = 5000
rate_time=10


# rates = []

# for i in range(5):

D_reactions_list, it_list, rate = simulate_activity(composition,P_CO,P_NO, metals, p_react, n=n, max_iter=int(4e+5),time_interval=time_interval,rate_time=rate_time,tol=1e-6, return_history=True)
# rates.append(rate)
a, b, *ignore = linregress(it_list[-rate_time:], D_reactions_list[-rate_time:])

# print(np.mean(rates),np.std(rates,ddof=1))

plt.plot(it_list,D_reactions_list)
plt.plot(it_list[-rate_time:],a*np.array(it_list[-rate_time:])+b)
plt.plot()
plt.show()