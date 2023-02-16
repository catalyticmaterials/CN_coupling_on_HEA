from methods import count_sites_equilibtrium, count_sites_dynamic, dyn_eq
import numpy as np
import matplotlib.pyplot as plt


#composition = np.array([0.2,0.2,0.2,0.2,0.2])
composition = np.array([0.0,0.0,1.0,0.0,0.0])
#composition = np.array([0.0,0.5,0.5,0.0,0.0])

np.random.seed(1)
n_pair_frac,CO_ads, NO_ads, H_ads = count_sites_dynamic(composition, P_CO=1, P_NO=0.1,return_ads=True)
print(n_pair_frac)

plt.figure()
plt.hist(CO_ads,bins=100,histtype="step",label="CO")
plt.hist(NO_ads,bins=100,histtype="step",label="NO")
plt.hist(H_ads,bins=100,histtype="step",label="H")
plt.legend()
print(np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads))


np.random.seed(1)
n_pair_frac,CO_ads, NO_ads, H_ads = count_sites_equilibtrium(composition, P_CO=1, P_NO=0.1,return_ads=True)
print(n_pair_frac)
plt.figure()
plt.hist(CO_ads,bins=100,histtype="step",label="CO")
plt.hist(NO_ads,bins=100,histtype="step",label="NO")
plt.hist(H_ads,bins=100,histtype="step",label="H")
plt.legend()

print(np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads))




np.random.seed(1)
n_pair_frac,CO_ads, NO_ads, H_ads = dyn_eq(composition, P_CO=1, P_NO=0.2,max_iterations=int(1e+5),return_ads=True)
print(n_pair_frac)
plt.figure()
plt.hist(CO_ads,bins=100,histtype="step",label="CO")
plt.hist(NO_ads,bins=100,histtype="step",label="NO")
plt.hist(H_ads,bins=100,histtype="step",label="H")
plt.legend()

print(np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads))