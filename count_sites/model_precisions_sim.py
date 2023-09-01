import numpy as np
import sys
sys.path.append('..')
from scripts.methods import count_sites
from scripts import metals

n_list = [10,50,100,200]

N_samples = 10

composition = np.array([0.2,0.2,0.2,0.2,0.2])



data = np.empty((8,0))
for method in ['eq','dyn','mc']:
    for P_NO in [1,0.1]:
        
        means = np.empty(0)
        stds = np.empty(0)

        np.random.seed(1)

        for n in n_list:
            
            active_sites = np.array([count_sites(composition,1,P_NO,metals,method,n=n) for _ in range(N_samples)])

            means = np.append(means,np.mean(active_sites))
            stds = np.append(stds,np.std(active_sites,ddof=1))

        
        
        data = np.hstack((data,np.append(means,stds).reshape(-1,1)))



np.savetxt('model_precisions.csv',data,delimiter=',',fmt='%.4f',header='eq_1,eq_01,dyn_1,dyn_01,mc_1,mc_01')





