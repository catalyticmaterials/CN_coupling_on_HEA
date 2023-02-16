import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
from methods import block_indices, adsorb_CO,adsorb_H,adsorb_NO, dyn_eq
import sys
sys.path.append("..")
from shared_params.surface import predict_energies, initiate_surface

np.random.seed(1)

composition = np.array([0.2,0.2,0.2,0.2,0.2])
#composition = np.array([0.0,0.0,1.0,0.0,0.0])

n_list=np.array([10,25,50,100,200])

active_sites = np.empty((10,5))

for i,n in enumerate(n_list):
    active_sites[:,i] = np.array([dyn_eq(composition, P_CO=1, P_NO=1,n=n) for _ in range(10)])
    
data = np.vstack((n_list,active_sites))

np.savetxt("dyn_eq_std_test.csv",data,delimiter=",",comments="")


means = np.mean(active_sites,axis=0)
stds = np.std(active_sites,axis=0,ddof=1)
means_err = stds/np.sqrt(10)



fig,ax=plt.subplots(dpi=400)

#ax.scatter(n_list,means,c="k",marker=".")
ax.errorbar(n_list,means,means_err,fmt=".",c="k",label="Mean")

polygon=Polygon(np.array([np.append(n_list,np.flip(n_list)),np.append(means+stds,np.flip(means-stds))]).T,alpha=0.5,label="std")
ax.add_patch(polygon)

ax.legend()
ax.set_ylabel("Active sites pr. surface atoms")
ax.set_xlabel(r"$\sqrt{N_{surface \, atoms}}$")
ax.set_xticks([10,25,50,100,200])
plt.title("Ag$_{20}$Au$_{20}$Cu$_{20}$Pd$_{20}$Pt$_{20}$")

plt.savefig("dyn_eq_std.png")