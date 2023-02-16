import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from simulate_activity_functions import simulate_activity
from shared_params.compositionspace_functions import get_molar_fractions, molar_fractions_to_cartesians
import shared_params.plotting_ternary_functions as pt

grid_mf = get_molar_fractions(0.05, 5)
mask = (grid_mf[:,0]==0) * (grid_mf[:,4]==0)
grid_mf = grid_mf[mask]  

np.random.seed(42)

rate1=np.empty(0)
rate05=np.empty(0)
for composition in tqdm(grid_mf):
    rate1 = np.append(rate1,simulate_activity(composition, P_CO=1.0, P_NO=1.0))
    rate05 = np.append(rate05,simulate_activity(composition, P_CO=1.0, P_NO=0.5))

grid_mf_ternary = grid_mf[:,[1,2,3]]

data = np.hstack((grid_mf_ternary,rate1.reshape(-1,1),rate05.reshape(-1,1)))


np.savetxt("AuCuPd_grid.csv",data,delimiter=",",header="Au,Cu,Pd,active sites (P_NO=1),active sites (P_NO=0.5)",comments="",fmt=["%.2f","%.2f","%.2f","%.6f","%.6f"])

grid = molar_fractions_to_cartesians(grid_mf_ternary)

pt.make_plot(grid.T, rate1, ["Au","Cu","Pd"],colorbar=True)
pt.make_plot(grid.T, rate05, ["Au","Cu","Pd"],colorbar=True)


