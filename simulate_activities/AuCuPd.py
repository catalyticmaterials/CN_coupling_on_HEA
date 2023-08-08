import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from simulate_activity_functions import simulate_activity
from scripts.compositionspace_functions import get_molar_fractions, molar_fractions_to_cartesians
import scripts.plotting_ternary_functions as pt
from scripts import metals

grid_mf = get_molar_fractions(0.05, 5)
mask = (grid_mf[:,0]==0) * (grid_mf[:,4]==0)
grid_mf = grid_mf[mask]  

np.random.seed(42)

rate1=np.empty(0)
rate01=np.empty(0)
for composition in tqdm(grid_mf):
    rate1 = np.append(rate1,simulate_activity(composition, P_CO=1.0, P_NO=1.0,metals=metals,p_react=1.0))
    rate01 = np.append(rate01,simulate_activity(composition, P_CO=1.0, P_NO=0.1,metals=metals,p_react=1.0))
    
    


grid_mf_ternary = grid_mf[:,[1,2,3]]

data = np.hstack((grid_mf_ternary,rate1.reshape(-1,1),rate01.reshape(-1,1)))


np.savetxt("AuCuPd_grid.csv",data,delimiter=",",header="Au,Cu,Pd,active sites (P_NO=1),active sites (P_NO=0.5)",comments="",fmt=["%.2f","%.2f","%.2f","%.4f","%.4f"])




