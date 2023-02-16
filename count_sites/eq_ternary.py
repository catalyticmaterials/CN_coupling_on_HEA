import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from methods import count_sites_equilibtrium as cse
from shared_params.compositionspace_functions import get_molar_fractions, molar_fractions_to_cartesians
import shared_params.plotting_ternary_functions as pt


grid_mf = get_molar_fractions(0.05, 5)
mask = (grid_mf[:,3]==0) * (grid_mf[:,4]==0)
grid_mf = grid_mf[mask]  

np.random.seed(42)

site1=np.empty(0)
site05=np.empty(0)
for composition in tqdm(grid_mf):
    site1 = np.append(site1,cse(composition, P_CO=1.0, P_NO=1.0))
    site05 = np.append(site05,cse(composition, P_CO=1.0, P_NO=0.5))



data = np.hstack((grid_mf[:,:3],site1.reshape(-1,1),site05.reshape(-1,1)))


np.savetxt("AgAuCu_grid_eq.csv",data,delimiter=",",header="Ag,Au,Cu,active sites (P_NO=1),active sites (P_NO=0.5)",comments="",fmt=["%.2f","%.2f","%.2f","%.6f","%.6f"])

grid = molar_fractions_to_cartesians(grid_mf[:,:3])

pt.make_plot(grid.T, site1, ["Ag","Au","Cu"],colorbar=True)
pt.make_plot(grid.T, site05, ["Ag","Au","Cu"],colorbar=True)





grid_mf = get_molar_fractions(0.05, 5)
mask = (grid_mf[:,0]==0) * (grid_mf[:,4]==0)
grid_mf = grid_mf[mask]  

np.random.seed(42)

site1=np.empty(0)
site05=np.empty(0)
for composition in tqdm(grid_mf):
    site1 = np.append(site1,cse(composition, P_CO=1.0, P_NO=1.0))
    site05 = np.append(site05,cse(composition, P_CO=1.0, P_NO=0.5))

grid_mf_ternary = grid_mf[:,[1,2,3]]
data = np.hstack((grid_mf_ternary,site1.reshape(-1,1),site05.reshape(-1,1)))


np.savetxt("AuCuPd_grid_eq.csv",data,delimiter=",",header="Au,Cu,Pd,active sites (P_NO=1),active sites (P_NO=0.5)",comments="",fmt=["%.2f","%.2f","%.2f","%.6f","%.6f"])

grid = molar_fractions_to_cartesians(grid_mf_ternary)

pt.make_plot(grid.T, site1, ["Au","Cu","Pd"],colorbar=True)
pt.make_plot(grid.T, site05, ["Au","Cu","Pd"],colorbar=True)
