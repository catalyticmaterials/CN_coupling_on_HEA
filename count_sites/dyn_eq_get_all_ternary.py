import numpy as np
from tqdm import tqdm
from itertools import combinations
import sys
sys.path.append("..")
from methods import dyn_eq
from shared_params.compositionspace_functions import get_molar_fractions, molar_fractions_to_cartesians
import shared_params.plotting_ternary_functions as pt


grid_mf = get_molar_fractions(0.05, 5)


mask = np.sum(grid_mf==0,axis=1)>=2

grid_mf = grid_mf[mask]  

np.random.seed(42)

site1=np.empty(0)
sites02 = np.empty(0)
for composition in tqdm(grid_mf):
    site1 = np.append(site1,dyn_eq(composition, P_CO=1.0, P_NO=1.0))
    sites02 = np.append(sites02,dyn_eq(composition, P_CO=1.0, P_NO=0.2))


data = np.hstack((grid_mf,site1.reshape(-1,1),sites02.reshape(-1,1)))


np.savetxt("ternary_grid_comp_dyn_eq.csv",data,delimiter=",",header="Ag,Au,Cu,Pd,Pt,active sites (P_NO=1,active sites (P_NO=0.2)",comments="",fmt=["%.2f","%.2f","%.2f","%.2f","%.2f","%.6f","%.6f"])


    