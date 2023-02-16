import numpy as np
from itertools import combinations
import sys
sys.path.append("..")
from shared_params.plotting_ternary_functions import molar_fractions_to_cartesians, make_plot


data = np.loadtxt("ternary_grid_comp_dyn_eq.csv",delimiter=",",skiprows=1)

mfs = data[:,:5]
active_sites_p1 = data[:,5]
active_sites_p02 = data[:,6]

metals = ['Ag','Au', 'Cu', 'Pd','Pt']
for comp in combinations(metals, 3):
    idx = [i for i in range(5) if metals[i] not in comp]
    mask = (mfs[:,idx[0]]==0)*(mfs[:,idx[1]]==0)
    
    ternary_grid = mfs[mask]
    ternary_grid = np.delete(ternary_grid,idx,axis=1)
    
    grid=molar_fractions_to_cartesians(ternary_grid)
    
    make_plot(grid,active_sites_p1[mask],comp,colorbar=True)
    make_plot(grid,active_sites_p02[mask],comp,colorbar=True)
