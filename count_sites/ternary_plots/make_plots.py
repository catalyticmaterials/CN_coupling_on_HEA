import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import sys
sys.path.append('..')
import scripts.plotting_ternary_functions as pt

metals = ['Ag','Au', 'Cu', 'Pd','Pt']


for method in ['equilibrium','dynamic','MC']:

    data = np.loadtxt(f'{method}/ternary_grid.csv',delimiter=',',skiprows=1)

    ternary_grid_mf_all = data[:,:5]
    n_sites = data[:,5:]

    # Get max sites at each pressure
    max_sites1 = np.max(n_sites[:,0])
    max_sites05 = np.max(n_sites[:,1])
    max_sites01 = np.max(n_sites[:,2])
    

    # Loop through all ternary alloys
    for elements in combinations(metals,3):
        # Get only molar fractions from quinary grid, which are in the ternary alloy
        excl_ind = [i for i in range(5) if metals[i] not in elements]
        mask = (ternary_grid_mf_all[:,excl_ind[0]]==0) * (ternary_grid_mf_all[:,excl_ind[1]]==0)
        ternary_grid_mf = ternary_grid_mf_all[mask]
        n_sites1,n_sites05,n_sites01 = n_sites[mask].T
        ternary_grid_mf = np.delete(ternary_grid_mf,excl_ind,axis=1)
        ternary_grid = pt.molar_fractions_to_cartesians(ternary_grid_mf)

        # Make directory of the alloy name
        alloy=''.join(elements)
        folder_name = f'ternary_plots/{alloy}'
        if os.path.isdir(folder_name)==False:
            os.makedirs(folder_name)

        
        # Make a plot for each pressure
        fig,ax = pt.make_plot(ternary_grid, n_sites1, elements,colorbar=True,vmax=max_sites1)
        plt.savefig(f'{folder_name}/{alloy}_{method}_PNO_1.png')
        plt.close()

        fig,ax = pt.make_plot(ternary_grid, n_sites05, elements,colorbar=True,vmax=max_sites05)
        plt.savefig(f'{folder_name}/{alloy}_{method}_PNO_05.png')
        plt.close()

        fig,ax = pt.make_plot(ternary_grid, n_sites01, elements,colorbar=True,vmax=max_sites01)
        plt.savefig(f'{folder_name}/{alloy}_{method}_PNO_01.png')
        plt.close()
        