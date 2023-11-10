import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import sys
sys.path.append('..')
import scripts.plotting_ternary_functions as pt

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Load data
data = np.loadtxt('pseudo_ternary/AgAuCuPd.csv',delimiter=',',skiprows=1)
mfs = data[:,:4]
eq1,dyn1,mc1,eq01,dyn01,mc01 = data[:,4:].T


# Combine Ag and Au
mfs_combined = np.hstack((mfs[:,:2].sum(axis=1).reshape(-1,1),mfs[:,2:]))

ternary_mfs = np.unique(mfs_combined,axis=0)
ternary_grid = pt.molar_fractions_to_cartesians(ternary_mfs)

for y,name in zip(data[:,4:].T,['eq1','dyn1','mc1','eq01','dyn01','mc01']):

    # Get the best y of each unique row
    y_best = [np.max(y[np.all(mfs_combined==row,axis=1)]) for row in ternary_mfs]
    
    fig,ax = pt.make_plot(ternary_grid, y_best, elements=['AgAu','Cu','Pd'],colorbar=True,vmax=np.max(y_best))
    # plt.subplots_adjust(left=0.2)
    # Get the current position of the plot in figure coordinates
    pos = ax.get_position()

    # Set the new position
    ax.set_position([pos.x0 + 0.02, pos.y0, pos.width, pos.height])
    plt.savefig(f'pseudo_ternary/AgAuCuPd_{name}.png',dpi=600)
    plt.close()
