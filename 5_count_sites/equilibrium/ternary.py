import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
from scripts.compositionspace_functions import get_molar_fractions, molar_fractions_to_cartesians
import scripts.plotting_ternary_functions as pt
from scripts.methods import count_sites


metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Get all quinary molar fractions with 0.05 spacing
quinary_grid_mf = get_molar_fractions(0.05,5)

# Prepare ternary grid used in plotting
ternary_grid_mf = get_molar_fractions(0.05,3)
ternary_grid = molar_fractions_to_cartesians(ternary_grid_mf).T


# Loop through all ternary alloys
for elements in combinations(metals,3):
    # Get only molar fractions from quinary grid, which are in the ternary alloy
    excl_ind = [i for i in range(5) if metals[i] not in elements]
    mask = (quinary_grid_mf[:,excl_ind[0]]==0) * (quinary_grid_mf[:,excl_ind[1]]==0)
    ternary_grid_mf = quinary_grid_mf[mask]

    # Set seed for reproduceability
    np.random.seed(42)

    # Initiate arrays
    n_sites1=np.empty(0)
    n_sites05=np.empty(0)
    n_sites01=np.empty(0)
    # Go through all compositions in the ternary grid for each pressure
    for composition in tqdm(ternary_grid_mf):
        n_sites1 = np.append(n_sites1,count_sites(composition, P_CO=1.0, P_NO=1.0,metals=metals,method='eq'))
        n_sites05 = np.append(n_sites05,count_sites(composition, P_CO=1.0, P_NO=0.5,metals=metals,method='eq'))
        n_sites01 = np.append(n_sites01,count_sites(composition, P_CO=1.0, P_NO=0.1,metals=metals,method='eq'))

    # Make directory of the alloy name
    alloy=''.join(elements)
    if os.path.isdir(f"ternary_plots/{alloy}")==False:
        os.makedirs(f"ternary_plots/{alloy}")

    # Make a plot for each pressure
    fig,ax = pt.make_plot(ternary_grid, n_sites1, elements,colorbar=True)
    plt.savefig(f'ternary_plots/{alloy}/{alloy}_dynamic_PNO_1.png')
    plt.close()

    fig,ax = pt.make_plot(ternary_grid, n_sites05, elements,colorbar=True)
    plt.savefig(f'ternary_plots/{alloy}/{alloy}_dynamic_PNO_05.png')
    plt.close()

    fig,ax = pt.make_plot(ternary_grid, n_sites01, elements,colorbar=True)
    plt.savefig(f'ternary_plots/{alloy}/{alloy}_dynamic_PNO_01.png')
    plt.close()

