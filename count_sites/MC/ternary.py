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


# Set seed for reproduceability
np.random.seed(42)

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Get all quinary molar fractions with 0.05 spacing
quinary_grid_mf = get_molar_fractions(0.05,5)

ternary_mask = np.sum(quinary_grid_mf == 0,axis=1)>=2 

ternary_grid_mf = quinary_grid_mf[ternary_mask]

# Initiate arrays
n_sites1=np.empty(0)
n_sites05=np.empty(0)
n_sites01=np.empty(0)
for composition in tqdm(ternary_grid_mf):
        n_sites1 = np.append(n_sites1,count_sites(composition, P_CO=1.0, P_NO=1.0,metals=metals,method='mc'))
        n_sites05 = np.append(n_sites05,count_sites(composition, P_CO=1.0, P_NO=0.5,metals=metals,method='mc'))
        n_sites01 = np.append(n_sites01,count_sites(composition, P_CO=1.0, P_NO=0.1,metals=metals,method='mc'))


data = np.hstack((ternary_grid_mf,n_sites1.reshape(-1,1),n_sites05.reshape(-1,1),n_sites01.reshape(-1,1)))


np.savetxt('MC/ternary_grid.csv',data,delimiter=',',fmt=['%0.2f','%0.2f','%0.2f','%0.2f','%0.2f','%.4f','%.4f','%.4f'],header='Ag,Au,Cu,Pd,Pt,P_NO=1,P_NO=05,P_NO=01')