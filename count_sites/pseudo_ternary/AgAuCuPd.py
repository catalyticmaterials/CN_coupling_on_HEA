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
methods = ['eq','dyn','mc']

# Get all quinary molar fractions with 0.05 spacing
quaternary_grid_mf = get_molar_fractions(0.05,4)
quinary_grid_mf = np.hstack((quaternary_grid_mf,np.zeros((len(quaternary_grid_mf),1))))

print(len(quaternary_grid_mf))

# Initiate arrays
n_sites1=np.empty((0,3))
n_sites01=np.empty((0,3))
for composition in tqdm(quinary_grid_mf):
        n_sites1 = np.vstack((n_sites1,[count_sites(composition, P_CO=1.0, P_NO=1.0,metals=metals,method=m) for m in methods]))
        n_sites01 = np.vstack((n_sites01,[count_sites(composition, P_CO=1.0, P_NO=0.1,metals=metals,method=m) for m in methods]))


data = np.hstack((quaternary_grid_mf,n_sites1,n_sites01))


np.savetxt('pseudo_ternary/AgAuCuPd.csv',data,delimiter=',',fmt=['%0.2f','%0.2f','%0.2f','%0.2f','%.4f','%.4f','%.4f','%.4f','%.4f','%.4f'],header='Ag,Au,Cu,Pd,eq1,dyn1,mc1,eq01,dyn01,mc01')