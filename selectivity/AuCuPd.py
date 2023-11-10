import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('..')
from scripts import metals
from scripts.methods import count_selectivity
from scripts import get_molar_fractions
from scripts.plotting_ternary_functions import make_plot, molar_fractions_to_cartesians

# Set seed CO pressure and method
np.random.seed(42)
P_CO = 1
method = 'mc'

# Get ternary grid and append zeros to make composition quinary
grid_mf = get_molar_fractions(0.05,3)
grid_mf_quinary = np.hstack((np.zeros((len(grid_mf),1)),grid_mf,np.zeros((len(grid_mf),1))))

# Get results for each composition
results_1 = np.empty((0,4))
results_01 = np.empty((0,4))
for composition in tqdm(grid_mf_quinary):
    results_1 = np.vstack((results_1,count_selectivity(composition,P_CO,1, metals, method)))
    results_01 = np.vstack((results_01,count_selectivity(composition,P_CO,0.1, metals, method)))

# Save results
None_array = np.full((len(results_1),1),'')
data = np.hstack((grid_mf,None_array,results_1,None_array,results_01))

np.savetxt('AuCuPd_mc_selectivity.csv',data,delimiter=',',fmt="%s",
           header = 'Au,Cu,Pd,P_NO=1,H,NO_NH3,NO_CN,CO_NO_pairs,P_NO=0.1,H,NO_NH3,NO_CN,CO_NO_pairs')
